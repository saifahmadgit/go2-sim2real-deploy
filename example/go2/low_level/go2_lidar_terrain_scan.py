#!/usr/bin/env python3
"""
go2_lidar_terrain_scan.py — v3 (Edge Detection + Geometric Extrapolation)
==========================================================================
LIDAR terrain scanner for Go2 RL policy deployment on stairs.

CORE IDEA:
  Don't try to bin sparse far-field LIDAR points into a grid.
  Instead:
    1. Use the DENSE near-field points to precisely detect step edges
    2. Given known stair_depth and stair_height, extrapolate the rest
    3. Fill the 5×3 grid from pure geometry

  This is much more robust because:
  - Near-field has 500+ points (reliable edge detection)
  - Far-field has <10 points (unreliable binning)
  - Stair geometry is KNOWN and CONSTANT

ALGORITHM:
  1. Transform LIDAR → body frame (with known LIDAR-to-base offset)
  2. Gravity-project z using IMU roll/pitch
  3. Build fine 1D z-profile (1cm bins, forward direction)
  4. Detect step edges (z-jumps ≈ stair_height)
  5. For each grid x-point, compute which step it's on → height
  6. Output: terrain_z - base_z (matching training exactly)

COORDINATE FRAME MATCHING:
  Training computes:  value = terrain_z_world - base_z_world

  We compute the same by:
  - Transforming LIDAR points to base_link frame (known offset)
  - Gravity-projecting z gives (point_world_z - base_world_z)
  - For a ground point, this IS (terrain_z - base_z)
  - The LIDAR offset ensures body-frame origin = base_link

Usage:
  python3 go2_lidar_terrain_scan.py              # default DDS
  python3 go2_lidar_terrain_scan.py eth0          # specific interface
"""

import math
import os
import sys
import time

import numpy as np

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_

# ================================================================
#  STAIR PARAMETERS — SET THESE FOR YOUR STAIRS
# ================================================================
STAIR_DEPTH = 0.25  # meters — horizontal depth of one tread
STAIR_HEIGHT = 0.15  # meters — vertical rise of one step

# ================================================================
#  ROBOT GEOMETRY — LIDAR position relative to base_link
# ================================================================
# base_link is at the robot's body center.
# LIDAR is under the chin, forward and slightly below.
# Measure these on your robot or calibrate from flat ground.
LIDAR_X_FROM_BASE = 0.28  # meters forward (positive = forward)
LIDAR_Z_FROM_BASE = -0.02  # meters vertical (negative = below base)

# Robot standing height (base_link above ground in default stance)
# This is used as initial guess; calibration corrects it.
NOMINAL_STANDING_HEIGHT = 0.35

# ================================================================
#  GRID — must match training EXACTLY
# ================================================================
X_POINTS = [0.0, 0.15, 0.30, 0.50, 0.80]
Y_POINTS = [-0.15, 0.0, 0.15]
NX = len(X_POINTS)
NY = len(Y_POINTS)
NUM_CELLS = NX * NY  # 15

# Cell ordering: meshgrid(x, y, indexing='ij').reshape(-1)
_gx, _gy = np.meshgrid(X_POINTS, Y_POINTS, indexing="ij")
CELL_X = _gx.reshape(-1)
CELL_Y = _gy.reshape(-1)

# ================================================================
#  EDGE DETECTION CONFIG
# ================================================================
PROFILE_X_MIN = -0.05  # start of 1D profile (slightly behind feet)
PROFILE_X_MAX = 0.50  # end (covers most of grid)
PROFILE_BIN_SIZE = 0.01  # 1cm resolution
Y_CAPTURE_HALF = 0.20  # lateral capture band

# Edge is a z-jump close to stair_height
EDGE_Z_THRESHOLD_MIN = STAIR_HEIGHT * 0.4  # must jump at least 40% of step
EDGE_Z_THRESHOLD_MAX = STAIR_HEIGHT * 1.8  # but not more than 180%
EDGE_MIN_POINTS_PER_BIN = 3  # bins need this many points

# Near-field ground reference (for measuring current standing z)
GROUND_REF_X_MIN = -0.05
GROUND_REF_X_MAX = 0.08  # within ~8cm of body center

# ================================================================
#  HEIGHT PROCESSING
# ================================================================
HEIGHT_CLIP = 1.0
HEIGHT_SCALE = 1.0
CALIBRATION_FILE = "lidar_calibration.npz"
CALIBRATION_DURATION = 5.0
CAPTURE_DURATION = 3.0

# ================================================================
#  DDS
# ================================================================
CLOUD_TOPIC = "rt/utlidar/cloud"
STATE_TOPIC = "rt/sportmodestate"
STRIDE = 1

# ================================================================
#  Z FILTERING
# ================================================================
Z_MIN = -2.0
Z_MAX = 1.0


# ================================================================
#  PRINT GRID INFO
# ================================================================
print(f"Stair params: depth={STAIR_DEPTH}m, height={STAIR_HEIGHT}m")
print(f"LIDAR offset from base: x={LIDAR_X_FROM_BASE}m, z={LIDAR_Z_FROM_BASE}m")
print(f"Grid: {NX}x{NY} = {NUM_CELLS} cells")
print(
    f"Edge detection: profile {PROFILE_X_MIN}→{PROFILE_X_MAX}m, "
    f"bin={PROFILE_BIN_SIZE * 100:.0f}cm"
)
print(
    f"Edge threshold: [{EDGE_Z_THRESHOLD_MIN * 100:.1f}, "
    f"{EDGE_Z_THRESHOLD_MAX * 100:.1f}]cm jump"
)


# ================================================================
#  POINT CLOUD DECODE
# ================================================================
_DTYPES = {5: np.int32, 6: np.uint32, 7: np.float32, 8: np.float64}


def decode_xyz(msg):
    offsets = {}
    for f in msg.fields:
        if f.name in ("x", "y", "z"):
            offsets[f.name] = (int(f.offset), _DTYPES.get(int(f.datatype), np.float32))
    if len(offsets) < 3:
        return np.zeros((0, 3), dtype=np.float32)

    dtype = np.dtype(
        {
            "names": ["x", "y", "z"],
            "formats": [offsets["x"][1], offsets["y"][1], offsets["z"][1]],
            "offsets": [offsets["x"][0], offsets["y"][0], offsets["z"][0]],
            "itemsize": int(msg.point_step),
        }
    )

    data = bytes(msg.data)
    num_pts = int(msg.width) * max(1, int(msg.height))
    if len(data) < num_pts * int(msg.point_step):
        num_pts = len(data) // int(msg.point_step)
    if num_pts == 0:
        return np.zeros((0, 3), dtype=np.float32)

    pts = np.frombuffer(data[: num_pts * int(msg.point_step)], dtype=dtype)[::STRIDE]
    return np.stack([pts["x"], pts["y"], pts["z"]], axis=1).astype(np.float32)


# ================================================================
#  IMU
# ================================================================
def quat_to_rpy(w, x, y, z):
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    sinp = max(-1.0, min(1.0, 2 * (w * y - z * x)))
    pitch = math.asin(sinp)
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def gravity_z_row(roll, pitch):
    """Third row of body-to-world rotation matrix.
    Projects body-frame point onto gravity axis."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    return np.array([-sp, cp * sr, cp * cr], dtype=np.float64)


# ================================================================
#  TERRAIN SCANNER
# ================================================================


class TerrainScanner:
    def __init__(self):
        self.calibration_ground_z = -NOMINAL_STANDING_HEIGHT
        self.calibrated = False

        # Profile bins (precompute)
        self.profile_edges = np.arange(
            PROFILE_X_MIN, PROFILE_X_MAX + PROFILE_BIN_SIZE, PROFILE_BIN_SIZE
        )
        self.profile_centers = (self.profile_edges[:-1] + self.profile_edges[1:]) / 2
        self.n_bins = len(self.profile_centers)

    def _lidar_to_body(self, pts_lidar):
        """Transform LIDAR frame → base_link body frame."""
        bx = -pts_lidar[:, 0] + LIDAR_X_FROM_BASE
        by = pts_lidar[:, 1]
        bz = -pts_lidar[:, 2] + LIDAR_Z_FROM_BASE
        return bx, by, bz

    def _build_profile(self, bx, z_grav, by):
        """
        Build fine 1D z-profile along x (forward direction).

        Uses ALL points within ±Y_CAPTURE_HALF lateral band.
        Each 1cm bin gets median z from all points in it.

        Returns:
            profile_z: (n_bins,) — median z per bin (NaN if empty)
            profile_n: (n_bins,) — point count per bin
        """
        y_ok = (by >= -Y_CAPTURE_HALF) & (by <= Y_CAPTURE_HALF)
        bx_f = bx[y_ok]
        zg_f = z_grav[y_ok]

        profile_z = np.full(self.n_bins, np.nan)
        profile_n = np.zeros(self.n_bins, dtype=np.int32)

        # Digitize: which bin does each point fall into
        bin_idx = np.digitize(bx_f, self.profile_edges) - 1
        valid_bin = (bin_idx >= 0) & (bin_idx < self.n_bins)

        for b in range(self.n_bins):
            mask = valid_bin & (bin_idx == b)
            n = np.sum(mask)
            profile_n[b] = n
            if n >= EDGE_MIN_POINTS_PER_BIN:
                profile_z[b] = np.median(zg_f[mask])

        return profile_z, profile_n

    def _detect_edges(self, profile_z):
        """
        Detect step edges as z-discontinuities in the profile.

        Returns list of (x_position, z_jump) tuples.
        z_jump > 0 = ascending step, < 0 = descending.
        """
        edges = []
        valid = ~np.isnan(profile_z)

        # Compute z-difference between consecutive valid bins
        for i in range(1, self.n_bins):
            if valid[i] and valid[i - 1]:
                dz = profile_z[i] - profile_z[i - 1]

                if abs(dz) >= EDGE_Z_THRESHOLD_MIN and abs(dz) <= EDGE_Z_THRESHOLD_MAX:
                    # Edge detected between bin i-1 and bin i
                    edge_x = (self.profile_centers[i - 1] + self.profile_centers[i]) / 2
                    edges.append((edge_x, dz))

        return edges

    def _measure_ground_z(self, bx, z_grav, by):
        """
        Measure current standing surface height from near-field points.

        Returns median z_grav of ground points near the robot.
        This value ≈ -(standing_height) on flat ground.
        """
        mask = (
            (bx >= GROUND_REF_X_MIN)
            & (bx <= GROUND_REF_X_MAX)
            & (by >= -Y_CAPTURE_HALF)
            & (by <= Y_CAPTURE_HALF)
        )
        n = np.sum(mask)
        if n < 5:
            return self.calibration_ground_z, n
        return float(np.median(z_grav[mask])), n

    def compute_scan(self, points_lidar, roll=0.0, pitch=0.0):
        """
        Compute 15-value terrain scan using edge detection + geometry.

        Steps:
          1. Transform to body frame (with LIDAR offset)
          2. Gravity-project z
          3. Measure current ground z (near-field)
          4. Build 1D profile, detect step edges
          5. For each grid x: determine which step → compute height
          6. Replicate to all 3 Y columns

        Returns:
          scan: (15,) float32
          info: dict with diagnostics
        """
        scan = np.zeros(NUM_CELLS, dtype=np.float64)
        info = {
            "ground_z": self.calibration_ground_z,
            "ground_n": 0,
            "edges": [],
            "profile_z": None,
            "profile_n": None,
        }

        if len(points_lidar) == 0:
            scan[:] = self.calibration_ground_z
            return self._finalize(scan), info

        # --- Transform and filter ---
        bx, by, bz = self._lidar_to_body(points_lidar)

        valid = (bz > Z_MIN) & (bz < Z_MAX) & np.all(np.isfinite(points_lidar), axis=1)
        bx, by, bz = bx[valid], by[valid], bz[valid]

        if len(bx) == 0:
            scan[:] = self.calibration_ground_z
            return self._finalize(scan), info

        # --- Gravity projection ---
        r2 = gravity_z_row(roll, pitch)
        z_grav = r2[0] * bx + r2[1] * by + r2[2] * bz

        # --- Measure current ground level ---
        ground_z, ground_n = self._measure_ground_z(bx, z_grav, by)
        info["ground_z"] = ground_z
        info["ground_n"] = ground_n

        # --- Build fine profile and detect edges ---
        profile_z, profile_n = self._build_profile(bx, z_grav, by)
        edges = self._detect_edges(profile_z)
        info["profile_z"] = profile_z
        info["profile_n"] = profile_n
        info["edges"] = edges

        # --- Construct grid from geometry ---
        for xi in range(NX):
            x = X_POINTS[xi]

            if len(edges) == 0:
                # No step detected — flat terrain
                h = ground_z
            else:
                # Start from current ground, accumulate step transitions
                h = ground_z
                for edge_x, edge_dz in edges:
                    if x >= edge_x:
                        # This grid point is past this edge
                        h += edge_dz

                        # Extrapolate further steps with known geometry
                        # How far past the edge are we?
                        dx_past = x - edge_x
                        direction = 1.0 if edge_dz > 0 else -1.0
                        step_h = abs(edge_dz)  # actual measured step height

                        # Additional full steps beyond the first
                        extra_steps = int(dx_past / STAIR_DEPTH)
                        h += extra_steps * step_h * direction

                        break  # Use first edge only (closest)

            # Replicate to all 3 Y columns
            for yi in range(NY):
                scan[xi * NY + yi] = h

        return self._finalize(scan), info

    def _finalize(self, scan):
        """Clip and scale."""
        scan = np.clip(scan, -HEIGHT_CLIP, HEIGHT_CLIP)
        scan *= HEIGHT_SCALE
        return scan.astype(np.float32)

    def calibrate(self, scan_list, ground_z_list):
        """
        Calibrate from flat-ground measurements.

        Determines the precise ground_z value on flat ground.
        This corrects for any LIDAR offset error, mounting tilt, etc.
        """
        if len(ground_z_list) < 5:
            print("[CALIB] ERROR: Need more measurements")
            return False

        measured_ground_z = float(np.median(ground_z_list))
        self.calibration_ground_z = measured_ground_z
        self.calibrated = True

        expected = -NOMINAL_STANDING_HEIGHT
        error = measured_ground_z - expected

        print("\n" + "=" * 60)
        print("  CALIBRATION COMPLETE")
        print("=" * 60)
        print(f"  Ground z measurements : {len(ground_z_list)}")
        print(f"  Measured ground z     : {measured_ground_z:.4f} m")
        print(f"  Expected              : {expected:.4f} m")
        print(f"  Error                 : {error * 100:+.1f} cm")

        if abs(error) > 0.05:
            print(
                f"\n  NOTE: {abs(error) * 100:.1f}cm error suggests "
                f"LIDAR_Z_FROM_BASE may need adjustment"
            )
            print(f"    Current: {LIDAR_Z_FROM_BASE:.3f}m")
            print(f"    Try:     {LIDAR_Z_FROM_BASE - error:.3f}m")
        else:
            print("  Status: good (<5cm error)")

        # Check scan consistency
        if len(scan_list) > 0:
            stacked = np.stack(scan_list, axis=0)
            spread = np.max(stacked) - np.min(stacked)
            print(f"  Scan spread: {spread * 100:.1f}cm across all frames")

        print("=" * 60 + "\n")

        # Save
        try:
            np.savez(
                CALIBRATION_FILE,
                ground_z=measured_ground_z,
                standing_height=NOMINAL_STANDING_HEIGHT,
                lidar_x=LIDAR_X_FROM_BASE,
                lidar_z=LIDAR_Z_FROM_BASE,
                stair_depth=STAIR_DEPTH,
                stair_height=STAIR_HEIGHT,
            )
            print(f"  Saved to {CALIBRATION_FILE}")
        except Exception as e:
            print(f"  Save failed: {e}")

        return True

    def load_calibration(self):
        if not os.path.exists(CALIBRATION_FILE):
            return False
        try:
            data = np.load(CALIBRATION_FILE)
            self.calibration_ground_z = float(data["ground_z"])
            self.calibrated = True
            print(
                f"[CALIB] Loaded: ground_z={self.calibration_ground_z:.4f}m "
                f"(from {CALIBRATION_FILE})"
            )
            return True
        except Exception as e:
            print(f"[CALIB] Load failed: {e}")
            return False


# ================================================================
#  DDS CALLBACKS
# ================================================================
latest_cloud_msg = None
cloud_count = 0
imu_roll = 0.0
imu_pitch = 0.0
imu_yaw = 0.0
imu_count = 0


def _on_cloud(msg):
    global latest_cloud_msg, cloud_count
    latest_cloud_msg = msg
    cloud_count += 1


def _on_state(msg):
    global imu_roll, imu_pitch, imu_yaw, imu_count
    try:
        imu = msg.imu_state
        rpy = imu.rpy
        if rpy is not None and len(rpy) >= 3:
            imu_roll = float(rpy[0])
            imu_pitch = float(rpy[1])
            imu_yaw = float(rpy[2])
            imu_count += 1
            return
        quat = imu.quaternion
        if quat is not None and len(quat) >= 4:
            imu_roll, imu_pitch, imu_yaw = quat_to_rpy(
                float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
            )
            imu_count += 1
    except Exception:
        pass


# ================================================================
#  INTERACTIVE FLOW
# ================================================================


def run_calibration(scanner):
    global latest_cloud_msg, cloud_count

    scan_list = []
    ground_z_list = []
    t0 = time.time()
    last_cc = cloud_count

    while time.time() - t0 < CALIBRATION_DURATION:
        if latest_cloud_msg is not None and cloud_count > last_cc:
            last_cc = cloud_count
            xyz = decode_xyz(latest_cloud_msg)
            if len(xyz) > 100:
                scan, info = scanner.compute_scan(xyz, imu_roll, imu_pitch)
                if info["ground_n"] >= 10:
                    scan_list.append(scan.copy())
                    ground_z_list.append(info["ground_z"])

                elapsed = time.time() - t0
                pct = elapsed / CALIBRATION_DURATION * 100
                print(
                    f"\r  {pct:5.1f}%  frames={len(scan_list)}  "
                    f"ground_z={info['ground_z']:+.4f}  "
                    f"pts={info['ground_n']}  "
                    f"roll={math.degrees(imu_roll):+.1f} "
                    f"pitch={math.degrees(imu_pitch):+.1f}",
                    end="",
                    flush=True,
                )
        time.sleep(0.02)

    print()
    return scanner.calibrate(scan_list, ground_z_list)


def run_measurement(scanner, num):
    global latest_cloud_msg, cloud_count

    frames = []
    t0 = time.time()
    last_cc = cloud_count

    print(f"  Capturing for {CAPTURE_DURATION:.0f}s", end="", flush=True)

    while time.time() - t0 < CAPTURE_DURATION:
        if latest_cloud_msg is not None and cloud_count > last_cc:
            last_cc = cloud_count
            xyz = decode_xyz(latest_cloud_msg)
            if len(xyz) > 100:
                scan, info = scanner.compute_scan(xyz, imu_roll, imu_pitch)
                frames.append((scan.copy(), info))
                print(".", end="", flush=True)
        time.sleep(0.02)

    print(f" done! ({len(frames)} frames)\n")

    if len(frames) < 3:
        print("  Too few frames.\n")
        return

    # --- Aggregate ---
    all_scans = np.stack([f[0] for f in frames], axis=0)
    mean_scan = np.mean(all_scans, axis=0)
    std_scan = np.std(all_scans, axis=0)

    # Aggregate edges across frames
    all_edges = []
    ground_zs = []
    for _, info in frames:
        all_edges.extend(info["edges"])
        ground_zs.append(info["ground_z"])

    # --- Print results ---
    print("=" * 72)
    print(f"  MEASUREMENT #{num}  ({len(frames)} frames)")
    print("=" * 72)

    # Ground reference
    mean_gz = np.mean(ground_zs)
    print(
        f"\n  Ground z (near-field): {mean_gz:+.4f} m  "
        f"(= ~{-mean_gz * 100:.1f}cm below base)"
    )

    # Edges
    if all_edges:
        edge_xs = [e[0] for e in all_edges]
        edge_dzs = [e[1] for e in all_edges]
        mean_edge_x = np.mean(edge_xs)
        mean_edge_dz = np.mean(edge_dzs)
        direction = "ASCENDING" if mean_edge_dz > 0 else "DESCENDING"
        print(
            f"  Step edge detected: x={mean_edge_x:.3f}m, "
            f"dz={mean_edge_dz:+.3f}m ({direction})"
        )
        print(
            f"    Edge consistency: x_std={np.std(edge_xs) * 100:.1f}cm, "
            f"dz_std={np.std(edge_dzs) * 100:.1f}cm "
            f"({len(all_edges)} detections in {len(frames)} frames)"
        )
    else:
        print("  No step edges detected (flat terrain)")

    # Grid table
    print("\n  5×3 Grid values (all Y columns identical — stair symmetry):")
    hdr = "          "
    for yv in Y_POINTS:
        hdr += f"   y={yv:+.2f}  "
    print(hdr)

    for i in range(NX):
        row = f"  x={X_POINTS[i]:.2f}:"
        for j in range(NY):
            idx = i * NY + j
            m = mean_scan[idx]
            row += f"  {m:+.4f}  "

        stds = [std_scan[i * NY + j] for j in range(NY)]
        row += f"  std={np.mean(stds):.4f}"
        print(row)

    # Forward profile
    print("\n  Forward profile:")
    for i in range(NX):
        v = mean_scan[i * NY]  # same for all Y
        rel = v - mean_scan[0]  # relative to x=0.00
        bar_n = int(round(abs(rel) * 200))

        if abs(rel) < 0.02:
            tag = "flat"
            bar = "─" * min(bar_n, 30)
        elif rel > 0:
            tag = f"+{rel * 100:5.1f}cm"
            bar = "█" * min(bar_n, 30)
        else:
            tag = f"{rel * 100:+5.1f}cm"
            bar = "▼" * min(bar_n, 30)

        print(f"    x={X_POINTS[i]:.2f}: {v:+.4f}  {tag:>8s}  {bar}")

    # Network vector
    print(f"\n  Network vector ({NUM_CELLS} values):")
    vec = mean_scan.tolist()
    print(f"  [{', '.join(f'{v:+.3f}' for v in vec)}]")

    # Quality
    max_std = np.max(std_scan)
    print(f"\n  Max std across cells: {max_std * 100:.2f}cm")
    if max_std < 0.01:
        print("  Quality: EXCELLENT (< 1cm variation)")
    elif max_std < 0.03:
        print("  Quality: GOOD (< 3cm variation)")
    else:
        print("  Quality: CHECK — high variation")

    print("=" * 72)

    # Profile visualization (if available from last frame)
    last_info = frames[-1][1]
    if last_info["profile_z"] is not None:
        pz = last_info["profile_z"]
        pn = last_info["profile_n"]
        print("\n  1D Profile (1cm bins, last frame):")
        # Show every 2cm for compact display
        for b in range(0, len(pz), 2):
            xc = PROFILE_X_MIN + b * PROFILE_BIN_SIZE + PROFILE_BIN_SIZE / 2
            if xc > PROFILE_X_MAX:
                break
            if np.isnan(pz[b]):
                bar = "  ?"
            else:
                rel = pz[b] - mean_gz
                if abs(rel) < 0.01:
                    bar = "─"
                elif rel > 0:
                    bar = "█" * min(int(rel * 100), 20)
                else:
                    bar = "▼" * min(int(-rel * 100), 10)
            print(f"    x={xc:+.2f}: n={pn[b]:4d}  {bar}")


def main():
    global latest_cloud_msg, cloud_count, imu_count

    print("\n" + "=" * 55)
    print("  GO2 LIDAR TERRAIN SCANNER v3")
    print("  Edge Detection + Geometric Extrapolation")
    print("=" * 55)

    # --- DDS ---
    if len(sys.argv) > 1 and sys.argv[1] not in ("--help", "-h"):
        iface = sys.argv[1]
        ChannelFactoryInitialize(0, iface)
        print(f"[INIT] DDS: {iface}")
    else:
        ChannelFactoryInitialize(0)
        print("[INIT] DDS: default")

    cloud_sub = ChannelSubscriber(CLOUD_TOPIC, PointCloud2_)
    cloud_sub.Init(_on_cloud, 10)
    state_sub = ChannelSubscriber(STATE_TOPIC, SportModeState_)
    state_sub.Init(_on_state, 10)

    print("\nWaiting for data...", end="", flush=True)
    t0 = time.time()
    while True:
        if cloud_count > 0:
            if imu_count > 0:
                print(f" OK (cloud={cloud_count}, imu={imu_count})")
                break
            elif time.time() - t0 > 8.0:
                print("\n[WARN] No IMU, continuing without tilt correction")
                break
        if time.time() - t0 > 15.0:
            print("\n[ERROR] No LIDAR data")
            sys.exit(1)
        time.sleep(0.3)
        print(".", end="", flush=True)

    scanner = TerrainScanner()
    loaded = scanner.load_calibration()

    if loaded:
        print(f"\nCalibration loaded. Delete {CALIBRATION_FILE} to recalibrate.")
    else:
        print()
        print("=" * 55)
        print("  STEP 1: CALIBRATION")
        print("  Place robot on FLAT GROUND, default stance, keep STILL.")
        print("=" * 55)
        input("\n>>> Press ENTER to start calibration...")
        print(f"\n  Collecting for {CALIBRATION_DURATION:.0f}s...")

        if not run_calibration(scanner):
            print("[ERROR] Calibration failed")
            sys.exit(1)

    print()
    print("=" * 55)
    print("  STEP 2: MEASUREMENTS")
    print("  Move robot with remote, press ENTER to measure.")
    print("  Ctrl+C to quit.")
    print("=" * 55)

    num = 0
    try:
        while True:
            num += 1
            input(f"\n>>> Measurement #{num}: position robot, press ENTER...")
            print()
            run_measurement(scanner, num)
    except KeyboardInterrupt:
        print("\n\n[STOP] Done.")


if __name__ == "__main__":
    main()
