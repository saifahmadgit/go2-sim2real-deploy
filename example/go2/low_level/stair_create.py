#!/usr/bin/env python3
"""
go2_stair_scan.py — Final 5×3 Grid for RL Policy
==================================================
Generates the exact 15-value terrain scan your actor network expects.

APPROACH:
  1. LIDAR near-field (0-27cm): detect WHERE the first stair edge is
  2. Known stair geometry (STAIR_HEIGHT, STAIR_DEPTH): construct the rest
  3. Output: terrain_z - base_z for each grid cell (matching training)

MATH:
  Robot stands on step 0. base_z_world = step0_z + standing_height.

  Grid point on step N (ascending):
    terrain_z = step0_z + N * STAIR_HEIGHT
    value = terrain_z - base_z = N * STAIR_HEIGHT - standing_height
    value = ground_z + N * STAIR_HEIGHT
    (because ground_z ≈ -standing_height)

  Grid point on step N (descending):
    value = ground_z - N * STAIR_HEIGHT

  ground_z is measured live from LIDAR near-field points.

COORDINATE FRAMES:
  - LIDAR detects edge at d_lidar (forward from LIDAR)
  - Grid x-points are in base_link frame
  - edge_x_base = d_lidar + LIDAR_X_FROM_BASE
  - Grid point on step N if: x >= edge_x_base + (N-1)*STAIR_DEPTH

  Training uses yaw-only rotation for grid points. Pitch effect on
  x-positions is <2% (cos(10deg)=0.985), negligible.
  z-values use gravity projection, already handled in ground_z.

Usage:
  python3 go2_stair_scan.py              # default DDS
  python3 go2_stair_scan.py eth0          # specific interface
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
#  STAIR PARAMETERS
# ================================================================
STAIR_DEPTH = 0.25  # meters — one tread depth
STAIR_HEIGHT = 0.18  # meters — one riser height

# ================================================================
#  ROBOT GEOMETRY
# ================================================================
LIDAR_X_FROM_BASE = 0.28  # LIDAR is this far forward of base_link
LIDAR_Z_FROM_BASE = -0.02  # LIDAR is this far below base_link (negative = below)

# ================================================================
#  GRID (must match training EXACTLY)
# ================================================================
X_POINTS = [0.0, 0.15, 0.30, 0.50, 0.80]
Y_POINTS = [-0.15, 0.0, 0.15]
NX = len(X_POINTS)
NY = len(Y_POINTS)
NUM_CELLS = NX * NY  # 15

# ================================================================
#  EDGE DETECTION CONFIG
# ================================================================
PROFILE_X_MIN = -0.10
PROFILE_X_MAX = 0.60
BIN_SIZE = 0.01
Y_HALF = 0.20
MIN_PTS_PER_BIN = 3

EDGE_Z_THRESHOLD = 0.04  # 4cm deviation from ground = edge
EDGE_SEARCH_MAX_X = 0.27  # reliable LIDAR range

# Ground reference: near-field bins
GROUND_X_MAX = 0.08

# ================================================================
#  CALIBRATION
# ================================================================
CALIBRATION_DURATION = 5.0
CAPTURE_DURATION = 3.0
CALIBRATION_FILE = "stair_scan_calib.npz"

HEIGHT_CLIP = 1.0
HEIGHT_SCALE = 1.0

# ================================================================
#  DDS
# ================================================================
CLOUD_TOPIC = "rt/utlidar/cloud"
STATE_TOPIC = "rt/sportmodestate"
STRIDE = 1
Z_MIN = -2.0
Z_MAX = 1.0


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
    n = int(msg.width) * max(1, int(msg.height))
    if len(data) < n * int(msg.point_step):
        n = len(data) // int(msg.point_step)
    if n == 0:
        return np.zeros((0, 3), dtype=np.float32)
    pts = np.frombuffer(data[: n * int(msg.point_step)], dtype=dtype)[::STRIDE]
    return np.stack([pts["x"], pts["y"], pts["z"]], axis=1).astype(np.float32)


# ================================================================
#  IMU
# ================================================================
imu_roll = 0.0
imu_pitch = 0.0
imu_yaw = 0.0
imu_count = 0


def _on_state(msg):
    global imu_roll, imu_pitch, imu_yaw, imu_count
    try:
        rpy = msg.imu_state.rpy
        if rpy is not None and len(rpy) >= 3:
            imu_roll, imu_pitch, imu_yaw = float(rpy[0]), float(rpy[1]), float(rpy[2])
            imu_count += 1
    except:
        pass


def gravity_z_row(roll, pitch):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    return np.array([-sp, cp * sr, cp * cr])


# ================================================================
#  CLOUD CALLBACK
# ================================================================
latest_msg = None
msg_count = 0


def _on_cloud(msg):
    global latest_msg, msg_count
    latest_msg = msg
    msg_count += 1


# ================================================================
#  CORE: process one LIDAR frame → 15 values
# ================================================================


def process_frame(pts_lidar, roll, pitch, ground_z_calib):
    """
    Process one LIDAR frame into the 15-value grid.

    Returns:
        scan: (15,) float32 — network input
        info: dict with edge_x_lidar, edge_x_base, edge_dz,
              ground_z, direction, step_assignment
    """
    scan = np.full(NUM_CELLS, ground_z_calib, dtype=np.float64)
    info = {
        "ground_z": ground_z_calib,
        "ground_n": 0,
        "edge_x_lidar": None,
        "edge_x_base": None,
        "edge_dz": None,
        "direction": "flat",
        "steps": [0] * NX,
    }

    if len(pts_lidar) == 0:
        return _clip(scan), info

    # --- LIDAR → body frame (LIDAR frame, no base offset yet) ---
    bx = -pts_lidar[:, 0]
    by = pts_lidar[:, 1]
    bz = -pts_lidar[:, 2]

    valid = (bz > Z_MIN) & (bz < Z_MAX) & np.all(np.isfinite(pts_lidar), axis=1)
    bx, by, bz = bx[valid], by[valid], bz[valid]

    if len(bx) == 0:
        return _clip(scan), info

    # --- Gravity projection ---
    r2 = gravity_z_row(roll, pitch)
    z_grav = r2[0] * bx + r2[1] * by + r2[2] * bz

    # --- Measure ground z from near-field ---
    y_ok = np.abs(by) <= Y_HALF
    gnd_mask = y_ok & (bx >= -0.05) & (bx <= GROUND_X_MAX)
    gnd_n = int(np.sum(gnd_mask))
    if gnd_n >= 5:
        ground_z = float(np.median(z_grav[gnd_mask]))
    else:
        ground_z = ground_z_calib

    info["ground_z"] = ground_z
    info["ground_n"] = gnd_n

    # --- Build profile and detect edge ---
    profile_edges = np.arange(PROFILE_X_MIN, PROFILE_X_MAX + BIN_SIZE, BIN_SIZE)
    profile_centers = (profile_edges[:-1] + profile_edges[1:]) / 2
    n_bins = len(profile_centers)

    filt = y_ok & (bz > Z_MIN) & (bz < Z_MAX)
    bx_f, zg_f = bx[filt], z_grav[filt]

    profile_z = np.full(n_bins, np.nan)
    bin_idx = np.digitize(bx_f, profile_edges) - 1
    for b in range(n_bins):
        mask = bin_idx == b
        if np.sum(mask) >= MIN_PTS_PER_BIN:
            profile_z[b] = np.median(zg_f[mask])

    # Find edge: first bin where z deviates from ground
    edge_x_lidar = None
    edge_dz = None
    valid_z = ~np.isnan(profile_z)

    for b in range(n_bins):
        if not valid_z[b]:
            continue
        if profile_centers[b] > EDGE_SEARCH_MAX_X:
            break
        dz = profile_z[b] - ground_z
        if abs(dz) > EDGE_Z_THRESHOLD:
            edge_x_lidar = profile_centers[b]
            edge_dz = dz
            break

    # --- Construct 5×3 grid ---
    if edge_x_lidar is not None:
        # --- Pitch correction ---
        # The robot pitches on stairs. We need to account for this
        # when converting LIDAR-frame edge to base_link frame.
        #
        # TRAINING uses yaw-only for grid x → world mapping.
        # So grid x IS the body-frame forward distance.
        # We need edge_x_base in the same body-frame-forward sense.
        #
        # LIDAR is at body position (Lx, 0, Lz).
        # Its horizontal forward offset from base when pitched:
        #   horiz = Lx * cos(pitch) - Lz * sin(pitch)
        # (Lz is negative → -Lz*sin(pitch) adds forward when nose-up)
        #
        # The edge at LIDAR distance d has body_x = d (from -lidar_x).
        # Its world horizontal distance from LIDAR = d * cos(pitch)
        # (body x-axis is tilted, so horizontal projection shrinks)
        #
        # Total horizontal from base = d*cos(pitch) + lidar_horiz
        # But training maps body_x directly to horizontal (yaw-only).
        # So edge_x_base in "training body frame" = this horizontal distance.
        #
        # Similarly, one stair tread (STAIR_DEPTH horizontal) appears as
        # STAIR_DEPTH / cos(pitch) in body x. But since training maps
        # body_x = horizontal, the tread in "training body x" IS STAIR_DEPTH.

        cp = math.cos(pitch)
        sp = math.sin(pitch)

        # LIDAR horizontal forward offset from base
        lidar_horiz = LIDAR_X_FROM_BASE * cp - LIDAR_Z_FROM_BASE * sp

        # Edge horizontal distance from base
        edge_x_base = edge_x_lidar * cp + lidar_horiz

        ascending = edge_dz > 0

        info["edge_x_lidar"] = edge_x_lidar
        info["edge_x_base"] = edge_x_base
        info["edge_dz"] = edge_dz
        info["direction"] = "ascending" if ascending else "descending"

        # Stair depth in the same frame as grid x-points
        # Training: body_x ≈ horizontal distance (yaw-only mapping)
        # One tread = STAIR_DEPTH horizontally = STAIR_DEPTH in training body x
        tread_x = STAIR_DEPTH

        for xi in range(NX):
            x = X_POINTS[xi]

            if x < edge_x_base:
                step_n = 0
                h = ground_z
            else:
                dx_past_edge = x - edge_x_base
                step_n = 1 + int(dx_past_edge / tread_x)

                if ascending:
                    h = ground_z + step_n * STAIR_HEIGHT
                else:
                    h = ground_z - step_n * STAIR_HEIGHT

            info["steps"][xi] = step_n if ascending else -step_n

            for yi in range(NY):
                scan[xi * NY + yi] = h
    else:
        # No edge: flat terrain
        for i in range(NUM_CELLS):
            scan[i] = ground_z

    return _clip(scan), info


def _clip(scan):
    return np.clip(scan * HEIGHT_SCALE, -HEIGHT_CLIP, HEIGHT_CLIP).astype(np.float32)


# ================================================================
#  CALIBRATION
# ================================================================


def run_calibration():
    global latest_msg, msg_count

    ground_zs = []
    t0 = time.time()
    last_mc = msg_count

    while time.time() - t0 < CALIBRATION_DURATION:
        if latest_msg is not None and msg_count > last_mc:
            last_mc = msg_count
            xyz = decode_xyz(latest_msg)
            if len(xyz) > 100:
                _, info = process_frame(xyz, imu_roll, imu_pitch, -0.35)
                if info["ground_n"] >= 10:
                    ground_zs.append(info["ground_z"])

                elapsed = time.time() - t0
                pct = elapsed / CALIBRATION_DURATION * 100
                print(
                    f"\r  {pct:5.1f}%  samples={len(ground_zs)}  "
                    f"ground_z={info['ground_z']:+.4f}",
                    end="",
                    flush=True,
                )
        time.sleep(0.02)

    print()

    if len(ground_zs) < 5:
        return None

    gz = float(np.median(ground_zs))
    gz_std = float(np.std(ground_zs))

    print(f"\n  Calibrated ground_z = {gz:+.4f} m (std={gz_std * 100:.2f}cm)")
    print(f"  Standing height ≈ {-gz:.3f} m")

    try:
        np.savez(CALIBRATION_FILE, ground_z=gz)
        print(f"  Saved to {CALIBRATION_FILE}")
    except:
        pass

    return gz


def load_calibration():
    if not os.path.exists(CALIBRATION_FILE):
        return None
    try:
        data = np.load(CALIBRATION_FILE)
        gz = float(data["ground_z"])
        print(f"[CALIB] Loaded ground_z={gz:+.4f} from {CALIBRATION_FILE}")
        return gz
    except:
        return None


# ================================================================
#  MEASUREMENT
# ================================================================


def run_measurement(num, ground_z_calib):
    global latest_msg, msg_count

    frames = []
    t0 = time.time()
    last_mc = msg_count

    print(f"  Capturing {CAPTURE_DURATION:.0f}s", end="", flush=True)

    while time.time() - t0 < CAPTURE_DURATION:
        if latest_msg is not None and msg_count > last_mc:
            last_mc = msg_count
            xyz = decode_xyz(latest_msg)
            if len(xyz) > 100:
                scan, info = process_frame(xyz, imu_roll, imu_pitch, ground_z_calib)
                frames.append((scan, info))
                print(".", end="", flush=True)
        time.sleep(0.02)

    print(f" ({len(frames)} frames)\n")

    if len(frames) < 3:
        print("  Too few frames.\n")
        return

    # --- Aggregate ---
    all_scans = np.stack([f[0] for f in frames], axis=0)
    mean_scan = np.mean(all_scans, axis=0)
    std_scan = np.std(all_scans, axis=0)

    # Aggregate info
    ground_zs = [f[1]["ground_z"] for f in frames]
    edge_xs_base = [
        f[1]["edge_x_base"] for f in frames if f[1]["edge_x_base"] is not None
    ]
    edge_dzs = [f[1]["edge_dz"] for f in frames if f[1]["edge_dz"] is not None]
    directions = [f[1]["direction"] for f in frames]
    last_steps = frames[-1][1]["steps"]

    # Most common direction
    from collections import Counter

    dir_counts = Counter(directions)
    dominant_dir = dir_counts.most_common(1)[0][0]

    mean_gz = np.mean(ground_zs)

    # --- Print ---
    print("=" * 70)
    print(f"  MEASUREMENT #{num}")
    print("=" * 70)

    print(f"\n  Ground z       : {mean_gz:+.4f} m  (standing height ≈ {-mean_gz:.3f}m)")
    print(
        f"  IMU            : roll={math.degrees(imu_roll):+.1f}°  "
        f"pitch={math.degrees(imu_pitch):+.1f}°"
    )
    print(f"  Stair params   : depth={STAIR_DEPTH}m, height={STAIR_HEIGHT}m")

    if edge_xs_base:
        mean_edge_base = np.mean(edge_xs_base)
        mean_edge_dz = np.mean(edge_dzs)
        edge_std = np.std(edge_xs_base)
        detections = len(edge_xs_base)

        print(
            f"\n  Edge detected  : x_base = {mean_edge_base:.3f} m  "
            f"(±{edge_std * 100:.1f}cm, {detections}/{len(frames)} frames)"
        )
        print(f"  Edge dz        : {mean_edge_dz:+.3f} m ({dominant_dir})")
        print(
            f"  Edge x_lidar   : {np.mean([f[1]['edge_x_lidar'] for f in frames if f[1]['edge_x_lidar'] is not None]):.3f} m"
        )
    else:
        print("\n  Edge           : none detected (flat terrain)")

    # Grid table
    print("\n  5×3 GRID OUTPUT (network input):")
    print("  All Y columns identical (stair Y-symmetry)")
    print()

    hdr = "          "
    for yv in Y_POINTS:
        hdr += f"  y={yv:+.2f}  "
    hdr += " step#"
    print(hdr)

    for i in range(NX):
        vals = [mean_scan[i * NY + j] for j in range(NY)]
        stds = [std_scan[i * NY + j] for j in range(NY)]
        step = last_steps[i]

        row = f"  x={X_POINTS[i]:.2f}:"
        for v in vals:
            row += f"  {v:+.4f}  "

        step_str = f"step {step:+d}" if step != 0 else "ground"
        row += f" {step_str}"
        print(row)

    # Verify: values should increase by STAIR_HEIGHT per step
    print("\n  Step-by-step height check:")
    for i in range(NX):
        v = mean_scan[i * NY]  # Y=0 column
        step = last_steps[i]
        expected = (
            mean_gz + step * STAIR_HEIGHT
            if step >= 0
            else mean_gz + step * STAIR_HEIGHT
        )
        error = (v - expected) * 100
        print(
            f"    x={X_POINTS[i]:.2f}: value={v:+.4f}  "
            f"expected={expected:+.4f}  "
            f"error={error:+.1f}cm  step={step}"
        )

    # Network vector
    print("\n  ┌─────────────────────────────────────────────────────────┐")
    print(f"  │  NETWORK INPUT ({NUM_CELLS} values):                              │")
    vec = mean_scan.tolist()
    vec_str = "[" + ", ".join(f"{v:+.3f}" for v in vec) + "]"
    print(f"  │  {vec_str:<57s}│")
    print("  └─────────────────────────────────────────────────────────┘")

    max_std = np.max(std_scan)
    print(f"\n  Max frame-to-frame std: {max_std * 100:.2f}cm")
    print("=" * 70)


# ================================================================
#  MAIN
# ================================================================


def main():
    global latest_msg, msg_count

    print("\n" + "=" * 55)
    print("  GO2 STAIR SCAN — 5×3 Grid Generator")
    print(f"  Stair: depth={STAIR_DEPTH}m, height={STAIR_HEIGHT}m")
    print(f"  LIDAR offset: {LIDAR_X_FROM_BASE}m forward of base")
    print(f"  Edge search:  0 to {EDGE_SEARCH_MAX_X}m (LIDAR frame)")
    print("=" * 55)

    # --- DDS ---
    if len(sys.argv) > 1 and sys.argv[1] not in ("--help", "-h"):
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)

    cloud_sub = ChannelSubscriber(CLOUD_TOPIC, PointCloud2_)
    cloud_sub.Init(_on_cloud, 10)
    state_sub = ChannelSubscriber(STATE_TOPIC, SportModeState_)
    state_sub.Init(_on_state, 10)

    print("\nWaiting for data...", end="", flush=True)
    t0 = time.time()
    while True:
        if msg_count > 0 and imu_count > 0:
            print(" OK")
            break
        if msg_count > 0 and time.time() - t0 > 8:
            print("\n[WARN] No IMU, continuing")
            break
        if time.time() - t0 > 15:
            print("\n[ERROR] No LIDAR")
            sys.exit(1)
        time.sleep(0.3)
        print(".", end="", flush=True)

    # --- Calibration ---
    ground_z = load_calibration()

    if ground_z is None:
        print()
        print("=" * 55)
        print("  STEP 1: CALIBRATION")
        print("  Robot on FLAT GROUND, default stance, keep STILL.")
        print("=" * 55)
        input("\n>>> Press ENTER to calibrate...")
        print()
        ground_z = run_calibration()
        if ground_z is None:
            print("[ERROR] Calibration failed")
            sys.exit(1)
    else:
        print(f"  (Delete {CALIBRATION_FILE} to recalibrate)")

    # --- Measurements ---
    print()
    print("=" * 55)
    print("  STEP 2: MEASUREMENTS")
    print("  Move robot with remote, press ENTER to scan.")
    print("  Values are in base_link frame, matching training.")
    print("  Ctrl+C to quit.")
    print("=" * 55)

    num = 0
    try:
        while True:
            num += 1
            input(f"\n>>> Measurement #{num}: press ENTER...")
            print()
            run_measurement(num, ground_z)
    except KeyboardInterrupt:
        print("\n\nDone.")


if __name__ == "__main__":
    main()
