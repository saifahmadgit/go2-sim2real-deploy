#!/usr/bin/env python3
"""
find_stair_edge.py
==================
Minimal test: just find where the first stair edge is.

Builds a fine 1D z-profile from LIDAR, looks for z-jump.
Prints: "First stair at x = 0.XX m" or "No stair detected."

Usage:
  python3 find_stair_edge.py              # default DDS
  python3 find_stair_edge.py eth0          # specific interface
"""

import math
import sys
import time

import numpy as np

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_

# ================================================================
#  CONFIG
# ================================================================
EXPECTED_STEP_HEIGHT = 0.10  # not used for detection anymore, just for display
# Detection: any bin that differs from ground by more than this = edge
EDGE_Z_THRESHOLD = 0.04  # 4cm — anything above this is "something changed"

# LIDAR z-data is reliable up to this distance. Beyond here, z drifts
# and always false-triggers at ~27.5cm. So we ONLY search within this range.
EDGE_SEARCH_MAX_X = 0.27  # meters — don't trust z beyond here

PROFILE_X_MIN = -0.10  # start scanning slightly behind
PROFILE_X_MAX = 0.60  # how far ahead to look
BIN_SIZE = 0.01  # 1cm bins
Y_HALF = 0.20  # lateral capture band

MIN_PTS_PER_BIN = 3

CLOUD_TOPIC = "rt/utlidar/cloud"
STATE_TOPIC = "rt/sportmodestate"
CAPTURE_SECONDS = 3.0

# ================================================================
#  DECODE
# ================================================================
_DTYPES = {5: np.int32, 6: np.uint32, 7: np.float32, 8: np.float64}


def decode_xyz(msg):
    offsets = {}
    for f in msg.fields:
        if f.name in ("x", "y", "z"):
            offsets[f.name] = (int(f.offset), _DTYPES.get(int(f.datatype), np.float32))
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
    pts = np.frombuffer(data[: n * int(msg.point_step)], dtype=dtype)
    return np.stack([pts["x"], pts["y"], pts["z"]], axis=1).astype(np.float32)


# ================================================================
#  IMU
# ================================================================
imu_roll = 0.0
imu_pitch = 0.0


def _on_state(msg):
    global imu_roll, imu_pitch
    try:
        rpy = msg.imu_state.rpy
        if rpy is not None and len(rpy) >= 3:
            imu_roll = float(rpy[0])
            imu_pitch = float(rpy[1])
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
#  BUILD PROFILE AND FIND EDGE
# ================================================================


def build_profile(pts_lidar, roll, pitch):
    """
    Build 1D z-profile from LIDAR points.

    Returns:
        centers: (N,) bin center x positions in body frame
        z_vals:  (N,) median z per bin (NaN if empty)
        counts:  (N,) points per bin
    """
    # LIDAR → body frame
    bx = -pts_lidar[:, 0]  # -lidar_x = forward
    by = pts_lidar[:, 1]
    bz = -pts_lidar[:, 2]  # -lidar_z = up

    # Filter
    valid = (
        (np.abs(by) <= Y_HALF)
        & (bz > -2.0)
        & (bz < 1.0)
        & np.all(np.isfinite(pts_lidar), axis=1)
    )
    bx, by, bz = bx[valid], by[valid], bz[valid]

    # Gravity projection
    r2 = gravity_z_row(roll, pitch)
    z_grav = r2[0] * bx + r2[1] * by + r2[2] * bz

    # Bin
    edges = np.arange(PROFILE_X_MIN, PROFILE_X_MAX + BIN_SIZE, BIN_SIZE)
    centers = (edges[:-1] + edges[1:]) / 2
    n_bins = len(centers)

    z_vals = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=np.int32)

    bin_idx = np.digitize(bx, edges) - 1
    for b in range(n_bins):
        mask = bin_idx == b
        n = np.sum(mask)
        counts[b] = n
        if n >= MIN_PTS_PER_BIN:
            z_vals[b] = np.median(z_grav[mask])

    return centers, z_vals, counts


def find_edge(centers, z_vals):
    """
    Find first x where z diverges from ground level.

    Ground = median of first few valid bins (near robot).
    Edge = first bin where |z - ground| > EDGE_Z_THRESHOLD.

    Returns (edge_x, dz_from_ground) or None.
    """
    valid = ~np.isnan(z_vals)

    # Establish ground reference from first valid bins (x < 0.10m)
    ground_bins = []
    for i in range(len(centers)):
        if valid[i] and centers[i] < 0.10:
            ground_bins.append(z_vals[i])

    if len(ground_bins) < 2:
        # Not enough near-field data, try any early bins
        for i in range(len(centers)):
            if valid[i] and len(ground_bins) < 5:
                ground_bins.append(z_vals[i])

    if len(ground_bins) == 0:
        return None

    ground_z = np.median(ground_bins)

    # Scan forward: find first bin that deviates from ground
    # Only trust z up to EDGE_SEARCH_MAX_X
    for i in range(len(centers)):
        if not valid[i]:
            continue
        if centers[i] > EDGE_SEARCH_MAX_X:
            break  # beyond reliable range, stop
        dz = z_vals[i] - ground_z
        if abs(dz) > EDGE_Z_THRESHOLD:
            return centers[i], dz, ground_z

    return None


# ================================================================
#  MAIN
# ================================================================


def main():
    global latest_msg, msg_count

    print("\n  STAIR EDGE DETECTOR")
    print(f"  Threshold: >{EDGE_Z_THRESHOLD * 100:.0f}cm change from ground level")
    print(f"  Reliable range: x < {EDGE_SEARCH_MAX_X}m (z drifts beyond)")
    print(
        f"  Profile: {PROFILE_X_MIN}m to {PROFILE_X_MAX}m, "
        f"{BIN_SIZE * 100:.0f}cm bins\n"
    )

    if len(sys.argv) > 1 and sys.argv[1] not in ("--help", "-h"):
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)

    cloud_sub = ChannelSubscriber(CLOUD_TOPIC, PointCloud2_)
    cloud_sub.Init(_on_cloud, 10)
    state_sub = ChannelSubscriber(STATE_TOPIC, SportModeState_)
    state_sub.Init(_on_state, 10)

    print("Waiting for LIDAR...", end="", flush=True)
    t0 = time.time()
    while latest_msg is None:
        if time.time() - t0 > 15:
            print(" TIMEOUT")
            sys.exit(1)
        time.sleep(0.3)
        print(".", end="", flush=True)
    print(" OK\n")

    measurement = 0
    try:
        while True:
            measurement += 1
            input(f">>> #{measurement}: position robot facing stairs, press ENTER...")

            # Capture frames
            frames = []
            t0 = time.time()
            last_id = id(latest_msg)
            print(f"  Capturing {CAPTURE_SECONDS:.0f}s", end="", flush=True)

            while time.time() - t0 < CAPTURE_SECONDS:
                if latest_msg is not None and id(latest_msg) != last_id:
                    last_id = id(latest_msg)
                    xyz = decode_xyz(latest_msg)
                    if len(xyz) > 100:
                        c, z, n = build_profile(xyz, imu_roll, imu_pitch)
                        frames.append((c, z, n))
                        print(".", end="", flush=True)
                time.sleep(0.02)
            print(f" ({len(frames)} frames)\n")

            if len(frames) < 3:
                print("  Not enough frames.\n")
                continue

            # Average profile across frames
            all_z = np.stack([f[1] for f in frames], axis=0)  # (M, bins)
            all_n = np.stack([f[2] for f in frames], axis=0)
            centers = frames[0][0]

            # Per-bin: median of medians (robust)
            avg_z = np.full(len(centers), np.nan)
            avg_n = np.mean(all_n, axis=0)
            for b in range(len(centers)):
                col = all_z[:, b]
                valid = col[~np.isnan(col)]
                if len(valid) >= 3:
                    avg_z[b] = np.median(valid)

            # Find edge
            result = find_edge(centers, avg_z)

            # Get ground reference for display
            ground_z_ref = None
            for b in range(len(centers)):
                if not np.isnan(avg_z[b]) and centers[b] < 0.10:
                    if ground_z_ref is None:
                        ground_z_ref = avg_z[b]
            if ground_z_ref is None:
                for b in range(len(centers)):
                    if not np.isnan(avg_z[b]):
                        ground_z_ref = avg_z[b]
                        break

            # --- Print profile ---
            print(
                f"  Ground reference z: {ground_z_ref:+.4f}"
                if ground_z_ref
                else "  Ground: ???"
            )
            print(f"  Edge threshold: ±{EDGE_Z_THRESHOLD * 100:.1f}cm from ground")
            print()
            print("  1D Height Profile (body x, forward):")
            print(f"  {'x(m)':>6}  {'z(m)':>8}  {'dz':>7}  {'pts':>5}  profile")
            print(f"  {'─' * 60}")

            ref_z = ground_z_ref

            for b in range(len(centers)):
                x = centers[b]
                z = avg_z[b]
                n = avg_n[b]

                if np.isnan(z):
                    print(f"  {x:+.3f}  {'---':>8}  {'':>7}  {n:5.0f}")
                    continue

                # dz from ground
                dz_str = ""
                bar = ""
                if ref_z is not None:
                    rel = z - ref_z
                    dz_str = f"{rel * 100:+5.1f}cm"
                    if abs(rel) < 0.01:
                        bar = "─"
                    elif rel > 0:
                        bar = "█" * min(int(rel * 100), 30)
                    else:
                        bar = "▿" * min(int(-rel * 100), 15)

                # Highlight edge or unreliable zone
                marker = ""
                if result and abs(x - result[0]) < BIN_SIZE * 1.5:
                    marker = "  ◄── EDGE"
                elif x > EDGE_SEARCH_MAX_X:
                    marker = "  (unreliable)"

                print(f"  {x:+.3f}  {z:+.4f}  {dz_str:>7}  {n:5.0f}  {bar}{marker}")

            # --- Result ---
            print()
            if result:
                edge_x, edge_dz, gnd = result
                direction = "UP" if edge_dz > 0 else "DOWN"
                print("  ╔══════════════════════════════════════════╗")
                print(f"  ║  CHANGE at x = {edge_x:.3f} m                  ║")
                print(
                    f"  ║  dz = {edge_dz:+.3f} m ({direction}, "
                    f"{abs(edge_dz) * 100:.1f}cm from ground)  ║"
                )
                print(f"  ║  ground z = {gnd:+.4f} m                     ║")
                print("  ╚══════════════════════════════════════════╝")
            else:
                print(f"  No significant z-change within {EDGE_SEARCH_MAX_X}m")
                print("  (flat terrain or stair beyond reliable range)")

            print()

    except KeyboardInterrupt:
        print("\n\nDone.")


if __name__ == "__main__":
    main()
