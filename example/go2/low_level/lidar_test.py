#!/usr/bin/env python3
"""
LIDAR 15×1 grid measurement tool for Go2 L1.

15 bins along x (forward), single wide y column (±0.15m).
Stairs only vary in x — put all resolution there.

Flow:
  1. Press Enter to start a measurement
  2. Collects frames for a few seconds
  3. Shows 1D height profile + stats
  4. Press Enter for next measurement
  5. Ctrl+C to quit

Usage:
  python lidar_measure.py              # default DDS
  python lidar_measure.py eth0         # specific interface
"""

import sys
import time

import numpy as np
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_

# --- 15×1 Grid: all resolution in forward direction ---
X_POINTS = [
    0.00,
    0.02,
    0.04,
    0.06,
    0.08,
    0.10,  # 6 pts, 2cm spacing (under feet)
    0.13,
    0.16,
    0.19,
    0.22,  # 4 pts, 3cm spacing (one tread ahead)
    0.26,
    0.30,
    0.34,
    0.38,
    0.42,
]  # 5 pts, 4cm spacing (look-ahead)
NX = len(X_POINTS)  # 15

Y_HALF = 0.15  # capture ±15cm laterally (full robot width, single column)


# Cell x half-widths: half distance to nearest neighbour
def _cell_x_halves():
    halves = []
    for i in range(NX):
        if i == 0:
            h = (X_POINTS[1] - X_POINTS[0]) / 2.0
        elif i == NX - 1:
            h = (X_POINTS[-1] - X_POINTS[-2]) / 2.0
        else:
            h = min(X_POINTS[i] - X_POINTS[i - 1], X_POINTS[i + 1] - X_POINTS[i]) / 2.0
        halves.append(h)
    return halves


X_HALVES = _cell_x_halves()

# --- Config ---
TOPIC = "rt/utlidar/cloud"
FORWARD_SIGN = -1
STRIDE = 1
MIN_PTS = 3
CAPTURE_SECONDS = 3.0

LIDAR_X_OFFSET = 0.0
LIDAR_Z_OFFSET = 0.0

# --- PointCloud2 decode ---
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
    pts = np.frombuffer(bytes(msg.data), dtype=dtype)[::STRIDE]
    return np.stack([pts["x"], pts["y"], pts["z"]], axis=1).astype(np.float32)


latest_msg = None


def _cb(msg):
    global latest_msg
    latest_msg = msg


def print_raw_bounds(xyz):
    print("\n  ┌─────────────────────────────────────────────────┐")
    print("  │  RAW POINT CLOUD (LIDAR frame, NO transforms)   │")
    print("  └─────────────────────────────────────────────────┘")
    print(f"  Total points (stride={STRIDE}): {len(xyz)}")
    if len(xyz) == 0:
        return

    print(f"\n  {'':>6} {'min':>10} {'max':>10} {'mean':>10} {'std':>10}")
    for axis, name in enumerate(["raw_x", "raw_y", "raw_z"]):
        v = xyz[:, axis]
        print(
            f"  {name:>6} {v.min():+10.3f} {v.max():+10.3f} "
            f"{v.mean():+10.3f} {v.std():10.3f}"
        )

    print("\n  Percentiles:")
    for axis, name in enumerate(["raw_x", "raw_y", "raw_z"]):
        v = xyz[:, axis]
        pcts = np.percentile(v, [5, 25, 50, 75, 95])
        print(
            f"  {name:>6}  5%={pcts[0]:+.3f}  25%={pcts[1]:+.3f}  "
            f"50%={pcts[2]:+.3f}  75%={pcts[3]:+.3f}  95%={pcts[4]:+.3f}"
        )


def print_transformed_coverage(xyz):
    x = FORWARD_SIGN * xyz[:, 0] + LIDAR_X_OFFSET
    y = xyz[:, 1]
    z = -xyz[:, 2] + LIDAR_Z_OFFSET

    print("\n  ┌─────────────────────────────────────────────────┐")
    print(f"  │  TRANSFORMED COVERAGE (FORWARD_SIGN={FORWARD_SIGN:+d})          │")
    print("  └─────────────────────────────────────────────────┘")
    print(f"  fwd_x: [{x.min():+.3f}, {x.max():+.3f}]  (positive = ahead)")
    print(f"  y:     [{y.min():+.3f}, {y.max():+.3f}]  (capturing ±{Y_HALF}m)")
    print(f"  z:     [{z.min():+.3f}, {z.max():+.3f}]")

    print(f"\n  Points per x-bin (y within ±{Y_HALF}m):")
    for i in range(NX):
        xc = X_POINTS[i]
        xh = X_HALVES[i]
        lo, hi = xc - xh, xc + xh
        in_bin = (
            (x >= lo)
            & (x < hi)
            & (y >= -Y_HALF)
            & (y <= Y_HALF)
            & (z > -2.0)
            & (z < 1.0)
        )
        n = np.sum(in_bin)
        bar = "█" * min(40, n // 5)
        status = "OK" if n >= MIN_PTS * 3 else ("LOW" if n > 0 else "DEAD")
        print(
            f"    [{i:2d}] x={xc:.2f}m  [{lo:+.3f},{hi:+.3f}]: "
            f"{n:5d} pts  {status:>4}  {bar}"
        )

    print("\n  Where are the points? (transformed x histogram)")
    bins = [-0.5, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 3.0]
    for b in range(len(bins) - 1):
        n = np.sum((x >= bins[b]) & (x < bins[b + 1]))
        bar = "█" * min(40, n // 20)
        print(f"    [{bins[b]:+5.1f}, {bins[b + 1]:+5.1f}): {n:6d}  {bar}")


def grid_from_cloud(xyz):
    """Bin one frame into 15×1 grid."""
    x = FORWARD_SIGN * xyz[:, 0] + LIDAR_X_OFFSET
    y = xyz[:, 1]
    z = -xyz[:, 2] + LIDAR_Z_OFFSET

    heights = np.full(NX, np.nan)
    counts = np.zeros(NX, dtype=int)

    for i in range(NX):
        xc = X_POINTS[i]
        xh = X_HALVES[i]
        mask = (
            (x >= xc - xh)
            & (x < xc + xh)
            & (y >= -Y_HALF)
            & (y <= Y_HALF)
            & (z > -2.0)
            & (z < 1.0)
        )
        cell_z = z[mask]
        counts[i] = len(cell_z)
        if len(cell_z) >= MIN_PTS:
            heights[i] = np.percentile(cell_z, 90)

    return heights, counts


def capture_frames():
    frames = []
    raw_clouds = []
    t0 = time.time()
    last_id = id(latest_msg)

    print(f"  Capturing for {CAPTURE_SECONDS:.0f}s", end="", flush=True)

    while time.time() - t0 < CAPTURE_SECONDS:
        if latest_msg is None or id(latest_msg) == last_id:
            time.sleep(0.02)
            continue
        last_id = id(latest_msg)

        xyz = decode_xyz(latest_msg)
        if len(xyz) > 0:
            h, c = grid_from_cloud(xyz)
            frames.append((h, c))
            raw_clouds.append(xyz)
            print(".", end="", flush=True)

    print(f" done! ({len(frames)} frames)\n")
    return frames, raw_clouds


def compute_stats(frames):
    n_frames = len(frames)
    all_h = np.array([f[0] for f in frames])  # (N, 15)
    all_c = np.array([f[1] for f in frames])  # (N, 15)

    mean_h = np.full(NX, np.nan)
    std_h = np.full(NX, np.nan)
    mean_c = np.mean(all_c, axis=0)
    valid_pct = np.zeros(NX)

    for i in range(NX):
        vals = all_h[:, i]
        valid = vals[~np.isnan(vals)]
        valid_pct[i] = len(valid) / n_frames * 100
        if len(valid) >= 3:
            mean_h[i] = np.mean(valid)
            std_h[i] = np.std(valid)

    return mean_h, std_h, mean_c, valid_pct


def make_relative(mean_h):
    if not np.isnan(mean_h[0]):
        ground = float(mean_h[0])
    else:
        valid = mean_h[~np.isnan(mean_h)]
        ground = float(valid[0]) if len(valid) > 0 else 0.0
    rel = mean_h - ground
    return np.clip(rel, -1.0, 1.0), ground


def print_results(measurement_num, frames, raw_clouds):
    print_raw_bounds(raw_clouds[-1])
    print_transformed_coverage(raw_clouds[-1])

    mean_h, std_h, mean_c, valid_pct = compute_stats(frames)
    rel_h, ground_z = make_relative(mean_h)

    print("\n" + "=" * 72)
    print(
        f"  MEASUREMENT #{measurement_num}  "
        f"({len(frames)} frames, ground_z={ground_z:.4f}m)"
    )
    print("=" * 72)

    # 1D profile table
    print(
        f"\n  {'idx':>4} {'x(m)':>6} {'height':>8} {'std':>8} "
        f"{'pts':>6} {'valid%':>7}  profile"
    )
    print(f"  {'─' * 68}")

    for i in range(NX):
        h = rel_h[i]
        s = std_h[i]
        c = mean_c[i]
        v = valid_pct[i]

        h_str = f"{h:+.4f}" if not np.isnan(h) else "   --- "
        s_str = f"{s:.4f}" if not np.isnan(s) else "  --- "

        # ASCII bar: each char ≈ 1cm
        if np.isnan(h):
            bar = "  ?"
        elif h >= 0:
            bar = "  │" + "█" * min(int(round(h * 100)), 30)
        else:
            ncm = min(int(round(-h * 100)), 15)
            bar = " " * (15 - ncm) + "▓" * ncm + "│"

        print(
            f"  [{i:2d}] {X_POINTS[i]:5.2f}  {h_str}  {s_str}  "
            f"{c:5.1f}  {v:5.0f}%  {bar}"
        )

    # Policy vector
    vec = []
    for i in range(NX):
        v = rel_h[i]
        vec.append(0.0 if np.isnan(v) else float(v))

    print(f"\n  POLICY VECTOR ({NX}-dim):")
    print(f"  [{', '.join(f'{v:+.3f}' for v in vec)}]")

    dead = sum(1 for i in range(NX) if np.isnan(mean_h[i]))
    noisy = sum(1 for i in range(NX) if not np.isnan(std_h[i]) and std_h[i] > 0.02)
    good = NX - dead - noisy
    print(f"\n  Summary: {good} good  |  {noisy} noisy (>2cm std)  |  {dead} dead")
    print("=" * 72)


def main():
    if len(sys.argv) > 1 and sys.argv[1] not in ("--help", "-h"):
        ChannelFactoryInitialize(0, sys.argv[1])
        print(f"DDS interface: {sys.argv[1]}")
    else:
        ChannelFactoryInitialize(0)

    sub = ChannelSubscriber(TOPIC, PointCloud2_)
    sub.Init(_cb, 1)

    print("\nLIDAR 15×1 Grid Measurement Tool")
    print(f"Topic: {TOPIC}")
    print(f"FORWARD_SIGN={FORWARD_SIGN}  y_capture=±{Y_HALF}m  stride={STRIDE}")
    print(f"X points: {[f'{x:.2f}' for x in X_POINTS]}")
    print(f"Cell x-halves: {[f'{h:.3f}' for h in X_HALVES]}\n")

    print("Waiting for LIDAR data...", end="", flush=True)
    while latest_msg is None:
        time.sleep(0.1)
    print(" connected!\n")

    measurement_num = 0

    try:
        while True:
            measurement_num += 1
            input(f">>> Scene #{measurement_num}: set up, then press ENTER...")
            print()

            frames, raw_clouds = capture_frames()

            if len(frames) < 5:
                print("  Too few frames. Try again.\n")
                continue

            print_results(measurement_num, frames, raw_clouds)
            print()

    except KeyboardInterrupt:
        print("\n\nDone.")


if __name__ == "__main__":
    main()
