#!/usr/bin/env python3
"""
Go2 L1 LiDAR → 5×3 Terrain Delta Grid
========================================
Shows height DELTA from expected ground at each cell.
  0  = flat ground
  +20 = something 20cm tall (box, stair step)
  -10 = 10cm drop (stair down, hole)

How it works:
  1. Fit ground plane from near-body points (x < 0.20m)
  2. For each cell, predict expected ground z from the plane
  3. Delta = expected_z - observed_z  (positive = obstacle above ground)
  4. Uses p10 of z per cell (captures highest surface robustly)

Usage:
  python lidar_grid_simple.py enp0s31f6
  python lidar_grid_simple.py enp0s31f6 --accum 1.0
"""

import argparse
import collections
import math
import threading
import time

import numpy as np

TOPIC = "rt/utlidar/cloud"

X_PTS = [0.00, 0.15, 0.30, 0.50, 0.80]
Y_PTS = [-0.15, 0.00, 0.15]
X_EDGES = [-0.075, 0.075, 0.225, 0.400, 0.650, 0.875]
Y_EDGES = [-0.225, -0.075, 0.075, 0.225]

_PF_DTYPES = {
    1: np.int8,
    2: np.uint8,
    3: np.int16,
    4: np.uint16,
    5: np.int32,
    6: np.uint32,
    7: np.float32,
    8: np.float64,
}


def pc2_to_xyz(msg):
    wanted = {}
    for f in msg.fields:
        if f.name in ("x", "y", "z"):
            wanted[f.name] = (int(f.offset), _PF_DTYPES.get(int(f.datatype)))
    ps = int(msg.point_step)
    dt = np.dtype(
        {
            "names": ["x", "y", "z"],
            "formats": [wanted["x"][1], wanted["y"][1], wanted["z"][1]],
            "offsets": [wanted["x"][0], wanted["y"][0], wanted["z"][0]],
            "itemsize": ps,
        }
    )
    c = np.frombuffer(bytes(msg.data), dtype=dt)
    xyz = np.stack([c["x"], c["y"], c["z"]], axis=1).astype(np.float32)
    if bool(msg.is_bigendian):
        xyz = xyz.byteswap().newbyteorder()
    return xyz[np.isfinite(xyz).all(axis=1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("interface")
    ap.add_argument("--accum", type=float, default=0.5)
    args = ap.parse_args()

    import unitree_sdk2py.idl.sensor_msgs.msg.dds_ as sd
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber

    ChannelFactoryInitialize(0, args.interface)

    Entry = collections.namedtuple("E", ["t", "xyz"])
    buf = collections.deque()
    lk = threading.Lock()

    def cb(msg):
        xyz = pc2_to_xyz(msg)
        with lk:
            buf.append(Entry(time.monotonic(), xyz))

    sub = ChannelSubscriber(TOPIC, sd.PointCloud2_)
    sub.Init(cb, 10)
    print(f"Listening on {args.interface}, accum={args.accum}s")
    print("Delta grid: 0 = flat ground, +N = Ncm obstacle, -N = Ncm drop")
    print("Waiting for data...\n")

    try:
        while True:
            time.sleep(0.2)
            now = time.monotonic()
            with lk:
                while buf and buf[0].t < now - args.accum:
                    buf.popleft()
                if not buf:
                    continue
                nf = len(buf)
                pts = np.vstack([e.xyz for e in buf])

            # Keep only grid region, minimal z filter
            mask = (
                (pts[:, 0] > X_EDGES[0])
                & (pts[:, 0] < X_EDGES[-1])
                & (pts[:, 1] > Y_EDGES[0])
                & (pts[:, 1] < Y_EDGES[-1])
                & (pts[:, 2] > 0.02)
            )
            fp = pts[mask]
            if len(fp) < 10:
                continue

            xi = np.digitize(fp[:, 0], X_EDGES) - 1
            yi = np.digitize(fp[:, 1], Y_EDGES) - 1

            # Fit ground plane from near-body points ONLY
            near = fp[fp[:, 0] < 0.20]
            if len(near) < 10:
                continue
            A = np.column_stack([near[:, 0], near[:, 1], np.ones(len(near))])
            try:
                abc = np.linalg.lstsq(A, near[:, 2], rcond=None)[0]
                a, b, c = float(abc[0]), float(abc[1]), float(abc[2])
            except:
                continue

            pitch_deg = -math.degrees(math.atan(a))

            lines = []
            lines.append(
                f"  Frames: {nf}  |  Pts: {len(fp):,}  |  Pitch: {pitch_deg:+.1f}°"
            )
            lines.append("  DELTA (cm):  0=ground  +N=obstacle  -N=drop")
            lines.append("         y=-15cm    y=0cm    y=+15cm")

            vec = []
            pop = 0
            for i in range(5):
                row = []
                for j in range(3):
                    m = (xi == i) & (yi == j)
                    n = m.sum()
                    if n > 0:
                        cell_pts = fp[m]
                        # Expected ground z at cell center
                        z_expected = a * X_PTS[i] + b * Y_PTS[j] + c
                        # Observed: p10 of z = highest surface
                        z_observed = np.percentile(cell_pts[:, 2], 10)
                        # Positive delta = above ground
                        delta_cm = (z_expected - z_observed) * 100

                        row.append(f"{delta_cm:+5.0f}({n:>3d})")
                        vec.append(delta_cm / 100)
                        pop += 1
                    else:
                        row.append("   --(  0)")
                        vec.append(0.0)
                lines.append(f"  x={X_PTS[i]:.2f} │{row[0]}│{row[1]}│{row[2]}│")

            lines.append(f"  Coverage: {pop}/15")
            lines.append(f"  Vec: [{', '.join(f'{v:+.2f}' for v in vec)}]")

            print("\033[2J\033[H" + "\n".join(lines), flush=True)

    except KeyboardInterrupt:
        print("\nDone.")


if __name__ == "__main__":
    main()
