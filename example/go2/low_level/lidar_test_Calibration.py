#!/usr/bin/env python3
"""
LIDAR 15×1 grid measurement tool with tilt calibration.

Flow:
  1. Place robot on FLAT ground, press Enter → captures tilt angle + residual
  2. All subsequent measurements are tilt-corrected
  3. Press Enter between scenes, Ctrl+C to quit

Usage:
  python lidar_measure.py              # default DDS
  python lidar_measure.py eth0         # specific interface
"""

import sys
import time

import numpy as np
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_

# --- 15×1 Grid ---
X_POINTS = [
    0.00,
    0.02,
    0.04,
    0.06,
    0.08,
    0.10,
    0.13,
    0.16,
    0.19,
    0.22,
    0.26,
    0.30,
    0.34,
    0.38,
    0.42,
]
NX = len(X_POINTS)

Y_HALF = 0.15


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

# --- Tilt state (computed during calibration) ---
TILT_RAD = 0.0  # rotation angle around y-axis
RESIDUAL_LUT = None  # per-bin residual after rotation (15,)

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


def transform_points(xyz):
    """Apply forward-sign flip, z-negation, and tilt correction."""
    x_raw = FORWARD_SIGN * xyz[:, 0] + LIDAR_X_OFFSET
    y = xyz[:, 1]
    z_raw = -xyz[:, 2] + LIDAR_Z_OFFSET

    # Tilt correction: rotate around y-axis
    if abs(TILT_RAD) > 1e-6:
        cos_t = np.cos(TILT_RAD)
        sin_t = np.sin(TILT_RAD)
        x = cos_t * x_raw + sin_t * z_raw
        z = -sin_t * x_raw + cos_t * z_raw
    else:
        x = x_raw
        z = z_raw

    return x, y, z


def grid_from_cloud(xyz):
    """Bin one frame into 15×1 grid."""
    x, y, z = transform_points(xyz)

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
    all_h = np.array([f[0] for f in frames])
    all_c = np.array([f[1] for f in frames])

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

    # Apply residual LUT correction
    if RESIDUAL_LUT is not None:
        for i in range(NX):
            if not np.isnan(rel[i]):
                rel[i] -= RESIDUAL_LUT[i]

    return np.clip(rel, -1.0, 1.0), ground


# =====================================================================
# Calibration
# =====================================================================


def calibrate(frames, raw_clouds):
    """Compute tilt angle and residual LUT from flat ground data."""
    global TILT_RAD, RESIDUAL_LUT

    # Step 1: Compute mean heights WITHOUT tilt correction
    mean_h, std_h, mean_c, valid_pct = compute_stats(frames)

    # Make relative to cell 0
    if np.isnan(mean_h[0]):
        print("  ERROR: Cell 0 has no data. Cannot calibrate.")
        return False

    ground = float(mean_h[0])
    rel = mean_h - ground

    print("  ┌──────────────────────────────────────────────┐")
    print("  │  CALIBRATION — Step 1: Measure tilt          │")
    print("  └──────────────────────────────────────────────┘")
    print("\n  Uncorrected flat-ground profile:")
    for i in range(NX):
        v = rel[i]
        v_str = f"{v:+.4f}" if not np.isnan(v) else "  --- "
        print(f"    [{i:2d}] x={X_POINTS[i]:.2f}m  {v_str}")

    # Step 2: Fit line to valid points → tilt angle
    xs = []
    zs = []
    for i in range(NX):
        if not np.isnan(rel[i]) and valid_pct[i] > 50:
            xs.append(X_POINTS[i])
            zs.append(rel[i])

    if len(xs) < 3:
        print("  ERROR: Too few valid cells for tilt fit.")
        return False

    xs = np.array(xs)
    zs = np.array(zs)

    # Linear fit: z = slope * x + intercept
    slope, intercept = np.polyfit(xs, zs, 1)
    tilt_angle = np.arctan(slope)  # slope = Δz/Δx ≈ -sin(tilt)/cos(tilt)

    print(f"\n  Linear fit: slope = {slope:.4f} m/m  (intercept = {intercept:.4f}m)")
    print(f"  Tilt angle: {np.degrees(tilt_angle):+.2f}°")

    # Step 3: Apply rotation and re-measure
    TILT_RAD = -tilt_angle  # negate: we rotate opposite to measured tilt
    print(f"  Applying rotation: {np.degrees(TILT_RAD):+.2f}°")

    # Re-process the same raw clouds with rotation active
    corrected_frames = []
    for xyz in raw_clouds:
        h, c = grid_from_cloud(xyz)
        corrected_frames.append((h, c))

    mean_h2, std_h2, _, _ = compute_stats(corrected_frames)
    ground2 = float(mean_h2[0]) if not np.isnan(mean_h2[0]) else 0.0
    rel2 = mean_h2 - ground2

    print("\n  ┌──────────────────────────────────────────────┐")
    print("  │  CALIBRATION — Step 2: After rotation         │")
    print("  └──────────────────────────────────────────────┘")
    print("\n  Post-rotation profile (residual):")

    RESIDUAL_LUT = np.zeros(NX)
    max_residual = 0.0

    for i in range(NX):
        v = rel2[i]
        v_str = f"{v:+.4f}" if not np.isnan(v) else "  --- "
        if not np.isnan(v):
            RESIDUAL_LUT[i] = v
            max_residual = max(max_residual, abs(v))
        print(f"    [{i:2d}] x={X_POINTS[i]:.2f}m  {v_str}")

    print("\n  ┌──────────────────────────────────────────────┐")
    print("  │  CALIBRATION — Step 3: Final result           │")
    print("  └──────────────────────────────────────────────┘")
    print(f"\n  Tilt rotation:  {np.degrees(TILT_RAD):+.2f}°")
    print(f"  Max residual:   {max_residual * 100:.2f}cm  (corrected via LUT)")
    print(f"  Residual LUT:   [{', '.join(f'{v:+.4f}' for v in RESIDUAL_LUT)}]")

    # Verify: show what flat ground reads after full correction
    print("\n  Verification (flat ground after full correction):")
    for i in range(NX):
        v = rel2[i]
        if not np.isnan(v):
            corrected = v - RESIDUAL_LUT[i]
        else:
            corrected = np.nan
        c_str = f"{corrected:+.4f}" if not np.isnan(corrected) else "  --- "
        print(f"    [{i:2d}] x={X_POINTS[i]:.2f}m  {c_str}  (should be ~0.000)")

    print("\n  Calibration complete. All further measurements are corrected.")
    return True


# =====================================================================
# Display
# =====================================================================


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


def print_coverage(xyz):
    x, y, z = transform_points(xyz)

    print("\n  Points per x-bin (tilt-corrected):")
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
        print(f"    [{i:2d}] x={xc:.2f}m: {n:5d} pts  {status:>4}  {bar}")


def print_results(measurement_num, frames, raw_clouds):
    print_raw_bounds(raw_clouds[-1])
    print_coverage(raw_clouds[-1])

    mean_h, std_h, mean_c, valid_pct = compute_stats(frames)
    rel_h, ground_z = make_relative(mean_h)

    print("\n" + "=" * 72)
    print(
        f"  MEASUREMENT #{measurement_num}  "
        f"({len(frames)} frames, ground_z={ground_z:.4f}m, "
        f"tilt={np.degrees(TILT_RAD):+.1f}°)"
    )
    print("=" * 72)

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


# =====================================================================
# Main
# =====================================================================


def main():
    if len(sys.argv) > 1 and sys.argv[1] not in ("--help", "-h"):
        ChannelFactoryInitialize(0, sys.argv[1])
        print(f"DDS interface: {sys.argv[1]}")
    else:
        ChannelFactoryInitialize(0)

    sub = ChannelSubscriber(TOPIC, PointCloud2_)
    sub.Init(_cb, 1)

    print("\nLIDAR 15×1 Grid Measurement Tool (with tilt calibration)")
    print(f"Topic: {TOPIC}  FORWARD_SIGN={FORWARD_SIGN}  y=±{Y_HALF}m")
    print(f"X points: {[f'{x:.2f}' for x in X_POINTS]}\n")

    print("Waiting for LIDAR data...", end="", flush=True)
    while latest_msg is None:
        time.sleep(0.1)
    print(" connected!\n")

    # ── Step 1: Calibration on flat ground ──
    print("=" * 72)
    print("  STEP 1: TILT CALIBRATION")
    print("  Place robot on FLAT ground with nothing in front.")
    print("=" * 72)
    input("\n>>> Press ENTER to capture flat-ground baseline...")
    print()

    frames, raw_clouds = capture_frames()
    if len(frames) < 5:
        print("  Too few frames. Restart script.")
        return

    ok = calibrate(frames, raw_clouds)
    if not ok:
        print("  Calibration failed. Continuing without correction.\n")

    # ── Step 2: Measurements ──
    print("\n" + "=" * 72)
    print("  CALIBRATION DONE — Now measure anything!")
    print("=" * 72 + "\n")

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
