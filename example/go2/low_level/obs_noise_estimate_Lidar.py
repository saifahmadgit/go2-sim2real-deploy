#!/usr/bin/env python3
"""
obs_noise_estimate_lidar_dds.py
DDS-only noise estimator for GO2: LowState + LiDAR PointCloud2 (rt/utlidar/cloud)

- Samples at SAMPLE_HZ for SECONDS
- Builds 11x5 height grid (max-z per cell, clipped) -> 55 dims
- Computes centered std per channel + simple Gaussianity stats (skew/kurt/JB p)

No ROS/rclpy required.
"""

import sys
import threading
import time

import numpy as np

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_

# ================= CONFIG =================
SAMPLE_HZ = 50
SECONDS = 20
NUM_JOINTS = 12

# LiDAR -> grid (match what you plan to feed actor)
LIDAR_NUM_X = 11
LIDAR_NUM_Y = 5
LIDAR_X_RANGE = (-0.2, 1.0)  # (behind, ahead)
LIDAR_Y_RANGE = (-0.25, 0.25)  # (left, right)
LIDAR_HEIGHT_CLIP = 1.0
NUM_LIDAR_SCAN = LIDAR_NUM_X * LIDAR_NUM_Y  # 55

# If your cloud has forward points with negative x, set -1 (your earlier sample looked like that)
LIDAR_FORWARD_SIGN = -1

# Optional constant offset to "body-ish" frame (constant offsets do NOT affect std, only mean)
LIDAR_SENSOR_OFFSET = np.array([0.0, 0.0, 0.30], dtype=np.float32)

# Point cloud decoding speed controls
PC_STRIDE = 6  # take every Nth point
PC_MAX_POINTS = 60000  # cap points after stride

# Filter points used for grid (tune to your pipeline)
Z_FILTER = (-2.0, 0.10)  # keep near ground band (example)
MIN_POINTS_PER_CELL = 3  # require at least this many points in cell or set cell=0

# ==========================================


# -------------------------
# Quaternion -> projected gravity (expects w,x,y,z from LowState)
# -------------------------
def quat_to_gravity_xyz(q_wxyz):
    w, x, y, z = q_wxyz
    R = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )
    g = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    return (R.T @ g).astype(np.float32)


# -------------------------
# PointCloud2 DDS decode (fast): x,y,z only
# -------------------------
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


def pc2_to_xyz(msg: PointCloud2_, stride=1, max_points=None):
    # map x/y/z fields
    wanted = {}
    for f in msg.fields:
        if f.name in ("x", "y", "z"):
            dt = _PF_DTYPES.get(int(f.datatype))
            if dt is None:
                raise RuntimeError(
                    f"Unsupported datatype {f.datatype} for field {f.name}"
                )
            wanted[f.name] = (int(f.offset), dt)

    if not all(k in wanted for k in ("x", "y", "z")):
        raise RuntimeError(f"Missing x/y/z fields. Got {[f.name for f in msg.fields]}")

    point_step = int(msg.point_step)
    dtype = np.dtype(
        {
            "names": ["x", "y", "z"],
            "formats": [wanted["x"][1], wanted["y"][1], wanted["z"][1]],
            "offsets": [wanted["x"][0], wanted["y"][0], wanted["z"][0]],
            "itemsize": point_step,
        }
    )

    buf = bytes(msg.data)
    cloud = np.frombuffer(buf, dtype=dtype)

    if stride and stride > 1:
        cloud = cloud[::stride]
    if max_points is not None:
        cloud = cloud[:max_points]

    xyz = np.stack([cloud["x"], cloud["y"], cloud["z"]], axis=1).astype(np.float32)
    if bool(msg.is_bigendian):
        xyz = xyz.byteswap().newbyteorder()

    # drop NaN/inf
    m = np.isfinite(xyz).all(axis=1)
    return xyz[m]


# -------------------------
# LiDAR grid processor: max-z per cell (flattened 55)
# -------------------------
class LidarProcessor:
    def __init__(self):
        self.nx = LIDAR_NUM_X
        self.ny = LIDAR_NUM_Y
        self.x0, self.x1 = LIDAR_X_RANGE
        self.y0, self.y1 = LIDAR_Y_RANGE
        self.dx = (self.x1 - self.x0) / self.nx
        self.dy = (self.y1 - self.y0) / self.ny

    def process(self, points_xyz):
        if points_xyz is None or points_xyz.shape[0] == 0:
            return np.zeros(NUM_LIDAR_SCAN, dtype=np.float32)

        # transform conventions
        pts = points_xyz.astype(np.float32).copy()
        pts[:, 0] *= float(LIDAR_FORWARD_SIGN)  # enforce +x forward
        pts += LIDAR_SENSOR_OFFSET  # constant offset (optional)

        px, py, pz = pts[:, 0], pts[:, 1], pts[:, 2]

        # in-range (slightly padded like you had)
        in_range = (
            (px >= self.x0 - self.dx)
            & (px <= self.x1 + self.dx)
            & (py >= self.y0 - self.dy)
            & (py <= self.y1 + self.dy)
        )
        z_ok = (pz >= Z_FILTER[0]) & (pz <= Z_FILTER[1])
        mask = in_range & z_ok
        if not np.any(mask):
            return np.zeros(NUM_LIDAR_SCAN, dtype=np.float32)

        px = px[mask]
        py = py[mask]
        pz = pz[mask]

        xi = np.clip(((px - self.x0) / self.dx).astype(np.int32), 0, self.nx - 1)
        yi = np.clip(((py - self.y0) / self.dy).astype(np.int32), 0, self.ny - 1)
        cell = xi * self.ny + yi

        height = np.zeros(NUM_LIDAR_SCAN, dtype=np.float32)
        cnt = np.zeros(NUM_LIDAR_SCAN, dtype=np.int32)

        np.maximum.at(height, cell, pz)
        np.add.at(cnt, cell, 1)

        height[cnt < MIN_POINTS_PER_CELL] = 0.0
        height = np.clip(height, -LIDAR_HEIGHT_CLIP, LIDAR_HEIGHT_CLIP)
        return height


# -------------------------
# Simple gaussianity check: skew/kurt + Jarque–Bera p (df=2 => p=exp(-JB/2))
# -------------------------
def skew_kurt_jb(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 20:
        return np.nan, np.nan, np.nan, np.nan
    x = x - np.mean(x)
    s2 = np.mean(x * x)
    if s2 < 1e-12:
        return 0.0, -3.0, 0.0, 1.0  # degenerate
    s = np.sqrt(s2)
    m3 = np.mean((x / s) ** 3)
    m4 = np.mean((x / s) ** 4)
    skew = m3
    kurt = m4  # raw kurtosis (Gaussian ~3)
    n = x.size
    jb = (n / 6.0) * (skew**2) + (n / 24.0) * ((kurt - 3.0) ** 2)
    p = float(np.exp(-jb / 2.0))  # chi-square df=2 tail
    return float(skew), float(kurt), float(jb), float(p)


# -------------------------
# Main recorder
# -------------------------
class NoiseEstimatorDDS:
    def __init__(self):
        self.lowstate = None
        self.pc2 = None
        self._lock = threading.Lock()

        self.lidar_proc = LidarProcessor()

        self.sub_ls = ChannelSubscriber("rt/lowstate", LowState_)
        self.sub_ls.Init(self._cb_lowstate, 10)

        self.sub_pc = ChannelSubscriber("rt/utlidar/cloud", PointCloud2_)
        self.sub_pc.Init(self._cb_pc2, 1)  # keep latest only

        # buffers
        self.buf_gyro = []
        self.buf_grav = []
        self.buf_q = []
        self.buf_dq = []
        self.buf_lidar = []

    def _cb_lowstate(self, msg):
        with self._lock:
            self.lowstate = msg

    def _cb_pc2(self, msg):
        with self._lock:
            self.pc2 = msg

    def run(self):
        print("Waiting for rt/lowstate ...")
        while True:
            with self._lock:
                ok = self.lowstate is not None
            if ok:
                break
            time.sleep(0.05)
        print("✓ lowstate received")

        print("Waiting up to 3s for rt/utlidar/cloud ...")
        t0 = time.time()
        while time.time() - t0 < 3.0:
            with self._lock:
                ok = self.pc2 is not None
            if ok:
                break
            time.sleep(0.05)

        with self._lock:
            has_lidar = self.pc2 is not None
        print(
            "✓ lidar received"
            if has_lidar
            else "⚠ no lidar yet (will still try sampling)"
        )

        total = int(SAMPLE_HZ * SECONDS)
        dt = 1.0 / SAMPLE_HZ

        print(f"\nRecording {SECONDS}s @ {SAMPLE_HZ}Hz  (target samples={total})")
        print("KEEP ROBOT STILL.\n")

        next_t = time.monotonic()
        for k in range(total):
            now = time.monotonic()
            if now < next_t:
                time.sleep(next_t - now)
            next_t += dt

            with self._lock:
                ls = self.lowstate
                pc = self.pc2

            # ---- proprio ----
            gyro = np.array(ls.imu_state.gyroscope, dtype=np.float32)  # (3,)
            quat = np.array(ls.imu_state.quaternion, dtype=np.float32)  # (4,) wxyz
            grav = quat_to_gravity_xyz(quat)

            q = np.array(
                [ls.motor_state[i].q for i in range(NUM_JOINTS)], dtype=np.float32
            )
            dq = np.array(
                [ls.motor_state[i].dq for i in range(NUM_JOINTS)], dtype=np.float32
            )

            self.buf_gyro.append(gyro)
            self.buf_grav.append(grav)
            self.buf_q.append(q)
            self.buf_dq.append(dq)

            # ---- lidar heightmap ----
            if pc is not None:
                try:
                    xyz = pc2_to_xyz(pc, stride=PC_STRIDE, max_points=PC_MAX_POINTS)
                    scan = self.lidar_proc.process(xyz)  # (55,)
                    self.buf_lidar.append(scan)
                except Exception:
                    # keep going; lidar can hiccup
                    pass

            if (k + 1) % (SAMPLE_HZ * 2) == 0:
                print(
                    f"  {int((k + 1) / SAMPLE_HZ):2d}s: proprio={k + 1}  lidar_scans={len(self.buf_lidar)}"
                )

        print("\nDone. Computing stats...\n")
        self.compute()

    @staticmethod
    def centered_std(arr):
        arr = np.asarray(arr, dtype=np.float64)
        mu = np.mean(arr, axis=0)
        return np.std(arr - mu, axis=0)

    def compute(self):
        gyro = np.asarray(self.buf_gyro)  # (N,3)
        grav = np.asarray(self.buf_grav)  # (N,3)
        q = np.asarray(self.buf_q)  # (N,12)
        dq = np.asarray(self.buf_dq)  # (N,12)

        s_gyro = self.centered_std(gyro)  # (3,)
        s_grav = self.centered_std(grav)  # (3,)
        s_q = self.centered_std(q)  # (12,)
        s_dq = self.centered_std(dq)  # (12,)

        print("========== Proprio noise (centered std) ==========")
        print(
            f"gyro std per-axis     : {np.round(s_gyro, 6)}   mean={float(np.mean(s_gyro)):.6f}"
        )
        print(
            f"gravity std per-axis  : {np.round(s_grav, 6)}   mean={float(np.mean(s_grav)):.6f}"
        )
        print(
            f"q std mean over joints: {float(np.mean(s_q)):.6f} rad   max={float(np.max(s_q)):.6f}"
        )
        print(
            f"dq std mean over joints:{float(np.mean(s_dq)):.6f} rad/s max={float(np.max(s_dq)):.6f}"
        )

        # Gaussianity (aggregate per channel)
        sk, ku, jb, p = skew_kurt_jb(gyro.reshape(-1))
        print(f"\n[gyro] skew={sk:.3f} kurt={ku:.3f} JBp≈{p:.3g}")
        sk, ku, jb, p = skew_kurt_jb(grav.reshape(-1))
        print(f"[grav] skew={sk:.3f} kurt={ku:.3f} JBp≈{p:.3g}")
        sk, ku, jb, p = skew_kurt_jb(q.reshape(-1))
        print(f"[q]    skew={sk:.3f} kurt={ku:.3f} JBp≈{p:.3g}")
        sk, ku, jb, p = skew_kurt_jb(dq.reshape(-1))
        print(f"[dq]   skew={sk:.3f} kurt={ku:.3f} JBp≈{p:.3g}")

        # LiDAR
        if len(self.buf_lidar) >= 10:
            scans = np.asarray(self.buf_lidar, dtype=np.float64)  # (M,55)
            s_cell = self.centered_std(scans)  # (55,)
            s_mean = float(np.mean(s_cell))
            s_med = float(np.median(s_cell))
            s_max = float(np.max(s_cell))

            grid = s_cell.reshape(LIDAR_NUM_X, LIDAR_NUM_Y)

            print(
                "\n========== LiDAR heightmap noise (centered std, meters) =========="
            )
            print(f"scans collected: {scans.shape[0]}")
            print(f"mean std/cell  : {s_mean:.6f} m")
            print(f"median std/cell: {s_med:.6f} m")
            print(f"max std/cell   : {s_max:.6f} m")

            print("\nPer-cell std grid (ahead at TOP):")
            print(np.round(grid[::-1, :], 4))

            # Gaussianity for lidar (aggregate all cells/time)
            sk, ku, jb, p = skew_kurt_jb(scans.reshape(-1))
            print(f"\n[lidar_all] skew={sk:.3f} kurt={ku:.3f} JBp≈{p:.3g}")

            # Recommend training noise (conservative)
            recommended = max(0.02, round(2.0 * s_mean, 3))
            print(
                f"\nSuggested height_noise_std (conservative): {recommended:.3f} m  (2x mean, floor 0.02)"
            )
        else:
            print("\n========== LiDAR ==========")
            print("Not enough lidar scans collected to estimate noise reliably.")

        # Copy/paste block (scalar per channel)
        print("\n========== Copy/paste (scalar noise levels) ==========")
        print("obs_noise = {")
        print(f'  "ang_vel": {float(np.mean(s_gyro)):.6g},')
        print(f'  "gravity": {float(np.mean(s_grav)):.6g},')
        print(f'  "dof_pos": {float(np.mean(s_q)):.6g},')
        print(f'  "dof_vel": {float(np.mean(s_dq)):.6g},')
        print('  "commands": 0.0,')
        print('  "actions": 0.0,')
        print("}")
        if len(self.buf_lidar) >= 10:
            print(
                f"lidar_height_noise_std = {max(0.02, 2.0 * float(np.mean(self.centered_std(np.asarray(self.buf_lidar))))):.3f}"
            )


def main():
    print("=" * 60)
    print("  GO2 DDS OBS + LIDAR (11x5) NOISE ESTIMATOR")
    print("=" * 60)
    print(f"SAMPLE_HZ={SAMPLE_HZ}  SECONDS={SECONDS}")
    print(
        f"LIDAR grid={LIDAR_NUM_X}x{LIDAR_NUM_Y}  x={LIDAR_X_RANGE}  y={LIDAR_Y_RANGE}"
    )
    print(f"LIDAR_FORWARD_SIGN={LIDAR_FORWARD_SIGN}  PC_STRIDE={PC_STRIDE}\n")

    if len(sys.argv) > 1:
        ChannelFactoryInitialize(0, sys.argv[1])
        print("DDS interface:", sys.argv[1])
    else:
        ChannelFactoryInitialize(0)
        print("DDS interface: default")

    NoiseEstimatorDDS().run()


if __name__ == "__main__":
    main()
