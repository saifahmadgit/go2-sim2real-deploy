#!/usr/bin/env python3
import sys
import time

import numpy as np

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_

# -------------------------
# Config
# -------------------------
TOPIC = "rt/utlidar/cloud"  # try "rt/utlidar/cloud_deskewed" if needed
STRIDE = 6  # decimation (bigger = faster, fewer points)
MAX_POINTS = 40000  # cap for safety

# Map ROI (robot frame)
X_BACK = 0.20  # meters behind
X_FRONT = 1.00  # meters ahead
Y_HALF = 0.25  # meters left/right

NX, NY = 11, 5  # grid size

# Z filtering (tune if needed)
Z_MIN, Z_MAX = -0.8, 2.0

# If your “front” points have negative x in the cloud, set to -1
FORWARD_SIGN = -1

# Cell validity / smoothing
MIN_POINTS_PER_CELL = 6
EMA_ALPHA = 0.35  # 0=no smoothing, closer to 1=more responsive

# ASCII rendering
CHARS = " .:-=+*#%@"

# -------------------------
# PointCloud2 decoding
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

    return xyz


# -------------------------
# Heightmap builder (11x5)
# height = max_z - min_z per cell
# -------------------------
def heightmap_11x5(xyz):
    if xyz is None or xyz.shape[0] == 0:
        return np.zeros((NX, NY), dtype=np.float32), np.zeros((NX, NY), dtype=np.int32)

    # Make +x be "forward"
    x = FORWARD_SIGN * xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]

    x_min, x_max = -X_BACK, X_FRONT
    y_min, y_max = -Y_HALF, Y_HALF

    m = (
        (x >= x_min)
        & (x < x_max)
        & (y >= y_min)
        & (y < y_max)
        & (z >= Z_MIN)
        & (z <= Z_MAX)
    )
    if not np.any(m):
        return np.zeros((NX, NY), dtype=np.float32), np.zeros((NX, NY), dtype=np.int32)

    x = x[m]
    y = y[m]
    z = z[m]

    dx = (x_max - x_min) / NX
    dy = (y_max - y_min) / NY

    ix = ((x - x_min) / dx).astype(np.int32)
    iy = ((y - y_min) / dy).astype(np.int32)
    ix = np.clip(ix, 0, NX - 1)
    iy = np.clip(iy, 0, NY - 1)

    min_z = np.full((NX, NY), np.inf, dtype=np.float32)
    max_z = np.full((NX, NY), -np.inf, dtype=np.float32)
    cnt = np.zeros((NX, NY), dtype=np.int32)

    np.minimum.at(min_z, (ix, iy), z)
    np.maximum.at(max_z, (ix, iy), z)
    np.add.at(cnt, (ix, iy), 1)

    h = (max_z - min_z).astype(np.float32)
    h[~np.isfinite(h)] = 0.0
    h = np.maximum(h, 0.0)

    # Zero-out sparse cells
    h[cnt < MIN_POINTS_PER_CELL] = 0.0
    return h, cnt


def ascii_heatmap(h, h_max=0.40):
    # map heights to CHARS (clamped)
    v = np.clip(h / max(h_max, 1e-6), 0.0, 1.0)
    idx = (v * (len(CHARS) - 1)).astype(np.int32)
    # Print with "ahead" at top: show x bins from front->back
    lines = []
    for i in range(NX - 1, -1, -1):
        row = "".join(CHARS[idx[i, j]] for j in range(NY))
        lines.append(row)
    return "\n".join(lines)


class LidarHMProbe:
    def __init__(self, topic):
        self.msg = None
        self.sub = ChannelSubscriber(topic, PointCloud2_)
        self.sub.Init(self._cb, 1)
        self.h_ema = np.zeros((NX, NY), dtype=np.float32)
        self.hz = 0.0
        self._t_last = None
        self.topic = topic

    def _cb(self, msg):
        now = time.time()
        if self._t_last is not None:
            dt = now - self._t_last
            if dt > 1e-6:
                inst = 1.0 / dt
                self.hz = inst if self.hz == 0 else (0.9 * self.hz + 0.1 * inst)
        self._t_last = now
        self.msg = msg

    def step(self):
        if self.msg is None:
            return None, None, None

        xyz = pc2_to_xyz(self.msg, stride=STRIDE, max_points=MAX_POINTS)
        h, cnt = heightmap_11x5(xyz)

        # EMA smoothing
        self.h_ema = (1.0 - EMA_ALPHA) * self.h_ema + EMA_ALPHA * h
        return self.h_ema, cnt, xyz


def main():
    if len(sys.argv) > 1:
        ChannelFactoryInitialize(0, sys.argv[1])
        print("DDS interface:", sys.argv[1])
    else:
        ChannelFactoryInitialize(0)
        print("DDS interface: default")

    probe = LidarHMProbe(TOPIC)
    print(f"Subscribing: {TOPIC}")
    print(
        f"ROI: x=[{-X_BACK:.2f},{X_FRONT:.2f}] m, y=[{-Y_HALF:.2f},{Y_HALF:.2f}] m, grid={NX}x{NY}"
    )
    print(f"FORWARD_SIGN={FORWARD_SIGN}  (flip to +1 if front is +x for you)\n")

    try:
        while True:
            h, cnt, xyz = probe.step()
            if h is None:
                print("Waiting for LiDAR...")
                time.sleep(0.2)
                continue

            # Quick “front danger” check (front-most 2 x-bins)
            front_slice = h[NX - 2 : NX, :]  # two most-forward rows (in x)
            front_max = float(np.max(front_slice))

            # Clear-ish screen (portable)
            print("\033[2J\033[H", end="")

            print("========== LIVE 11x5 HEIGHTMAP (meters) ==========")
            print(
                f"topic={TOPIC}  lidar_rate~{probe.hz:.1f}Hz  stride={STRIDE}  pts_used~{xyz.shape[0]}"
            )
            print(f"front_max_height~{front_max:.3f} m   (front 2 rows)")
            print("")
            print("ASCII (ahead at TOP, behind at BOTTOM; left->right across columns):")
            print(ascii_heatmap(h, h_max=0.40))
            print("")
            print("Numeric grid (ahead at TOP):")
            # print numeric with ahead on top (reverse x)
            print(np.round(h[::-1, :], 3))
            print("=================================================")

            time.sleep(0.10)

    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
