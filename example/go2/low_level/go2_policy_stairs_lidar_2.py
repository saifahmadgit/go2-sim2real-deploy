"""
go2_deploy_stairs_info.py
==========================
Go2 stair climbing deployment with 4-value stair info.

Much simpler than the 5×3 LIDAR grid version. The policy receives:
  [0] edge_distance : from LIDAR edge detector (0..0.27m)
  [1] stair_height  : user-provided constant (meters)
  [2] stair_depth   : user-provided constant (meters)
  [3] direction     : auto from LIDAR edge sign (+1/-1/0)

Usage:
  1. Set STAIR_HEIGHT and STAIR_DEPTH at the top of this file
  2. python go2_deploy_stairs_info.py

The LIDAR edge detector scans the near-field (0-27cm) for height
changes. If found, reports distance and direction. If not found,
reports 0.27m (sentinel) and direction 0.

CHECKPOINT: Must be from train_stair5_info.py (53-dim actor obs).
"""

import math as pymath
import os
import sys
import time

import numpy as np
import torch

# ============================================================
# MODE
# ============================================================
MODE = "robot_run"  # "dummy" | "robot_print" | "robot_run"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_PATH = os.path.join(SCRIPT_DIR, "stairs_lidar2_40500.pt")

# ============================================================
# STAIR DIMENSIONS — SET THESE TO MATCH YOUR STAIRCASE
# ============================================================
STAIR_HEIGHT = 0.15  # meters (riser height, always positive)
STAIR_DEPTH = 0.03  # meters (tread depth, always positive)

# ============================================================
# VELOCITY COMMANDS
# ============================================================
FORWARD_VX = 0.4
BACKWARD_VX = 0.3
LEFT_VY = 0.5
RIGHT_VY = 0.5
YAW_CW_WZ = 0.7
YAW_CCW_WZ = 0.7

# ============================================================
# ACTION / TIMING
# ============================================================
ACTION_CLIP = 100.0
ACTION_SCALE = 0.25

POLICY_HZ = 50.0
LOWCMD_HZ = 500.0

STAND_SECONDS = 4.0
STAND_KP = 40.0
STAND_KD = 0.5

# ============================================================
# PLS (Per-Leg Stiffness)
# ============================================================
PLS_ENABLE = True
PLS_KP_DEFAULT = 40.0
PLS_KP_ACTION_SCALE = 20.0
PLS_KP_RANGE = [10.0, 70.0]

KP_FACTOR = 1.0
KD_FACTOR = 1.5

POLICY_KP_FALLBACK = 40.0
POLICY_KD_FALLBACK = 2.0

MAX_STEP_RAD = 0.1
PRINT_EVERY_N = 10
SIMULATE_1STEP_ACTION_LATENCY = False
TRANSITION_SECONDS = 2.0

# ============================================================
# EDGE DETECTION CONFIG
# ============================================================
EDGE_MAX_RANGE = 0.27  # reliable LIDAR range
EDGE_Z_THRESHOLD = 0.04  # 4cm z-change = edge
PROFILE_BIN_SIZE = 0.01  # 1cm profile resolution
PROFILE_X_MIN = -0.05
PROFILE_X_MAX = 0.30
Y_CAPTURE_HALF = 0.20
MIN_PTS_PER_BIN = 3
GROUND_X_MAX = 0.08  # near-field ground reference

# Calibration
LIDAR_CALIB_FILE = "stair_scan_calib.npz"
NOMINAL_STANDING_HEIGHT = 0.35
LIDAR_CALIB_DURATION = 5.0

# ============================================================
# OBSERVATION SCALING (must match training)
# ============================================================
EDGE_DIST_SCALE = 3.7  # 1/0.27
HEIGHT_SCALE = 5.0  # 1/0.20
DEPTH_SCALE = 3.3  # 1/0.30
DIRECTION_SCALE = 1.0

# ============================================================
# OBSERVATION DIMENSIONS
# ============================================================
NUM_POS_ACTIONS = 12
NUM_STIFFNESS_ACTIONS = 4 if PLS_ENABLE else 0
NUM_ACT = NUM_POS_ACTIONS + NUM_STIFFNESS_ACTIONS  # 16

NUM_STAIR_INFO = 4
NUM_PROPRIO = 3 + 3 + 3 + 12 + 12 + NUM_ACT  # 49
NUM_OBS = NUM_PROPRIO + NUM_STAIR_INFO  # 53

OBS_SCALES = {"lin_vel": 2.0, "ang_vel": 0.25, "dof_pos": 1.0, "dof_vel": 0.05}

JOINT_NAMES = [
    "FR_hip",
    "FR_thigh",
    "FR_calf",
    "FL_hip",
    "FL_thigh",
    "FL_calf",
    "RR_hip",
    "RR_thigh",
    "RR_calf",
    "RL_hip",
    "RL_thigh",
    "RL_calf",
]
LEG_NAMES = ["FR", "FL", "RR", "RL"]
LEG_JOINT_MAP = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]

DEFAULT_DOF_POS = torch.tensor(
    [0.0, 0.8, -1.5, 0.0, 0.8, -1.5, 0.0, 1.0, -1.5, 0.0, 1.0, -1.5],
    dtype=torch.float32,
)
STAND_DOF_POS = DEFAULT_DOF_POS.clone()


# ============================================================
# LIDAR: Point Cloud Decode
# ============================================================

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
    pts = np.frombuffer(data[: n * int(msg.point_step)], dtype=dtype)
    return np.stack([pts["x"], pts["y"], pts["z"]], axis=1).astype(np.float32)


# ============================================================
# LIDAR: Edge Detector
# ============================================================


class EdgeDetector:
    """
    Detects nearest stair edge from LIDAR point cloud.

    Returns:
      edge_distance : 0..0.27m (or 0.27 if no edge)
      direction     : +1 ascending, -1 descending, 0 no edge
    """

    def __init__(self):
        self.ground_z_calib = -NOMINAL_STANDING_HEIGHT
        self.calibrated = False

        self.profile_edges = np.arange(
            PROFILE_X_MIN, PROFILE_X_MAX + PROFILE_BIN_SIZE, PROFILE_BIN_SIZE
        )
        self.profile_centers = (self.profile_edges[:-1] + self.profile_edges[1:]) / 2
        self.n_bins = len(self.profile_centers)

    def detect(self, pts_lidar, roll=0.0, pitch=0.0):
        """
        Returns (edge_distance, direction) from latest LIDAR cloud.
        """
        if len(pts_lidar) == 0:
            return EDGE_MAX_RANGE, 0.0

        # Transform LIDAR → body frame
        bx = -pts_lidar[:, 0]
        by = pts_lidar[:, 1]
        bz = -pts_lidar[:, 2]

        valid = (bz > -2.0) & (bz < 1.0) & np.all(np.isfinite(pts_lidar), axis=1)
        bx, by, bz = bx[valid], by[valid], bz[valid]

        if len(bx) == 0:
            return EDGE_MAX_RANGE, 0.0

        # Gravity projection for pitch correction
        cr, sr = pymath.cos(roll), pymath.sin(roll)
        cp, sp = pymath.cos(pitch), pymath.sin(pitch)
        r2 = np.array([-sp, cp * sr, cp * cr], dtype=np.float64)
        z_grav = r2[0] * bx + r2[1] * by + r2[2] * bz

        # Measure ground z (near-field reference)
        y_ok = np.abs(by) <= Y_CAPTURE_HALF
        gnd_mask = y_ok & (bx >= -0.05) & (bx <= GROUND_X_MAX)
        if np.sum(gnd_mask) >= 5:
            ground_z = float(np.median(z_grav[gnd_mask]))
        else:
            ground_z = self.ground_z_calib

        # Build 1D forward profile
        filt = y_ok
        bx_f, zg_f = bx[filt], z_grav[filt]
        profile_z = np.full(self.n_bins, np.nan)
        bin_idx = np.digitize(bx_f, self.profile_edges) - 1
        for b in range(self.n_bins):
            mask = bin_idx == b
            if np.sum(mask) >= MIN_PTS_PER_BIN:
                profile_z[b] = np.median(zg_f[mask])

        # Find first significant height change
        valid_z = ~np.isnan(profile_z)
        for b in range(self.n_bins):
            if not valid_z[b]:
                continue
            if self.profile_centers[b] > EDGE_MAX_RANGE:
                break
            if self.profile_centers[b] < 0.0:
                continue
            dz = profile_z[b] - ground_z
            if abs(dz) > EDGE_Z_THRESHOLD:
                edge_distance = float(self.profile_centers[b])
                direction = 1.0 if dz > 0 else -1.0
                return edge_distance, direction

        return EDGE_MAX_RANGE, 0.0

    def calibrate_from_flat(self, ground_z_list):
        if len(ground_z_list) < 5:
            return False
        self.ground_z_calib = float(np.median(ground_z_list))
        self.calibrated = True
        print(f"  [LIDAR] Calibrated ground_z = {self.ground_z_calib:+.4f}m")
        try:
            np.savez(LIDAR_CALIB_FILE, ground_z=self.ground_z_calib)
        except Exception:
            pass
        return True

    def load_calibration(self):
        if not os.path.exists(LIDAR_CALIB_FILE):
            return False
        try:
            data = np.load(LIDAR_CALIB_FILE)
            self.ground_z_calib = float(data["ground_z"])
            self.calibrated = True
            print(f"  [LIDAR] Loaded calib: ground_z={self.ground_z_calib:+.4f}")
            return True
        except Exception:
            return False


# ============================================================
# LIDAR: DDS callback + global state
# ============================================================

_lidar_cloud_msg = None
_lidar_cloud_count = 0


def _on_lidar_cloud(msg):
    global _lidar_cloud_msg, _lidar_cloud_count
    _lidar_cloud_msg = msg
    _lidar_cloud_count += 1


_edge_detector = None
_cached_edge_distance = EDGE_MAX_RANGE
_cached_direction = 0.0
_lidar_last_cloud_count = 0


def update_edge_detection(roll, pitch):
    """
    Update cached edge detection from latest LIDAR cloud.
    Returns (edge_distance, direction).
    """
    global _cached_edge_distance, _cached_direction, _lidar_last_cloud_count

    if _edge_detector is None or _lidar_cloud_msg is None:
        return _cached_edge_distance, _cached_direction

    if _lidar_cloud_count == _lidar_last_cloud_count:
        return _cached_edge_distance, _cached_direction

    _lidar_last_cloud_count = _lidar_cloud_count

    try:
        xyz = decode_xyz(_lidar_cloud_msg)
        if len(xyz) > 100:
            _cached_edge_distance, _cached_direction = _edge_detector.detect(
                xyz, roll, pitch
            )
    except Exception:
        pass

    return _cached_edge_distance, _cached_direction


# ============================================================
# Build stair info observation (4 scaled values)
# ============================================================


def build_stair_info(edge_distance, stair_height, stair_depth, direction):
    """
    Build 4-value stair info tensor matching training exactly.

    Returns: (4,) float32 tensor [edge_dist*scale, height*scale, depth*scale, direction*scale]
    """
    return torch.tensor(
        [
            edge_distance * EDGE_DIST_SCALE,
            stair_height * HEIGHT_SCALE,
            stair_depth * DEPTH_SCALE,
            direction * DIRECTION_SCALE,
        ],
        dtype=torch.float32,
    )


# ============================================================
# PLS: compute per-joint Kp/Kd
# ============================================================


def compute_pls_kp_kd(stiffness_actions_4):
    kp_per_leg = PLS_KP_DEFAULT + stiffness_actions_4 * PLS_KP_ACTION_SCALE
    kp_per_leg = torch.clamp(kp_per_leg, PLS_KP_RANGE[0], PLS_KP_RANGE[1])
    kp_12 = torch.zeros(12, dtype=torch.float32)
    for leg_idx in range(4):
        for joint_idx in LEG_JOINT_MAP[leg_idx]:
            kp_12[joint_idx] = kp_per_leg[leg_idx]
    kd_12 = 0.2 * torch.sqrt(kp_12)
    kp_12 = kp_12 * KP_FACTOR
    kd_12 = kd_12 * KD_FACTOR
    return kp_12, kd_12


# ============================================================
# Quaternion helpers
# ============================================================


def quat_conj(q_wxyz):
    return torch.tensor(
        [q_wxyz[0], -q_wxyz[1], -q_wxyz[2], -q_wxyz[3]], dtype=torch.float32
    )


def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return torch.tensor(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=torch.float32,
    )


def rotate_vec_by_quat(v_xyz, q_wxyz):
    vq = torch.tensor([0.0, v_xyz[0], v_xyz[1], v_xyz[2]], dtype=torch.float32)
    return quat_mul(quat_mul(q_wxyz, vq), quat_conj(q_wxyz))[1:]


def projected_gravity_from_quat_body_in_world(q_wxyz):
    g_world = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32)
    q = torch.tensor(q_wxyz, dtype=torch.float32)
    return rotate_vec_by_quat(g_world, quat_conj(q))


def pitch_roll_from_quat(q_wxyz):
    w, x, y, z = q_wxyz
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll_rad = torch.atan2(torch.tensor(sinr_cosp), torch.tensor(cosr_cosp))
    sinp = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch_rad = torch.asin(torch.tensor(sinp))
    return float(pitch_rad) * 57.2958, float(roll_rad) * 57.2958


def get_imu_rp_from_raw(raw):
    quat = raw["imu"]["quat_wxyz"]
    w, x, y, z = quat
    roll = pymath.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = pymath.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    return roll, pitch


# ============================================================
# Build observation vector (53 dims = 49 proprio + 4 stair info)
# ============================================================


def build_obs(raw, command_3, last_action, stair_info_4):
    """
    Build actor observation matching train_stair5_info.py exactly.

    Layout (53 dims):
      [0:3]   angular velocity * 0.25
      [3:6]   projected gravity
      [6:9]   commands * [2.0, 2.0, 0.25]
      [9:21]  (joint_pos - default) * 1.0
      [21:33] joint_vel * 0.05
      [33:49] last_actions (16)
      [49:53] stair_info (4 scaled values)
    """
    gyro = torch.tensor(raw["imu"]["gyro_rad_s"], dtype=torch.float32)
    proj_g = projected_gravity_from_quat_body_in_world(raw["imu"]["quat_wxyz"])
    q = torch.tensor([m["q_rad"] for m in raw["motors"]], dtype=torch.float32)
    dq = torch.tensor([m["dq_rad_s"] for m in raw["motors"]], dtype=torch.float32)
    cmd = torch.tensor(command_3, dtype=torch.float32)
    cmd_scale = torch.tensor(
        [OBS_SCALES["lin_vel"], OBS_SCALES["lin_vel"], OBS_SCALES["ang_vel"]],
        dtype=torch.float32,
    )

    obs = torch.cat(
        [
            gyro * OBS_SCALES["ang_vel"],  # 3
            proj_g,  # 3
            cmd * cmd_scale,  # 3
            (q - DEFAULT_DOF_POS) * OBS_SCALES["dof_pos"],  # 12
            dq * OBS_SCALES["dof_vel"],  # 12
            last_action,  # 16
            stair_info_4,  # 4
        ],
        dim=0,
    )

    if obs.shape[0] != NUM_OBS:
        raise RuntimeError(f"obs should be {NUM_OBS}, got {obs.shape[0]}")
    return obs


# ============================================================
# Load policy
# ============================================================


def load_policy(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["model_state_dict"]

    try:
        from rsl_rl.modules import ActorCritic
    except Exception:
        from rsl_rl.modules.actor_critic import ActorCritic

    num_critic_obs = sd["critic.0.weight"].shape[1]

    policy = ActorCritic(
        num_actor_obs=NUM_OBS,
        num_critic_obs=num_critic_obs,
        num_actions=NUM_ACT,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        init_noise_std=1.0,
    )
    policy.load_state_dict(sd, strict=True)
    policy.eval()
    print(
        f"  Loaded checkpoint: actor_obs={NUM_OBS}, "
        f"critic_obs={num_critic_obs}, actions={NUM_ACT}"
    )
    return policy


# ============================================================
# Debug print
# ============================================================


def print_status_line(
    step,
    command_3,
    target_q_12,
    edge_dist,
    direction,
    kp_12=None,
    pitch_deg=0.0,
    roll_deg=0.0,
):
    vx, vy, wz = command_3
    kp_str = ""
    if kp_12 is not None:
        kps = [float(kp_12[LEG_JOINT_MAP[i][0]]) for i in range(4)]
        kp_str = f" Kp=[{kps[0]:.0f},{kps[1]:.0f},{kps[2]:.0f},{kps[3]:.0f}]"

    dir_sym = {1.0: "↑", -1.0: "↓"}.get(direction, "—")

    print(
        f"\r  step={step:06d}  cmd=[{vx:+.2f},{vy:+.2f},{wz:+.2f}]  "
        f"pitch={pitch_deg:+5.1f}° roll={roll_deg:+5.1f}°"
        f"  edge={edge_dist:.2f}m {dir_sym}"
        f"{kp_str}  ",
        end="",
        flush=True,
    )


# ============================================================
# Robot lowstate → raw dict
# ============================================================


def lowstate_to_raw(low_state):
    imu = low_state.imu_state
    gyro = list(imu.gyroscope)
    quat = list(imu.quaternion)
    quat_wxyz = [float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])]
    gyro_xyz = [float(gyro[0]), float(gyro[1]), float(gyro[2])]
    motors = []
    for i in range(12):
        ms = low_state.motor_state[i]
        motors.append({"q_rad": float(ms.q), "dq_rad_s": float(ms.dq)})
    return {"imu": {"gyro_rad_s": gyro_xyz, "quat_wxyz": quat_wxyz}, "motors": motors}


# ============================================================
# Keyboard input
# ============================================================

import select
import termios
import tty


class RawTerminal:
    def __init__(self):
        self.fd = sys.stdin.fileno()
        self.old_settings = None

    def __enter__(self):
        self.old_settings = termios.tcgetattr(self.fd)
        tty.setraw(self.fd)
        return self

    def __exit__(self, *args):
        if self.old_settings is not None:
            termios.tcsetattr(self.fd, termios.TCSANOW, self.old_settings)

    def get_key(self):
        rlist, _, _ = select.select([sys.stdin], [], [], 0)
        if not rlist:
            return None
        ch = sys.stdin.read(1)
        if ch == "\x1b":
            rlist2, _, _ = select.select([sys.stdin], [], [], 0.005)
            if rlist2:
                ch2 = sys.stdin.read(1)
                if ch2 == "[":
                    ch3 = sys.stdin.read(1)
                    return {"A": "UP", "B": "DOWN", "C": "RIGHT", "D": "LEFT"}.get(ch3)
            return "ESC"
        elif ch == "\x03":
            return "CTRL_C"
        return ch


command_state = {"vx": 0.0, "vy": 0.0, "wz": 0.0}


def make_command_list():
    return [command_state["vx"], command_state["vy"], command_state["wz"]]


def handle_key(key):
    if key is None:
        return True
    if isinstance(key, str) and len(key) == 1:
        key = key.lower()
    if key in ("x", "CTRL_C"):
        return False
    if key == "w":
        command_state.update({"vx": FORWARD_VX, "vy": 0.0, "wz": 0.0})
    elif key == "s":
        command_state.update({"vx": -BACKWARD_VX, "vy": 0.0, "wz": 0.0})
    elif key == "d":
        command_state.update({"vx": 0.0, "vy": RIGHT_VY, "wz": 0.0})
    elif key == "a":
        command_state.update({"vx": 0.0, "vy": -LEFT_VY, "wz": 0.0})
    elif key == "q":
        command_state.update({"vx": 0.0, "vy": 0.0, "wz": -YAW_CW_WZ})
    elif key == "r":
        command_state.update({"vx": 0.0, "vy": 0.0, "wz": YAW_CCW_WZ})
    elif key == " ":
        command_state.update({"vx": 0.0, "vy": 0.0, "wz": 0.0})
    return True


def print_controls():
    print("\n============ STAIR CLIMBING (4-value info) ============")
    print(f"  W : forward (vx={FORWARD_VX:.2f})")
    print(f"  S : backward (vx={-BACKWARD_VX:.2f})")
    print("  D : right   A : left")
    print("  Q : yaw CW  R : yaw CCW")
    print("  SPACE : zero velocity")
    print("  E : return to stand")
    print("  X : quit")
    print(
        f"\n  Stair: height={STAIR_HEIGHT * 100:.1f}cm  depth={STAIR_DEPTH * 100:.1f}cm"
    )
    print(f"  Edge detection: 0..{EDGE_MAX_RANGE * 100:.0f}cm range")
    print("=======================================================\n")


def slew_limit(prev_q, new_q, max_step_rad):
    delta = torch.clamp(new_q - prev_q, -max_step_rad, max_step_rad)
    return prev_q + delta


# ============================================================
# DDS init
# ============================================================


def init_dds():
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize

    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)


def init_lidar_subscriber():
    from unitree_sdk2py.core.channel import ChannelSubscriber
    from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_

    sub = ChannelSubscriber("rt/utlidar/cloud", PointCloud2_)
    sub.Init(_on_lidar_cloud, 10)
    print("  LIDAR subscriber started (rt/utlidar/cloud)")
    return sub


def wait_for_lowstate():
    from unitree_sdk2py.core.channel import ChannelSubscriber
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_

    latest = {"msg": None}

    def cb(msg):
        latest["msg"] = msg

    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init(cb, 10)
    print("Waiting for rt/lowstate...")
    while latest["msg"] is None:
        time.sleep(0.05)
    print("Got first lowstate.")
    return latest


def release_sport_and_highlevel():
    from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import (
        MotionSwitcherClient,
    )
    from unitree_sdk2py.go2.sport.sport_client import SportClient

    sc = SportClient()
    sc.SetTimeout(5.0)
    sc.Init()
    msc = MotionSwitcherClient()
    msc.SetTimeout(5.0)
    msc.Init()
    print("\nReleasing high-level control...")
    status, result = msc.CheckMode()
    while result.get("name"):
        sc.StandDown()
        msc.ReleaseMode()
        time.sleep(1.0)
        status, result = msc.CheckMode()
    print("High-level control released.")


# ============================================================
# LIDAR calibration
# ============================================================


def run_lidar_calibration(detector):
    global _lidar_cloud_msg, _lidar_cloud_count

    ground_zs = []
    t0 = time.time()
    last_cc = _lidar_cloud_count

    print(f"\n  [LIDAR] Calibrating for {LIDAR_CALIB_DURATION:.0f}s on flat ground...")

    while time.time() - t0 < LIDAR_CALIB_DURATION:
        if _lidar_cloud_msg is not None and _lidar_cloud_count > last_cc:
            last_cc = _lidar_cloud_count
            try:
                xyz = decode_xyz(_lidar_cloud_msg)
                if len(xyz) > 100:
                    bx = -xyz[:, 0]
                    by = xyz[:, 1]
                    bz = -xyz[:, 2]
                    valid = (bz > -2.0) & (bz < 1.0) & np.all(np.isfinite(xyz), axis=1)
                    bx, by, bz = bx[valid], by[valid], bz[valid]
                    y_ok = np.abs(by) <= Y_CAPTURE_HALF
                    gnd = y_ok & (bx >= -0.05) & (bx <= GROUND_X_MAX)
                    if np.sum(gnd) >= 10:
                        gz = float(np.median(bz[gnd]))
                        ground_zs.append(gz)
            except Exception:
                pass
            pct = (time.time() - t0) / LIDAR_CALIB_DURATION * 100
            print(f"\r  {pct:5.1f}%  samples={len(ground_zs)}", end="", flush=True)
        time.sleep(0.02)

    print()
    return detector.calibrate_from_flat(ground_zs)


# ============================================================
# robot_run: full deployment
# ============================================================


def run_robot_run(policy):
    global _edge_detector

    from unitree_sdk2py.core.channel import ChannelPublisher
    from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
    from unitree_sdk2py.utils.crc import CRC
    from unitree_sdk2py.utils.thread import RecurrentThread

    import unitree_legged_const as go2

    init_dds()
    latest = wait_for_lowstate()

    lidar_sub = init_lidar_subscriber()

    print("  Waiting for LIDAR data...", end="", flush=True)
    t0 = time.time()
    while _lidar_cloud_count == 0 and time.time() - t0 < 10.0:
        time.sleep(0.3)
        print(".", end="", flush=True)
    if _lidar_cloud_count > 0:
        print(f" OK ({_lidar_cloud_count} clouds)")
    else:
        print("\n  [WARN] No LIDAR — edge detection will return defaults")

    _edge_detector = EdgeDetector()
    loaded = _edge_detector.load_calibration()
    if not loaded and _lidar_cloud_count > 0:
        run_lidar_calibration(_edge_detector)

    release_sport_and_highlevel()

    # Publisher
    pub = ChannelPublisher("rt/lowcmd", LowCmd_)
    pub.Init()
    crc = CRC()
    low_cmd = unitree_go_msg_dds__LowCmd_()
    low_cmd.head[0] = 0xFE
    low_cmd.head[1] = 0xEF
    low_cmd.level_flag = 0xFF
    low_cmd.gpio = 0
    for i in range(20):
        low_cmd.motor_cmd[i].mode = 0x01
        low_cmd.motor_cmd[i].q = go2.PosStopF
        low_cmd.motor_cmd[i].dq = go2.VelStopF
        low_cmd.motor_cmd[i].kp = 0.0
        low_cmd.motor_cmd[i].kd = 0.0
        low_cmd.motor_cmd[i].tau = 0.0

    shared = {
        "target_q": None,
        "kp_per_joint": torch.full((12,), STAND_KP, dtype=torch.float32),
        "kd_per_joint": torch.full((12,), STAND_KD, dtype=torch.float32),
    }

    raw0 = lowstate_to_raw(latest["msg"])
    start_q = torch.tensor([m["q_rad"] for m in raw0["motors"]], dtype=torch.float32)
    shared["target_q"] = start_q.clone()

    def write_lowcmd():
        tq = shared["target_q"]
        if tq is None:
            return
        kp_arr = shared["kp_per_joint"]
        kd_arr = shared["kd_per_joint"]
        for i in range(12):
            low_cmd.motor_cmd[i].mode = 0x01
            low_cmd.motor_cmd[i].q = float(tq[i])
            low_cmd.motor_cmd[i].dq = 0.0
            low_cmd.motor_cmd[i].kp = float(kp_arr[i])
            low_cmd.motor_cmd[i].kd = float(kd_arr[i])
            low_cmd.motor_cmd[i].tau = 0.0
        low_cmd.crc = crc.Crc(low_cmd)
        pub.Write(low_cmd)

    writer = RecurrentThread(
        interval=1.0 / LOWCMD_HZ, target=write_lowcmd, name="lowcmd_writer"
    )
    writer.Start()

    # Ramp to stand
    print("\nRamping to STAND pose...")
    ramp_steps = max(1, int(STAND_SECONDS * POLICY_HZ))
    prev_q = start_q.clone()
    for k in range(ramp_steps):
        alpha = (k + 1) / float(ramp_steps)
        desired = (1 - alpha) * start_q + alpha * STAND_DOF_POS
        desired = slew_limit(prev_q, desired, MAX_STEP_RAD)
        shared["target_q"] = desired.clone()
        prev_q = desired.clone()
        time.sleep(1.0 / POLICY_HZ)
    print("Stand pose reached.")

    print(
        f"\n*** ROBOT STANDING — LIDAR {'ACTIVE' if _lidar_cloud_count > 0 else 'INACTIVE'} ***"
    )
    print(
        f"  Obs: {NUM_OBS} dims ({NUM_PROPRIO} proprio + {NUM_STAIR_INFO} stair_info)"
    )
    print(
        f"  Stair: height={STAIR_HEIGHT * 100:.1f}cm  depth={STAIR_DEPTH * 100:.1f}cm"
    )
    print("Type 'go' to enable keyboard control.")
    user = input("> ").strip().lower()
    if user != "go":
        print("Aborted.")
        return

    print_controls()

    STATE_STANDING = "standing"
    STATE_POLICY = "policy"
    STATE_TRANSITION = "transition"
    current_state = STATE_STANDING

    dt = 1.0 / POLICY_HZ
    step = 0
    last_action_for_obs = torch.zeros(NUM_ACT, dtype=torch.float32)
    prev_policy_action = torch.zeros(NUM_ACT, dtype=torch.float32)
    prev_target_q = shared["target_q"].clone()

    transition_start_q = None
    transition_step = 0
    transition_steps = int(TRANSITION_SECONDS * POLICY_HZ)

    MOVEMENT_KEYS = ["w", "a", "s", "d", "q", "r"]
    running = True

    try:
        with RawTerminal() as term:
            next_tick = time.monotonic()

            while running:
                now = time.monotonic()
                sleep_time = next_tick - now
                if sleep_time > 0:
                    time.sleep(sleep_time)
                next_tick += dt

                key = term.get_key()
                if key and len(key) == 1:
                    key = key.lower()
                if key in ("x", "CTRL_C"):
                    running = False
                    break

                is_movement = key in MOVEMENT_KEYS
                is_stop = key == "e"

                if current_state == STATE_STANDING:
                    if is_movement or key == " ":
                        print("\n→ POLICY mode (stair climbing)")
                        current_state = STATE_POLICY
                        shared["kp_per_joint"][:] = POLICY_KP_FALLBACK
                        shared["kd_per_joint"][:] = POLICY_KD_FALLBACK
                        last_action_for_obs = torch.zeros(NUM_ACT, dtype=torch.float32)
                        prev_policy_action = torch.zeros(NUM_ACT, dtype=torch.float32)
                        prev_target_q = STAND_DOF_POS.clone()
                        step = 0
                        handle_key(key)
                    else:
                        shared["target_q"] = STAND_DOF_POS.clone()
                        prev_target_q = STAND_DOF_POS.clone()
                        if key:
                            handle_key(key)
                        continue

                elif current_state == STATE_POLICY:
                    if is_stop:
                        print("\n→ Returning to STAND")
                        current_state = STATE_TRANSITION
                        transition_start_q = prev_target_q.clone()
                        transition_step = 0
                        command_state.update({"vx": 0, "vy": 0, "wz": 0})
                        continue

                    if not handle_key(key):
                        running = False
                        break

                    raw = lowstate_to_raw(latest["msg"])
                    command = make_command_list()

                    # Edge detection from LIDAR
                    roll_rad, pitch_rad = get_imu_rp_from_raw(raw)
                    edge_dist, direction = update_edge_detection(roll_rad, pitch_rad)

                    # Build stair info (4 scaled values)
                    stair_info = build_stair_info(
                        edge_dist, STAIR_HEIGHT, STAIR_DEPTH, direction
                    )

                    # Build obs (49 + 4 = 53)
                    obs = build_obs(raw, command, last_action_for_obs, stair_info)

                    with torch.no_grad():
                        action_raw = policy.act_inference(obs.unsqueeze(0)).squeeze(0)

                    action_clip = torch.clamp(action_raw, -ACTION_CLIP, ACTION_CLIP)

                    if SIMULATE_1STEP_ACTION_LATENCY:
                        exec_action = prev_policy_action.clone()
                        prev_policy_action = action_clip.clone()
                    else:
                        exec_action = action_clip.clone()

                    pos_action = exec_action[:NUM_POS_ACTIONS]
                    policy_target_q = DEFAULT_DOF_POS + ACTION_SCALE * pos_action
                    target_q = slew_limit(prev_target_q, policy_target_q, MAX_STEP_RAD)
                    prev_target_q = target_q.clone()
                    shared["target_q"] = target_q.clone()

                    if PLS_ENABLE and exec_action.shape[0] > NUM_POS_ACTIONS:
                        stiffness_action = exec_action[NUM_POS_ACTIONS:]
                        kp_12, kd_12 = compute_pls_kp_kd(stiffness_action)
                        shared["kp_per_joint"] = kp_12.clone()
                        shared["kd_per_joint"] = kd_12.clone()

                    last_action_for_obs = action_clip.clone()

                    if step % PRINT_EVERY_N == 0:
                        pitch_deg, roll_deg = pitch_roll_from_quat(
                            raw["imu"]["quat_wxyz"]
                        )
                        print_status_line(
                            step,
                            command,
                            target_q,
                            edge_dist,
                            direction,
                            kp_12=shared["kp_per_joint"] if PLS_ENABLE else None,
                            pitch_deg=pitch_deg,
                            roll_deg=roll_deg,
                        )

                    step += 1

                elif current_state == STATE_TRANSITION:
                    alpha = min(1.0, (transition_step + 1) / float(transition_steps))
                    desired = (1 - alpha) * transition_start_q + alpha * STAND_DOF_POS
                    desired = slew_limit(prev_target_q, desired, MAX_STEP_RAD)
                    shared["target_q"] = desired.clone()
                    prev_target_q = desired.clone()
                    shared["kp_per_joint"] = (1 - alpha) * shared[
                        "kp_per_joint"
                    ] + alpha * torch.full((12,), STAND_KP)
                    shared["kd_per_joint"] = (1 - alpha) * shared[
                        "kd_per_joint"
                    ] + alpha * torch.full((12,), STAND_KD)
                    transition_step += 1
                    if transition_step >= transition_steps:
                        print("\n→ STAND ready")
                        current_state = STATE_STANDING
                        shared["kp_per_joint"][:] = STAND_KP
                        shared["kd_per_joint"][:] = STAND_KD

    except KeyboardInterrupt:
        pass

    print("\n\nStopping...")
    for _ in range(200):
        for i in range(12):
            low_cmd.motor_cmd[i].q = go2.PosStopF
            low_cmd.motor_cmd[i].dq = go2.VelStopF
            low_cmd.motor_cmd[i].kp = 0.0
            low_cmd.motor_cmd[i].kd = 0.0
            low_cmd.motor_cmd[i].tau = 0.0
        low_cmd.crc = crc.Crc(low_cmd)
        pub.Write(low_cmd)
        time.sleep(0.002)
    print("Stopped.")


# ============================================================
# MAIN
# ============================================================


def main():
    print(f"\n{'=' * 60}")
    print("  Go2 STAIR CLIMBING — 4-Value Stair Info Deployment")
    print(
        f"  NUM_OBS:       {NUM_OBS} ({NUM_PROPRIO} proprio + {NUM_STAIR_INFO} stair_info)"
    )
    print(f"  NUM_ACT:       {NUM_ACT}")
    print(f"  PLS:           {'ON' if PLS_ENABLE else 'OFF'}")
    print(f"  Stair height:  {STAIR_HEIGHT * 100:.1f}cm")
    print(f"  Stair depth:   {STAIR_DEPTH * 100:.1f}cm")
    print(f"  Edge range:    0..{EDGE_MAX_RANGE * 100:.0f}cm")
    print(f"  Checkpoint:    {CKPT_PATH}")
    print(f"{'=' * 60}\n")

    policy = load_policy(CKPT_PATH)

    if MODE == "robot_run":
        run_robot_run(policy)
    elif MODE == "robot_print":
        # Simplified print mode
        global _edge_detector
        init_dds()
        latest = wait_for_lowstate()
        init_lidar_subscriber()
        _edge_detector = EdgeDetector()
        _edge_detector.load_calibration()

        last_action = torch.zeros(NUM_ACT, dtype=torch.float32)
        step = 0
        while True:
            raw = lowstate_to_raw(latest["msg"])
            roll_rad, pitch_rad = get_imu_rp_from_raw(raw)
            edge_dist, direction = update_edge_detection(roll_rad, pitch_rad)
            stair_info = build_stair_info(
                edge_dist, STAIR_HEIGHT, STAIR_DEPTH, direction
            )
            obs = build_obs(raw, make_command_list(), last_action, stair_info)
            with torch.no_grad():
                action_raw = policy.act_inference(obs.unsqueeze(0)).squeeze(0)
            action_clip = torch.clamp(action_raw, -ACTION_CLIP, ACTION_CLIP)
            last_action = action_clip.clone()

            if step % PRINT_EVERY_N == 0:
                dir_sym = {1.0: "UP", -1.0: "DOWN"}.get(direction, "FLAT")
                print(
                    f"[step {step}] edge={edge_dist:.3f}m dir={dir_sym}  "
                    f"stair_info={stair_info.tolist()}"
                )
            step += 1
            time.sleep(1.0 / POLICY_HZ)
    else:
        print("Dummy mode — no robot needed")
        stair_info = build_stair_info(EDGE_MAX_RANGE, STAIR_HEIGHT, STAIR_DEPTH, 0.0)
        print(f"Stair info (flat ground): {stair_info.tolist()}")


if __name__ == "__main__":
    main()
