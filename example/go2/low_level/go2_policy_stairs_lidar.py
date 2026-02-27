"""
deploy_real_robot_lidar.py
==========================
Real-robot deployment of a LIDAR-aware Go2 stair-climbing policy.

Adapted from the non-LIDAR deployment script.  Key differences:
  - Actor observation is 104 dims:  49 proprioceptive + 55 LIDAR terrain scan
  - Integrates LidarProcessor from lidar_real_robot.py to convert L1/L2
    point cloud → 11×5 body-frame height grid matching training
  - Subscribes to /utlidar/cloud (ROS2 PointCloud2) for LIDAR data
  - build_obs() now takes lidar_scan parameter and appends it
  - All modes (dummy, robot_print, robot_run) handle LIDAR scan

The LIDAR processing pipeline:
  1. Subscribe to /utlidar/cloud at ~10-20Hz
  2. Transform point cloud: sensor frame → body frame
  3. Filter & bin into 11×5 grid, compute max height per cell
  4. Subtract robot base z → relative heights, clip ±1.0m
  5. Append 55-dim scan to 49-dim proprioceptive obs → 104-dim actor obs
  6. Feed to actor network → joint position targets
"""

import os
import sys
import time

import numpy as np
import torch
import yaml

#

MODE = "robot_run"  # "dummy" | "robot_print" | "robot_run"

DUMMY_YAML_PATH = "dummy_state.yaml"  ## to test off the robot
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_PATH = os.path.join(SCRIPT_DIR, "stairs4_lidar_68000.pt")  # ← LIDAR checkpoint

# ============================================================
# PREDEFINED VELOCITIES
# ============================================================
FORWARD_VX = 0.4
BACKWARD_VX = 0.3
LEFT_VY = 0.5
RIGHT_VY = 0.5
YAW_CW_WZ = 0.7
YAW_CCW_WZ = 0.7

# ============================================================

ACTION_CLIP = 100.0
ACTION_SCALE = 0.25

POLICY_HZ = 50.0
LOWCMD_HZ = 500.0

# Stand pose phase (before policy)
STAND_SECONDS = 4.0
STAND_KP = 40.0
STAND_KD = 0.5

# ============================================================
# PLS (Per-Leg Stiffness) — deployment config
# ============================================================
PLS_ENABLE = True
PLS_KP_DEFAULT = 40.0
PLS_KP_ACTION_SCALE = 20.0
PLS_KP_RANGE = [10.0, 70.0]

# ============================================================
# RUNTIME TUNING FACTORS
# ============================================================
KP_FACTOR = 1.0
KD_FACTOR = 0.6

POLICY_KP_FALLBACK = 40.0
POLICY_KD_FALLBACK = 2.0

MAX_STEP_RAD = 0.1

PRINT_EVERY_N = 10

SIMULATE_1STEP_ACTION_LATENCY = False

TRANSITION_SECONDS = 2.0

# ============================================================
# LIDAR CONFIGURATION — must match training exactly
# ============================================================
LIDAR_ENABLE = True

# Actor scan grid (must match go2_env_stair4_lidar.py LidarScanConfig)
LIDAR_NUM_X = 11
LIDAR_NUM_Y = 5
LIDAR_X_RANGE = (-0.2, 1.0)  # body frame, meters
LIDAR_Y_RANGE = (-0.25, 0.25)  # body frame, meters
LIDAR_HEIGHT_CLIP = 1.0
LIDAR_HEIGHT_SCALE = 1.0
NUM_LIDAR_SCAN = LIDAR_NUM_X * LIDAR_NUM_Y  # 55

# Sensor-to-body transform (L1 is mounted on top of robot)
# Adjust these based on your specific Go2 model mounting
LIDAR_SENSOR_OFFSET = np.array([0.0, 0.0, 0.30])  # meters in body frame

# ============================================================
# TRAINING CONSTANTS — must match stair v4 LIDAR training config
# ============================================================

NUM_POS_ACTIONS = 12
NUM_STIFFNESS_ACTIONS = 4 if PLS_ENABLE else 0
NUM_ACT = NUM_POS_ACTIONS + NUM_STIFFNESS_ACTIONS  # 16

# Actor obs: ang_vel(3) + gravity(3) + commands(3) + dof_pos(12)
#            + dof_vel(12) + actions(NUM_ACT) + lidar_scan(55)
NUM_PROPRIO = 3 + 3 + 3 + 12 + 12 + NUM_ACT  # 49
NUM_OBS = NUM_PROPRIO + (NUM_LIDAR_SCAN if LIDAR_ENABLE else 0)  # 104

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
# LIDAR Processor (imported or inline)
# ============================================================

try:
    from lidar_real_robot import LidarProcessor
except ImportError:
    # Inline minimal processor if module not available
    class LidarProcessor:
        """Minimal inline LIDAR processor matching training config."""

        def __init__(
            self,
            num_x=11,
            num_y=5,
            x_range=(-0.2, 1.0),
            y_range=(-0.25, 0.25),
            height_clip=1.0,
            height_scale=1.0,
            default_height=0.0,
            max_point_height=0.1,
            min_point_height=-2.0,
        ):
            self.num_x = num_x
            self.num_y = num_y
            self.x_range = x_range
            self.y_range = y_range
            self.height_clip = height_clip
            self.height_scale = height_scale
            self.default_height = default_height
            self.max_point_height = max_point_height
            self.min_point_height = min_point_height
            self.num_points = num_x * num_y
            self.dx = (x_range[1] - x_range[0]) / num_x
            self.dy = (y_range[1] - y_range[0]) / num_y

        def process(self, points_body, base_z=0.0, debug=False):
            if points_body.shape[0] == 0:
                return np.full(self.num_points, self.default_height, dtype=np.float32)
            px, py, pz = points_body[:, 0], points_body[:, 1], points_body[:, 2]
            in_range = (
                (px >= self.x_range[0] - self.dx)
                & (px <= self.x_range[1] + self.dx)
                & (py >= self.y_range[0] - self.dy)
                & (py <= self.y_range[1] + self.dy)
            )
            height_ok = (pz >= self.min_point_height) & (pz <= self.max_point_height)
            mask = in_range & height_ok
            if mask.sum() == 0:
                return np.full(self.num_points, self.default_height, dtype=np.float32)
            px_f, py_f, pz_f = px[mask], py[mask], pz[mask]
            xi = np.clip(
                ((px_f - self.x_range[0]) / self.dx).astype(int), 0, self.num_x - 1
            )
            yi = np.clip(
                ((py_f - self.y_range[0]) / self.dy).astype(int), 0, self.num_y - 1
            )
            cell_idx = xi * self.num_y + yi
            height_grid = np.full(
                self.num_points, self.default_height, dtype=np.float32
            )
            np.maximum.at(height_grid, cell_idx, pz_f)
            height_grid = np.clip(height_grid, -self.height_clip, self.height_clip)
            height_grid *= self.height_scale
            return height_grid


# Global LIDAR processor instance
_lidar_processor = None


def get_lidar_processor():
    global _lidar_processor
    if _lidar_processor is None:
        _lidar_processor = LidarProcessor(
            num_x=LIDAR_NUM_X,
            num_y=LIDAR_NUM_Y,
            x_range=LIDAR_X_RANGE,
            y_range=LIDAR_Y_RANGE,
            height_clip=LIDAR_HEIGHT_CLIP,
            height_scale=LIDAR_HEIGHT_SCALE,
        )
    return _lidar_processor


# ============================================================
# LIDAR point cloud state (updated by subscriber thread)
# ============================================================

import threading

_lidar_lock = threading.Lock()
_lidar_state = {
    "points_body": np.zeros(
        (0, 3), dtype=np.float32
    ),  # latest point cloud in body frame
    "timestamp": 0.0,
    "received": False,
}


def get_latest_lidar_scan(base_z: float = 0.0) -> np.ndarray:
    """Get the latest LIDAR scan processed into actor observation format.

    Returns:
        (55,) numpy array — terrain heights relative to robot base.
        If no LIDAR data received, returns zeros.
    """
    processor = get_lidar_processor()

    with _lidar_lock:
        points = _lidar_state["points_body"].copy()
        received = _lidar_state["received"]

    if not received or points.shape[0] == 0:
        return np.zeros(NUM_LIDAR_SCAN, dtype=np.float32)

    return processor.process(points, base_z=base_z)


def sensor_to_body_frame(points_sensor: np.ndarray) -> np.ndarray:
    """Transform L1/L2 sensor-frame points to body frame.

    The L1 sensor is mounted on TOP of the robot, typically ~0.3m above
    the body origin (COM). The sensor frame has z-up.

    This is a simple translation (no rotation needed if sensor is
    aligned with body axes). Adjust if your sensor is tilted.

    Args:
        points_sensor: (N, 3) in sensor frame

    Returns:
        (N, 3) in body frame (x=forward, y=left, z=up, origin at body COM)
    """
    points_body = points_sensor.copy()
    # Sensor is above body origin — shift z down
    points_body[:, 0] += LIDAR_SENSOR_OFFSET[0]
    points_body[:, 1] += LIDAR_SENSOR_OFFSET[1]
    points_body[:, 2] += LIDAR_SENSOR_OFFSET[2]
    # Now z=0 is at sensor, we want z=0 at body COM
    # Points on the ground will have z ≈ -0.3 in body frame
    # The height scan is relative to robot base, so this is correct
    return points_body


# ============================================================
# ROS2 LIDAR subscriber (started when using real robot)
# ============================================================

_lidar_thread = None


def start_lidar_subscriber():
    """Start a background thread subscribing to /utlidar/cloud.

    Uses ROS2 (rclpy) if available. Falls back to unitree_sdk2 if not.
    """
    global _lidar_thread

    def _ros2_subscriber():
        """ROS2 subscriber thread for LIDAR point cloud."""
        try:
            import struct

            import rclpy
            from rclpy.node import Node
            from sensor_msgs.msg import PointCloud2

            class LidarSubscriber(Node):
                def __init__(self):
                    super().__init__("go2_lidar_sub")
                    self.sub = self.create_subscription(
                        PointCloud2,
                        "/utlidar/cloud",
                        self.callback,
                        10,
                    )
                    self.get_logger().info("Subscribed to /utlidar/cloud")

                def callback(self, msg):
                    # Parse PointCloud2 → numpy (N, 3)
                    points = self._parse_pc2(msg)
                    if points is not None and points.shape[0] > 0:
                        # Transform sensor → body frame
                        points_body = sensor_to_body_frame(points)
                        with _lidar_lock:
                            _lidar_state["points_body"] = points_body
                            _lidar_state["timestamp"] = time.time()
                            _lidar_state["received"] = True

                def _parse_pc2(self, msg):
                    """Extract XYZ points from PointCloud2 message."""
                    # Find field offsets
                    field_map = {}
                    for f in msg.fields:
                        field_map[f.name] = f.offset

                    if (
                        "x" not in field_map
                        or "y" not in field_map
                        or "z" not in field_map
                    ):
                        return None

                    ox, oy, oz = field_map["x"], field_map["y"], field_map["z"]
                    step = msg.point_step
                    data = bytes(msg.data)
                    n_points = msg.width * msg.height

                    points = np.zeros((n_points, 3), dtype=np.float32)
                    for i in range(n_points):
                        base = i * step
                        points[i, 0] = struct.unpack_from("f", data, base + ox)[0]
                        points[i, 1] = struct.unpack_from("f", data, base + oy)[0]
                        points[i, 2] = struct.unpack_from("f", data, base + oz)[0]

                    # Filter out NaN/inf
                    valid = np.isfinite(points).all(axis=1)
                    return points[valid]

            rclpy.init()
            node = LidarSubscriber()
            rclpy.spin(node)
        except ImportError:
            print("[LIDAR] rclpy not available. Trying unitree_sdk2 LIDAR reader...")
            _unitree_sdk_subscriber()
        except Exception as e:
            print(f"[LIDAR] ROS2 subscriber error: {e}")

    def _unitree_sdk_subscriber():
        """Fallback: read LIDAR via unitree_sdk2py PointCloud subscriber."""
        try:
            from unitree_sdk2py.core.channel import ChannelSubscriber

            # The exact message type depends on your unitree SDK version
            # Common: PointCloud2_ or utlidar cloud topic
            print("[LIDAR] unitree_sdk2 LIDAR subscriber not yet implemented.")
            print("[LIDAR] Please use ROS2 (rclpy) or implement SDK subscriber.")
            print("[LIDAR] Policy will run with zero LIDAR scan (blind mode).")
        except ImportError:
            print("[LIDAR] Neither rclpy nor unitree_sdk2 available.")
            print("[LIDAR] Policy will run with zero LIDAR scan (blind mode).")

    _lidar_thread = threading.Thread(target=_ros2_subscriber, daemon=True)
    _lidar_thread.start()
    print("[LIDAR] Background subscriber started.")

    # Wait briefly for first message
    for _ in range(20):
        time.sleep(0.1)
        with _lidar_lock:
            if _lidar_state["received"]:
                print("[LIDAR] First point cloud received.")
                return True

    print("[LIDAR] WARNING: No LIDAR data received after 2s.")
    print("[LIDAR] Policy will run with zero scan until data arrives.")
    return False


# ============================================================
# PLS: compute per-joint Kp/Kd from network stiffness actions
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
# Quaternion helpers (expects w,x,y,z)
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
    sinp = 2.0 * (w * y - z * x)
    sinp = max(-1.0, min(1.0, sinp))
    pitch_rad = torch.asin(torch.tensor(sinp))
    return float(pitch_rad) * 57.2958, float(roll_rad) * 57.2958


# ============================================================
# Build the observation vector (NUM_OBS = 104 dims)
# ============================================================


def build_obs(raw, command_3, last_action, lidar_scan=None):
    """
    Build actor observation matching LIDAR stair training env exactly.

    Observation layout (104 dims):
      [0:3]     angular velocity * 0.25
      [3:6]     projected gravity (unit vector in body frame)
      [6:9]     commands * [2.0, 2.0, 0.25]
      [9:21]    (joint_pos - default) * 1.0
      [21:33]   joint_vel * 0.05
      [33:49]   last_actions (all 16: 12 pos + 4 stiffness)
      [49:104]  lidar_scan (55: 11×5 terrain height grid)

    Args:
        raw: dict with imu + motors data
        command_3: [vx, vy, wz] list
        last_action: (NUM_ACT,) tensor of last applied actions
        lidar_scan: (55,) numpy array or torch tensor of terrain heights.
                    If None and LIDAR_ENABLE, uses latest from subscriber.
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

    obs_parts = [
        gyro * OBS_SCALES["ang_vel"],  # 3
        proj_g,  # 3
        cmd * cmd_scale,  # 3
        (q - DEFAULT_DOF_POS) * OBS_SCALES["dof_pos"],  # 12
        dq * OBS_SCALES["dof_vel"],  # 12
        last_action,  # 16 (NUM_ACT)
    ]

    # Append LIDAR scan  ← NEW
    if LIDAR_ENABLE:
        if lidar_scan is None:
            # Get from subscriber
            lidar_scan = get_latest_lidar_scan()

        if isinstance(lidar_scan, np.ndarray):
            lidar_scan = torch.from_numpy(lidar_scan).float()

        if lidar_scan.shape[0] != NUM_LIDAR_SCAN:
            raise RuntimeError(
                f"LIDAR scan should be {NUM_LIDAR_SCAN}, got {lidar_scan.shape[0]}"
            )
        obs_parts.append(lidar_scan)  # 55

    obs = torch.cat(obs_parts, dim=0)

    if obs.shape[0] != NUM_OBS:
        raise RuntimeError(f"obs should be {NUM_OBS}, got {obs.shape[0]}")
    return obs


# ============================================================
# Load policy checkpoint
# ============================================================


def load_policy(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["model_state_dict"]

    try:
        from rsl_rl.modules import ActorCritic
    except Exception:
        from rsl_rl.modules.actor_critic import ActorCritic

    num_critic_obs = sd["critic.0.weight"].shape[1]

    # Verify actor input dim matches
    saved_actor_in = sd["actor.0.weight"].shape[1]
    if saved_actor_in != NUM_OBS:
        print(f"\n  ⚠ WARNING: Checkpoint actor expects {saved_actor_in} obs")
        print(f"  ⚠ But NUM_OBS = {NUM_OBS}")
        if saved_actor_in == NUM_PROPRIO:
            print(
                f"  ⚠ This looks like a NON-LIDAR checkpoint (proprio only = {NUM_PROPRIO})"
            )
            print("  ⚠ Set LIDAR_ENABLE = False for this checkpoint.")
        raise RuntimeError(
            f"Actor obs dim mismatch: checkpoint={saved_actor_in} vs config={NUM_OBS}"
        )

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
        f"  Loaded checkpoint: actor_obs={NUM_OBS} "
        f"(proprio={NUM_PROPRIO}+lidar={NUM_LIDAR_SCAN}), "
        f"critic_obs={num_critic_obs}, actions={NUM_ACT}"
    )
    return policy


# ============================================================
# DEBUG PRINT
# ============================================================


def debug_print_all(
    raw,
    command_3,
    last_action,
    obs,
    action_raw,
    action_clipped,
    target_q_12,
    kp_12=None,
    kd_12=None,
    lidar_scan=None,
    note="",
):
    quat = raw["imu"]["quat_wxyz"]
    proj_g = projected_gravity_from_quat_body_in_world(quat).tolist()
    q = [m["q_rad"] for m in raw["motors"]]
    dq = [m["dq_rad_s"] for m in raw["motors"]]
    pitch_deg, roll_deg = pitch_roll_from_quat(quat)

    if note:
        print(f"\n==================== {note} ====================")

    print("\n==================== RAW SENSORS USED ======================")
    print("IMU gyro (rad/s)         :", raw["imu"]["gyro_rad_s"])
    print("IMU quat (wxyz)          :", quat)
    print("Projected gravity (unit) :", proj_g)
    print(f"Pitch / Roll (deg)       : {pitch_deg:+.1f} / {roll_deg:+.1f}")
    print("Commands [vx,vy,wz]      :", command_3)

    print("\nJoint q (rad):")
    for i, name in enumerate(JOINT_NAMES):
        print(f"  {i:02d} {name:>8s} : {q[i]: .6f}")

    print("\nJoint dq (rad/s):")
    for i, name in enumerate(JOINT_NAMES):
        print(f"  {i:02d} {name:>8s} : {dq[i]: .6f}")

    # LIDAR scan info  ← NEW
    if LIDAR_ENABLE and lidar_scan is not None:
        scan = lidar_scan if isinstance(lidar_scan, np.ndarray) else lidar_scan.numpy()
        print(
            f"\n==================== LIDAR SCAN ({NUM_LIDAR_SCAN}) =================="
        )
        print(f"  Shape : {LIDAR_NUM_X}×{LIDAR_NUM_Y} = {NUM_LIDAR_SCAN}")
        print(f"  Min   : {scan.min():.4f} m")
        print(f"  Max   : {scan.max():.4f} m")
        print(f"  Mean  : {scan.mean():.4f} m")
        nonzero = np.sum(np.abs(scan) > 0.01)
        print(f"  Active: {nonzero}/{NUM_LIDAR_SCAN} cells with |h|>1cm")
        # Print as grid
        grid = scan.reshape(LIDAR_NUM_X, LIDAR_NUM_Y)
        print("  Grid (rows=x forward→, cols=y left→right):")
        for xi in range(LIDAR_NUM_X):
            row_str = " ".join(f"{grid[xi, yi]:+.2f}" for yi in range(LIDAR_NUM_Y))
            x_val = LIDAR_X_RANGE[0] + xi * (LIDAR_X_RANGE[1] - LIDAR_X_RANGE[0]) / (
                LIDAR_NUM_X - 1
            )
            print(f"    x={x_val:+.2f}m : [{row_str}]")

    print(f"\n==================== OBSERVATION VECTOR ({NUM_OBS}) =================")
    labels = []
    labels += ["ang_vel_x_scaled", "ang_vel_y_scaled", "ang_vel_z_scaled"]
    labels += ["grav_x", "grav_y", "grav_z"]
    labels += ["cmd_vx_scaled", "cmd_vy_scaled", "cmd_wz_scaled"]
    labels += [f"dof_pos_err_{n}_scaled" for n in JOINT_NAMES]
    labels += [f"dof_vel_{n}_scaled" for n in JOINT_NAMES]
    labels += [f"last_act_pos_{n}" for n in JOINT_NAMES]
    if PLS_ENABLE:
        labels += [f"last_act_stiff_{leg}" for leg in LEG_NAMES]
    if LIDAR_ENABLE:
        for xi in range(LIDAR_NUM_X):
            for yi in range(LIDAR_NUM_Y):
                labels.append(f"lidar_h_{xi:02d}_{yi:02d}")

    for i in range(min(NUM_OBS, len(labels))):
        print(f"{i:03d}  {labels[i]:>30s} : {float(obs[i]): .6f}")

    print(
        f"\n==================== ACTIONS ({NUM_ACT}) / ROBOT COMMAND ================="
    )

    print("Position actions RAW (dimensionless):")
    for i, name in enumerate(JOINT_NAMES):
        print(f"  {i:02d} {name:>8s} : {float(action_raw[i]): .6f}")

    print("\nPosition actions CLIPPED (dimensionless):")
    for i, name in enumerate(JOINT_NAMES):
        print(f"  {i:02d} {name:>8s} : {float(action_clipped[i]): .6f}")

    if PLS_ENABLE and action_raw.shape[0] > 12:
        print("\nStiffness actions RAW (dimensionless):")
        for i, leg in enumerate(LEG_NAMES):
            raw_val = float(action_raw[12 + i])
            clip_val = float(action_clipped[12 + i])
            kp_val = PLS_KP_DEFAULT + clip_val * PLS_KP_ACTION_SCALE
            kp_val = max(PLS_KP_RANGE[0], min(PLS_KP_RANGE[1], kp_val))
            print(
                f"  {leg:>4s} : raw={raw_val: .4f}  clipped={clip_val: .4f}  → Kp_network={kp_val:.1f}"
            )

    print("\nTarget joint position q (rad):")
    for i, name in enumerate(JOINT_NAMES):
        print(f"  {i:02d} {name:>8s} : {float(target_q_12[i]): .6f}")

    if kp_12 is not None and kd_12 is not None:
        print(
            f"\nPer-joint Kp/Kd (after KP_FACTOR={KP_FACTOR}, KD_FACTOR={KD_FACTOR}):"
        )
        for i, name in enumerate(JOINT_NAMES):
            print(
                f"  {i:02d} {name:>8s} : Kp={float(kp_12[i]):.2f}  Kd={float(kd_12[i]):.3f}"
            )
        print("\nPer-leg summary:")
        for leg_idx, leg in enumerate(LEG_NAMES):
            j = LEG_JOINT_MAP[leg_idx][0]
            print(f"  {leg:>4s} : Kp={float(kp_12[j]):.2f}  Kd={float(kd_12[j]):.3f}")


def print_status_line(
    step,
    command_3,
    target_q_12,
    kp_12=None,
    pitch_deg=0.0,
    roll_deg=0.0,
    lidar_active=False,
):
    vx, vy, wz = command_3
    kp_str = ""
    if kp_12 is not None:
        kps = [float(kp_12[LEG_JOINT_MAP[i][0]]) for i in range(4)]
        kp_str = f"  Kp=[{kps[0]:.0f},{kps[1]:.0f},{kps[2]:.0f},{kps[3]:.0f}]"
    lidar_str = " L✓" if lidar_active else " L✗"
    print(
        f"\r  step={step:06d}  cmd=[{vx:+.2f},{vy:+.2f},{wz:+.2f}]  "
        f"pitch={pitch_deg:+5.1f}° roll={roll_deg:+5.1f}°"
        f"{kp_str}{lidar_str}  ",
        end="",
        flush=True,
    )


# ============================================================
# Robot lowstate -> raw dict
# ============================================================


def lowstate_to_raw(low_state):
    imu = low_state.imu_state
    gyro = list(imu.gyroscope)
    quat = list(imu.quaternion)
    if len(quat) != 4:
        raise RuntimeError(f"Expected quaternion length 4, got {len(quat)}")

    quat_wxyz = [float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])]
    gyro_xyz = [float(gyro[0]), float(gyro[1]), float(gyro[2])]

    motors = []
    for i in range(12):
        ms = low_state.motor_state[i]
        motors.append({"q_rad": float(ms.q), "dq_rad_s": float(ms.dq)})

    return {"imu": {"gyro_rad_s": gyro_xyz, "quat_wxyz": quat_wxyz}, "motors": motors}


# ============================================================
# KEYBOARD INPUT
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
                    arrow_map = {"A": "UP", "B": "DOWN", "C": "RIGHT", "D": "LEFT"}
                    return arrow_map.get(ch3, None)
            return "ESC"
        elif ch == "\x03":
            return "CTRL_C"
        else:
            return ch


command_state = {
    "vx": 0.0,
    "vy": 0.0,
    "wz": 0.0,
}


def make_command_list():
    return [command_state["vx"], command_state["vy"], command_state["wz"]]


def handle_key(key):
    if key is None:
        return True
    if isinstance(key, str) and len(key) == 1:
        key = key.lower()
    if key == "x" or key == "CTRL_C":
        return False

    if key == "w":
        command_state["vx"] = FORWARD_VX
        command_state["vy"] = 0.0
        command_state["wz"] = 0.0
    elif key == "s":
        command_state["vx"] = -BACKWARD_VX
        command_state["vy"] = 0.0
        command_state["wz"] = 0.0
    elif key == "d":
        command_state["vx"] = 0.0
        command_state["vy"] = RIGHT_VY
        command_state["wz"] = 0.0
    elif key == "a":
        command_state["vx"] = 0.0
        command_state["vy"] = -LEFT_VY
        command_state["wz"] = 0.0
    elif key == "q":
        command_state["vx"] = 0.0
        command_state["vy"] = 0.0
        command_state["wz"] = -YAW_CW_WZ
    elif key == "r":
        command_state["vx"] = 0.0
        command_state["vy"] = 0.0
        command_state["wz"] = YAW_CCW_WZ
    elif key == " ":
        command_state["vx"] = 0.0
        command_state["vy"] = 0.0
        command_state["wz"] = 0.0

    return True


def print_controls():
    print("\n=========== LIDAR STAIR CLIMBING CONTROLS ===========")
    print(f"  W           : forward   (vx={FORWARD_VX:.2f})  ← approach stairs")
    print(f"  S           : backward  (vx={-BACKWARD_VX:.2f})")
    print(f"  D           : right     (vy={RIGHT_VY:.2f})")
    print(f"  A           : left      (vy={-LEFT_VY:.2f})")
    print(f"  Q           : yaw CW    (wz={-YAW_CW_WZ:.2f})")
    print(f"  R           : yaw CCW   (wz={YAW_CCW_WZ:.2f})")
    print("  SPACE       : zero velocity (policy keeps running)")
    print("  E           : return to stand")
    print("  X           : quit")
    print("=====================================================")
    print("  LIDAR: terrain scan active — policy sees ahead")
    print("  Status line shows L✓ (LIDAR active) or L✗ (no data)")
    print("=====================================================\n")


# ============================================================


def slew_limit(prev_q, new_q, max_step_rad):
    delta = new_q - prev_q
    delta = torch.clamp(delta, -max_step_rad, max_step_rad)
    return prev_q + delta


def init_dds():
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize

    if len(sys.argv) > 1:
        ChannelFactoryInitialize(0, sys.argv[1])
        print("DDS interface:", sys.argv[1])
    else:
        ChannelFactoryInitialize(0)
        print("DDS interface: default")


def wait_for_lowstate():
    from unitree_sdk2py.core.channel import ChannelSubscriber
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_

    latest = {"msg": None}

    def cb(msg: LowState_):
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
# robot_print — read-only mode, prints policy output + LIDAR
# ============================================================


def run_robot_print(policy):
    init_dds()
    latest = wait_for_lowstate()

    # Start LIDAR subscriber
    if LIDAR_ENABLE:
        start_lidar_subscriber()

    dt = 1.0 / POLICY_HZ
    step = 0
    last_action = torch.zeros(NUM_ACT, dtype=torch.float32)

    while True:
        raw = lowstate_to_raw(latest["msg"])

        command = make_command_list()

        # Get LIDAR scan
        lidar_scan = None
        if LIDAR_ENABLE:
            lidar_scan = get_latest_lidar_scan()

        obs = build_obs(raw, command, last_action, lidar_scan=lidar_scan)

        with torch.no_grad():
            action_raw = policy.act_inference(obs.unsqueeze(0)).squeeze(0)

        action_clip = torch.clamp(action_raw, -ACTION_CLIP, ACTION_CLIP)

        pos_action = action_clip[:NUM_POS_ACTIONS]
        target_q = DEFAULT_DOF_POS + ACTION_SCALE * pos_action

        kp_12, kd_12 = None, None
        if PLS_ENABLE and action_clip.shape[0] > NUM_POS_ACTIONS:
            stiffness_action = action_clip[NUM_POS_ACTIONS:]
            kp_12, kd_12 = compute_pls_kp_kd(stiffness_action)

        last_action = action_clip.clone()

        if step % PRINT_EVERY_N == 0:
            debug_print_all(
                raw,
                command,
                last_action,
                obs,
                action_raw,
                action_clip,
                target_q,
                kp_12=kp_12,
                kd_12=kd_12,
                lidar_scan=lidar_scan,
                note=f"robot_print step {step}",
            )

        step += 1
        time.sleep(dt)


# ============================================================
# robot_run: release -> stand -> prompt -> state machine
# ============================================================


def run_robot_run(policy):
    from unitree_sdk2py.core.channel import ChannelPublisher
    from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
    from unitree_sdk2py.utils.crc import CRC
    from unitree_sdk2py.utils.thread import RecurrentThread

    import unitree_legged_const as go2

    init_dds()
    latest = wait_for_lowstate()

    # Start LIDAR subscriber
    if LIDAR_ENABLE:
        lidar_ok = start_lidar_subscriber()
        if not lidar_ok:
            print("\n⚠ LIDAR not receiving data. Continue anyway? (y/n)")
            resp = input("> ").strip().lower()
            if resp != "y":
                print("Aborted.")
                return

    # 1) Release high-level first
    release_sport_and_highlevel()

    # 2) Setup publisher + command packet
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

    # 500Hz writer
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
    print(f"LowCmd writer started at {LOWCMD_HZ} Hz.")

    # 3) Ramp to stand pose
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

    # 4) Prompt user
    print("\n*** ROBOT IS STANDING - READY FOR LIDAR STAIR CLIMBING ***")
    print(f"PLS: {'ENABLED' if PLS_ENABLE else 'DISABLED'}")
    print(f"LIDAR: {'ENABLED' if LIDAR_ENABLE else 'DISABLED'}")
    if LIDAR_ENABLE:
        with _lidar_lock:
            lidar_active = _lidar_state["received"]
        print(f"LIDAR status: {'receiving data ✓' if lidar_active else 'NO DATA ✗'}")
    print(f"Runtime tuning: KP_FACTOR={KP_FACTOR}, KD_FACTOR={KD_FACTOR}")
    print("Type 'go' and press Enter to enable keyboard control.")
    print("Anything else will abort.\n")
    user = input("> ").strip().lower()
    if user != "go":
        print("Aborted. Robot will hold stand pose until you kill the script.")
        return

    print_controls()
    print("\nRobot in STAND mode. Press W to walk forward toward stairs.")
    print("Press 'E' to return to stand.\n")

    # State machine
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

                if key == "x" or key == "CTRL_C":
                    running = False
                    break

                is_movement_cmd = key in MOVEMENT_KEYS
                is_stop_cmd = key == "e"

                # ===== STATE MACHINE =====
                if current_state == STATE_STANDING:
                    if is_movement_cmd or key == " ":
                        print("\n→ Activating POLICY mode (LIDAR stair climbing)")
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
                    if is_stop_cmd:
                        print("\n→ Returning to STAND mode")
                        current_state = STATE_TRANSITION
                        transition_start_q = prev_target_q.clone()
                        transition_step = 0
                        command_state["vx"] = 0.0
                        command_state["vy"] = 0.0
                        command_state["wz"] = 0.0
                        continue
                    else:
                        if not handle_key(key):
                            running = False
                            break

                        raw = lowstate_to_raw(latest["msg"])
                        command = make_command_list()

                        # Get LIDAR scan  ← NEW
                        lidar_scan = None
                        if LIDAR_ENABLE:
                            lidar_scan = get_latest_lidar_scan()

                        obs = build_obs(
                            raw,
                            command,
                            last_action_for_obs,
                            lidar_scan=lidar_scan,
                        )

                        with torch.no_grad():
                            action_raw = policy.act_inference(obs.unsqueeze(0)).squeeze(
                                0
                            )

                        action_clip = torch.clamp(action_raw, -ACTION_CLIP, ACTION_CLIP)

                        if SIMULATE_1STEP_ACTION_LATENCY:
                            exec_action = prev_policy_action.clone()
                            prev_policy_action = action_clip.clone()
                        else:
                            exec_action = action_clip.clone()

                        pos_action = exec_action[:NUM_POS_ACTIONS]
                        policy_target_q = DEFAULT_DOF_POS + ACTION_SCALE * pos_action
                        target_q = slew_limit(
                            prev_target_q, policy_target_q, MAX_STEP_RAD
                        )
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
                            # Check LIDAR status
                            with _lidar_lock:
                                lidar_active = _lidar_state["received"]
                            print_status_line(
                                step,
                                command,
                                target_q,
                                kp_12=shared["kp_per_joint"] if PLS_ENABLE else None,
                                pitch_deg=pitch_deg,
                                roll_deg=roll_deg,
                                lidar_active=lidar_active,
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
                    ] + alpha * torch.full((12,), STAND_KP, dtype=torch.float32)
                    shared["kd_per_joint"] = (1 - alpha) * shared[
                        "kd_per_joint"
                    ] + alpha * torch.full((12,), STAND_KD, dtype=torch.float32)

                    transition_step += 1

                    if transition_step >= transition_steps:
                        print("\n→ STAND mode ready")
                        current_state = STATE_STANDING
                        shared["kp_per_joint"][:] = STAND_KP
                        shared["kd_per_joint"][:] = STAND_KD
                        if key and key in MOVEMENT_KEYS:
                            print("→ Movement detected, activating POLICY mode")
                            current_state = STATE_POLICY
                            shared["kp_per_joint"][:] = POLICY_KP_FALLBACK
                            shared["kd_per_joint"][:] = POLICY_KD_FALLBACK
                            last_action_for_obs = torch.zeros(
                                NUM_ACT, dtype=torch.float32
                            )
                            prev_policy_action = torch.zeros(
                                NUM_ACT, dtype=torch.float32
                            )
                            step = 0
                            handle_key(key)

    except KeyboardInterrupt:
        pass

    print("\n\nStopping policy. Sending safe stop packets...")
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
    print("  Go2 LIDAR STAIR CLIMBING Deployment")
    print(f"  PLS:        {'ENABLED' if PLS_ENABLE else 'DISABLED'}")
    print(
        f"  NUM_ACT:    {NUM_ACT} ({'12 pos + 4 stiffness' if PLS_ENABLE else '12 pos'})"
    )
    print(f"  NUM_OBS:    {NUM_OBS} (proprio={NUM_PROPRIO} + lidar={NUM_LIDAR_SCAN})")
    print(f"  LIDAR:      {'ENABLED' if LIDAR_ENABLE else 'DISABLED'}")
    if LIDAR_ENABLE:
        print(f"  LIDAR grid: {LIDAR_NUM_X}×{LIDAR_NUM_Y} = {NUM_LIDAR_SCAN} points")
        print(f"  LIDAR X:    {LIDAR_X_RANGE} m (body frame)")
        print(f"  LIDAR Y:    {LIDAR_Y_RANGE} m (body frame)")
        print(f"  Sensor off: {LIDAR_SENSOR_OFFSET.tolist()} m")
    print(f"  KP_FACTOR:  {KP_FACTOR}")
    print(f"  KD_FACTOR:  {KD_FACTOR}")
    if PLS_ENABLE:
        print(f"  Kp range:   {PLS_KP_RANGE}")
        print(f"  Kp default: {PLS_KP_DEFAULT}")
        print(f"  Kp scale:   {PLS_KP_ACTION_SCALE}")
        print("  Kd formula: 0.2 * sqrt(Kp)")
    print(f"  Checkpoint: {CKPT_PATH}")
    print(f"{'=' * 60}\n")

    policy = load_policy(CKPT_PATH)

    if MODE == "dummy":
        with open(DUMMY_YAML_PATH, "r") as f:
            raw = yaml.safe_load(f)

        last_action = torch.zeros(NUM_ACT, dtype=torch.float32)
        command = make_command_list()

        # For dummy mode: generate a synthetic LIDAR scan (flat ground)
        lidar_scan = (
            np.zeros(NUM_LIDAR_SCAN, dtype=np.float32) if LIDAR_ENABLE else None
        )

        obs = build_obs(raw, command, last_action, lidar_scan=lidar_scan)

        with torch.no_grad():
            action_raw = policy.act_inference(obs.unsqueeze(0)).squeeze(0)

        action_clip = torch.clamp(action_raw, -ACTION_CLIP, ACTION_CLIP)

        pos_action = action_clip[:NUM_POS_ACTIONS]
        target_q = DEFAULT_DOF_POS + ACTION_SCALE * pos_action

        kp_12, kd_12 = None, None
        if PLS_ENABLE and action_clip.shape[0] > NUM_POS_ACTIONS:
            stiffness_action = action_clip[NUM_POS_ACTIONS:]
            kp_12, kd_12 = compute_pls_kp_kd(stiffness_action)

        debug_print_all(
            raw,
            command,
            last_action,
            obs,
            action_raw,
            action_clip,
            target_q,
            kp_12=kp_12,
            kd_12=kd_12,
            lidar_scan=lidar_scan,
            note="dummy (LIDAR=zeros)",
        )
        return

    if MODE == "robot_print":
        run_robot_print(policy)
        return

    if MODE == "robot_run":
        run_robot_run(policy)
        return

    print("Unknown MODE:", MODE)


if __name__ == "__main__":
    main()
