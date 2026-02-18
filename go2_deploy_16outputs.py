import os
import sys
import time

import torch
import yaml

#

MODE = "robot_run"  # "dummy" | "robot_print" | "robot_run"

DUMMY_YAML_PATH = "dummy_state.yaml"  ## to test off the robot
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_PATH = os.path.join(SCRIPT_DIR, "model_69600.pt")

# ============================================================
# PREDEFINED VELOCITIES (edit these to change speed)
# ============================================================
FORWARD_VX = 0.0  # W key: forward speed
BACKWARD_VX = 0.5  # S key: backward speed
LEFT_VY = 1.0  # A key: negative vy (left)
RIGHT_VY = 1.0  # D key: positive vy (right)
YAW_CCW_WZ = 0.4  # Q key: counter-clockwise yaw
YAW_CW_WZ = 0.4  # E key: clockwise yaw

# ============================================================

ACTION_CLIP = 100.0
ACTION_SCALE = 0.25

POLICY_HZ = 50.0
LOWCMD_HZ = 500.0

# Stand pose phase (before policy)
STAND_SECONDS = 4.0
STAND_KP = 40.0
STAND_KD = 0.5

# Policy phase gains
POLICY_KP = 60.0
POLICY_KD = 3.0

MAX_STEP_RAD = 0.1

PRINT_EVERY_N = 10

SIMULATE_1STEP_ACTION_LATENCY = False

# Transition timing
TRANSITION_SECONDS = 2.0  # Time to transition back to stand

# TRAINING CONSTANTS

NUM_OBS = 45
NUM_ACT = 12

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

DEFAULT_DOF_POS = torch.tensor(
    [0.0, 0.8, -1.5, 0.0, 0.8, -1.5, 0.0, 1.0, -1.5, 0.0, 1.0, -1.5],
    dtype=torch.float32,
)
STAND_DOF_POS = DEFAULT_DOF_POS.clone()


# Quaternion helpers (expects w,x,y,z)


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


# Build the 45-dim observation


def build_obs(raw, command_3, last_action_12):
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
            gyro * OBS_SCALES["ang_vel"],
            proj_g,
            cmd * cmd_scale,
            (q - DEFAULT_DOF_POS) * OBS_SCALES["dof_pos"],
            dq * OBS_SCALES["dof_vel"],
            last_action_12,
        ],
        dim=0,
    )

    if obs.shape[0] != NUM_OBS:
        raise RuntimeError(f"obs should be 45, got {obs.shape[0]}")
    return obs


# Load policy checkpoint


def load_policy(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["model_state_dict"]

    try:
        from rsl_rl.modules import ActorCritic
    except Exception:
        from rsl_rl.modules.actor_critic import ActorCritic

    policy = ActorCritic(
        num_actor_obs=NUM_OBS,
        num_critic_obs=NUM_OBS,
        num_actions=NUM_ACT,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        init_noise_std=1.0,
    )
    policy.load_state_dict(sd, strict=True)
    policy.eval()
    return policy


# DEBUG PRINT


def debug_print_all(
    raw,
    command_3,
    last_action_12,
    obs_45,
    action_raw_12,
    action_clipped_12,
    target_q_12,
    note="",
):
    quat = raw["imu"]["quat_wxyz"]
    proj_g = projected_gravity_from_quat_body_in_world(quat).tolist()
    q = [m["q_rad"] for m in raw["motors"]]
    dq = [m["dq_rad_s"] for m in raw["motors"]]

    if note:
        print(f"\n==================== {note} ====================")

    print("\n==================== RAW SENSORS USED ======================")
    print("IMU gyro (rad/s)         :", raw["imu"]["gyro_rad_s"])
    print("IMU quat (wxyz)          :", quat)
    print("Projected gravity (unit) :", proj_g)
    print("Commands [vx,vy,wz]      :", command_3)

    print("\nJoint q (rad):")
    for i, name in enumerate(JOINT_NAMES):
        print(f"  {i:02d} {name:>8s} : {q[i]: .6f}")

    print("\nJoint dq (rad/s):")
    for i, name in enumerate(JOINT_NAMES):
        print(f"  {i:02d} {name:>8s} : {dq[i]: .6f}")

    print("\n==================== OBSERVATION VECTOR (45) =================")
    labels = []
    labels += ["ang_vel_x_scaled", "ang_vel_y_scaled", "ang_vel_z_scaled"]
    labels += ["grav_x", "grav_y", "grav_z"]
    labels += ["cmd_vx_scaled", "cmd_vy_scaled", "cmd_wz_scaled"]
    labels += [f"dof_pos_err_{n}_scaled" for n in JOINT_NAMES]
    labels += [f"dof_vel_{n}_scaled" for n in JOINT_NAMES]
    labels += [f"last_action_{n}" for n in JOINT_NAMES]

    for i in range(45):
        print(f"{i:02d}  {labels[i]:>28s} : {float(obs_45[i]): .6f}")

    print("\n==================== ACTIONS / ROBOT COMMAND =================")
    print("Policy action RAW (dimensionless):")
    for i, name in enumerate(JOINT_NAMES):
        print(f"  {i:02d} {name:>8s} : {float(action_raw_12[i]): .6f}")

    print("\nPolicy action CLIPPED (dimensionless):")
    for i, name in enumerate(JOINT_NAMES):
        print(f"  {i:02d} {name:>8s} : {float(action_clipped_12[i]): .6f}")

    print("\nTarget joint position q (rad):")
    for i, name in enumerate(JOINT_NAMES):
        print(f"  {i:02d} {name:>8s} : {float(target_q_12[i]): .6f}")


# Compact one-line status (used during policy loop instead of full debug dump)
def print_status_line(step, command_3, target_q_12):
    vx, vy, wz = command_3
    tq = [float(target_q_12[i]) for i in range(12)]
    print(
        f"\r  step={step:06d}  cmd=[{vx:+.2f},{vy:+.2f},{wz:+.2f}]  "
        f"tq0={tq[0]:+.3f} tq1={tq[1]:+.3f} tq2={tq[2]:+.3f}  ",
        end="",
        flush=True,
    )


# Robot lowstate -> raw dict


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
# KEYBOARD INPUT (raw terminal mode set ONCE, not per-call)
# ============================================================

import select
import termios
import tty


class RawTerminal:
    """Context manager: sets terminal to raw mode once, restores on exit."""

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
        """Non-blocking key read. Returns key string or None."""
        rlist, _, _ = select.select([sys.stdin], [], [], 0)
        if not rlist:
            return None
        ch = sys.stdin.read(1)
        if ch == "\x1b":
            # Handle escape sequences (arrow keys) if you want to keep them
            rlist2, _, _ = select.select([sys.stdin], [], [], 0.005)
            if rlist2:
                ch2 = sys.stdin.read(1)
                if ch2 == "[":
                    ch3 = sys.stdin.read(1)
                    # You can remove this if you don't want arrow key support
                    arrow_map = {"A": "UP", "B": "DOWN", "C": "RIGHT", "D": "LEFT"}
                    return arrow_map.get(ch3, None)
            return "ESC"
        elif ch == "\x03":
            return "CTRL_C"
        else:
            return ch


# Shared mutable command state
command_state = {
    "vx": 0.0,
    "vy": 0.0,
    "wz": 0.0,
}


def make_command_list():
    return [command_state["vx"], command_state["vy"], command_state["wz"]]


def handle_key(key):
    """Process a keypress and update command_state. Returns False to quit."""
    if key is None:
        return True

    # Convert to lowercase for case-insensitive handling
    if isinstance(key, str) and len(key) == 1:
        key = key.lower()

    if (
        key == "x" or key == "CTRL_C"
    ):  # Changed quit key to 'x' to avoid conflict with 's'
        return False

    # WASD controls
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
        command_state["wz"] = YAW_CCW_WZ
    elif key == "e":
        command_state["vx"] = 0.0
        command_state["vy"] = 0.0
        command_state["wz"] = -YAW_CW_WZ
    elif key == " " or key == "r":  # Space or 'r' for stop/reset
        command_state["vx"] = 0.0
        command_state["vy"] = 0.0
        command_state["wz"] = 0.0

    return True


def print_controls():
    print("\n============ CONTROLS ============")
    print(f"  W           : forward   (vx={FORWARD_VX:.2f})")
    print(f"  S           : backward  (vx={-BACKWARD_VX:.2f})")
    print(f"  D           : right     (vy={RIGHT_VY:.2f})")
    print(f"  A           : left      (vy={-LEFT_VY:.2f})")
    print(f"  Q           : yaw CCW   (wz={YAW_CCW_WZ:.2f})")
    print(f"  E           : yaw CW    (wz={-YAW_CW_WZ:.2f})")
    print("  SPACE / R   : stop all (return to stand)")
    print("  X           : quit")
    print("==================================\n")


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


# robot_print


def run_robot_print(policy):
    init_dds()
    latest = wait_for_lowstate()

    dt = 1.0 / POLICY_HZ
    step = 0
    last_action = torch.zeros(NUM_ACT, dtype=torch.float32)

    while True:
        raw = lowstate_to_raw(latest["msg"])

        command = make_command_list()
        obs = build_obs(raw, command, last_action)

        with torch.no_grad():
            action_raw = policy.act_inference(obs.unsqueeze(0)).squeeze(0)

        action_clip = torch.clamp(action_raw, -ACTION_CLIP, ACTION_CLIP)
        target_q = DEFAULT_DOF_POS + ACTION_SCALE * action_clip

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
                note=f"robot_print step {step}",
            )

        step += 1
        time.sleep(dt)


# robot_run:
# release -> go to stand -> prompt -> state machine (stand/policy/transition)


def run_robot_run(policy):
    from unitree_sdk2py.core.channel import ChannelPublisher
    from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
    from unitree_sdk2py.utils.crc import CRC
    from unitree_sdk2py.utils.thread import RecurrentThread

    import unitree_legged_const as go2

    init_dds()
    latest = wait_for_lowstate()

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

    # shared command streamed at 500Hz
    shared = {
        "target_q": None,
        "kp": float(STAND_KP),
        "kd": float(STAND_KD),
    }

    # start from current joints
    raw0 = lowstate_to_raw(latest["msg"])
    start_q = torch.tensor([m["q_rad"] for m in raw0["motors"]], dtype=torch.float32)
    shared["target_q"] = start_q.clone()

    # 500Hz writer
    def write_lowcmd():
        tq = shared["target_q"]
        if tq is None:
            return
        for i in range(12):
            low_cmd.motor_cmd[i].mode = 0x01
            low_cmd.motor_cmd[i].q = float(tq[i])
            low_cmd.motor_cmd[i].dq = 0.0
            low_cmd.motor_cmd[i].kp = float(shared["kp"])
            low_cmd.motor_cmd[i].kd = float(shared["kd"])
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

    # 4) Prompt user to confirm before enabling control
    print("\n*** ROBOT IS STANDING - READY TO START ***")
    print("Type 'go' and press Enter to enable keyboard control.")
    print("Anything else will abort.\n")
    user = input("> ").strip().lower()
    if user != "go":
        print("Aborted. Robot will hold stand pose until you kill the script.")
        return

    print_controls()
    print("\nRobot in STAND mode. Press movement keys to activate policy.")
    print("Press 'SPACE' or 'R' to return to stand.\n")

    # State machine
    STATE_STANDING = "standing"
    STATE_POLICY = "policy"
    STATE_TRANSITION = "transition"

    current_state = STATE_STANDING

    # Policy variables
    dt = 1.0 / POLICY_HZ
    step = 0
    last_action_for_obs = torch.zeros(NUM_ACT, dtype=torch.float32)
    prev_policy_action = torch.zeros(NUM_ACT, dtype=torch.float32)
    prev_target_q = shared["target_q"].clone()

    # Transition variables
    transition_start_q = None
    transition_step = 0
    transition_steps = int(TRANSITION_SECONDS * POLICY_HZ)

    running = True
    try:
        with RawTerminal() as term:
            next_tick = time.monotonic()

            while running:
                # Precise timing
                now = time.monotonic()
                sleep_time = next_tick - now
                if sleep_time > 0:
                    time.sleep(sleep_time)
                next_tick += dt

                # Poll keyboard
                key = term.get_key()

                # Convert to lowercase for comparison
                if key and len(key) == 1:
                    key = key.lower()

                # Handle quit
                if key == "x" or key == "CTRL_C":
                    running = False
                    break

                # Check if it's a movement command (WASD + QE)
                is_movement_cmd = key in ["w", "a", "s", "d", "q", "e"]
                is_stop_cmd = key in [" ", "r"]

                # ===== STATE MACHINE =====
                if current_state == STATE_STANDING:
                    if is_movement_cmd:
                        print("\n→ Activating POLICY mode")
                        current_state = STATE_POLICY
                        shared["kp"] = float(POLICY_KP)
                        shared["kd"] = float(POLICY_KD)
                        last_action_for_obs = torch.zeros(NUM_ACT, dtype=torch.float32)
                        prev_policy_action = torch.zeros(NUM_ACT, dtype=torch.float32)
                        prev_target_q = STAND_DOF_POS.clone()
                        step = 0
                        # Process the movement key
                        handle_key(key)
                    else:
                        # Stay in stand pose
                        shared["target_q"] = STAND_DOF_POS.clone()
                        prev_target_q = STAND_DOF_POS.clone()
                        if key:
                            handle_key(key)  # Update command even if not moving
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
                        # Normal policy execution
                        if not handle_key(key):
                            running = False
                            break

                        raw = lowstate_to_raw(latest["msg"])
                        command = make_command_list()
                        obs = build_obs(raw, command, last_action_for_obs)

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

                        policy_target_q = DEFAULT_DOF_POS + ACTION_SCALE * exec_action
                        target_q = slew_limit(
                            prev_target_q, policy_target_q, MAX_STEP_RAD
                        )
                        prev_target_q = target_q.clone()

                        shared["target_q"] = target_q.clone()
                        last_action_for_obs = action_clip.clone()

                        if step % PRINT_EVERY_N == 0:
                            print_status_line(step, command, target_q)

                        step += 1

                elif current_state == STATE_TRANSITION:
                    # Smoothly ramp back to stand pose
                    alpha = min(1.0, (transition_step + 1) / float(transition_steps))
                    desired = (1 - alpha) * transition_start_q + alpha * STAND_DOF_POS
                    desired = slew_limit(prev_target_q, desired, MAX_STEP_RAD)

                    shared["target_q"] = desired.clone()
                    prev_target_q = desired.clone()

                    transition_step += 1

                    if transition_step >= transition_steps:
                        print("\n→ STAND mode ready")
                        current_state = STATE_STANDING
                        shared["kp"] = float(STAND_KP)
                        shared["kd"] = float(STAND_KD)
                        # Check if there's already a movement key pressed
                        if key and key in ["w", "a", "s", "d", "q", "e"]:
                            print("→ Movement detected, activating POLICY mode")
                            current_state = STATE_POLICY
                            shared["kp"] = float(POLICY_KP)
                            shared["kd"] = float(POLICY_KD)
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

    # Restore terminal before printing stop messages
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
    policy = load_policy(CKPT_PATH)

    if MODE == "dummy":
        with open(DUMMY_YAML_PATH, "r") as f:
            raw = yaml.safe_load(f)

        last_action = torch.zeros(NUM_ACT, dtype=torch.float32)
        command = make_command_list()
        obs = build_obs(raw, command, last_action)

        with torch.no_grad():
            action_raw = policy.act_inference(obs.unsqueeze(0)).squeeze(0)

        action_clip = torch.clamp(action_raw, -ACTION_CLIP, ACTION_CLIP)
        target_q = DEFAULT_DOF_POS + ACTION_SCALE * action_clip

        debug_print_all(
            raw,
            command,
            last_action,
            obs,
            action_raw,
            action_clip,
            target_q,
            note="dummy",
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
