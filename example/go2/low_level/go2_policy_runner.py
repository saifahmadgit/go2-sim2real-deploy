"""
GO2 PPO Runner — SIM MATCHED (Genesis → Real Robot)

MODES:
1 = OFFLINE (YAML dummy states)
2 = MONITOR (Robot connected, prints obs/action, NO motor commands)
3 = LIVE (Robot connected, prints obs/action, SENDS motor commands)

MATCHES TRAINING:
- obs scaling
- gravity projection
- base-frame angular velocity
- default pose residual control
- action latency (1 step)
- action_scale = env_cfg["action_scale"]

YAML FILE:
offline_states.yaml (same directory)
"""

import copy
import os
import signal
import sys
import time
import yaml

import numpy as np
import torch
import torch.nn as nn

from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import (
    MotionSwitcherClient,
)
from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.go2.sport.sport_client import SportClient
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC

# ==========================================================
# CONFIG (MATCH GENESIS)
# ==========================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "crouch.pt")
YAML_PATH = os.path.join(SCRIPT_DIR, "offline_states.yaml")

DT = 0.02

# From env_cfg
ACTION_SCALE = 0.65

KP = 80.0
KD = 4.0

# Observation scaling (obs_cfg)
OBS_SCALES = {
    "lin_vel": 2.0,
    "ang_vel": 0.25,
    "dof_pos": 1.0,
    "dof_vel": 0.05,
}

COMMANDS_SCALE = np.array(
    [OBS_SCALES["lin_vel"], OBS_SCALES["lin_vel"], OBS_SCALES["ang_vel"]],
    dtype=np.float32,
)

# Simulate action latency like Genesis
SIMULATE_ACTION_LATENCY = True

POLICY_JOINT_NAMES = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]

ROBOT_MOTOR_ORDER = POLICY_JOINT_NAMES.copy()

DEFAULT_Q = np.array(
    [
        0.0, 0.8, -1.5,
        0.0, 0.8, -1.5,
        0.0, 1.0, -1.5,
        0.0, 1.0, -1.5,
    ],
    dtype=np.float32,
)

# ==========================================================
# POLICY
# ==========================================================
class SimplePolicy(nn.Module):
    def __init__(self, obs_dim=45, act_dim=12):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, act_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)

# ==========================================================
# OBS BUILDER (SIM MATCHED)
# ==========================================================
class ObsBuilder:
    def __init__(self):
        self.last_actions = np.zeros(12, dtype=np.float32)
        self.cmd = np.zeros(3, dtype=np.float32)

    # ---------------------------
    # Quaternion math
    # ---------------------------
    @staticmethod
    def quat_rotate(q, v):
        x, y, z, w = q
        qvec = np.array([x, y, z], dtype=np.float32)
        uv = np.cross(qvec, v)
        uuv = np.cross(qvec, uv)
        return v + 2.0 * (w * uv + uuv)

    @staticmethod
    def quat_rotate_inverse(q, v):
        x, y, z, w = q
        q_conj = np.array([-x, -y, -z, w], dtype=np.float32)
        return ObsBuilder.quat_rotate(q_conj, v)

    # ---------------------------
    # Build 45D observation
    # ---------------------------
    def build(self, ls):
        q = np.zeros(12, dtype=np.float32)
        dq = np.zeros(12, dtype=np.float32)

        for i in range(12):
            m = ls.motor_state[i]
            q[i] = m.q
            dq[i] = m.dq

        # IMU
        gyro = np.array(ls.imu_state.gyroscope, dtype=np.float32)
        quat = np.array(ls.imu_state.quaternion, dtype=np.float32)

        quat = quat / (np.linalg.norm(quat) + 1e-8)

        # Gravity in base frame
        world_g = np.array([0, 0, -1], dtype=np.float32)
        gravity_body = self.quat_rotate_inverse(quat, world_g)

        # Angular velocity in base frame
        gyro_body = self.quat_rotate_inverse(quat, gyro)

        obs = np.concatenate(
            [
                gyro_body * OBS_SCALES["ang_vel"],        # 3
                gravity_body,                             # 3
                self.cmd * COMMANDS_SCALE,               # 3
                (q - DEFAULT_Q) * OBS_SCALES["dof_pos"], # 12
                dq * OBS_SCALES["dof_vel"],              # 12
                self.last_actions,                       # 12
            ]
        )

        return obs.astype(np.float32)

# ==========================================================
# FAKE LOWSTATE (OFFLINE MODE)
# ==========================================================
class FakeMotor:
    def __init__(self, q, dq):
        self.q = q
        self.dq = dq

class FakeIMU:
    def __init__(self, gyro, quat):
        self.gyroscope = gyro
        self.quaternion = quat

class FakeLowState:
    def __init__(self, gyro, quat, q, dq):
        self.motor_state = [FakeMotor(q[i], dq[i]) for i in range(12)]
        self.imu_state = FakeIMU(gyro, quat)

# ==========================================================
# CORE RUNNER
# ==========================================================
class Go2Runner:
    def __init__(self, mode):
        self.mode = mode
        self.low_state = None
        self.cmd_template = None
        self.running = True

        self.policy = self._load_policy()
        self.obs_builder = ObsBuilder()
        self.crc = CRC()

        self.last_actions = np.zeros(12, dtype=np.float32)

        if self.mode in [2, 3]:
            self._init_dds()
            self._ensure_low_level_free()

        if self.mode == 3:
            self.pub = ChannelPublisher("rt/lowcmd", LowCmd_)
            self.pub.Init()

        signal.signal(signal.SIGINT, self._handle_sigint)

    # ======================================================
    # LOAD POLICY
    # ======================================================
    def _load_policy(self):
        print("Loading model:", MODEL_PATH)
        policy = SimplePolicy()
        ckpt = torch.load(MODEL_PATH, map_location="cpu")

        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            ckpt = ckpt["model_state_dict"]

        actor_state = {}
        for k, v in ckpt.items():
            if k.startswith("actor."):
                actor_state[k.replace("actor.", "net.")] = v

        policy.load_state_dict(actor_state, strict=True)
        policy.eval()
        print("Model loaded\n")
        return policy

    # ======================================================
    # DDS
    # ======================================================
    def _init_dds(self):
        self.sub_state = ChannelSubscriber("rt/lowstate", LowState_)
        self.sub_state.Init(self.cb_state, 10)

        self.sub_cmd = ChannelSubscriber("rt/lowcmd", LowCmd_)
        self.sub_cmd.Init(self.cb_cmd, 10)

    def cb_state(self, msg):
        self.low_state = msg

    def cb_cmd(self, msg):
        if self.cmd_template is None:
            self.cmd_template = copy.deepcopy(msg)
            print("Captured LowCmd template")

    # ======================================================
    # MODE HANDSHAKE
    # ======================================================
    def _ensure_low_level_free(self):
        print("Releasing high-level control...")
        self.sc = SportClient()
        self.sc.SetTimeout(5.0)
        self.sc.Init()

        self.msc = MotionSwitcherClient()
        self.msc.SetTimeout(5.0)
        self.msc.Init()

        _, result = self.msc.CheckMode()
        while result.get("name"):
            print("Active mode:", result["name"])
            try:
                self.sc.StandDown()
                self.msc.ReleaseMode()
            except:
                pass
            time.sleep(1.0)
            _, result = self.msc.CheckMode()

        print("LOW LEVEL MODE CONFIRMED\n")

    # ======================================================
    # CTRL+C
    # ======================================================
    def _handle_sigint(self, signum, frame):
        print("\nCTRL+C — Exiting safely")
        self.running = False
        sys.exit(0)

    # ======================================================
    # SEND MOTOR COMMAND
    # ======================================================
    def _send_action(self, action):
        if self.cmd_template is None:
            return

        # simulate action latency
        exec_action = self.last_actions if SIMULATE_ACTION_LATENCY else action

        q_targets = DEFAULT_Q + exec_action * ACTION_SCALE
        cmd = copy.deepcopy(self.cmd_template)

        for i in range(12):
            mc = cmd.motor_cmd[i]
            mc.mode = 0x01
            mc.q = float(q_targets[i])
            mc.dq = 0.0
            mc.kp = KP
            mc.kd = KD
            mc.tau = 0.0

        cmd.crc = self.crc.Crc(cmd)
        self.pub.Write(cmd)

        self.last_actions = action.copy()

    # ======================================================
    # OFFLINE MODE
    # ======================================================
    def run_offline(self):
        print("\n=== OFFLINE MODE ===\n")
        print("Using YAML:", YAML_PATH)

        with open(YAML_PATH, "r") as f:
            data = yaml.safe_load(f)

        for s in data["states"]:
            fake = FakeLowState(
                s["gyro"],
                s["quat"],
                s["q"],
                s["dq"],
            )

            obs = self.obs_builder.build(fake)

            with torch.no_grad():
                action = self.policy(torch.tensor(obs).unsqueeze(0))[0].numpy()

            print("STATE:", s["name"])
            print("OBS (45):", np.round(obs, 3))
            print("ACTION (12):", np.round(action, 3))
            print("TARGET_Q:", np.round(DEFAULT_Q + action * ACTION_SCALE, 3))
            print("-" * 60)

            self.obs_builder.last_actions = action

    # ======================================================
    # ROBOT MODE LOOP
    # ======================================================
    def run_robot(self):
        print("\nWaiting for LowState...")
        while self.low_state is None:
            time.sleep(0.1)

        print("Waiting for LowCmd template...")
        while self.cmd_template is None:
            time.sleep(0.1)

        print("\n=== ROBOT LOOP STARTED ===\n")

        while self.running:
            obs = self.obs_builder.build(self.low_state)

            with torch.no_grad():
                action = self.policy(torch.tensor(obs).unsqueeze(0))[0].numpy()

            print("OBS (45):", np.round(obs, 3))
            print("ACTION (12):", np.round(action, 3))

            if self.mode == 3:
                self._send_action(action)

            self.obs_builder.last_actions = action
            time.sleep(DT)

# ==========================================================
# MAIN
# ==========================================================
if __name__ == "__main__":
    print("\nSelect Mode:")
    print("1 = OFFLINE (YAML dummy states)")
    print("2 = MONITOR (Robot, NO commands sent)")
    print("3 = LIVE (Robot, SEND commands)")
    mode = int(input("Enter mode number: ").strip())

    if mode not in [1, 2, 3]:
        print("Invalid mode")
        sys.exit(1)

    if mode in [2, 3]:
        if len(sys.argv) > 1:
            ChannelFactoryInitialize(0, sys.argv[1])
        else:
            ChannelFactoryInitialize(0)

    runner = Go2Runner(mode)

    if mode == 1:
        runner.run_offline()
    else:
        runner.run_robot()
