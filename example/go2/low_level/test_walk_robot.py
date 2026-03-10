import copy
import os
import time
import sys
import signal
import atexit

import numpy as np
import torch
import torch.nn as nn

from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)

from unitree_sdk2py.idl.unitree_go.msg.dds_ import (
    LowCmd_,
    LowState_,
)

from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.go2.sport.sport_client import SportClient

# ==========================================================
# CONFIG
# ==========================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "walk.pt")

DT = 0.02
ACTION_SCALE = 0.2   # Start small. Robot on stand first.

DEFAULT_Q = np.array(
    [
        0.0, 0.8, -1.5,   # FR
        0.0, 0.8, -1.5,   # FL
        0.0, 1.0, -1.5,   # RR
        0.0, 1.0, -1.5    # RL
    ],
    dtype=np.float32,
)

MOTOR_IDX = list(range(12))

# ==========================================================
# SIMPLE POLICY NETWORK
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
# OBS BUILDER
# ==========================================================
class ObsBuilder:
    def __init__(self):
        self.last_actions = np.zeros(12, dtype=np.float32)
        self.cmd = np.zeros(3, dtype=np.float32)

    def build(self, ls):
        q = np.zeros(12, dtype=np.float32)
        dq = np.zeros(12, dtype=np.float32)

        for i in MOTOR_IDX:
            m = ls.motor_state[i]
            q[i] = m.q
            dq[i] = m.dq

        gyro = np.array(ls.imu_state.gyroscope, dtype=np.float32)

        obs = np.concatenate(
            [
                gyro,                                    # 3
                np.array([0, 0, -1], dtype=np.float32), # fake gravity
                self.cmd,                               # 3
                q - DEFAULT_Q,                         # 12
                dq,                                    # 12
                self.last_actions,                    # 12
            ]
        )
        return obs.astype(np.float32)

# ==========================================================
# MAIN RUNNER
# ==========================================================
class Go2Runner:
    def __init__(self):
        self.low_state = None
        self.cmd_template = None
        self.safe = False  # safety latch

        print("Loading model:", MODEL_PATH)

        # -------------------------
        # CREATE NETWORK
        # -------------------------
        self.policy = SimplePolicy(obs_dim=45, act_dim=12)

        # -------------------------
        # LOAD CHECKPOINT
        # -------------------------
        checkpoint = torch.load(MODEL_PATH, map_location="cpu")

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            print("Found checkpoint bundle, extracting model_state_dict")
            full_state = checkpoint["model_state_dict"]
        else:
            print("Found raw state_dict")
            full_state = checkpoint

        # -------------------------
        # EXTRACT ACTOR WEIGHTS
        # -------------------------
        actor_state = {}
        for k, v in full_state.items():
            if k.startswith("actor."):
                actor_state[k.replace("actor.", "net.")] = v

        print("Loading actor weights:")
        for k in actor_state:
            print(" ", k)

        self.policy.load_state_dict(actor_state)
        self.policy.eval()

        print("Policy test output:", self.policy(torch.zeros(1, 45)))

        # -------------------------
        # DDS
        # -------------------------
        self.sub_state = ChannelSubscriber("rt/lowstate", LowState_)
        self.sub_state.Init(self.cb_state, 10)

        # Subscribe to LOWCMD to steal a valid template
        self.sub_cmd = ChannelSubscriber("rt/lowcmd", LowCmd_)
        self.sub_cmd.Init(self.cb_cmd, 10)

        self.pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.pub.Init()

        self.obs_builder = ObsBuilder()
        self.crc = CRC()

        # -------------------------
        # MODE HANDSHAKE (OFFICIAL WAY)
        # -------------------------
        print("\nSwitching robot to LOW LEVEL MODE (official handshake)...")

        self.sc = SportClient()
        self.sc.SetTimeout(5.0)
        self.sc.Init()

        self.msc = MotionSwitcherClient()
        self.msc.SetTimeout(5.0)
        self.msc.Init()

        status, result = self.msc.CheckMode()
        while result["name"]:
            print("High-level mode active:", result["name"])
            print("Standing down + releasing mode...")
            self.sc.StandDown()
            self.msc.ReleaseMode()
            time.sleep(1.0)
            status, result = self.msc.CheckMode()

        print("LOW LEVEL MODE CONFIRMED")

        # -------------------------
        # SAFETY HOOKS
        # -------------------------
        self._register_safety_hooks()

        print("\nMode: LIVE ROBOT MODE")
        print("Waiting for LowCmd template (run go2_sport_client once)...")

    # ======================================================
    # CALLBACKS
    # ======================================================
    def cb_state(self, msg):
        self.low_state = msg

    def cb_cmd(self, msg):
        if self.cmd_template is None:
            self.cmd_template = copy.deepcopy(msg)
            print("Captured LowCmd template from robot")

    # ======================================================
    # SAFETY: FREEZE ON EXIT
    # ======================================================
    def _register_safety_hooks(self):
        def cleanup():
            if self.safe:
                return
            self.safe = True

            print("\n[SAFETY] Policy stopped. FREEZING robot in current pose.")

            try:
                # Stop sending new commands
                time.sleep(0.1)

                # Release low-level mode so robot holds last pose
                self.msc.ReleaseMode()
                print("[SAFETY] Low-level mode released. Robot should hold position.")
            except Exception as e:
                print("[SAFETY] Cleanup error:", e)

        atexit.register(cleanup)
        signal.signal(signal.SIGINT, lambda sig, frame: cleanup())
        signal.signal(signal.SIGTERM, lambda sig, frame: cleanup())

    # ======================================================
    # SEND ACTION
    # ======================================================
    def send_action(self, action):
        if self.safe:
            return

        if self.cmd_template is None:
            print("Waiting for LowCmd template...")
            return

        cmd = copy.deepcopy(self.cmd_template)

        for i in range(12):
            mc = cmd.motor_cmd[i]
            mc.mode = 0x01  # position control
            mc.q = float(DEFAULT_Q[i] + action[i] * ACTION_SCALE)
            mc.dq = 0.0
            mc.kp = 20.0
            mc.kd = 0.5
            mc.tau = 0.0

        cmd.crc = self.crc.Crc(cmd)
        self.pub.Write(cmd)

    # ======================================================
    # MAIN LOOP
    # ======================================================
    def loop(self):
        while self.low_state is None:
            time.sleep(0.1)

        print("\nLowstate received. Running policy loop...\n")

        while True:
            obs = self.obs_builder.build(self.low_state)

            with torch.no_grad():
                obs_t = torch.tensor(obs).unsqueeze(0)
                action = self.policy(obs_t)[0].cpu().numpy()

            self.obs_builder.last_actions = action

            print("OBS[0:6]:", np.round(obs[:6], 3))
            print("ACTION   :", np.round(action, 3))

            self.send_action(action)
            time.sleep(DT)

# ==========================================================
# START
# ==========================================================
if __name__ == "__main__":
    print("\n=== GO2 POLICY RUNNER ===")
    print("WARNING: Please ensure there are no obstacles around the robot.")
    input("Press Enter to continue...")

    if len(sys.argv) > 1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)

    runner = Go2Runner()
    runner.loop()
