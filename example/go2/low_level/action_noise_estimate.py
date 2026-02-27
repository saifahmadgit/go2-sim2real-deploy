import sys
import time

import numpy as np
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import (
    MotionSwitcherClient,
)
from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread

import unitree_legged_const as go2

# ========= CONFIG =========
JOINT = 3
AMP = 0.25
FREQ = 1.0
KP = 60
KD = 2
SECONDS = 15
ACT_MULT = 1.0
DT = 0.002
LOWLEVEL = 0xFF
# ===========================


class ActNoise:
    def __init__(self):
        self.cmd = unitree_go_msg_dds__LowCmd_()
        self.state = None
        self.crc = CRC()

        self.targets = []
        self.measured = []
        self.t0 = None

    def cb(self, msg):
        self.state = msg

    def start(self):
        print("Switching LOWLEVEL...")
        msc = MotionSwitcherClient()
        msc.Init()
        msc.SelectMode(LOWLEVEL)
        time.sleep(0.5)

        self.pub = ChannelPublisher("rt/lowcmd", type(self.cmd))
        self.pub.Init()

        sub = ChannelSubscriber("rt/lowstate", LowState_)
        sub.Init(self.cb, 10)

        while self.state is None:
            time.sleep(0.1)

        self.init_cmd()
        RecurrentThread(DT, self.loop, "loop").Start()

        print("Recording action noise...")
        time.sleep(SECONDS)

        self.compute()

    def init_cmd(self):
        self.cmd.head[0] = 0xFE
        self.cmd.head[1] = 0xEF
        self.cmd.level_flag = LOWLEVEL

        for i in range(20):
            self.cmd.motor_cmd[i].mode = 0x0A
            self.cmd.motor_cmd[i].q = go2.PosStopF
            self.cmd.motor_cmd[i].dq = go2.VelStopF

        self.center = self.state.motor_state[JOINT].q
        self.t0 = time.time()

    def loop(self):
        t = time.time() - self.t0
        target = self.center + AMP * np.sin(2 * np.pi * FREQ * t)

        self.cmd.motor_cmd[JOINT].q = float(target)
        self.cmd.motor_cmd[JOINT].kp = KP
        self.cmd.motor_cmd[JOINT].kd = KD

        self.cmd.crc = self.crc.Crc(self.cmd)
        self.pub.Write(self.cmd)

        if self.state:
            q = self.state.motor_state[JOINT].q
            self.targets.append(target)
            self.measured.append(q)

    def compute(self):
        err = np.array(self.measured) - np.array(self.targets)
        std = np.std(err)

        print("\n=== ACTION NOISE ===")
        print("action_noise_std =", ACT_MULT * std)


if __name__ == "__main__":
    print("GO2 ACTION NOISE (Robot on Rig)")
    input("Press ENTER")

    if len(sys.argv) > 1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)

    ActNoise().start()
