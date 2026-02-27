import sys
import threading
import time

import unitree_legged_const as go2

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

# ===== CONFIG =====
JOINT = 3  # 0–11
TARGET = 1.0  # radians
KP = 10
KD = 10
DT = 0.002
# =================

HIGHLEVEL = 0xEE
LOWLEVEL = 0xFF


class JointTest:
    def __init__(self):
        self.cmd = unitree_go_msg_dds__LowCmd_()  # <-- CORRECT CLASS
        self.state = None
        self.crc = CRC()
        self.active = False
        self.last_print = 0

    def start(self):
        # ---- SWITCH TO LOW LEVEL MODE ----
        print("Switching to LOW LEVEL mode...")
        self.msc = MotionSwitcherClient()
        self.msc.SetTimeout(5.0)
        self.msc.Init()
        self.msc.SelectMode(LOWLEVEL)
        time.sleep(0.5)

        # ---- DDS ----
        self.pub = ChannelPublisher("rt/lowcmd", type(self.cmd))
        self.pub.Init()

        sub = ChannelSubscriber("rt/lowstate", LowState_)
        sub.Init(self.cb, 10)

        print("Waiting for robot state...")
        while self.state is None:
            time.sleep(0.1)

        print("Connected")

        # ---- HEADER ----
        self.cmd.head[0] = 0xFE
        self.cmd.head[1] = 0xEF
        self.cmd.level_flag = LOWLEVEL
        self.cmd.gpio = 0

        # ---- SAFE DEFAULTS ----
        for i in range(20):
            self.cmd.motor_cmd[i].mode = 0x0A  # SERVO (PD mode)
            self.cmd.motor_cmd[i].q = go2.PosStopF
            self.cmd.motor_cmd[i].dq = go2.VelStopF
            self.cmd.motor_cmd[i].kp = 0
            self.cmd.motor_cmd[i].kd = 0
            self.cmd.motor_cmd[i].tau = 0

        print("Low-level control active")
        print("Motor slots:", len(self.cmd.motor_cmd))  # sanity check (should be 20)

        RecurrentThread(DT, self.loop, "joint_loop").Start()
        threading.Thread(target=self.keyboard, daemon=True).start()

    def cb(self, msg):
        self.state = msg

    def get_joint_pos(self):
        if self.state:
            return self.state.motor_state[JOINT].q
        return None

    def loop(self):
        if self.active:
            self.cmd.motor_cmd[JOINT].q = TARGET
            self.cmd.motor_cmd[JOINT].dq = 0
            self.cmd.motor_cmd[JOINT].kp = KP
            self.cmd.motor_cmd[JOINT].kd = KD
            self.cmd.motor_cmd[JOINT].tau = 0
        else:
            self.cmd.motor_cmd[JOINT].q = go2.PosStopF
            self.cmd.motor_cmd[JOINT].dq = go2.VelStopF
            self.cmd.motor_cmd[JOINT].kp = 0
            self.cmd.motor_cmd[JOINT].kd = 0
            self.cmd.motor_cmd[JOINT].tau = 0

        self.cmd.crc = self.crc.Crc(self.cmd)
        self.pub.Write(self.cmd)

        # ---- LIVE PRINT (10Hz) ----
        now = time.time()
        if self.active and self.state and (now - self.last_print) > 0.1:
            pos = self.get_joint_pos()
            if pos is not None:
                print(f"\rJoint {JOINT}: {pos:.3f} → {TARGET:.3f} rad", end="")
            self.last_print = now

    def keyboard(self):
        print("\nControls:")
        print("  m = move joint")
        print("  r = release joint")
        print("  q = quit (freeze motors)\n")

        while True:
            key = input("> ").strip().lower()
            if key == "m":
                pos = self.get_joint_pos()
                if pos is not None:
                    print(f"\nMoving joint {JOINT} from {pos:.3f} → {TARGET:.3f} rad")
                else:
                    print(f"\nMoving joint {JOINT} to {TARGET:.3f} rad")
                self.active = True

            elif key == "r":
                self.active = False
                print(f"\nJoint {JOINT} released")

            elif key == "q":
                print("\nFreezing all joints and exiting...")
                self.freeze_all()
                sys.exit(0)

    def freeze_all(self):
        for i in range(20):
            self.cmd.motor_cmd[i].q = go2.PosStopF
            self.cmd.motor_cmd[i].dq = go2.VelStopF
            self.cmd.motor_cmd[i].kp = 0
            self.cmd.motor_cmd[i].kd = 0
            self.cmd.motor_cmd[i].tau = 0

        self.cmd.crc = self.crc.Crc(self.cmd)
        self.pub.Write(self.cmd)


if __name__ == "__main__":
    print("Robot must be on rig")
    input("Press ENTER to start")

    if len(sys.argv) > 1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)

    JointTest().start()

    while True:
        time.sleep(1)
