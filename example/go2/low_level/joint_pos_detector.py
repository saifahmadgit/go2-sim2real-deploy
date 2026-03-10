import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_

# Motor order for Go2 (first 12 are leg joints)
MOTOR_IDX = list(range(12))


class JointProbe:
    def __init__(self):
        self.low_state = None

        self.sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.sub.Init(self.cb, 10)

    def cb(self, msg):
        self.low_state = msg

    def print_joints(self):
        if self.low_state is None:
            print("Waiting for lowstate...")
            return

        joints = []
        for i in MOTOR_IDX:
            m = self.low_state.motor_state[i]
            joints.append(round(m.q, 4))

        print("Joint positions:", joints)


if __name__ == "__main__":
    print("\n=== GO2 JOINT POSITION PROBE ===")
    print("Move the robot by hand and watch values change\n")

    ChannelFactoryInitialize(0)

    probe = JointProbe()

    while True:
        probe.print_joints()
        time.sleep(0.1)  # 10 Hz
