import time

import numpy as np
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_

MOTOR_IDX = list(range(12))


class RawObsProbe:
    def __init__(self):
        self.ls = None
        self.last_actions = np.zeros(12, dtype=np.float32)
        self.cmd = np.zeros(3, dtype=np.float32)

        self.sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.sub.Init(self.cb, 10)

    def cb(self, msg):
        self.ls = msg

    def print_obs(self):
        if self.ls is None:
            print("Waiting for lowstate...")
            return

        ls = self.ls

        # Joint states
        q = np.zeros(12)
        dq = np.zeros(12)
        for i in MOTOR_IDX:
            m = ls.motor_state[i]
            q[i] = m.q
            dq[i] = m.dq

        # Raw IMU
        gyro = np.array(ls.imu_state.gyroscope)
        quat = np.array(ls.imu_state.quaternion)

        obs = np.concatenate(
            [
                gyro,  # 3
                quat[:3],  # just first 3 for dimension check
                self.cmd,  # 3
                q,  # 12
                dq,  # 12
                self.last_actions,  # 12
            ]
        )

        print("\n=========== RAW OBS CHECK ===========")
        print(f"Gyro (3)     : {np.round(gyro, 4)}")
        print(f"Quat (4)    : {np.round(quat, 4)}")
        print(f"Cmd (3)     : {self.cmd}")
        print(f"Joint q (12): {np.round(q, 4)}")
        print(f"Joint dq(12): {np.round(dq, 4)}")
        print(f"Last a(12)  : {self.last_actions}")
        print("\nOBS DIM:", obs.shape[0])
        print("====================================")


if __name__ == "__main__":
    print("\n=== GO2 RAW LOWSTATE PROBE ===")
    ChannelFactoryInitialize(0)

    probe = RawObsProbe()
    while True:
        probe.print_obs()
        time.sleep(0.2)
