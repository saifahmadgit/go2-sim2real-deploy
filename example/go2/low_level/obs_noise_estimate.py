import sys
import time

import numpy as np
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_

# ================= CONFIG =================
SAMPLE_HZ = 200
SECONDS = 20
OBS_MULT = 1.0
NUM_JOINTS = 12
# ==========================================


def quat_to_gravity_xyz(q):
    w, x, y, z = q
    R = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    g = np.array([0, 0, -1])
    return R.T @ g


class ObsNoise:
    def __init__(self):
        self.state = None
        self.sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.sub.Init(self.cb, 10)

        self.gyro = []
        self.grav = []
        self.q = []
        self.dq = []

    def cb(self, msg):
        self.state = msg

    def run(self):

        print("Waiting for state...")
        while self.state is None:
            time.sleep(0.1)

        print("Recording centered observation noise...")
        start = time.time()

        while time.time() - start < SECONDS:
            ls = self.state

            self.gyro.append(list(ls.imu_state.gyroscope))
            self.grav.append(quat_to_gravity_xyz(ls.imu_state.quaternion))

            q = [ls.motor_state[i].q for i in range(NUM_JOINTS)]
            dq = [ls.motor_state[i].dq for i in range(NUM_JOINTS)]

            self.q.append(q)
            self.dq.append(dq)

            time.sleep(1 / SAMPLE_HZ)

        self.compute()

    # ================= CORE =================

    def centered_std(self, arr):
        arr = np.array(arr)
        mean = np.mean(arr, axis=0)
        centered = arr - mean
        std = np.std(centered, axis=0)
        return np.mean(std)  # single scalar for RL

    def compute(self):

        s1 = self.centered_std(self.gyro)
        s2 = self.centered_std(self.grav)
        s3 = self.centered_std(self.q)
        s4 = self.centered_std(self.dq)

        print("\n=== SIMULATION READY VALUES ===")
        print("obs_noise_level = 1.0")
        print("obs_noise = {")
        print(f'  "ang_vel": {OBS_MULT * s1:.6g},')
        print(f'  "gravity": {OBS_MULT * s2:.6g},')
        print(f'  "dof_pos": {OBS_MULT * s3:.6g},')
        print(f'  "dof_vel": {OBS_MULT * s4:.6g},')
        print("}")


# ================= ENTRY =================

if __name__ == "__main__":
    print("GO2 OBSERVATION NOISE ESTIMATION (Centered)")

    if len(sys.argv) > 1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)

    ObsNoise().run()
