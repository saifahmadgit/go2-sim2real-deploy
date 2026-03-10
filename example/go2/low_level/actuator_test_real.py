"""
actuator_test_real.py  —  Real Go2 actuator step-response test

The user holds the robot in the air by hand. The script sends the
exact same joint-position step commands as the sim test, recording
commanded vs actual joint positions from the motor encoders.

Saves results to  actuator_test_real_results.pkl

Usage:
    python actuator_test_real.py [NETWORK_INTERFACE]

    Example:  python actuator_test_real.py eth0
              python actuator_test_real.py          # default interface
"""

import os
import pickle
import sys
import time

import torch

# =====================================================================
# TEST CONFIGURATION  — MUST MATCH actuator_test_sim.py exactly
# =====================================================================

SHORT_NAMES = [
    "FR_hip", "FR_thigh", "FR_calf",
    "FL_hip", "FL_thigh", "FL_calf",
    "RR_hip", "RR_thigh", "RR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
]

DEFAULT_DOF_POS = [
    0.0, 0.8, -1.5,   # FR
    0.0, 0.8, -1.5,   # FL
    0.0, 1.0, -1.5,   # RR
    0.0, 1.0, -1.5,   # RL
]

# Which joints to test (index 0-11)
TEST_JOINTS = [0, 1, 2]  # FR_hip, FR_thigh, FR_calf

# Step amplitude (rad)
STEP_AMPLITUDES = [0.2, 0.4]

# Kp / Kd combos to test — same as sim
KP_KD_COMBOS = [
    (20.0, 0.5),
    (30.0, 0.5),
    (40.0, 0.5),
    (40.0, 1.0),
    (40.0, 2.0),
    (60.0, 0.5),
    (60.0, 2.0),
]

# Timing — same as sim
SETTLE_SECONDS = 1.0
STEP_SECONDS   = 2.0
RETURN_SECONDS = 1.5

# Control rate — the rate we send commands and record
# Real robot lowcmd runs at 500 Hz, we record at this rate
CONTROL_HZ = 500.0
CONTROL_DT = 1.0 / CONTROL_HZ

# How often to record (every N control steps)
# 1 = record every step (500 Hz), 5 = record at 100 Hz, etc.
RECORD_DECIMATION = 1

# Output file
OUTPUT_FILE = "actuator_test_real_results.pkl"


# =====================================================================
# SDK helpers
# =====================================================================

def init_dds():
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    if len(sys.argv) > 1:
        ChannelFactoryInitialize(0, sys.argv[1])
        print(f"DDS interface: {sys.argv[1]}")
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


def read_joint_state(low_state_msg):
    """Read current joint positions and velocities from lowstate."""
    q = []
    dq = []
    for i in range(12):
        ms = low_state_msg.motor_state[i]
        q.append(float(ms.q))
        dq.append(float(ms.dq))
    return q, dq


# =====================================================================
# Main
# =====================================================================

def main():
    import unitree_legged_const as go2
    from unitree_sdk2py.core.channel import ChannelPublisher
    from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
    from unitree_sdk2py.utils.crc import CRC

    init_dds()
    latest = wait_for_lowstate()

    # Release high-level control
    release_sport_and_highlevel()

    # Setup publisher
    pub = ChannelPublisher("rt/lowcmd", LowCmd_)
    pub.Init()
    crc = CRC()
    low_cmd = unitree_go_msg_dds__LowCmd_()

    low_cmd.head[0] = 0xFE
    low_cmd.head[1] = 0xEF
    low_cmd.level_flag = 0xFF
    low_cmd.gpio = 0

    # Initialize all motors to safe stop
    for i in range(20):
        low_cmd.motor_cmd[i].mode = 0x01
        low_cmd.motor_cmd[i].q = go2.PosStopF
        low_cmd.motor_cmd[i].dq = go2.VelStopF
        low_cmd.motor_cmd[i].kp = 0.0
        low_cmd.motor_cmd[i].kd = 0.0
        low_cmd.motor_cmd[i].tau = 0.0

    # ------------------------------------------------------------------
    # Helper: send one low-level command frame
    # ------------------------------------------------------------------
    def send_cmd(target_q_12, kp_12, kd_12):
        """Send position command to all 12 joints."""
        for i in range(12):
            low_cmd.motor_cmd[i].mode = 0x01
            low_cmd.motor_cmd[i].q = float(target_q_12[i])
            low_cmd.motor_cmd[i].dq = 0.0
            low_cmd.motor_cmd[i].kp = float(kp_12[i])
            low_cmd.motor_cmd[i].kd = float(kd_12[i])
            low_cmd.motor_cmd[i].tau = 0.0
        low_cmd.crc = crc.Crc(low_cmd)
        pub.Write(low_cmd)

    # ------------------------------------------------------------------
    # Helper: send safe stop
    # ------------------------------------------------------------------
    def send_safe_stop():
        for i in range(12):
            low_cmd.motor_cmd[i].q = go2.PosStopF
            low_cmd.motor_cmd[i].dq = go2.VelStopF
            low_cmd.motor_cmd[i].kp = 0.0
            low_cmd.motor_cmd[i].kd = 0.0
            low_cmd.motor_cmd[i].tau = 0.0
        low_cmd.crc = crc.Crc(low_cmd)
        pub.Write(low_cmd)

    # ------------------------------------------------------------------
    # Ramp to default pose first (safely)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  ACTUATOR STEP-RESPONSE TEST — REAL ROBOT")
    print("=" * 60)
    print(f"\nTesting joints: {[SHORT_NAMES[j] for j in TEST_JOINTS]}")
    print(f"Amplitudes:     {STEP_AMPLITUDES} rad")
    print(f"Kp/Kd combos:   {len(KP_KD_COMBOS)} combos")
    total_tests = len(TEST_JOINTS) * len(STEP_AMPLITUDES) * len(KP_KD_COMBOS)
    total_time = total_tests * (SETTLE_SECONDS + STEP_SECONDS + RETURN_SECONDS)
    print(f"Total tests:    {total_tests}")
    print(f"Estimated time: {total_time:.0f} seconds ({total_time/60:.1f} min)")

    print("\n*** HOLD THE ROBOT IN THE AIR WITH ALL LEGS FREE ***")
    print("*** Make sure legs can move freely without hitting anything ***")
    print("\nType 'go' to start, anything else to abort:")
    user = input("> ").strip().lower()
    if user != "go":
        print("Aborted.")
        return

    # Ramp to default pose over 3 seconds
    RAMP_KP = 30.0
    RAMP_KD = 1.0
    RAMP_SECONDS = 3.0
    ramp_steps = int(RAMP_SECONDS * CONTROL_HZ)

    current_q, _ = read_joint_state(latest["msg"])
    start_q = list(current_q)

    print(f"\nRamping to default pose over {RAMP_SECONDS}s...")
    for step in range(ramp_steps):
        alpha = (step + 1) / float(ramp_steps)
        target = [
            (1 - alpha) * start_q[i] + alpha * DEFAULT_DOF_POS[i]
            for i in range(12)
        ]
        kp_arr = [RAMP_KP] * 12
        kd_arr = [RAMP_KD] * 12
        send_cmd(target, kp_arr, kd_arr)
        time.sleep(CONTROL_DT)

    print("Default pose reached. Starting tests in 1 second...")
    time.sleep(1.0)

    # ------------------------------------------------------------------
    # Run all tests
    # ------------------------------------------------------------------
    all_results = []
    test_num = 0

    settle_steps = int(SETTLE_SECONDS * CONTROL_HZ)
    step_steps   = int(STEP_SECONDS * CONTROL_HZ)
    return_steps = int(RETURN_SECONDS * CONTROL_HZ)

    try:
        for joint_idx in TEST_JOINTS:
            for amplitude in STEP_AMPLITUDES:
                for kp, kd in KP_KD_COMBOS:
                    test_num += 1
                    jname = SHORT_NAMES[joint_idx]
                    print(f"\n[{test_num}/{total_tests}] Joint={jname}  "
                          f"amp={amplitude:.2f}rad  Kp={kp}  Kd={kd}")

                    # Build command arrays (all joints at default, test joint will change)
                    target_q = list(DEFAULT_DOF_POS)
                    kp_arr = [kp] * 12   # same Kp for all joints (they hold default)
                    kd_arr = [kd] * 12

                    record = {
                        "joint_idx": joint_idx,
                        "joint_name": jname,
                        "amplitude": amplitude,
                        "kp": kp,
                        "kd": kd,
                        "dt": CONTROL_DT,
                        "default_pos": DEFAULT_DOF_POS[joint_idx],
                        "target_pos": DEFAULT_DOF_POS[joint_idx] + amplitude,
                        "phase": [],
                        "t": [],
                        "cmd_pos": [],
                        "actual_pos": [],
                        "actual_vel": [],
                    }

                    # --- Phase 1: settle at default ---
                    target_q[joint_idx] = DEFAULT_DOF_POS[joint_idx]
                    for s in range(settle_steps):
                        send_cmd(target_q, kp_arr, kd_arr)
                        time.sleep(CONTROL_DT)

                    # --- Phase 2: step to target ---
                    stepped_value = DEFAULT_DOF_POS[joint_idx] + amplitude
                    target_q[joint_idx] = stepped_value

                    t = 0.0
                    for s in range(step_steps):
                        send_cmd(target_q, kp_arr, kd_arr)

                        if s % RECORD_DECIMATION == 0:
                            q_actual, dq_actual = read_joint_state(latest["msg"])
                            record["phase"].append("step")
                            record["t"].append(t)
                            record["cmd_pos"].append(stepped_value)
                            record["actual_pos"].append(q_actual[joint_idx])
                            record["actual_vel"].append(dq_actual[joint_idx])

                        t += CONTROL_DT
                        time.sleep(CONTROL_DT)

                    # --- Phase 3: return to default ---
                    target_q[joint_idx] = DEFAULT_DOF_POS[joint_idx]

                    for s in range(return_steps):
                        send_cmd(target_q, kp_arr, kd_arr)

                        if s % RECORD_DECIMATION == 0:
                            q_actual, dq_actual = read_joint_state(latest["msg"])
                            record["phase"].append("return")
                            record["t"].append(t)
                            record["cmd_pos"].append(DEFAULT_DOF_POS[joint_idx])
                            record["actual_pos"].append(q_actual[joint_idx])
                            record["actual_vel"].append(dq_actual[joint_idx])

                        t += CONTROL_DT
                        time.sleep(CONTROL_DT)

                    all_results.append(record)
                    n_pts = len(record["t"])
                    final_err = abs(record["actual_pos"][-1] - record["cmd_pos"][-1])
                    print(f"  Recorded {n_pts} points, final err = {final_err:.4f} rad")

    except KeyboardInterrupt:
        print("\n\nInterrupted! Saving partial results...")

    finally:
        # Safe stop
        print("\nSending safe stop...")
        for _ in range(500):
            send_safe_stop()
            time.sleep(CONTROL_DT)
        print("Motors released.")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump({
            "source": "real_robot",
            "control_dt": CONTROL_DT,
            "joint_names": SHORT_NAMES,
            "default_dof_pos": DEFAULT_DOF_POS,
            "tests": all_results,
        }, f)

    print(f"\n{'='*60}")
    print(f"  Saved {len(all_results)} test results to: {OUTPUT_FILE}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
