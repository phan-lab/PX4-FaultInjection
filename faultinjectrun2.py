#!/usr/bin/env python3

import asyncio
import csv
import logging
import math
import random
import subprocess
import time
from pathlib import Path

import pandas as pd
from mavsdk import System
from mavsdk.offboard import OffboardError, AttitudeRate

logging.basicConfig(level=logging.INFO)

NUM_ROTORS = 4
NUM_RUNS = 100
MAX_FAILURES = 3

PX4_ROOT = Path("/home/sshenoy/PX4-Autopilot")
PX4_CMD = ["make", "px4_sitl", "gz_x500", "HEADLESS=1"]
LOG_DIR = Path("/home/sshenoy/PX4-Autopilot/build/px4_sitl_default/rootfs/log/2026-04-18")

FAULT_LOG_FILE = Path("fault_metadata.csv")
PX4_PROCESS = None


# --------------------------------------------------
# PX4 / Gazebo process management
# --------------------------------------------------
def start_px4():
    global PX4_PROCESS
    print("[PX4] Starting PX4 SITL...")
    PX4_PROCESS = subprocess.Popen(
        PX4_CMD,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    )
    time.sleep(12)


def stop_px4():
    global PX4_PROCESS

    print("[PX4] Stopping PX4 + Gazebo...")
    try:
        if PX4_PROCESS is not None:
            PX4_PROCESS.terminate()
            try:
                PX4_PROCESS.wait(timeout=5)
            except subprocess.TimeoutExpired:
                PX4_PROCESS.kill()
    except Exception:
        pass

    subprocess.run(["pkill", "-9", "-f", "px4"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["pkill", "-9", "-f", "gz"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    PX4_PROCESS = None
    time.sleep(8)


async def restart_px4():
    stop_px4()
    start_px4()
    await asyncio.sleep(10)


def px4_shell(command: str):
    global PX4_PROCESS
    if PX4_PROCESS is None or PX4_PROCESS.stdin is None:
        return
    try:
        PX4_PROCESS.stdin.write(command + "\n")
        PX4_PROCESS.stdin.flush()
        print(f"[PX4 SHELL] {command}")
    except Exception as e:
        print(f"[WARN] Failed to send PX4 shell command: {e}")


def get_all_logs():
    return set(f.name for f in LOG_DIR.rglob("*.ulg"))


# def get_latest_log():
#     logs = list(get_all_logs())
#     if not logs:
#         return None
#     return max(logs, key=lambda f: f.stat().st_mtime)


# async def wait_for_new_log(logs_before, timeout_s=15.0):
#     """
#     Wait after landing/disarm for PX4 to finish and flush a new ULog.
#     """
#     t0 = time.time()
#     while time.time() - t0 < timeout_s:
#         logs_after = get_all_logs()
#         new_logs = logs_after - logs_before
#         if new_logs:
#             latest = max(new_logs, key=lambda f: f.stat().st_mtime)
#             return latest
#         await asyncio.sleep(1.0)

#     # Fallback: maybe logger reused timing in a way that makes set-diff unreliable
#     latest = get_latest_log()
#     return latest


# --------------------------------------------------
# Fault model
# --------------------------------------------------
def sample_fault():
    """
    Synthetic severity-scaled MAVSDK disturbance model.

    fault_type:
      - healthy
      - single_partial
      - double_partial

    severity means 0.0 -> healthy, 1.0 -> strongest disturbance
    """
    scenario = random.choices(
        ["healthy", "single_partial", "double_partial"],
        weights=[0.25, 0.55, 0.20]
    )[0]

    rotor_ids = []
    fault_times = [100.0] * NUM_ROTORS
    severities = [0.0] * NUM_ROTORS

    if scenario == "healthy":
        return {
            "fault_type": scenario,
            "fault_times": fault_times,
            "severities": severities,
            "rotor_ids": rotor_ids
        }

    n_faults = 1 if scenario == "single_partial" else 2
    rotor_ids = random.sample(range(NUM_ROTORS), n_faults)

    for rid in rotor_ids:
        fault_times[rid] = random.uniform(2.0, 5.0)
        severities[rid] = random.uniform(0.2, 1.0)

    return {
        "fault_type": scenario,
        "fault_times": fault_times,
        "severities": severities,
        "rotor_ids": rotor_ids
    }


def rotor_to_disturbance_signs(rotor_id: int):
    """
    Approximate quad geometry mapping for synthetic disturbance injection.

    Assumed rotor layout:
      0: front-left
      1: front-right
      2: rear-right
      3: rear-left

    Returns signs for roll, pitch, yaw disturbance contribution.
    """
    mapping = {
        0: (+1.0, -1.0, +1.0),
        1: (-1.0, -1.0, -1.0),
        2: (-1.0, +1.0, +1.0),
        3: (+1.0, +1.0, -1.0),
    }
    return mapping[rotor_id]


def compute_disturbance_rates(t, fault_cfg):
    """
    Build a severity-scaled synthetic fault as body-rate + thrust disturbance.

    This is still synthetic, but better than modifying the attitude setpoint directly.
    """
    roll_rate = 0.0
    pitch_rate = 0.0
    yaw_rate = 0.0
    thrust_drop = 0.0

    for rid in range(NUM_ROTORS):
        t_fault = fault_cfg["fault_times"][rid]
        sev = fault_cfg["severities"][rid]

        if t < t_fault or sev <= 0.0:
            continue

        # smooth ramp in over 300 ms
        ramp = min((t - t_fault) / 0.30, 1.0)
        s = sev * ramp

        roll_sign, pitch_sign, yaw_sign = rotor_to_disturbance_signs(rid)

        # base bias component
        roll_rate  += roll_sign  * (18.0 * s)
        pitch_rate += pitch_sign * (18.0 * s)
        yaw_rate   += yaw_sign   * (35.0 * s)

        # small oscillatory component to make IMU windows richer
        roll_rate  += roll_sign  * (5.0 * s * math.sin(2 * math.pi * 2.2 * t))
        pitch_rate += pitch_sign * (5.0 * s * math.sin(2 * math.pi * 1.8 * t + 0.4))
        yaw_rate   += yaw_sign   * (8.0 * s * math.sin(2 * math.pi * 1.2 * t + 0.8))

        # collective thrust reduction grows with severity
        thrust_drop += 0.05 * s

    # keep thrust within safe-ish range
    return roll_rate, pitch_rate, yaw_rate, thrust_drop


# --------------------------------------------------
# MAVSDK helpers
# --------------------------------------------------
async def reconnect():
    for attempt in range(5):
        try:
            drone = System()
            await drone.connect(system_address="udpin://0.0.0.0:14540")

            print("[MAVSDK] Waiting for connection...")
            async for state in drone.core.connection_state():
                if state.is_connected:
                    print("[MAVSDK] Connected")
                    return drone
        except Exception as e:
            print(f"[WARN] Reconnect attempt {attempt + 1} failed: {e}")
            await asyncio.sleep(2)

    raise RuntimeError("Failed to connect to PX4")


async def wait_until_ready(drone, timeout_s=25.0):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            health_gen = drone.telemetry.health()
            health = await health_gen.__anext__()

            if (
                health.is_global_position_ok
                and health.is_home_position_ok
                and health.is_armable
            ):
                print("[READY] Global position OK, home OK, armable")
                return True
        except StopAsyncIteration:
            pass
        except Exception:
            pass

        await asyncio.sleep(0.2)

    return False


async def wait_until_landed(drone, timeout_s=25.0):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            async for landed_state in drone.telemetry.landed_state():
                if str(landed_state) == "LandedState.ON_GROUND":
                    return True
                break
        except Exception:
            pass
        await asyncio.sleep(0.2)

    return False


async def start_offboard(drone):
    """
    Prime Offboard with body-rate + thrust before start().
    """
    await drone.offboard.set_attitude_rate(
        AttitudeRate(0.0, 0.0, 0.0, 0.60)
    )
    await asyncio.sleep(0.2)

    try:
        await drone.offboard.start()
        print("[OFFBOARD] Started")
    except OffboardError as e:
        print(f"[ERROR] Offboard start failed: {e}")
        try:
            await drone.action.land()
        except Exception:
            pass
        raise


async def send_nominal_hover(drone, duration_s, dt=0.05):
    steps = int(duration_s / dt)
    for _ in range(steps):
        await drone.offboard.set_attitude_rate(
            AttitudeRate(0.0, 0.0, 0.0, 0.60)
        )
        await asyncio.sleep(dt)


async def send_small_nominal_excitation(drone, duration_s=4.0, dt=0.05):
    """
    Mild healthy excitation block so the dataset isn't only pure hover.
    """
    steps = int(duration_s / dt)
    for k in range(steps):
        t = k * dt
        roll_rate = 4.0 * math.sin(2 * math.pi * 0.8 * t)
        pitch_rate = 4.0 * math.sin(2 * math.pi * 1.1 * t)
        yaw_rate = 8.0 * math.sin(2 * math.pi * 0.5 * t)

        await drone.offboard.set_attitude_rate(
            AttitudeRate(roll_rate, pitch_rate, yaw_rate, 0.60)
        )
        await asyncio.sleep(dt)


async def send_faulted_segment(drone, fault_cfg, duration_s=6.0, dt=0.05):
    steps = int(duration_s / dt)
    t = 0.0

    for _ in range(steps):
        # healthy base excitation
        base_roll = 3.0 * math.sin(2 * math.pi * 0.7 * t)
        base_pitch = 3.0 * math.sin(2 * math.pi * 0.9 * t)
        base_yaw = 6.0 * math.sin(2 * math.pi * 0.4 * t)
        base_thrust = 0.60

        d_roll, d_pitch, d_yaw, d_thrust = compute_disturbance_rates(t, fault_cfg)

        cmd_roll = base_roll + d_roll
        cmd_pitch = base_pitch + d_pitch
        cmd_yaw = base_yaw + d_yaw
        cmd_thrust = max(0.50, min(0.68, base_thrust - d_thrust))

        await drone.offboard.set_attitude_rate(
            AttitudeRate(cmd_roll, cmd_pitch, cmd_yaw, cmd_thrust)
        )

        await asyncio.sleep(dt)
        t += dt


# --------------------------------------------------
# Per-run execution
# --------------------------------------------------
async def run_flight(drone, run_id, fault_cfg):
    logs_before = get_all_logs()

    try:
        # disable battery sim to reduce nuisance interruptions
        await drone.param.set_param_int("SIM_BAT_ENABLE", 0)

        # optional: start logger explicitly as a belt-and-suspenders measure
        px4_shell("logger on")

        ready = await wait_until_ready(drone, timeout_s=25.0)
        if not ready:
            print(f"[WARN] Run {run_id}: vehicle not ready")
            return False, None

        print("[ACTION] Arming")
        await drone.action.arm()
        await asyncio.sleep(1.0)

        print("[ACTION] Takeoff")
        await drone.action.takeoff()
        await asyncio.sleep(4.0)

        await start_offboard(drone)

        print("[PHASE] Healthy hover")
        await send_nominal_hover(drone, duration_s=2.0)

        print("[PHASE] Faulted segment")
        await send_faulted_segment(drone, fault_cfg, duration_s=6.0)

        print("[PHASE] Small post-fault excitation")
        await send_small_nominal_excitation(drone, duration_s=3.0)

        print("[ACTION] Stop offboard")
        try:
            await drone.offboard.stop()
        except Exception:
            pass

        print("[ACTION] Land")
        await drone.action.land()

        landed = await wait_until_landed(drone, timeout_s=25.0)
        if landed:
            print("[STATE] Landed")
        else:
            print("[WARN] Landed state not confirmed in time")

        try:
            await drone.action.disarm()
        except Exception:
            pass

        # optional explicit logger off after disarm
        px4_shell("logger off")

        # give PX4 time to flush the ulg
        # latest_log = await wait_for_new_log(logs_before, timeout_s=20.0)
        logs_after = get_all_logs()
        new_logs = logs_after - logs_before

        if new_logs:
            latest_log = max(new_logs, key=lambda f: (LOG_DIR / f).stat().st_mtime)
            print(f"[LOG] Captured log: {latest_log}")
        else:
            print("[WARN] No new log detected")
            latest_log = None
        return True, latest_log

    except Exception as e:
        print(f"[ERROR] Run {run_id} failed: {e}")

        try:
            await drone.offboard.stop()
        except Exception:
            pass
        try:
            await drone.action.land()
        except Exception:
            pass
        try:
            await drone.action.disarm()
        except Exception:
            pass

        # latest_log = await wait_for_new_log(logs_before, timeout_s=10.0)
        logs_after = get_all_logs()
        new_logs = logs_after - logs_before

        if new_logs:
            latest_log = max(new_logs, key=lambda f: (LOG_DIR / f).stat().st_mtime)
            print(f"[LOG] Captured log: {latest_log}")
        else:
            print("[WARN] No new log detected")
            latest_log = None
        return False, latest_log


# --------------------------------------------------
# Metadata
# --------------------------------------------------
def init_fault_log():
    if not FAULT_LOG_FILE.exists():
        with open(FAULT_LOG_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "run_id",
                "log_path",
                "fault_type",
                "rotor_id",
                "fault_time",
                "fault_severity",
            ])
        return 0

    df = pd.read_csv(FAULT_LOG_FILE)
    if len(df) == 0:
        return 0
    return int(df.iloc[-1]["run_id"]) + 1


def append_fault_metadata(run_id, log_path, fault_cfg):
    """
    One row per faulty rotor.
    Healthy runs get a single row with rotor_id=-1.
    """
    with open(FAULT_LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)

        if not fault_cfg["rotor_ids"]:
            writer.writerow([
                run_id,
                str(log_path) if log_path else "unknown",
                fault_cfg["fault_type"],
                -1,
                -1.0,
                0.0,
            ])
            return

        for rid in fault_cfg["rotor_ids"]:
            writer.writerow([
                run_id,
                str(log_path) if log_path else "unknown",
                fault_cfg["fault_type"],
                rid,
                fault_cfg["fault_times"][rid],
                fault_cfg["severities"][rid],
            ])


# --------------------------------------------------
# Main
# --------------------------------------------------
async def main():
    current_run_no = init_fault_log()

    start_px4()
    drone = await reconnect()

    failure_count = 0

    for i in range(NUM_RUNS):
        print(f"\n================ RUN {current_run_no} ================\n")
        fault_cfg = sample_fault()
        print(f"[PLAN] {fault_cfg}")

        try:
            success, latest_log = await run_flight(drone, current_run_no, fault_cfg)

            if success:
                failure_count = 0
            else:
                failure_count += 1
                print(f"[WARN] Run {current_run_no} unsuccessful")

            if latest_log is not None:
                print(f"[LOG] Detected ULog: {latest_log}")
            else:
                print("[WARN] No ULog detected")

            append_fault_metadata(current_run_no, latest_log, fault_cfg)
            print("[META] Metadata logged")

        except Exception as e:
            failure_count += 1
            print(f"[CRASH] Run {current_run_no} crashed: {e}")

        if failure_count >= MAX_FAILURES:
            print("[RECOVERY] Too many failures, restarting PX4")
            await restart_px4()
            drone = await reconnect()
            failure_count = 0

        if i != 0 and i % 10 == 0:
            print("[MAINT] Periodic PX4 restart")
            await restart_px4()
            drone = await reconnect()

        current_run_no += 1
        await asyncio.sleep(3.0)

    stop_px4()


if __name__ == "__main__":
    asyncio.run(main())
