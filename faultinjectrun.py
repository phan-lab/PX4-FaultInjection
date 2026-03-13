#!/usr/bin/env python3

import asyncio
import logging
import math
import random
import sys
import csv
import pandas as pd

from pathlib import Path

from mavsdk import System
from mavsdk.offboard import OffboardError, Attitude

logging.basicConfig(level=logging.INFO)

NUM_ROTORS = 4


def sample_faults():
    """Sample per-rotor fault start times and severities for a single run."""
    fault_times = [random.uniform(2.0, 8.0) for _ in range(NUM_ROTORS)]
    severities = [0.2 + 0.8 * random.betavariate(2, 5) for _ in range(NUM_ROTORS)]
    ''' print(f"[Faults] fault_times={fault_times}")
    print(f"[Faults] severities={severities}") '''
    return fault_times, severities


def apply_faults(attitude, t, fault_times, severities):
    """
    Apply rotor faults by scaling roll/pitch/yaw components.
    This is a simplified approximation:
    - Roll/pitch/yaw commands are scaled by average active fault severity.
    - This simulates partial rotor failure.
    """
    scale_factors = []
    for i in range(NUM_ROTORS):
        scale = 1.0
        if t >= fault_times[i]:
            # Smooth ramp over 0.3 s
            ramp = min((t - fault_times[i]) / 0.3, 1.0)
            scale = 1.0 + ramp * (severities[i] - 1.0)
        scale_factors.append(scale)

    # Approximate fault effect on attitude commands
    avg_scale = sum(scale_factors) / NUM_ROTORS
    return Attitude(
        roll_deg=attitude.roll_deg * avg_scale,
        pitch_deg=attitude.pitch_deg * avg_scale,
        yaw_deg=attitude.yaw_deg * avg_scale,
        thrust_value=attitude.thrust_value * avg_scale
    )


async def run_flight(drone, fault_times, severities, run_id):
    """Run one flight sequence with fault injection."""
    await drone.param.set_param_int("SIM_BAT_ENABLE", 0)

    print("Waiting for global position estimate...")
    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok:
            break

    print("-- Arming")
    await drone.action.arm()

    print("-- Takeoff")
    await drone.action.takeoff()
    await asyncio.sleep(3)

    # Start offboard
    for _ in range(20):
        await drone.offboard.set_attitude(Attitude(0, 0, 0, 0.6))
        await asyncio.sleep(0.05)
    try:
        await drone.offboard.start()
        async for imu in drone.telemetry.imu():
            px4_offboard_start_us = imu.timestamp_us
            break
        fault_times_us = [
            px4_offboard_start_us + int(fault_time_s * 1e6) for fault_time_s in fault_times
        ]
        print("Offboard started!")
        fault_log_file = "fault_metadata.csv"
        # print(fault_times_us, severities)
        with open(fault_log_file, mode="a", newline="") as f:
            writer = csv.writer(f)
            for rotor_id in range(NUM_ROTORS):
                writer.writerow([
                    run_id,
                    rotor_id,
                    fault_times_us[rotor_id],
                    severities[rotor_id]
                ])
        print("Run metadata logged")
    except OffboardError as e:
        print(f"Offboard start failed: {e}")
        await drone.action.land()
        sys.exit()

    # Stabilizing hover
    print("-- Stablizing hover")
    for _ in range(40):
        att = Attitude(0, 0, 0, 0.55)
        att = apply_faults(att, 0.0, fault_times, severities)
        await drone.offboard.set_attitude(att)
        await asyncio.sleep(0.05)

    # Attitude excitation
    print("-- Attitude excitation")
    T = 6.0
    dt = 0.05
    t = 0.0
    while t < T:
        roll = 6.0 * math.sin(2 * math.pi * 0.6 * t)
        pitch = 6.0 * math.sin(2 * math.pi * 0.4 * t)
        yaw = 20.0 * math.sin(2 * math.pi * 0.3 * t)
        att = Attitude(roll, pitch, yaw, 0.55)
        att = apply_faults(att, t, fault_times, severities)
        await drone.offboard.set_attitude(att)
        await asyncio.sleep(dt)
        t += dt

    # Translational motion
    print("-- Translational motion (tilt-based)")
    for _ in range(120):
        att = Attitude(0, 6.0, 0, 0.6)
        att = apply_faults(att, T, fault_times, severities)
        await drone.offboard.set_attitude(att)
        await asyncio.sleep(0.05)

    # Landing
    print("-- Landing")
    await drone.offboard.stop()
    await drone.action.land()
    await asyncio.sleep(5)

    print("-- Disarming")
    await drone.action.disarm()


async def main():
    drone = System()
    await drone.connect(system_address="udpin://0.0.0.0:14540")

    print("Waiting for drone to connect...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("-- Connected!")
            break

    fault_log_file = "fault_metadata.csv"

    if not(Path(fault_log_file).is_file()):
        current_run_no = 0
        fieldnames = ["run_id", "rotor_id", "fault_time", "fault_severity"]
        with open(fault_log_file, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(fieldnames)
    else:
        fault_metadata = pd.read_csv(fault_log_file)
        current_run_no = int(fault_metadata.iloc[-1]['run_id']) + 1

    NUM_RUNS = 5
    for i in range(NUM_RUNS):
        print(f"=== Run {current_run_no} ===")
        fault_times, severities = sample_faults()
        # fieldnames = ["run_id", "rotor_id", "fault_time", "fault_severity"]
        ''' with open(fault_log_file, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(fieldnames) '''
        ''' with open(fault_log_file, mode="a", newline="") as f:
            writer = csv.writer(f)
            for j in range(NUM_ROTORS):
                writer.writerow([i, j, fault_times[j], severities[j]])
            print(f"Run {i} metadata logged") '''
        await run_flight(drone, fault_times, severities, current_run_no)
        await asyncio.sleep(5)
        current_run_no += 1

    """ fault_times, severities = sample_faults()
    fieldnames = ["run_id", "rotor_id", "fault_time", "fault_severity"]
    fault_log_file = "fault_metadata.csv"
    with open(fault_log_file, mode="a", newline="") as f:
        writer = csv.writer(f)
        for j in range(NUM_ROTORS):
            writer.writerow([i, j, fault_times[j], severities[j]])
        print(f"Run metadata logged")
    await run_flight(drone, fault_times, severities) """


if __name__ == "__main__":
    asyncio.run(main())
