#!/usr/bin/env python3

import asyncio
import logging
import math
import sys
import random
import time

from mavsdk import System
from mavsdk.offboard import (
    OffboardError,
    Attitude,
    VelocityBodyYawspeed
)

# Enable INFO level logging by default so that INFO messages are shown
logging.basicConfig(level=logging.INFO)


async def run():
    drone = System()
    await drone.connect(system_address="udpin://0.0.0.0:14540")

    status_text_task = asyncio.ensure_future(print_status_text(drone))

    print("Waiting for drone to connect...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("-- Connected to drone!")
            break

    await drone.param.set_param_int("SIM_BAT_ENABLE", 0)
    # await drone.param.set_param_int("COM_ARM_BAT_MIN", 0.0)

    print("Waiting for drone to have a global position estimate...")
    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok:
            print("-- Global position estimate OK")
            break

    print("-- Arming")
    await drone.action.arm()

    print("-- Taking off")
    await drone.action.takeoff()
    await asyncio.sleep(3)

    for _ in range(20):
        await drone.offboard.set_attitude(
            Attitude(roll_deg=0, pitch_deg=0, yaw_deg=0, thrust_value=0.6)
        )
        asyncio.sleep(0.05)

    try:
        await drone.offboard.start()
        print("Offboard started!")
    except OffboardError as e:
        print(f"Offboard start failed: {e}")
        await drone.action.land()
        sys.exit()

    print("-- Stablizing hover")
    for _ in range(40):
        await drone.offboard.set_attitude(
            Attitude(0, 0, 0, 0.55)
        )
        await asyncio.sleep(0.05)


    print("-- Attitude excitation")
    T = 6.0
    dt = 0.05
    t = 0.0

    while t < T:
        roll = 6.0 * math.sin(2 * math.pi * 0.6 * t)
        pitch = 6.0 * math.sin(2 * math.pi * 0.4 * t)
        yaw = 20.0 * math.sin(2 * math.pi * 0.3 * t)

        await drone.offboard.set_attitude(
            Attitude(roll, pitch, yaw, 0.55)
        )
        await asyncio.sleep(dt)
        t += dt

    print("-- Translational motion")
    for _ in range(100):
        await drone.offboard.set_velocity_body(
            VelocityBodyYawspeed(
                forward_m_s=1.0,
                right_m_s=0.0,
                down_m_s=0.0,
                yawspeed_deg_s=0.0
            )
        )
        await asyncio.sleep(dt)

    print("-- Yaw sweep")
    for _ in range(80):
        await drone.offboard.set_velocity_body(
            VelocityBodyYawspeed(
                0.0, 0.0, 0.0, 40.0
            )
        )
        await asyncio.sleep(dt)

    for _ in range(80):
        await drone.offboard.set_velocity_body(
            VelocityBodyYawspeed(
                0.0, 0.0, 0.0, -40.0
            )
        )
        await asyncio.sleep(dt)

    print("-- Landing")
    await drone.offboard.stop()
    await drone.action.land()

    await asyncio.sleep(5)

    print("-- Disarming")
    await drone.action.disarm()

    status_text_task.cancel()


async def print_status_text(drone):
    try:
        async for status_text in drone.telemetry.status_text():
            print(f"Status: {status_text.type}: {status_text.text}")
    except asyncio.CancelledError:
        return


async def main():
    for i in range(5):
        print(f"Run {i}")
        await run()
        await asyncio.sleep(5)


if __name__ == "__main__":
    # Run the asyncio loop
    asyncio.run(main())
