import asyncio
import csv
import logging
import math
import random
import subprocess
import time
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from mavsdk import System
from mavsdk.offboard import OffboardError, AttitudeRate

logging.basicConfig(level=logging.INFO)

NUM_ROTORS = 4
NUM_RUNS = 5
MAX_FAILURES = 3
WINDOW_LEN = 100

inference_times = []

PX4_CMD = ["make", "px4_sitl", "gz_x500", "HEADLESS=1"]

FAULT_LOG_FILE = Path("fault_metadata_with_recovery.csv")

MODEL_PATH = Path("cnn-lstm-sim.pt")

IMU_CHANNELS = [
    "accelerometer_x", "accelerometer_y", "accelerometer_z",
    "gyroscope_x", "gyroscope_y", "gyroscope_z",
]

LOW_SEVERITY = 0.30
HIGH_SEVERITY = 0.65
FAULT_STREAK_REQUIRED = 3

PX4_PROCESS = None

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print("Using model device:", DEVICE)

feature_buffer = deque(maxlen=500)

latest_prediction = {
    "fault_confirmed": False,
    "rotor_id": -1,
    "severity": 0.0,
    "eff": np.ones(4, dtype=np.float32),
    "mode": "NOMINAL",
}

prediction_lock = asyncio.Lock()


class RotorRegressor(nn.Module):
    def __init__(self, n_channels=6, hidden_size=128, lstm_layers=2, dropout=0.3):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=128,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=False
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 4)
        )


    def forward(self, X, hc=None):
        X = X.permute(0, 2, 1)
        cnn_out = self.conv(X)
        cnn_out = cnn_out.transpose(1, 2)

        if hc is None:
            lstm_out, _ = self.lstm(cnn_out)
        else:
            lstm_out, hc = self.lstm(cnn_out, hc)

        last_hidden = lstm_out[:, -1, :]
        out = self.fc(last_hidden)
        return out


def load_fault_model(model_path: Path):
    model = RotorRegressor().to(DEVICE)
    model.load_state_dict(torch.load("cnn-lstm-sim.pt", map_location=DEVICE))
    model.eval()

    print("[MODEL] Loaded:", model_path)
    return model


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
        print(f"[WARN] Failed PX4 shell command: {e}")


def sample_fault():
    # scenario = random.choices(
    #     ["healthy", "single_partial", "double_partial"],
    #     weights=[0.25, 0.55, 0.20],
    # )[0]

    scenario = random.choices(
        ["single_partial", "double_partial"],
        weights=[0.55, 0.45],
    )[0]

    rotor_ids = []
    fault_times = [100.0] * NUM_ROTORS
    severities = [0.0] * NUM_ROTORS

    # if scenario == "healthy":
    #     return {
    #         "fault_type": scenario,
    #         "fault_times": fault_times,
    #         "severities": severities,
    #         "rotor_ids": rotor_ids,
    #     }

    n_faults = 1 if scenario == "single_partial" else 2
    rotor_ids = random.sample(range(NUM_ROTORS), n_faults)

    for rid in rotor_ids:
        fault_times[rid] = random.uniform(2.0, 5.0)
        # severities[rid] = random.uniform(0.2, 1.0)
        severities[rid] = random.uniform(0.4, 1.0)

    return {
        "fault_type": scenario,
        "fault_times": fault_times,
        "severities": severities,
        "rotor_ids": rotor_ids,
    }


def rotor_to_disturbance_signs(rotor_id: int):
    mapping = {
        0: (+1.0, -1.0, +1.0),
        1: (-1.0, -1.0, -1.0),
        2: (-1.0, +1.0, +1.0),
        3: (+1.0, +1.0, -1.0),
    }
    return mapping[rotor_id]


def compute_disturbance_rates(t, fault_cfg):
    roll_rate = 0.0
    pitch_rate = 0.0
    yaw_rate = 0.0
    thrust_drop = 0.0

    for rid in range(NUM_ROTORS):
        t_fault = fault_cfg["fault_times"][rid]
        sev = fault_cfg["severities"][rid]

        if t < t_fault or sev <= 0.0:
            continue

        ramp = min((t - t_fault) / 0.30, 1.0)
        s = sev * ramp

        roll_sign, pitch_sign, yaw_sign = rotor_to_disturbance_signs(rid)

        roll_rate += roll_sign * (18.0 * s)
        pitch_rate += pitch_sign * (18.0 * s)
        yaw_rate += yaw_sign * (35.0 * s)

        roll_rate += roll_sign * (5.0 * s * math.sin(2 * math.pi * 2.2 * t))
        pitch_rate += pitch_sign * (5.0 * s * math.sin(2 * math.pi * 1.8 * t + 0.4))
        yaw_rate += yaw_sign * (8.0 * s * math.sin(2 * math.pi * 1.2 * t + 0.8))

        thrust_drop += 0.05 * s

    return roll_rate, pitch_rate, yaw_rate, thrust_drop


def choose_recovery_mode(severity):
    if severity < LOW_SEVERITY:
        return "NOMINAL"
    elif severity < HIGH_SEVERITY:
        return "DEGRADED"
    else:
        return "UNRECOVERABLE"


def compensation_from_prediction(rotor_id, severity):
    mapping = {
        0: (+1.0, -1.0, +1.0),
        1: (-1.0, -1.0, -1.0),
        2: (-1.0, +1.0, +1.0),
        3: (+1.0, +1.0, -1.0),
    }

    if rotor_id not in mapping:
        return 0.0, 0.0, 0.0, 0.0

    r_sign, p_sign, y_sign = mapping[rotor_id]

    comp_roll = -r_sign * 5.0 * severity
    comp_pitch = -p_sign * 5.0 * severity
    comp_yaw = -y_sign * 8.0 * severity
    comp_thrust = 0.03 * severity

    return comp_roll, comp_pitch, comp_yaw, comp_thrust


async def imu_collector(drone):
    try:
        async for imu in drone.telemetry.imu():
            # print(imu.acceleration_frd)
            # print(imu.angular_velocity_frd)
            sample = {
                "accelerometer_x": imu.acceleration_frd.forward_m_s2,
                "accelerometer_y": imu.acceleration_frd.right_m_s2,
                "accelerometer_z": imu.acceleration_frd.down_m_s2,
                "gyroscope_x": imu.angular_velocity_frd.forward_rad_s,
                "gyroscope_y": imu.angular_velocity_frd.right_rad_s,
                "gyroscope_z": imu.angular_velocity_frd.down_rad_s
            }

            feature_buffer.append(sample)
    except Exception as e:
        print(f"[IMU ERROR]: {e}")


def predict_effectiveness(model):
    if len(feature_buffer) < WINDOW_LEN:
        return None

    recent = list(feature_buffer)[-WINDOW_LEN:]

    imu_window = np.array(
        [[sample[ch] for ch in IMU_CHANNELS] for sample in recent],
        dtype=np.float32,
    )

    # imu_window = np.array(feature_buffer, dtype=np.float32)

    # x = (x - mean[None, :]) / std[None, :]
    x = torch.tensor(imu_window, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    start_time = time.perf_counter()

    with torch.no_grad():
        pred_eff = model(x)

    end_time = time.perf_counter()

    inference_time = end_time - start_time
    inference_times.append(inference_time)

    eff = pred_eff.squeeze(0).detach().cpu().numpy()

    rotor_id = int(np.argmin(eff))
    severity = float(1.0 - eff[rotor_id])

    return {
        "eff": eff,
        "rotor_id": rotor_id,
        "severity": severity,
    }


async def inference_loop(model):
    try:
        fault_streak = 0
        smoothed_severity = 0.0
        latched_rotor = -1
        latched_mode = "NOMINAL"

        while True:
            pred = predict_effectiveness(model)

            if pred is None:
                await asyncio.sleep(0.05)
                continue

            severity = pred["severity"]
            rotor_id = pred["rotor_id"]

            smoothed_severity = 0.7 * smoothed_severity + 0.3 * severity

            if smoothed_severity >= LOW_SEVERITY:
                fault_streak += 1
            else:
                fault_streak = max(0, fault_streak - 1)

            fault_confirmed = fault_streak >= FAULT_STREAK_REQUIRED

            if fault_confirmed and latched_rotor == -1:
                latched_rotor = rotor_id

            if fault_confirmed:
                proposed_mode = choose_recovery_mode(smoothed_severity)

                if latched_mode != "UNRECOVERABLE":
                    latched_mode = proposed_mode

                if proposed_mode == "UNRECOVERABLE":
                    latched_mode = "UNRECOVERABLE"

            async with prediction_lock:
                latest_prediction["fault_confirmed"] = fault_confirmed
                latest_prediction["rotor_id"] = latched_rotor if latched_rotor != -1 else rotor_id
                latest_prediction["severity"] = smoothed_severity
                latest_prediction["eff"] = pred["eff"]
                latest_prediction["mode"] = latched_mode

            await asyncio.sleep(0.05)
    except Exception as e:
        print(f"[INFERENCE ERROR: {e}]")


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


async def send_small_nominal_excitation(drone, duration_s=3.0, dt=0.05):
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


async def send_faulted_segment_with_recovery(drone, fault_cfg, duration_s=8.0, dt=0.05):
    steps = int(duration_s / dt)
    t = 0.0

    for _ in range(steps):
        base_roll = 3.0 * math.sin(2 * math.pi * 0.7 * t)
        base_pitch = 3.0 * math.sin(2 * math.pi * 0.9 * t)
        base_yaw = 6.0 * math.sin(2 * math.pi * 0.4 * t)
        base_thrust = 0.60

        d_roll, d_pitch, d_yaw, d_thrust = compute_disturbance_rates(t, fault_cfg)

        async with prediction_lock:
            pred_state = latest_prediction.copy()

        mode = pred_state["mode"]
        pred_rotor = pred_state["rotor_id"]
        pred_severity = pred_state["severity"]

        if mode == "NOMINAL":
            cmd_roll = base_roll + d_roll
            cmd_pitch = base_pitch + d_pitch
            cmd_yaw = base_yaw + d_yaw
            cmd_thrust = base_thrust - d_thrust

        elif mode == "DEGRADED":
            cr, cp, cy, ct = compensation_from_prediction(pred_rotor, pred_severity)

            cmd_roll = 0.5 * base_roll + d_roll + cr
            cmd_pitch = 0.5 * base_pitch + d_pitch + cp
            cmd_yaw = 0.4 * base_yaw + d_yaw + cy
            cmd_thrust = base_thrust - d_thrust + ct

            print(
                f"[COMPENSATE] rotor={pred_rotor}, "
                f"severity={pred_severity:.2f}, "
                f"eff={pred_state['eff']}"
            )

        elif mode == "UNRECOVERABLE":
            print(
                f"[LAND] Severe fault detected | "
                f"rotor={pred_rotor}, severity={pred_severity:.2f}, "
                f"eff={pred_state['eff']}"
            )

            await drone.offboard.set_attitude_rate(
                AttitudeRate(0.0, 0.0, 0.0, 0.58)
            )
            await asyncio.sleep(1.0)

            try:
                await drone.offboard.stop()
            except Exception:
                pass

            await drone.action.land()
            return "Landed"

        else:
            cmd_roll = base_roll + d_roll
            cmd_pitch = base_pitch + d_pitch
            cmd_yaw = base_yaw + d_yaw
            cmd_thrust = base_thrust - d_thrust

        cmd_thrust = max(0.50, min(0.70, cmd_thrust))

        await drone.offboard.set_attitude_rate(
            AttitudeRate(cmd_roll, cmd_pitch, cmd_yaw, cmd_thrust)
        )

        await asyncio.sleep(dt)
        t += dt

    return "Completed"


def init_fault_log():
    if not FAULT_LOG_FILE.exists():
        with open(FAULT_LOG_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "run_id",
                "fault_type",
                "rotor_id",
                "fault_time",
                "fault_severity",
                "pred_rotor_id",
                "pred_severity",
                "recovery_mode"
            ])
        return 0

    df = pd.read_csv(FAULT_LOG_FILE)
    if len(df) == 0:
        return 0
    return int(df.iloc[-1]["run_id"]) + 1


async def get_prediction_snapshot():
    async with prediction_lock:
        return latest_prediction.copy()


def append_fault_metadata(run_id, fault_cfg, pred_snapshot):
    with open(FAULT_LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)

        pred_rotor = pred_snapshot["rotor_id"]
        pred_sev = pred_snapshot["severity"]
        mode = pred_snapshot["mode"]

        if not fault_cfg["rotor_ids"]:
            writer.writerow([
                run_id,
                fault_cfg["fault_type"],
                -1,
                -1.0,
                0.0,
                pred_rotor,
                pred_sev,
                mode
            ])
            return

        for rid in fault_cfg["rotor_ids"]:
            writer.writerow([
                run_id,
                fault_cfg["fault_type"],
                rid,
                fault_cfg["fault_times"][rid],
                fault_cfg["severities"][rid],
                pred_rotor,
                pred_sev,
                mode
            ])


def reset_prediction_state():
    feature_buffer.clear()
    latest_prediction["fault_confirmed"] = False
    latest_prediction["rotor_id"] = -1
    latest_prediction["severity"] = 0.0
    latest_prediction["eff"] = np.ones(4, dtype=np.float32)
    latest_prediction["mode"] = "NOMINAL"


async def run_flight(drone, run_id, fault_cfg, model):
    reset_prediction_state()

    imu_task = None
    inference_task = None

    try:
        await drone.param.set_param_int("SIM_BAT_ENABLE", 0)
        # px4_shell("logger on")

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

        imu_task = asyncio.create_task(imu_collector(drone))
        inference_task = asyncio.create_task(
            inference_loop(model)
        )

        print("[PHASE] Healthy hover")
        await send_nominal_hover(drone, duration_s=2.0)

        print("[PHASE] Faulted segment with model-based recovery")
        result = await send_faulted_segment_with_recovery(
            drone,
            fault_cfg,
            duration_s=8.0,
            dt=0.05,
        )

        if result != "Landed":
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
            print("[WARN] Landed state not confirmed")

        try:
            await drone.action.disarm()
        except Exception:
            pass

        # px4_shell("logger off")
        await asyncio.sleep(3.0)

        return True

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

        return False

    finally:
        if imu_task is not None:
            imu_task.cancel()
        if inference_task is not None:
            inference_task.cancel()


async def main():
    model = load_fault_model(MODEL_PATH)

    current_run_no = init_fault_log()

    start_px4()
    drone = await reconnect()

    failure_count = 0

    for i in range(NUM_RUNS):
        print(f"\n================ RUN {current_run_no} ================\n")

        inference_times.clear()

        fault_cfg = sample_fault()
        print(f"[PLAN] {fault_cfg}")

        try:
            success = await run_flight(
                drone,
                current_run_no,
                fault_cfg,
                model
            )

            if success:
                failure_count = 0
            else:
                failure_count += 1
                print(f"[WARN] Run {current_run_no} unsuccessful")

            pred_snapshot = await get_prediction_snapshot()

            append_fault_metadata(
                current_run_no,
                fault_cfg,
                pred_snapshot,
            )

            # print("[META] Metadata logged")
            print(
                f"[PRED] mode={pred_snapshot['mode']} | "
                f"faulty_rotor_index={pred_snapshot['rotor_id']} | "
                f"severity={pred_snapshot['severity']:.2f} | "
                f"eff={pred_snapshot['eff']}"
            )

            if len(inference_times) > 0:
                print(f"[TIMING] Last Inference Time: {inference_times[-1] * 1000:.2f} ms | Mean: {np.mean(inference_times) * 1000:.2f} ms")

        except Exception as e:
            failure_count += 1
            print(f"[CRASH] Run {current_run_no} crashed: {e}")

        if failure_count >= MAX_FAILURES:
            # print("[RECOVERY] Too many failures, restarting PX4")
            await restart_px4()
            drone = await reconnect()
            failure_count = 0

        if i != 0 and i % 10 == 0:
            # print("[MAINT] Periodic PX4 restart")
            await restart_px4()
            drone = await reconnect()

        current_run_no += 1
        await asyncio.sleep(3.0)

    stop_px4()


if __name__ == "__main__":
    asyncio.run(main())
