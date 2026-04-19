from pathlib import Path
import subprocess
import shutil
import pandas as pd
import numpy as np


# ============================================================
# Configuration
# ============================================================
LOG_DIR = Path("/home/sshenoy/PX4-Autopilot/build/px4_sitl_default/rootfs/log/2026-04-18")
METADATA_PATH = Path("fault_metadata.csv")
OUTPUT_DIR = Path("processed_imu_logs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TOPICS = [
    "sensor_combined",
    "vehicle_angular_velocity",
    "actuator_motors",
]

FINAL_COLUMNS = [
    "time",
    "gyroscope_x",
    "gyroscope_y",
    "gyroscope_z",
    "accelerometer_x",
    "accelerometer_y",
    "accelerometer_z",
    "angular_velocity_x",
    "angular_velocity_y",
    "angular_velocity_z",
    "angular_acceleration_x",
    "angular_acceleration_y",
    "angular_acceleration_z",
    "motor0",
    "motor1",
    "motor2",
    "motor3",
    "run_id",
    "rotor0_eff",
    "rotor1_eff",
    "rotor2_eff",
    "rotor3_eff",
]


# ============================================================
# Simple helpers
# ============================================================
def get_log_filename(run_metadata: pd.DataFrame):
    """
    Metadata has a log_path column containing either:
      - 'unknown'
      - or a filename like '14_28_27.ulg'
    """
    if "log_path" not in run_metadata.columns:
        return None

    value = run_metadata.iloc[0]["log_path"]
    if pd.isna(value):
        return None

    value = str(value).strip()
    if value in ("", "unknown", "None"):
        return None

    return value


def get_ulg_path(run_metadata: pd.DataFrame):
    filename = get_log_filename(run_metadata)
    if filename is None:
        return None

    ulg_path = LOG_DIR / filename
    if not ulg_path.exists():
        print(f"[WARN] Missing ULog file: {ulg_path}")
        return None

    return ulg_path


def run_ulog2csv(ulg_path: Path, topic: str):
    if shutil.which("ulog2csv") is None:
        raise RuntimeError("ulog2csv not found in PATH")

    result = subprocess.run(
        ["ulog2csv", "-m", topic, ulg_path.name],
        cwd=ulg_path.parent,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"[WARN] ulog2csv failed for topic={topic}, file={ulg_path.name}")
        stderr = result.stderr.strip()
        if stderr:
            print(stderr)
        return None

    matches = sorted(ulg_path.parent.glob(f"{ulg_path.stem}_{topic}_*.csv"))
    if not matches:
        print(f"[WARN] No CSV created for topic={topic}, file={ulg_path.name}")
        return None

    return matches[0]


def get_effective_fault_arrays(run_metadata: pd.DataFrame):
    rotor_fault_times_us = [10**15] * 4
    rotor_effectiveness = [1.0] * 4

    for rotor_id in range(4):
        rows = run_metadata[run_metadata["rotor_id"] == rotor_id]
        if len(rows) == 0:
            continue

        row = rows.iloc[0]

        fault_time = row.get("fault_time", -1.0)
        fault_severity = row.get("fault_severity", 1.0)

        if pd.notna(fault_time) and float(fault_time) >= 0.0:
            rotor_fault_times_us[rotor_id] = int(float(fault_time) * 1e6)

        if pd.notna(fault_severity) and float(fault_severity) >= 0.0:
            rotor_effectiveness[rotor_id] = round(1.0 - float(fault_severity), 3)

    return rotor_fault_times_us, rotor_effectiveness


def normalize_time_column(df: pd.DataFrame, start_time_us: int):
    df = df.copy()
    df["timestamp"] = df["timestamp"].astype(np.int64) - int(start_time_us)
    return df


def merge_topics_asof(base_df: pd.DataFrame, other_df: pd.DataFrame, tolerance_us: int = 20000):
    return pd.merge_asof(
        base_df.sort_values("time"),
        other_df.sort_values("time"),
        on="time",
        direction="nearest",
        tolerance=tolerance_us,
    )


# ============================================================
# Topic loaders
# ============================================================
def load_sensor_combined(csv_path: Path, start_time_us: int):
    df = pd.read_csv(csv_path)

    required = [
        "timestamp",
        "gyro_rad[0]",
        "gyro_rad[1]",
        "gyro_rad[2]",
        "accelerometer_m_s2[0]",
        "accelerometer_m_s2[1]",
        "accelerometer_m_s2[2]",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} missing sensor_combined columns: {missing}")

    df = normalize_time_column(df, start_time_us)
    df = df[required].copy()

    df = df.rename(columns={
        "timestamp": "time",
        "gyro_rad[0]": "gyroscope_x",
        "gyro_rad[1]": "gyroscope_y",
        "gyro_rad[2]": "gyroscope_z",
        "accelerometer_m_s2[0]": "accelerometer_x",
        "accelerometer_m_s2[1]": "accelerometer_y",
        "accelerometer_m_s2[2]": "accelerometer_z",
    })

    return df.sort_values("time").reset_index(drop=True)


def load_vehicle_angular_velocity(csv_path: Path, start_time_us: int):
    df = pd.read_csv(csv_path)

    if "timestamp" not in df.columns:
        raise ValueError(f"{csv_path} missing timestamp")

    df = normalize_time_column(df, start_time_us)

    rename_map = {
        "xyz[0]": "angular_velocity_x",
        "xyz[1]": "angular_velocity_y",
        "xyz[2]": "angular_velocity_z",
        "xyz_derivative[0]": "angular_acceleration_x",
        "xyz_derivative[1]": "angular_acceleration_y",
        "xyz_derivative[2]": "angular_acceleration_z",
    }
    existing = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df.rename(columns=existing)

    keep = ["timestamp"]
    for col in [
        "angular_velocity_x",
        "angular_velocity_y",
        "angular_velocity_z",
        "angular_acceleration_x",
        "angular_acceleration_y",
        "angular_acceleration_z",
    ]:
        if col in df.columns:
            keep.append(col)

    if len(keep) == 1:
        return None

    df = df[keep].copy()
    df = df.rename(columns={"timestamp": "time"})
    return df.sort_values("time").reset_index(drop=True)


def load_actuator_motors(csv_path: Path, start_time_us: int):
    df = pd.read_csv(csv_path)

    if "timestamp" not in df.columns:
        raise ValueError(f"{csv_path} missing timestamp")

    df = normalize_time_column(df, start_time_us)

    rename_map = {
        "control[0]": "motor0",
        "control[1]": "motor1",
        "control[2]": "motor2",
        "control[3]": "motor3",
        "output[0]": "motor0",
        "output[1]": "motor1",
        "output[2]": "motor2",
        "output[3]": "motor3",
    }
    existing = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df.rename(columns=existing)

    keep = ["timestamp"]
    for col in ["motor0", "motor1", "motor2", "motor3"]:
        if col in df.columns:
            keep.append(col)

    if len(keep) == 1:
        return None

    df = df[keep].copy()
    df = df.rename(columns={"timestamp": "time"})
    return df.sort_values("time").reset_index(drop=True)


# ============================================================
# Main per-run processing
# ============================================================
def process_one_run(run_id: int, run_metadata: pd.DataFrame):
    ulg_path = get_ulg_path(run_metadata)

    if ulg_path is None:
        print(f"[SKIP] Run {run_id}: no valid ULog file")
        return None

    print(f"[INFO] Processing run {run_id}: {ulg_path}")

    topic_csvs = {}

    for topic in TOPICS:
        csv_path = run_ulog2csv(ulg_path, topic)
        if csv_path is not None:
            topic_csvs[topic] = csv_path
        else:
            print(f"[WARN] Run {run_id}: missing topic {topic}")

    if "sensor_combined" not in topic_csvs:
        print(f"[SKIP] Run {run_id}: sensor_combined missing")
        return None

    # earliest timestamp among available topic csvs
    start_times = []
    for csv_path in topic_csvs.values():
        df_time = pd.read_csv(csv_path, usecols=["timestamp"])
        if len(df_time) > 0:
            start_times.append(int(df_time["timestamp"].iloc[0]))

    if not start_times:
        print(f"[SKIP] Run {run_id}: no timestamps found")
        return None

    start_time_us = min(start_times)

    merged_df = load_sensor_combined(topic_csvs["sensor_combined"], start_time_us)

    if "vehicle_angular_velocity" in topic_csvs:
        ang_df = load_vehicle_angular_velocity(topic_csvs["vehicle_angular_velocity"], start_time_us)
        if ang_df is not None:
            merged_df = merge_topics_asof(merged_df, ang_df, tolerance_us=20000)

    if "actuator_motors" in topic_csvs:
        motor_df = load_actuator_motors(topic_csvs["actuator_motors"], start_time_us)
        if motor_df is not None:
            merged_df = merge_topics_asof(merged_df, motor_df, tolerance_us=20000)

    merged_df["run_id"] = int(run_id)

    rotor_fault_times_us, rotor_effectiveness = get_effective_fault_arrays(run_metadata)

    for rotor_id in range(4):
        col = f"rotor{rotor_id}_eff"
        merged_df[col] = np.where(
            merged_df["time"] >= rotor_fault_times_us[rotor_id],
            rotor_effectiveness[rotor_id],
            1.0,
        )

    # fill missing optional columns
    for col in FINAL_COLUMNS:
        if col not in merged_df.columns:
            if col == "run_id":
                merged_df[col] = int(run_id)
            elif col.startswith("rotor") and col.endswith("_eff"):
                merged_df[col] = 1.0
            else:
                merged_df[col] = np.nan

    merged_df = merged_df[FINAL_COLUMNS].copy()

    for col in [
        "angular_velocity_x",
        "angular_velocity_y",
        "angular_velocity_z",
        "angular_acceleration_x",
        "angular_acceleration_y",
        "angular_acceleration_z",
        "motor0",
        "motor1",
        "motor2",
        "motor3",
    ]:
        if col in merged_df.columns:
            merged_df[col] = merged_df[col].interpolate(limit_direction="both")

    output_csv = OUTPUT_DIR / f"run_{run_id:05d}_merged.csv"
    merged_df.to_csv(output_csv, index=False)

    return output_csv


# ============================================================
# Main
# ============================================================
def main():
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Missing metadata file: {METADATA_PATH}")

    if not LOG_DIR.exists():
        raise FileNotFoundError(f"Missing hard-coded log directory: {LOG_DIR}")

    metadata = pd.read_csv(METADATA_PATH)

    processed_files = []

    for run_id, run_metadata in metadata.groupby("run_id", sort=True):
        try:
            output_csv = process_one_run(run_id, run_metadata)
            if output_csv is not None:
                processed_files.append(output_csv)
        except Exception as e:
            print(f"[ERROR] Run {run_id} failed: {e}")

    if not processed_files:
        print("[WARN] No runs were processed")
        return

    combined_df = pd.concat(
        [pd.read_csv(p) for p in processed_files],
        ignore_index=True
    )

    combined_output = OUTPUT_DIR / "imu_readings_merged.csv"
    combined_df.to_csv(combined_output, index=False)

    print(f"[DONE] Wrote merged dataset: {combined_output}")


if __name__ == "__main__":
    main()
