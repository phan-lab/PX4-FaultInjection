from pathlib import Path
import subprocess
import pandas as pd
import numpy as np


log_directory = Path("/home/sshenoy/PX4-Autopilot/build/px4_sitl_default/rootfs/log/2026-03-12")

# convert ulg log files to csv
ulg_file_names = [file.name for file in log_directory.glob("*.ulg")]
for ulg_file_name in ulg_file_names:
	result = subprocess.run(["ulog2csv", "-m", "sensor_combined", ulg_file_name], cwd=log_directory, capture_output=True, text=True)


csv_file_names = [file.name for file in log_directory.glob("*.csv")]
csv_file_names.sort()

metadata = pd.read_csv("fault_metadata.csv")
columns = ["timestamp", "gyro_rad[0]", "gyro_rad[1]", "gyro_rad[2]", "accelerometer_m_s2[0]", "accelerometer_m_s2[1]", "accelerometer_m_s2[2]", "run_id", "rotor0_eff", "rotor1_eff", "rotor2_eff", "rotor3_eff"]

run_no = 0
for csv_file in csv_file_names:
	run_metadata = metadata[metadata['run_id'] == run_no]

	print(f"Run: {run_no}")

	df = pd.read_csv(Path(f"{log_directory}/{csv_file}"))
	df['run_id'] = run_no

	# add fault info for each run in csv file
	rotor0_eff = run_metadata.loc[run_metadata["rotor_id"] == 0, "fault_severity"].iat[0]
	rotor1_eff = run_metadata.loc[run_metadata["rotor_id"] == 1, "fault_severity"].iat[0]
	rotor2_eff = run_metadata.loc[run_metadata["rotor_id"] == 2, "fault_severity"].iat[0]
	rotor3_eff = run_metadata.loc[run_metadata["rotor_id"] == 3, "fault_severity"].iat[0]

	rotor0_eff = round(rotor0_eff, 3)
	rotor1_eff = round(rotor1_eff, 3)
	rotor2_eff = round(rotor2_eff, 3)
	rotor3_eff = round(rotor3_eff, 3)

	rotor0_fault_time = run_metadata.loc[run_metadata["rotor_id"] == 0, "fault_time"].iat[0]
	rotor1_fault_time = run_metadata.loc[run_metadata["rotor_id"] == 1, "fault_time"].iat[0]
	rotor2_fault_time = run_metadata.loc[run_metadata["rotor_id"] == 2, "fault_time"].iat[0]
	rotor3_fault_time = run_metadata.loc[run_metadata["rotor_id"] == 3, "fault_time"].iat[0]

	df['rotor0_eff'] = np.where(df["timestamp"] >= rotor0_fault_time, rotor0_eff, 1.0)
	df['rotor1_eff'] = np.where(df["timestamp"] >= rotor1_fault_time, rotor1_eff, 1.0)
	df['rotor2_eff'] = np.where(df["timestamp"] >= rotor2_fault_time, rotor2_eff, 1.0)
	df['rotor3_eff'] = np.where(df["timestamp"] >= rotor3_fault_time, rotor3_eff, 1.0)

	run_no += 1

	# keep required columns for the model
	df = df[columns]

	# rename columns as expected by the model
	df.rename(columns={'timestamp': 'time', 'gyro_rad[0]': 'gyroscope_x', 'gyro_rad[1]': 'gyroscope_y', 'gyro_rad[2]': 'gyroscope_z', 'accelerometer_m_s2[0]': 'accelerometer_x', 'accelerometer_m_s2[1]': 'accelerometer_y', 'accelerometer_m_s2[2]': 'accelerometer_z'}, inplace=True)

	# overwrite original csv file
	df.to_csv(Path(f"{log_directory}/{csv_file}"), index=False)

# merge all csv files into one single file
combined_csv = pd.concat([pd.read_csv(Path(f"{log_directory}/{csv_file}")) for csv_file in csv_file_names])
combined_csv.to_csv(Path(f"{log_directory}/imu_readings.csv"), index=False)
