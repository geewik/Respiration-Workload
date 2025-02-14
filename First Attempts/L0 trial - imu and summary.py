import pandas as pd
import numpy as np

# Step 1: Load the data
imu_file = "D:\\aria-seated\\S12\\L0_imu_df.csv"
summary_file = "D:\\aria-seated\\S12\\L0_summary_df.csv"
output_file = "D:\\aria-seated\\cnn_input_data_with_labels.csv"

imu_data = pd.read_csv(imu_file)
summary_data = pd.read_csv(summary_file)

# Step 2: Drop unnecessary columns in IMU data (e.g., timestamps if they exist)
imu_data = imu_data.drop(columns=['timestamp'], errors='ignore')

# Step 3: Extract integer seconds and decimal parts for IMU
imu_data['whole_seconds'] = imu_data['sec'].astype(int)
imu_data['decimal'] = imu_data['sec'] - imu_data['whole_seconds']

# Step 4: Extract integer seconds for Summary Data
summary_data['whole_seconds'] = summary_data['sec'].astype(int)

# Step 5: Filter IMU data based on the range of summary seconds
start_second = summary_data['whole_seconds'].min()
end_second = summary_data['whole_seconds'].max()

imu_data_filtered = imu_data[
    (imu_data['whole_seconds'] >= start_second) & (imu_data['whole_seconds'] <= end_second)
]

# Step 6: Group IMU data to match summary timestamps
imu_chunks = []
for _, summary_row in summary_data.iterrows():
    summary_whole_second = summary_row['whole_seconds']

    # Get IMU rows matching the whole second
    imu_chunk = imu_data_filtered[imu_data_filtered['whole_seconds'] == summary_whole_second]

    # Drop unnecessary columns for chunk processing
    imu_chunk = imu_chunk.drop(columns=['whole_seconds', 'decimal', 'sec'], errors='ignore')

    # Add the chunk to the list as is
    imu_chunks.append(imu_chunk.to_numpy())  # Retain exact values

# Step 7: Prepare for CNN1D
fixed_size = 100  # Example fixed size for CNN1D input
processed_chunks = []

for chunk in imu_chunks:
    # Truncate or pad to fixed size
    if len(chunk) > fixed_size:
        processed_chunk = chunk[:fixed_size]  # Truncate
    else:
        padding = np.zeros((fixed_size - len(chunk), chunk.shape[1]))
        processed_chunk = np.vstack((chunk, padding))  # Pad
    processed_chunks.append(processed_chunk)

# Convert to NumPy array for training
cnn_input_data = np.array(processed_chunks)

# Step 8: Flatten and label data with BR from summary
total_chunks = len(cnn_input_data)
br_values = summary_data['BR'].values[:total_chunks]

cnn_input_data_flat = cnn_input_data.reshape(total_chunks, -1)
labeled_data = np.column_stack((cnn_input_data_flat, br_values))

# Step 9: Create column headers
imu_headers = imu_data.columns.drop(['whole_seconds', 'decimal', 'sec'], errors='ignore')
headers = [f"{col}_t{i+1}" for i in range(fixed_size) for col in imu_headers]
headers.append("BR")  # Add the label column

# Step 10: Save the labeled data to CSV
pd.DataFrame(labeled_data, columns=headers).to_csv(output_file, index=False)

print(f"Labeled data saved successfully to {output_file}!")

