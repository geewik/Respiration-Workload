import os
import pandas as pd
import numpy as np

# Define constants
base_directory = "D:\\aria-seated"
output_directory = os.path.join(base_directory, "output_files")  # Central folder for output files
fixed_size = 120  # Fixed size for CNN1D input

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

# Function to generate repeating column headers
def generate_column_headers(base_columns, total_columns):
    headers = []
    num_repeats = total_columns // len(base_columns)
    for i in range(num_repeats):
        for col in base_columns:
            headers.append(f"{col}_t{i+1}")
    return headers

# Function to process individual IMU and Summary file pair
def process_files(imu_file, summary_file, folder_name, prefix):
    try:
        # Load the data
        imu_data = pd.read_csv(imu_file)
        summary_data = pd.read_csv(summary_file)

        # Drop unnecessary columns in IMU data (e.g., timestamps if they exist)
        imu_data = imu_data.drop(columns=['timestamp'], errors='ignore')

        # Extract integer seconds and decimal parts for IMU
        imu_data['whole_seconds'] = imu_data['sec'].astype(int)
        imu_data['decimal'] = imu_data['sec'] - imu_data['whole_seconds']

        # Extract integer seconds for Summary Data
        summary_data['whole_seconds'] = summary_data['sec'].astype(int)

        # Filter IMU data based on the range of summary seconds
        start_second = summary_data['whole_seconds'].min()
        end_second = summary_data['whole_seconds'].max()

        imu_data_filtered = imu_data[
            (imu_data['whole_seconds'] >= start_second) & (imu_data['whole_seconds'] <= end_second)
        ]

        # Group IMU data to match summary timestamps
        imu_chunks = []
        for _, summary_row in summary_data.iterrows():
            summary_whole_second = summary_row['whole_seconds']

            # Get IMU rows matching the whole second
            imu_chunk = imu_data_filtered[imu_data_filtered['whole_seconds'] == summary_whole_second]

            # Drop unnecessary columns for chunk processing
            imu_chunk = imu_chunk.drop(columns=['whole_seconds', 'decimal', 'sec'], errors='ignore')

            # Add the chunk to the list as is
            imu_chunks.append(imu_chunk.to_numpy())  # Retain exact values

        # Prepare for CNN1D
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

        # Flatten and label data with BR from summary
        total_chunks = len(cnn_input_data)
        br_values = summary_data['BR'].values[:total_chunks]

        cnn_input_data_flat = cnn_input_data.reshape(total_chunks, -1)
        labeled_data = np.column_stack((cnn_input_data_flat, br_values))

        # Define base columns for headers
        base_columns = [
            "type",
            "accelerometer",
            "gyroscope",
            "magnetometer",
            "timestamp_interp"
        ]

        # Generate headers dynamically
        imu_headers_formatted = generate_column_headers(base_columns, cnn_input_data_flat.shape[1])
        headers = imu_headers_formatted + ["BR"]  # Append summary label

        # Validate header and data shape match
        if labeled_data.shape[1] != len(headers):
            print(f"ERROR: Data shape {labeled_data.shape}, Header length {len(headers)}")
            print(f"Generated Headers: {imu_headers_formatted}")
            raise ValueError("Mismatch between labeled data shape and headers.")

        # Save the labeled data to CSV
        output_file = os.path.join(output_directory, f"{folder_name}_{prefix}_chunked_with_labels.csv")
        pd.DataFrame(labeled_data, columns=headers).to_csv(output_file, index=False)
        print(f"Labeled data saved successfully to {output_file}!")

    except Exception as e:
        print(f"Error processing {imu_file} and {summary_file}: {e}")

# Function to process all folders and files
def process_all_files(base_directory, fixed_size):
    for folder in os.listdir(base_directory):
        folder_path = os.path.join(base_directory, folder)
        if os.path.isdir(folder_path):
            for prefix in ['L0', 'L1', 'L2', 'L3', 'M', 'R']:
                imu_file = os.path.join(folder_path, f"{prefix}_imu_df.csv")
                summary_file = os.path.join(folder_path, f"{prefix}_summary_df.csv")

                if os.path.exists(imu_file) and os.path.exists(summary_file):
                    print(f"Processing: {imu_file} and {summary_file}")
                    process_files(imu_file, summary_file, folder, prefix)

# Execute the processing
process_all_files(base_directory, fixed_size)
