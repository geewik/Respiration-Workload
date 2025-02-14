import os
import pandas as pd

# Directories for input and output files
input_directory = "D:\\aria-seated\\output_files\\chunked attempt two"
output_directory = "D:\\aria-seated\\processed_files"
os.makedirs(output_directory, exist_ok=True)

def process_file(file_path, output_path):
    """
    Process a single file: remove 'type_tX' columns, split accelerometer/gyroscope/magnetometer
    into _x, _y, _z, and preserve the original order.
    """
    try:
        # Load the dataset
        data = pd.read_csv(file_path)

        # Remove 'type_tX' columns
        type_columns = [col for col in data.columns if col.startswith("type_t")]
        data = data.drop(columns=type_columns, errors="ignore")

        # Extract the time steps in numerical order
        time_steps = sorted({
            int(col.split("_t")[1]) for col in data.columns if "_t" in col
        })

        # Define sensors to split
        sensors = ["accelerometer", "gyroscope", "magnetometer"]
        reordered_columns = []

        # Process each time step
        for t in time_steps:
            for sensor in sensors:
                # Match the current sensor column for this time step
                sensor_col = f"{sensor}_t{t}"
                if sensor_col in data.columns:
                    # Split the column into _x, _y, _z components
                    expanded_data = pd.DataFrame(
                        data[sensor_col].apply(
                            lambda x: eval(x) if isinstance(x, str) and "[" in x else [None, None, None]
                        ).tolist(),
                        columns=[f"{sensor_col}_x", f"{sensor_col}_y", f"{sensor_col}_z"]
                    )
                    # Add the expanded columns to the dataset
                    data = pd.concat([data, expanded_data], axis=1)
                    data.drop(columns=[sensor_col], inplace=True)

                # Append the new column names to the reordered list
                reordered_columns.extend([f"{sensor_col}_x", f"{sensor_col}_y", f"{sensor_col}_z"])

            # Add the timestamp column back in order
            timestamp_col = f"timestamp_interp_t{t}"
            if timestamp_col in data.columns:
                reordered_columns.append(timestamp_col)

        # Add the 'BR' column at the end
        if "BR" in data.columns:
            reordered_columns.append("BR")

        # Reorder the dataset
        data = data[reordered_columns]

        # Save the processed file
        data.to_csv(output_path, index=False)
        print(f"Processed and saved: {output_path}")

    except Exception as e:
        print(f"Error processing {file_path}: {e}")

# Process all files in the input directory
for file_name in os.listdir(input_directory):
    if file_name.endswith(".csv"):
        input_file = os.path.join(input_directory, file_name)
        output_file = os.path.join(output_directory, file_name)
        process_file(input_file, output_file)

print("Processing complete!")
