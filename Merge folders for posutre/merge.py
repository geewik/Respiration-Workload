import os
import pandas as pd


def merge_marker_summary(marker_folder_path, output_folder, subjects, levels):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for subject in subjects:
        subject_path = os.path.join(marker_folder_path, subject)
        output_subject_path = os.path.join(output_folder, subject)
        if not os.path.exists(output_subject_path):
            os.makedirs(output_subject_path)

        if not os.path.exists(subject_path):
            print(f"Skipping {subject}, folder not found.")
            continue

        for level in levels:
            marker_file = os.path.join(subject_path, f"{level}_marker_df.csv")
            summary_folder_path = "D:/aria-seated"
            summary_file = os.path.join(summary_folder_path, subject, f"{level}_summary_df.csv")

            if not os.path.exists(marker_file) or not os.path.exists(summary_file):
                print(f"Skipping {subject} {level}, required files not found.")
                continue

            # Check if files are empty
            if os.path.getsize(marker_file) == 0:
                print(f"Skipping {subject} {level}, marker file is empty.")
                continue
            if os.path.getsize(summary_file) == 0:
                print(f"Skipping {subject} {level}, summary file is empty.")
                continue

            try:
                # Read marker file and select necessary columns
                marker_df = pd.read_csv(marker_file)
                marker_columns = ['sec', 'Glasses_Rotation_Y', 'Glasses_Rotation_Z', 'Glasses_Rotation',
                                  'Glasses_Position_X', 'Glasses_Position_Y', 'Glasses_Position_Z']
                marker_df = marker_df[marker_columns]

                # Read summary file and select 'sec' and 'BR' column
                summary_df = pd.read_csv(summary_file)
                summary_df = summary_df[['sec', 'BR']]

                # Merge on 'sec' column using outer join to keep all timestamps
                merged_df = pd.merge(marker_df, summary_df, on='sec', how='outer')

                # Sort by sec column
                merged_df = merged_df.sort_values(by='sec')

                # Forward fill missing BR values
                merged_df['BR'] = merged_df['BR'].ffill()

                # Fill missing rotation and position values
                posture_columns = ['Glasses_Rotation_Y', 'Glasses_Rotation_Z', 'Glasses_Rotation',
                                   'Glasses_Position_X', 'Glasses_Position_Y', 'Glasses_Position_Z']
                merged_df[posture_columns] = merged_df[posture_columns].ffill(limit=100)

                # Save merged file
                output_file = os.path.join(output_subject_path, f"{level}_merged.csv")
                merged_df.to_csv(output_file, index=False)
                print(f"Merged file saved: {output_file}")

            except pd.errors.EmptyDataError:
                print(f"Skipping {subject} {level}, file contains no valid data.")
            except pd.errors.ParserError:
                print(f"Skipping {subject} {level}, file is corrupted and cannot be parsed.")
            except KeyError as e:
                print(f"Skipping {subject} {level}, missing expected column: {e}")


# Define parameters
marker_folder_path = "D:/aria-marker"
output_folder = os.path.join(marker_folder_path, "processed")
subjects = [f"s{i}" for i in range(12, 31)]
levels = ['L0', 'L1', 'L2', 'L3', 'M', 'R']

# Run the function
merge_marker_summary(marker_folder_path, output_folder, subjects, levels)