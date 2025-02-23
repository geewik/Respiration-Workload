import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess_data(folder_path, save_path="D:/aria-marker/gan_preprocessed"):
    # Check if the folder exists
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"❌ Folder NOT found: {folder_path}")

    # Ensure save directory exists
    os.makedirs(save_path, exist_ok=True)

    # List subject folders in the directory
    subject_folders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    print("📂 Found Subject Folders:", subject_folders)

    for subject in subject_folders:
        subject_path = os.path.join(folder_path, subject)
        subject_save_path = os.path.join(save_path, subject)
        os.makedirs(subject_save_path, exist_ok=True)

        merged_files = [os.path.join(subject_path, file)
                        for file in os.listdir(subject_path) if "_merged.csv" in file]

        print(f"✅ Processing Subject: {subject}")

        if not merged_files:
            print(f"❌ No `_merged.csv` files found for subject {subject}!")
            continue

        data_list = []

        for file in merged_files:
            print(f"📥 Loading: {file}")
            df = pd.read_csv(file)

            if df.empty:
                print(f"⚠️ Skipping empty file: {file}")
                continue

            # Normalize numerical columns
            posture_columns = ['Glasses_Rotation_Y', 'Glasses_Rotation_Z', 'Glasses_Rotation',
                               'Glasses_Position_X', 'Glasses_Position_Y', 'Glasses_Position_Z']
            scaler = MinMaxScaler()
            df[posture_columns] = scaler.fit_transform(df[posture_columns])

            # Convert 'BR' to categorical if necessary
            if df['BR'].dtype == 'object':
                df['BR'] = pd.factorize(df['BR'])[0]

            # Classify postures
            df['Posture'] = 'Straight'
            df.loc[df['Glasses_Position_Y'] > df['Glasses_Position_Y'].median() + 0.1, 'Posture'] = 'Looking Up'
            df.loc[df['Glasses_Position_Y'] < df['Glasses_Position_Y'].median() - 0.1, 'Posture'] = 'Looking Down'

            data_list.append(df)

        # Combine all preprocessed data for the subject
        processed_data = pd.concat(data_list, ignore_index=True)

        # Save the processed dataset in the subject's folder
        output_file = os.path.join(subject_save_path, "processed_dataset.csv")
        processed_data.to_csv(output_file, index=False)
        print(f"✅ Processed dataset saved for {subject} to: {output_file}")

    print("✅ All subjects processed!")

# Define dataset path
processed_folder = "D:/aria-marker/processed"
load_and_preprocess_data(processed_folder)
