import os
import numpy as np
import pandas as pd
import re
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Function to calculate MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# Directories
input_directory = r"D:\aria-seated\processed_files"
output_directory = r"D:\aria-seated\Results"

# Create output directory if not exists
os.makedirs(output_directory, exist_ok=True)

# Step 1: Load all files and extract subjects
datasets = {}
file_names = {}

for filename in os.listdir(input_directory):
    if filename.endswith(".csv"):
        file_path = os.path.join(input_directory, filename)
        data = pd.read_csv(file_path).to_numpy()

        if data.shape[1] < 2:
            print(f"File {filename} has insufficient columns.")
            continue

        # Extract subject ID using regex (assuming filenames have "subjectX")
        match = re.search(r"subject(\d+)", filename)
        if match:
            subject_id = match.group(1)  # Extract subject number as a string
        else:
            print(f"Warning: Could not extract subject ID from filename {filename}")
            continue

        # Store dataset by subject ID
        if subject_id not in datasets:
            datasets[subject_id] = []
            file_names[subject_id] = []

        datasets[subject_id].append(data)
        file_names[subject_id].append(filename)

# Convert lists to numpy arrays
for subject in datasets:
    datasets[subject] = np.concatenate(datasets[subject], axis=0)

# Prepare results storage
metrics_results = []

# --- 1️⃣ Within-Subject Testing (80% Train, 20% Test Per Subject) ---
for subject, data in datasets.items():
    X = data[:, :-1]  # Features
    y = data[:, -1]  # Target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Test model
    y_pred = model.predict(X_test)

    # Compute metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    metrics_results.append({
        "Test_Type": "Within-Subject",
        "Subject": subject,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R^2": r2,
        "MAPE": mape
    })

    print(f"Within-Subject {subject}: MAE={mae}, MSE={mse}, RMSE={rmse}, R^2={r2}, MAPE={mape}")

# --- 2️⃣ Cross-Subject Leave-One-Subject-Out (LOSO) ---
for subject_out in datasets.keys():
    print(f"Testing on subject {subject_out} (Leave-One-Subject-Out)...")

    # Test set: The entire subject-out dataset
    X_test = datasets[subject_out][:, :-1]
    y_test = datasets[subject_out][:, -1]

    # Train set: All other subjects
    train_data = [datasets[sub] for sub in datasets.keys() if sub != subject_out]
    train_data = np.concatenate(train_data, axis=0)

    X_train = train_data[:, :-1]
    y_train = train_data[:, -1]

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Test model
    y_pred = model.predict(X_test)

    # Compute metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    metrics_results.append({
        "Test_Type": "Cross-Subject LOSO",
        "Subject": subject_out,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R^2": r2,
        "MAPE": mape
    })

    print(f"Cross-Subject LOSO {subject_out}: MAE={mae}, MSE={mse}, RMSE={rmse}, R^2={r2}, MAPE={mape}")

# --- 3️⃣ Cross-Subject Train-All, Test-On-New (`subject_test` files) ---
print("Performing Cross-Subject Train-All-Conditions Testing...")

# Identify test files (containing "subject_test" in filename)
test_subjects = [sub for sub in datasets.keys() if any("subject_test" in fn for fn in file_names[sub])]

# Train on all available training data
train_subjects = [sub for sub in datasets.keys() if sub not in test_subjects]
train_data = [datasets[sub] for sub in train_subjects]
train_data = np.concatenate(train_data, axis=0)

X_train = train_data[:, :-1]
y_train = train_data[:, -1]

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Test on all `subject_test` data
for subject_test in test_subjects:
    X_test = datasets[subject_test][:, :-1]
    y_test = datasets[subject_test][:, -1]

    y_pred = model.predict(X_test)

    # Compute metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    metrics_results.append({
        "Test_Type": "Cross-Subject Train-All",
        "Subject": subject_test,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R^2": r2,
        "MAPE": mape
    })

    print(f"Cross-Subject Train-All {subject_test}: MAE={mae}, MSE={mse}, RMSE={rmse}, R^2={r2}, MAPE={mape}")

# Save results to CSV
output_file_path = os.path.join(output_directory, "cross_subject_testing_results.csv")
metrics_df = pd.DataFrame(metrics_results)
metrics_df.to_csv(output_file_path, index=False)
print(f"Cross-Subject Testing metrics saved to '{output_file_path}'.")
