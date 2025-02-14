import os
import numpy as np
import pandas as pd
import re
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Function to Calculate MAPE
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

        # Extract subject ID using regex
        match = re.search(r"S(\d+)", filename)
        if match:
            subject_id = match.group(1)  # Extract subject number
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

# --- 1️⃣ Cross-Subject Leave-One-Subject-Out (LOSO) ---
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

    # Train Decision Tree model
    model = DecisionTreeRegressor(random_state=42)
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

# --- 2️⃣ Cross-Subject Train-All, Test-On-New (`subject_test` files) ---
print("Performing Cross-Subject Train-All-Conditions Testing...")

# Identify test files (containing "subject_test" in filename)
test_subjects = [sub for sub in datasets.keys() if any("subject_test" in fn for fn in file_names[sub])]

# Train on all available training data
train_subjects = [sub for sub in datasets.keys() if sub not in test_subjects]
train_data = [datasets[sub] for sub in train_subjects]

# Prevent crashing if no train data is available
if len(train_data) > 0:
    train_data = np.concatenate(train_data, axis=0)
    X_train = train_data[:, :-1]
    y_train = train_data[:, -1]

    # Train Decision Tree model
    model = DecisionTreeRegressor(random_state=42)
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
else:
    print("🚨 No training data found for Cross-Subject Train-All testing. Skipping this step.")

# Save results to CSV
output_file_path = os.path.join(output_directory, "cross_subject_testing_results_decision_tree_no_bp.csv")
metrics_df = pd.DataFrame(metrics_results)
metrics_df.to_csv(output_file_path, index=False)
print(f"Cross-Subject Testing metrics saved to '{output_file_path}'.")

