import os
import numpy as np
import pandas as pd
import re
from scipy.fftpack import fft
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Function to Calculate MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# Feature Extraction Functions
def extract_statistical_features(data):
    """ Extracts statistical features while preventing divide-by-zero errors. """
    mean = np.mean(data, axis=0)
    std_dev = np.std(data, axis=0)
    std_dev[std_dev == 0] = 1e-10  # Prevent division by zero

    skewness = np.mean((data - mean) ** 3, axis=0) / (std_dev ** 3)
    kurtosis = np.mean((data - mean) ** 4, axis=0) / (std_dev ** 4)

    return np.array([
        mean,  # Mean
        std_dev,  # Standard deviation
        np.var(data, axis=0),  # Variance
        skewness,  # Skewness
        kurtosis,  # Kurtosis
        np.ptp(data, axis=0),  # Peak-to-peak amplitude
        np.sqrt(np.mean(data ** 2, axis=0))  # RMS
    ]).flatten()


def extract_frequency_features(data, fs=100):
    """ Extracts frequency-based features using FFT. """
    fft_values = np.abs(fft(data, axis=0))  # Compute FFT
    freqs = np.fft.fftfreq(len(data), d=1 / fs)  # Frequency bins

    dominant_freq = freqs[np.argmax(fft_values, axis=0)]
    spectral_entropy = -np.sum((fft_values ** 2) * np.log(fft_values ** 2 + 1e-10), axis=0)  # Prevent log(0)

    return np.concatenate([dominant_freq, spectral_entropy])


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
        df = pd.read_csv(file_path)

        # Ensure 'BR' is present in the dataset
        if 'BR' not in df.columns:
            print(f"File {filename} does not contain 'BR'. Skipping.")
            continue

        # Drop 'BR' and any column containing 'timestamp'
        X = df.drop(columns=[col for col in df.columns if 'timestamp' in col.lower() or col == 'BR']).to_numpy()
        y = df['BR'].to_numpy()  # Target is 'BR'

        if X.shape[1] < 1:
            print(
                f"File {filename} has insufficient feature columns after removing 'BR' and timestamp-related columns. Skipping.")
            continue

        # Apply Feature Extraction
        try:
            X_stat_features = extract_statistical_features(X)
            X_freq_features = extract_frequency_features(X)
            X_final = np.concatenate([X_stat_features, X_freq_features])  # Flatten extracted features
        except Exception as e:
            print(f"Feature extraction error in file {filename}: {e}")
            continue

        # Aggregate y to match extracted feature shape
        y_mean = np.mean(y)  # Mean BR value as target
        data = np.append(X_final, y_mean)  # Ensure matching dimensions

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
    datasets[subject] = np.array(datasets[subject])  # Convert to NumPy array

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

    if len(train_data) > 0:
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
        r2 = r2_score(y_test, y_pred) if len(y_test) > 1 else "N/A"  # Avoid NaN
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

test_subjects = [sub for sub in datasets.keys() if any("subject_test" in fn for fn in file_names[sub])]
train_subjects = [sub for sub in datasets.keys() if sub not in test_subjects]

if len(train_subjects) > 0:
    train_data = np.concatenate([datasets[sub] for sub in train_subjects], axis=0)
    X_train = train_data[:, :-1]
    y_train = train_data[:, -1]

    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)

    for subject_test in test_subjects:
        X_test = datasets[subject_test][:, :-1]
        y_test = datasets[subject_test][:, -1]

        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred) if len(y_test) > 1 else "N/A"
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

# Save results to CSV
output_file_path = os.path.join(output_directory, "cross_subject_testing_results_with_features_bp.csv")
pd.DataFrame(metrics_results).to_csv(output_file_path, index=False)
print(f"Cross-Subject Testing metrics saved to '{output_file_path}'.")
