import os
import numpy as np
import pandas as pd
from scipy.fftpack import fft
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Function to calculate MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Feature Extraction Functions
def extract_statistical_features(data):
    """ Extracts statistical features while preventing divide-by-zero errors. """
    mean = np.mean(data, axis=0)
    std_dev = np.std(data, axis=0)
    std_dev[std_dev == 0] = 1e-10  # Prevent division by zero

    skewness = np.mean((data - mean)**3, axis=0) / (std_dev**3)
    kurtosis = np.mean((data - mean)**4, axis=0) / (std_dev**4)

    return np.array([
        mean,  # Mean
        std_dev,  # Standard deviation
        np.var(data, axis=0),  # Variance
        skewness,  # Skewness (Fixed)
        kurtosis,  # Kurtosis (Fixed)
        np.ptp(data, axis=0),  # Peak-to-peak amplitude
        np.sqrt(np.mean(data**2, axis=0))  # RMS
    ]).flatten()

def extract_frequency_features(data, fs=100):
    """ Extracts frequency-based features using FFT. """
    fft_values = np.abs(fft(data, axis=0))  # Compute FFT
    freqs = np.fft.fftfreq(len(data), d=1/fs)  # Frequency bins

    dominant_freq = freqs[np.argmax(fft_values, axis=0)]
    spectral_entropy = -np.sum((fft_values**2) * np.log(fft_values**2 + 1e-10), axis=0)  # Prevent log(0)

    return np.concatenate([dominant_freq, spectral_entropy])

# Directories
input_directory = r"D:\aria-seated\processed_files"
output_directory = r"D:\aria-seated\Results"

# Create the results directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Step 1: Load all files
datasets = []
file_names = []

for filename in os.listdir(input_directory):
    if filename.endswith(".csv"):
        file_path = os.path.join(input_directory, filename)
        df = pd.read_csv(file_path)

        # Ensure 'BR' is present in the dataset
        if 'BR' not in df.columns:
            print(f"File {filename} does not contain 'BR'. Skipping.")
            continue

        # Drop 'BR' and any column containing 'magnetometer'
        X = df.drop(columns=[col for col in df.columns if 'magnetometer' in col.lower() or col == 'BR']).to_numpy()
        y = df['BR'].to_numpy()  # Target is 'BR'

        if X.shape[1] < 1:
            print(f"File {filename} has insufficient feature columns after removing 'BR' and magnetometer-related columns. Skipping.")
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
        datasets.append(data)
        file_names.append(filename)

# Step 2: Perform LOOCV
metrics_results = []

datasets = np.array(datasets)  # Convert to NumPy array
X_all = datasets[:, :-1]  # Extract features
y_all = datasets[:, -1]   # Extract target

for i in range(len(datasets)):
    # Leave one dataset out for testing
    X_test = X_all[i].reshape(1, -1)
    y_test = np.array([y_all[i]])

    # Use all other datasets for training
    X_train = np.delete(X_all, i, axis=0)
    y_train = np.delete(y_all, i, axis=0)

    # Train Decision Tree Regressor
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    # Handle R² issue (avoid nan for single test samples)
    if len(y_test) > 1:
        r2 = r2_score(y_test, y_pred)
    else:
        r2 = None  # Avoid NaN warnings

    mape = mean_absolute_percentage_error(y_test, y_pred)

    # Store results
    metrics_results.append({
        "File": file_names[i],
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R^2": r2 if r2 is not None else "N/A",  # Avoid NaN
        "MAPE": mape
    })

    print(f"File {file_names[i]}: MAE={mae}, MSE={mse}, RMSE={rmse}, R^2={'N/A' if r2 is None else r2}, MAPE={mape}")

# Save results to CSV
output_file_path = os.path.join(output_directory, "loocv_metrics_results_DecisionTree_with_fixed_features.csv")
metrics_df = pd.DataFrame(metrics_results)
metrics_df.to_csv(output_file_path, index=False)
print(f"LOOCV metrics saved to '{output_file_path}'.")
