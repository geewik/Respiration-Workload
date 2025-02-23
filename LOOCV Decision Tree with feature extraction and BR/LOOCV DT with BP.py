import os
import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter
from scipy.fftpack import fft
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Define the butter_bandpass_filter function
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data, axis=0)
    return y

# Define the moving average function for multidimensional NumPy arrays
def movingaverage(data, window_size, shift, axis=0):
    pad_width = [(0, 0)] * data.ndim
    pad_width[axis] = (window_size // 2, window_size // 2)
    padded_data = np.pad(data, pad_width, mode='edge')

    cumsum = np.cumsum(padded_data, axis=axis)
    ma = (cumsum[window_size:] - cumsum[:-window_size]) / window_size

    return np.roll(ma, shift, axis=axis)

# Define the IMU signal processing function
def imu_signal_processing(data, fs: int = 100):
    bp = butter_bandpass_filter(data, 3 / 60, 45 / 60, fs=fs, order=2)
    ma = movingaverage(bp, window_size=120, shift=1, axis=0)
    return ma

# Feature Extraction
def extract_statistical_features(data):
    return np.array([
        np.mean(data, axis=0),  # Mean
        np.std(data, axis=0),  # Standard deviation
        np.var(data, axis=0),  # Variance
        np.mean((data - np.mean(data, axis=0))**3, axis=0) / (np.std(data, axis=0)**3),  # Skewness
        np.mean((data - np.mean(data, axis=0))**4, axis=0) / (np.std(data, axis=0)**4),  # Kurtosis
        np.ptp(data, axis=0),  # Peak-to-peak amplitude
        np.sqrt(np.mean(data**2, axis=0))  # RMS
    ]).flatten()

def extract_frequency_features(data, fs=100):
    fft_values = np.abs(fft(data, axis=0))  # Compute FFT
    freqs = np.fft.fftfreq(len(data), d=1/fs)  # Frequency bins

    dominant_freq = freqs[np.argmax(fft_values, axis=0)]
    spectral_entropy = -np.sum((fft_values**2) * np.log(fft_values**2 + 1e-10), axis=0)  # Avoid log(0)

    return np.concatenate([dominant_freq, spectral_entropy])

# Directories
input_directory = r"D:\aria-seated\processed_files"
output_directory = r"D:\aria-seated\Results"

# Create output directory if not exists
os.makedirs(output_directory, exist_ok=True)

# Step 1: Load all files
datasets = []
file_names = []

for filename in os.listdir(input_directory):
    if filename.endswith(".csv"):
        file_path = os.path.join(input_directory, filename)
        df = pd.read_csv(file_path)

        if "BR" not in df.columns:
            print(f"Skipping {filename}: 'BR' column not found.")
            continue

        # Drop any column containing "timestamp" (case-insensitive)
        df = df.loc[:, ~df.columns.str.contains("timestamp", case=False)]

        # Define X (all columns except "BR") and y (target = "BR")
        X = df.drop(columns=["BR"]).to_numpy()
        y = df["BR"].to_numpy()

        # Apply IMU signal processing only to feature columns
        try:
            X = imu_signal_processing(X)
        except Exception as e:
            print(f"Error processing file {filename}: {e}")
            continue

        # Extract features (both statistical & frequency-based)
        try:
            X_stat_features = extract_statistical_features(X)
            X_freq_features = extract_frequency_features(X)
            X_final = np.concatenate([X_stat_features, X_freq_features])
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
    # Leave out one file for testing
    X_test = X_all[i].reshape(1, -1)
    y_test = np.array([y_all[i]])

    # Use all other files for training
    X_train = np.delete(X_all, i, axis=0)
    y_train = np.delete(y_all, i, axis=0)

    # Train model
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Test model
    y_pred = model.predict(X_test)

    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    # Handle R² issue
    try:
        r2 = r2_score(y_test, y_pred)
    except:
        r2 = None  # Set to None if undefined

    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100  # MAPE calculation

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
output_file_path = os.path.join(output_directory, "loocv_metrics_results_w_features.csv")
metrics_df = pd.DataFrame(metrics_results)
metrics_df.to_csv(output_file_path, index=False)
print(f"LOOCV metrics saved to '{output_file_path}'.")
z