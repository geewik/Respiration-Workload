import os
import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter
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
        data = pd.read_csv(file_path).to_numpy()

        if data.shape[1] < 2:
            print(f"File {filename} has insufficient columns.")
            continue

        # Apply IMU signal processing to features (excluding the target column)
        try:
            data[:, :-1] = imu_signal_processing(data[:, :-1])
        except Exception as e:
            print(f"Error processing file {filename}: {e}")
            continue

        datasets.append(data)
        file_names.append(filename)

# Step 2: Perform LOOCV
metrics_results = []

for i in range(len(datasets)):
    # Leave out one file for testing
    test_data = datasets[i]
    X_test = test_data[:, :-1]
    y_test = test_data[:, -1]

    # Use all other files for training
    train_data = [datasets[j] for j in range(len(datasets)) if j != i]
    train_data = np.concatenate(train_data, axis=0)

    X_train = train_data[:, :-1]
    y_train = train_data[:, -1]

    # Train model
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Test model
    y_pred = model.predict(X_test)

    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100  # MAPE calculation

    metrics_results.append({
        "File": file_names[i],
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R^2": r2,
        "MAPE": mape
    })

    print(f"File {file_names[i]}: MAE={mae}, MSE={mse}, RMSE={rmse}, R^2={r2}, MAPE={mape}")

# Save results to CSV
output_file_path = os.path.join(output_directory, "loocv_metrics_results_w_bp_2.csv")
metrics_df = pd.DataFrame(metrics_results)
metrics_df.to_csv(output_file_path, index=False)
print(f"LOOCV metrics saved to '{output_file_path}'.")