import os
import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter
from sklearn.tree import DecisionTreeRegressor
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return lfilter(b, a, data, axis=0)


def imu_signal_processing(data, fs=100):
    return butter_bandpass_filter(data, 3 / 60, 45 / 60, fs=fs, order=2)


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


input_directory = r"D:\\aria-seated\\processed_files"
output_directory = r"D:\\aria-seated\\Results"
os.makedirs(output_directory, exist_ok=True)

datasets = []
file_names = []

for filename in os.listdir(input_directory):
    if filename.endswith(".csv"):
        file_path = os.path.join(input_directory, filename)
        df = pd.read_csv(file_path)
        if 'BR' not in df.columns:
            continue
        df = df.drop(columns=[col for col in df.columns if "timestamp" in col.lower()], errors='ignore')
        imu_data = df.drop(columns=['BR']).to_numpy()
        processed_imu = imu_signal_processing(imu_data)
        datasets.append((processed_imu, df['BR'].values, filename))

metrics_results = []

for i in range(len(datasets)):
    X_test = datasets[i][0]
    y_test = datasets[i][1]
    X_train = np.vstack([datasets[j][0] for j in range(len(datasets)) if j != i])
    y_train = np.hstack([datasets[j][1] for j in range(len(datasets)) if j != i])

    # Convert to DataFrame to drop fully NaN columns
    X_train_df = pd.DataFrame(X_train)
    X_test_df = pd.DataFrame(X_test)
    X_train_df = X_train_df.dropna(axis=1, how="all")
    X_test_df = X_test_df[X_train_df.columns]
    X_train = X_train_df.to_numpy()
    X_test = X_test_df.to_numpy()

    # Imputation
    imputer = SimpleImputer(strategy="mean")
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    # PCA
    pca_full = PCA(n_components=0.95)
    pca_full.fit(X_train)
    n_components = pca_full.n_components_
    pca = IncrementalPCA(n_components=n_components, batch_size=500)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    print(f"Processing LOOCV iteration {i + 1}/{len(datasets)}")
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError(f"Mismatch detected: X_train={X_train.shape}, y_train={y_train.shape}")

    # Train model
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train_pca, y_train)
    y_pred = model.predict(X_test_pca)

    # Evaluate model
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    metrics_results.append({"File": datasets[i][2], "MAE": mae, "MSE": mse, "RMSE": rmse, "R^2": r2, "MAPE": mape})

output_file_path = os.path.join(output_directory, "pca_loocv_metrics_results_with_bp.csv")
pd.DataFrame(metrics_results).to_csv(output_file_path, index=False)
print(f"✅ LOOCV metrics saved to '{output_file_path}'.")
