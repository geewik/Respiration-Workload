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
            print(f"File {filename} does not contain 'BR'. Skipping.")
            continue
        df = df.drop(columns=[col for col in df.columns if "timestamp" in col.lower()], errors='ignore')
        X = df.drop(columns=['BR']).to_numpy()
        y = df['BR'].to_numpy()
        if X.shape[1] < 1:
            print(f"File {filename} has insufficient feature columns.")
            continue
        datasets.append((X, y))
        file_names.append(filename)

metrics_results = []

for i in range(len(datasets)):
    X_test, y_test = datasets[i]
    X_train = np.concatenate([datasets[j][0] for j in range(len(datasets)) if j != i], axis=0)
    y_train = np.concatenate([datasets[j][1] for j in range(len(datasets)) if j != i], axis=0)

    # Convert to DataFrame to drop fully NaN columns
    X_train_df = pd.DataFrame(X_train)
    X_test_df = pd.DataFrame(X_test)
    X_train_df = X_train_df.dropna(axis=1, how="all")
    X_test_df = X_test_df[X_train_df.columns]
    X_train = X_train_df.to_numpy()
    X_test = X_test_df.to_numpy()

    # Handle NaNs by filling with zero
    imputer = SimpleImputer(strategy="constant", fill_value=0, keep_empty_features=True)
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    # Determine optimal PCA components using full PCA first
    pca_full = PCA(n_components=0.95)
    pca_full.fit(X_train)
    n_components = pca_full.n_components_

    # Apply Incremental PCA with determined components
    pca = IncrementalPCA(n_components=n_components, batch_size=500)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    print(f"\n🔄 Processing LOOCV iteration {i + 1}/{len(datasets)}: Leaving out {file_names[i]}...")
    print(f"📊 Dataset before PCA ({file_names[i]}):")
    print(f"X_train - min: {X_train.min()}, max: {X_train.max()}, mean: {X_train.mean()}")
    print(f"X_test - min: {X_test.min()}, max: {X_test.max()}, mean: {X_test.mean()}")
    print(f"📉 Dataset after PCA ({file_names[i]}):")
    print(f"X_train_pca - min: {X_train_pca.min()}, max: {X_train_pca.max()}, mean: {X_train_pca.mean()}")
    print(f"X_test_pca - min: {X_test_pca.min()}, max: {X_test_pca.max()}, mean: {X_test_pca.mean()}")

    # Train model
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train_pca, y_train)
    y_pred = model.predict(X_test_pca)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    metrics_results.append({
        "File": file_names[i], "MAE": mae, "MSE": mse, "RMSE": rmse, "R^2": r2, "MAPE": mape
    })

output_file_path = os.path.join(output_directory, "loocv_metrics_results_DecisionTree_with_PCA_replaceZero.csv")
pd.DataFrame(metrics_results).to_csv(output_file_path, index=False)
print(f"LOOCV metrics saved to '{output_file_path}'.")