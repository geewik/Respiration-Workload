import os
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

input_directory = r"D:\aria-seated\processed_files"
output_directory = r"D:\aria-seated\Results"
os.makedirs(output_directory, exist_ok=True)

# Step 1: Collect all possible feature column names across all files
all_feature_columns = set()

for filename in os.listdir(input_directory):
    if filename.endswith(".csv"):
        file_path = os.path.join(input_directory, filename)
        df = pd.read_csv(file_path)

        if 'BR' not in df.columns:
            print(f"⚠️ File {filename} does not contain 'BR'. Skipping.")
            continue

        # Drop timestamp columns
        df = df.drop(columns=[col for col in df.columns if "timestamp" in col.lower()], errors='ignore')

        # Collect feature column names (excluding 'BR')
        feature_cols = set(df.columns) - {"BR"}
        all_feature_columns.update(feature_cols)

all_feature_columns = sorted(all_feature_columns)  # Sort for consistency

datasets = []
file_names = []

# Step 2: Load and standardize datasets to have the same columns
for filename in os.listdir(input_directory):
    if filename.endswith(".csv"):
        file_path = os.path.join(input_directory, filename)
        df = pd.read_csv(file_path)

        if 'BR' not in df.columns:
            continue

        df = df.drop(columns=[col for col in df.columns if "timestamp" in col.lower()], errors='ignore')

        # Ensure all datasets have the same columns by adding missing columns as NaN
        for col in all_feature_columns:
            if col not in df.columns:
                df[col] = np.nan  # Add missing columns as NaN

        df = df[all_feature_columns + ['BR']]  # Reorder columns

        X = df.drop(columns=['BR'])
        y = df['BR'].to_numpy()

        datasets.append((X, y))
        file_names.append(filename)

# Step 3: Drop columns that are fully NaN across all datasets
df_combined = pd.concat([d[0] for d in datasets], axis=0)
nan_columns = df_combined.columns[df_combined.isna().all()].tolist()

if nan_columns:
    print(f"🛑 Dropping fully NaN columns: {nan_columns}")
    for i in range(len(datasets)):
        datasets[i] = (datasets[i][0].drop(columns=nan_columns), datasets[i][1])

# Ensure that all datasets now have the same number of features
if len(datasets) == 0:
    raise ValueError("No valid datasets found after processing. Please check the input files.")

metrics_results = []

for i in range(len(datasets)):
    X_test, y_test = datasets[i]
    X_train = pd.concat([datasets[j][0] for j in range(len(datasets)) if j != i], axis=0).to_numpy()
    y_train = np.concatenate([datasets[j][1] for j in range(len(datasets)) if j != i], axis=0)

    # Handle NaNs by replacing with column mean
    imputer = SimpleImputer(strategy="mean")
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test.to_numpy())

    # Determine optimal PCA components
    pca_full = PCA(n_components=0.95)
    pca_full.fit(X_train)
    n_components = pca_full.n_components_

    # Apply Incremental PCA
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

output_file_path = os.path.join(output_directory, "loocv_metrics_results_DecisionTree_with_PCA_removeNan.csv")
pd.DataFrame(metrics_results).to_csv(output_file_path, index=False)
print(f"✅ LOOCV metrics saved to '{output_file_path}'.")
