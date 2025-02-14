import os
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Function to calculate MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

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

        # Ensure 'BR' is excluded from features
        if 'BR' not in df.columns:
            print(f"File {filename} does not contain 'BR'. Skipping.")
            continue

        X = df.drop(columns=['BR']).to_numpy()  # Exclude 'BR' from features
        y = df['BR'].to_numpy()  # Target is 'BR'

        if X.shape[1] < 1:
            print(f"File {filename} has insufficient feature columns after removing 'BR'.")
            continue

        datasets.append((X, y))
        file_names.append(filename)

# Step 2: Perform LOOCV
metrics_results = []

for i in range(len(datasets)):
    # Leave out one file for testing
    X_test, y_test = datasets[i]

    # Use all other files for training
    X_train = np.concatenate([datasets[j][0] for j in range(len(datasets)) if j != i], axis=0)
    y_train = np.concatenate([datasets[j][1] for j in range(len(datasets)) if j != i], axis=0)

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
    mape = mean_absolute_percentage_error(y_test, y_pred)

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
output_file_path = os.path.join(output_directory, "loocv_metrics_results_DecisionTree_no_bp.csv")
metrics_df = pd.DataFrame(metrics_results)
metrics_df.to_csv(output_file_path, index=False)
print(f"LOOCV metrics saved to '{output_file_path}'.")




