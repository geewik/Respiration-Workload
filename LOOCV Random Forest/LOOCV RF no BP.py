import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Function to calculate MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


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

        datasets.append(data)
        file_names.append(filename)

# Step 2: Perform LOOCV
metrics_results = []

for i in range(len(datasets)):
    # Leave one file out for testing
    test_data = datasets[i]
    X_test = test_data[:, :-1]
    y_test = test_data[:, -1]

    # Use all other files for training
    train_data = [datasets[j] for j in range(len(datasets)) if j != i]
    train_data = np.concatenate(train_data, axis=0)

    X_train = train_data[:, :-2]
    y_train = train_data[:, -2]

    # Train Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
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
output_file_path = os.path.join(output_directory, "loocv_metrics_results_random_forest_NO_BP.csv")
metrics_df = pd.DataFrame(metrics_results)
metrics_df.to_csv(output_file_path, index=False)
print(f"LOOCV metrics saved to '{output_file_path}'.")
