import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics

# Define the folder path and output file
folder_path = r"D:\aria-seated\output_files"
output_file = r"D:\aria-seated\output_files\decision_tree_results.csv"

# List all relevant CSV files in the folder (matching the naming pattern)
files = [
    file for file in os.listdir(folder_path)
    if file.endswith('.csv') and '_chunked_with_labels' in file
]

# Initialize a list to store results
results = []

# Loop through each file and process it
for csv_file in files:
    file_path = os.path.join(folder_path, csv_file)
    print(f"Processing file: {csv_file}")

    # Load the dataset
    data = pd.read_csv(file_path)

    # Drop non-relevant columns (e.g., type columns)
    data = data.drop(columns=[col for col in data.columns if col.startswith('type')])

    # Safely parse stringified arrays into numerical values (mean for simplicity)
    def parse_array_column(col):
        if col.apply(lambda x: isinstance(x, str) and x.startswith('[')).any():
            return col.apply(lambda x: np.mean(eval(x)) if isinstance(x, str) and x.startswith('[') else np.nan)
        return col

    # Apply parsing to all object columns
    array_columns = [col for col in data.columns if data[col].dtype == 'object']
    for col in array_columns:
        data[col] = parse_array_column(data[col])

    # Drop columns with excessive missing values (e.g., more than 50%)
    threshold = 0.5
    data = data.loc[:, data.isnull().mean() < threshold]

    # Fill remaining NaN values with column mean
    data = data.fillna(data.mean(numeric_only=True))

    # Ensure the dataset is not empty
    if data.empty:
        print(f"Skipping {csv_file}: dataset is empty after processing.")
        results.append({
            "File": csv_file,
            "Mean Absolute Error": "N/A",
            "R^2 Score": "N/A",
            "MSE": "N/A",
            "RMSE": "N/A",
            "MAPE (%)": "N/A"
        })
        continue

    # Ensure the 'BR' column exists
    if 'BR' not in data.columns:
        print(f"Skipping {csv_file}: 'BR' column not found.")
        results.append({
            "File": csv_file,
            "Mean Absolute Error": "N/A",
            "R^2 Score": "N/A",
            "MSE": "N/A",
            "RMSE": "N/A",
            "MAPE (%)": "N/A"
        })
        continue

    # Separate features and target
    X = data.drop("BR", axis=1)
    Y = data["BR"]

    # Split into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Save future samples for additional testing
    futureSample_X = X_test.iloc[-2:]
    futureSample_Y = Y_test.iloc[-2:]

    # Update test sets to exclude future samples
    X_test = X_test.iloc[:-2]
    Y_test = Y_test.iloc[:-2]
    Y_test = Y_test.reset_index(drop=True)

    # Initialize and fit the Decision Tree model
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, Y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Calculate metrics
    mean_absolute_error_test = metrics.mean_absolute_error(Y_test, y_pred)
    r2_test = metrics.r2_score(Y_test, y_pred)
    mse_test = metrics.mean_squared_error(Y_test, y_pred)
    rmse_test = np.sqrt(mse_test)

    # Define MAPE function
    def MAPE(y, y_predict):
        y = np.where(y == 0, np.finfo(float).eps, y)
        return np.mean(np.abs((y - y_predict) / y)) * 100

    mape_test = MAPE(Y_test, y_pred)

    # Append results to the list
    results.append({
        "File": csv_file,
        "Mean Absolute Error": round(mean_absolute_error_test, 4),
        "R^2 Score": round(r2_test, 4),
        "MSE": round(mse_test, 4),
        "RMSE": round(rmse_test, 4),
        "MAPE (%)": round(mape_test, 2)
    })

# Save results to a CSV file
results_df = pd.DataFrame(results)
results_df.to_csv(output_file, index=False)

print(f"Results saved to {output_file}")
