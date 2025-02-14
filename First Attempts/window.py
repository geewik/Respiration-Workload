import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt


def gather_seated_data(base_dir=r"D:\aria-seated"):

    print(f"Base directory: {base_dir}")
    all_data = {}

    try:
        items = os.listdir(base_dir)
        has_subdirs = any(os.path.isdir(os.path.join(base_dir, item)) for item in items)

        if has_subdirs:
            print("Found subdirectories. Processing each...")
            subjects = [f for f in items if os.path.isdir(os.path.join(base_dir, f))]
            print(f"Subjects found: {subjects}")

            for sbj in subjects:
                sbj_dir = os.path.join(base_dir, sbj)
                print(f"Checking folder: {sbj_dir}")
                print(f"All files in {sbj}: {os.listdir(sbj_dir)}")  # List all files for debugging

                sbj_data = {}
                try:
                    # Match files ending specifically with '_summary_df.csv'
                    summary_files = [f for f in os.listdir(sbj_dir) if f.lower().endswith('_summary_df.csv')]
                    print(f"Matching files in {sbj}: {summary_files}")

                    for summary_file in summary_files:
                        file_path = os.path.join(sbj_dir, summary_file)
                        print(f"Reading file: {file_path}")
                        df = pd.read_csv(file_path)
                        if df.empty:
                            print(f"File {file_path} is empty.")
                        else:
                            print(f"Data in {summary_file}:\n{df.head()}")
                            sbj_data[summary_file] = df
                except Exception as e:
                    print(f"Error processing folder {sbj}: {e}")

                all_data[sbj] = sbj_data
        else:
            print("No subdirectories found. Processing files directly...")
            summary_files = [f for f in items if f.lower().endswith('_summary_df.csv')]
            print(f"Files found: {summary_files}")

            for summary_file in summary_files:
                file_path = os.path.join(base_dir, summary_file)
                print(f"Reading file: {file_path}")
                try:
                    df = pd.read_csv(file_path)
                    if df.empty:
                        print(f"File {file_path} is empty.")
                    else:
                        print(f"Data in {summary_file}:\n{df.head()}")
                        all_data[summary_file] = df
                except Exception as e:
                    print(f"Error reading file {summary_file}: {e}")

    except Exception as e:
        print(f"Error accessing base directory: {e}")

    return all_data


def extract_X_Y(data, target_column="BR"):

    results = {}

    for sbj, files in data.items():
        sbj_results = {}

        for file_name, df in files.items():
            # Check if the target column exists in the DataFrame
            if target_column in df.columns:
                # Separate X (features) and Y (target)
                X = df.drop(columns=[target_column])
                Y = df[target_column]

                # Handle timestamp column if it exists
                if 'DateTime' in X.columns:  # Replace 'DateTime' with the actual timestamp column name
                    print(f"Processing timestamp in {file_name}")
                    X['DateTime'] = pd.to_datetime(X['DateTime'], format='%d/%m/%Y %H:%M:%S.%f')
                    X['DateTime'] = (X['DateTime'] - X['DateTime'].min()).dt.total_seconds()

                # Keep only numeric columns
                X = X.select_dtypes(include=["number"])

                sbj_results[file_name] = (X, Y)
            else:
                print(f"Warning: Column '{target_column}' not found in {file_name}")

        results[sbj] = sbj_results

    return results


def split_train_test(data, test_size=0.2, random_state=42):

    # Initialize a dictionary to store the results
    split_data = {}

    # Iterate through all subjects
    for sbj, files in data.items():
        sbj_splits = {}

        # Iterate through all files for the current subject
        for file_name, (X, Y) in files.items():
            try:
                # Split the data into training and testing sets
                X_train, X_test, Y_train, Y_test = train_test_split(
                    X, Y, test_size=test_size, random_state=random_state
                )
                sbj_splits[file_name] = (X_train, X_test, Y_train, Y_test)
            except Exception as e:
                print(f"Error processing {file_name} for {sbj}: {e}")

        # Store splits for the current subject
        split_data[sbj] = sbj_splits

    return split_data


def adjust_test_data(train_test_splits):

    adjusted_splits = {}

    for sbj, files in train_test_splits.items():
        sbj_adjusted = {}

        for file_name, (X_train, X_test, Y_train, Y_test) in files.items():
            try:
                # Extract the last two rows as future samples
                futureSample_X = X_test.iloc[-2:]
                futureSample_Y = Y_test.iloc[-2:]

                # Remove the last two rows from X_test and Y_test
                X_test = X_test.iloc[:-2]
                Y_test = Y_test.iloc[:-2].reset_index(drop=True)

                # Store adjusted data and future samples
                sbj_adjusted[file_name] = {
                    "X_train": X_train,
                    "X_test": X_test,
                    "Y_train": Y_train,
                    "Y_test": Y_test,
                    "futureSample_X": futureSample_X,
                    "futureSample_Y": futureSample_Y,
                }
            except Exception as e:
                print(f"Error adjusting data for {file_name} in {sbj}: {e}")

        adjusted_splits[sbj] = sbj_adjusted

    return adjusted_splits

def fit_linear_models(adjusted_splits):

    models = {}

    for sbj, files in adjusted_splits.items():
        sbj_models = {}

        for file_name, data in files.items():
            try:
                # Extract training data
                X_train = data["X_train"]
                Y_train = data["Y_train"]

                # Create and fit the model
                model = LinearRegression()
                model.fit(X_train, Y_train)

                # Store the trained model
                sbj_models[file_name] = model
            except Exception as e:
                print(f"Error fitting model for {file_name} in {sbj}: {e}")

        models[sbj] = sbj_models

    return models

def compute_metrics(trained_models, adjusted_splits):

    metrics_results = {}

    for sbj, files in trained_models.items():
        sbj_metrics = {}

        for file_name, model in files.items():
            try:
                # Retrieve the test data for this file
                data = adjusted_splits[sbj][file_name]
                X_test = data["X_test"]
                Y_test = data["Y_test"]

                # Make predictions
                y_pred = model.predict(X_test)

                # Compute metrics
                mean_absolute_error_test = metrics.mean_absolute_error(Y_test, y_pred)
                r2_test = metrics.r2_score(Y_test, y_pred)

                def MAPE(y, y_predict):
                    return np.mean(np.abs((y - y_predict) / y)) * 100

                mape_test = MAPE(Y_test, y_pred)

                # Store metrics for this file
                sbj_metrics[file_name] = {
                    "MAE": mean_absolute_error_test,
                    "R²": r2_test,
                    "MAPE": mape_test,
                }

            except Exception as e:
                print(f"Error computing metrics for {file_name} in {sbj}: {e}")

        metrics_results[sbj] = sbj_metrics

    return metrics_results

def compute_metrics_table(trained_models, adjusted_splits):

    results = []

    for sbj, files in trained_models.items():
        for file_name, model in files.items():
            try:
                # Retrieve the test data for this file
                data = adjusted_splits[sbj][file_name]
                X_test = data["X_test"]
                Y_test = data["Y_test"]

                # Make predictions
                y_pred = model.predict(X_test)

                # Compute metrics
                mean_absolute_error_test = metrics.mean_absolute_error(Y_test, y_pred)
                r2_test = metrics.r2_score(Y_test, y_pred)

                def MAPE(y, y_predict):
                    return np.mean(np.abs((y - y_predict) / y)) * 100

                mape_test = MAPE(Y_test, y_pred)

                # Append results as a dictionary
                results.append({
                    "Subject": sbj,
                    "File": file_name,
                    "MAE": mean_absolute_error_test,
                    "R²": r2_test,
                    "MAPE": mape_test
                })

            except Exception as e:
                print(f"Error computing metrics for {file_name} in {sbj}: {e}")

    # Convert the results to a DataFrame
    return pd.DataFrame(results)


def plot_metrics_separately(metrics_table):
    # Separate metrics for plotting
    metrics = ["MAE", "R²", "MAPE"]

    for metric in metrics:
        # Group by Subject/File or aggregate as needed (e.g., mean across all files)
        metric_means = metrics_table.groupby("Subject")[metric].mean()  # Example: Aggregated by Subject

        # Create a bar chart for the metric
        plt.figure(figsize=(8, 6))
        metric_means.plot(kind="bar", color="skyblue", edgecolor="black")

        # Add labels and title
        plt.title(f"Average {metric} Across Subjects", fontsize=14)
        plt.ylabel(metric, fontsize=12)
        plt.xlabel("Subject", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        # Show plot
        plt.show()

def main():
    # Step 1: Gather all the seated data
    data = gather_seated_data()

    # Step 2: Extract X (features) and Y (target) for each file
    X_Y_data = extract_X_Y(data)

    # Step 3: Split into train-test datasets
    train_test_splits = split_train_test(X_Y_data)

    # Step 4: Adjust the test data to include future samples
    adjusted_splits = adjust_test_data(train_test_splits)

    # Step 5: Fit linear models for each file
    trained_models = fit_linear_models(adjusted_splits)

    # Step 6: Compute metrics and return a table
    metrics_table = compute_metrics_table(trained_models, adjusted_splits)

    print(metrics_table)

    # Save the table to the specified directory
    metrics_table.to_csv(r"D:\aria-seated\metrics_results.csv", index=False)

    plot_metrics_separately(metrics_table)

    # Example: Access a trained model and make predictions
    subject = "S12"
    file_name = "L0_summary.csv"
    if subject in trained_models and file_name in trained_models[subject]:
        model = trained_models[subject][file_name]
        print(f"Model coefficients for {subject} - {file_name}: {model.coef_}")
        print(f"Model intercept for {subject} - {file_name}: {model.intercept_}")

# Run the main function
if __name__ == "__main__":
    main()
