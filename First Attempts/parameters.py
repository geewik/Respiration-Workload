import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer  # Ensure this is imported
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt


# Function to gather seated data
def gather_seated_data(base_dir=r"D:\aria-seated"):
    all_data = {}

    try:
        subjects = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
        for sbj in subjects:
            sbj_dir = os.path.join(base_dir, sbj)
            summary_files = [f for f in os.listdir(sbj_dir) if f.lower().endswith('_summary_df.csv')]

            sbj_data = {}
            for summary_file in summary_files:
                file_path = os.path.join(sbj_dir, summary_file)
                df = pd.read_csv(file_path)
                sbj_data[summary_file] = df

            all_data[sbj] = sbj_data
    except Exception as e:
        print(f"Error accessing base directory: {e}")

    return all_data


# Function to segment breathing rate
def segment_breathing_rate(data):
    for subject, files in data.items():
        for filename, df in files.items():
            if isinstance(df, pd.DataFrame) and "BR" in df.columns:
                df["Segmented_BR"] = pd.cut(
                    df["BR"],
                    bins=[0, 10, 20, np.inf],
                    labels=["Low", "Medium", "High"],
                    include_lowest=True
                )
                data[subject][filename] = df
    return data


# Function to prepare data for modeling
def prepare_segmented_data_for_modeling(data, segmented_col="Segmented_BR"):
    X, Y = [], []

    for subject, files in data.items():
        for filename, df in files.items():
            if isinstance(df, pd.DataFrame) and segmented_col in df.columns:
                X.append(df.drop(columns=[segmented_col, "BR"], errors="ignore"))
                Y.append(df[segmented_col])

    if X and Y:
        X = pd.concat(X, ignore_index=True)
        Y = pd.concat(Y, ignore_index=True)
        return X, Y
    else:
        print("No valid data found.")
        return None, None


# Function to preprocess data by handling missing values
def preprocess_data(X):
    X = X.dropna(axis=1, how='all')  # Drop columns with all NaN values
    imputer = SimpleImputer(strategy="mean")  # Impute missing values with mean
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    return X_imputed


# Function to perform linear regression
def perform_linear_regression(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, Y_train)

    Y_pred = model.predict(X_test)

    print("Linear Regression Results:")
    print(f"Mean Squared Error: {mean_squared_error(Y_test, Y_pred):.4f}")
    print(f"Mean Absolute Error: {mean_absolute_error(Y_test, Y_pred):.4f}")
    print(f"R^2 Score: {r2_score(Y_test, Y_pred):.4f}")

    return model


# Main function
def main():
    base_dir = r"D:\aria-seated"
    all_data = gather_seated_data(base_dir)

    # Segment breathing rate
    all_data = segment_breathing_rate(all_data)

    # Prepare data for modeling
    X, Y = prepare_segmented_data_for_modeling(all_data, segmented_col="Segmented_BR")

    if X is not None and Y is not None:
        # Preprocess features (X) to handle missing values
        X = preprocess_data(X)

        # Perform linear regression
        print(f"Features (X):\n{X.head()}")
        print(f"Target (Y):\n{Y.head()}")
        model = perform_linear_regression(X, Y)
    else:
        print("Data preparation failed. Cannot proceed with linear regression.")


if __name__ == "__main__":
    main()