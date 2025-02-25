import os
import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
file_path = "D:\\aria-seated\\Results\\loocv_metrics_results_DecisionTree_with_fixed_features_and_no_bp.csv"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File not found: {file_path}")

df = pd.read_csv(file_path)

# Ensure numeric columns are correctly parsed
numeric_columns = ["MAE", "MSE", "RMSE", "R^2", "MAPE"]
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Compute mean and median values
mean_values = df[numeric_columns].mean()
median_values = df[numeric_columns].median()

# Plot mean values
plt.figure(figsize=(10, 5))
plt.bar(mean_values.index, mean_values.values, color='skyblue')
plt.title("Mean of LOOCV Metrics")
plt.ylabel("Value")
plt.xlabel("Metrics")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Plot median values
plt.figure(figsize=(10, 5))
plt.bar(median_values.index, median_values.values, color='salmon')
plt.title("Median of LOOCV Metrics")
plt.ylabel("Value")
plt.xlabel("Metrics")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

print("Plots generated successfully.")