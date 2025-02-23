import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# File path
csv_file = r"D:\aria-seated\Results\cross_subject_testing_results_with_features_bp.csv"

# Check if file exists
if not os.path.exists(csv_file):
    print(f"🚨 Error: File {csv_file} not found!")
    exit()

# Load data
df = pd.read_csv(csv_file)

# Convert "N/A" in R² column to NaN
df["R^2"] = pd.to_numeric(df["R^2"], errors="coerce")

# Set Seaborn theme
sns.set_style("whitegrid")

# Create output directory for plots
output_dir = r"D:\aria-seated\Results\Plots"
os.makedirs(output_dir, exist_ok=True)

# 📌 1️⃣ Boxplots for Metrics
plt.figure(figsize=(10, 6))
sns.boxplot(data=df[["MAE", "MSE", "RMSE", "MAPE"]])
plt.title("Distribution of Error Metrics Across Subjects")
plt.ylabel("Error Value")
plt.xticks(rotation=45)
plt.savefig(os.path.join(output_dir, "boxplot_metrics.png"))
plt.show()

# 📌 2️⃣ Bar Plots for Each Metric Per Subject
metrics = ["MAE", "MSE", "RMSE", "MAPE"]

for metric in metrics:
    plt.figure(figsize=(12, 6))
    sns.barplot(x=df["Subject"], y=df[metric], hue=df["Test_Type"], palette="coolwarm")
    plt.title(f"{metric} Across Subjects")
    plt.xlabel("Subject")
    plt.ylabel(metric)
    plt.xticks(rotation=45)
    plt.legend(title="Test Type")
    plt.savefig(os.path.join(output_dir, f"{metric}_barplot.png"))
    plt.show()

# 📌 3️⃣ Scatter Plot for R² Values (if applicable)
if df["R^2"].notna().sum() > 0:
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x=df["Subject"], y=df["R^2"], hue=df["Test_Type"], style=df["Test_Type"], s=100)
    plt.axhline(y=0, color="red", linestyle="--", label="R^2 = 0")
    plt.title("R^2 Scores Across Subjects")
    plt.xlabel("Subject")
    plt.ylabel("R^2 Score")
    plt.xticks(rotation=45)
    plt.legend(title="Test Type")
    plt.savefig(os.path.join(output_dir, "r2_scatter.png"))
    plt.show()
else:
    print("⚠️ No valid R² values to plot!")

print(f"✅ Plots saved in {output_dir}")
