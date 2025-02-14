import pandas as pd
import matplotlib.pyplot as plt

# File path
file_path = r"D:\aria-seated\Results\80-20_Cross-Subject_DecisionTree_wbp_2025-01-30_23-56-19.csv"

# Load CSV file
df = pd.read_csv(file_path)

# Select only the numerical columns (excluding 'Test_Type' and 'Subject')
columns = ["MAE", "MSE", "RMSE", "R^2", "MAPE"]
df_numeric = df[columns]

# Compute mean and median
mean_values = df_numeric.mean()
median_values = df_numeric.median()

# Create a DataFrame for statistics
summary_df = pd.DataFrame({"Mean": mean_values, "Median": median_values})

# Display the table
print("\nSummary Statistics:")
print(summary_df)

# Save the table as a CSV file
summary_df.to_csv("80-20_cross_subject_metrics_summary.csv")

# Plot Mean and Median as a Bar Chart
plt.figure(figsize=(10, 5))
summary_df.plot(kind='bar', figsize=(10, 6))
plt.title("Mean and Median of Evaluation Metrics (80-20 Cross-Subject Decision Tree WBP)")
plt.xlabel("Metrics")
plt.ylabel("Values")
plt.xticks(rotation=45)
plt.legend(["Mean", "Median"])
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Save and show the bar chart
plt.savefig("80-20_cross_subject_metrics_summary.png", dpi=300)
plt.show()

# Chunk the data into 2 parts (each with ~10 instances)
chunk_size = 10
num_chunks = (len(df_numeric) + chunk_size - 1) // chunk_size  # Calculate number of chunks

# Plot multiple line charts in chunks of 10
for i in range(num_chunks):
    start_idx = i * chunk_size
    end_idx = min((i + 1) * chunk_size, len(df_numeric))  # Ensure last chunk captures all remaining values

    plt.figure(figsize=(12, 6))

    for col in columns:
        plt.plot(df_numeric.index[start_idx:end_idx], df_numeric[col][start_idx:end_idx], marker='o', label=col)

    plt.title(f"Line Chart of Instances {start_idx} to {end_idx - 1} (80-20 Cross-Subject Decision Tree WBP)")
    plt.xlabel("Instance (Row Index)")
    plt.ylabel("Metric Values")
    plt.legend()
    plt.grid(True)

    # Save each chunk as a separate image
    plt.savefig(f"80-20_cross_subject_metrics_line_chart_{i + 1}.png", dpi=300)
    plt.show()
