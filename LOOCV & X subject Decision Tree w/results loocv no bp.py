import pandas as pd
import matplotlib.pyplot as plt

# Define file paths
file_paths = {
    "With BP": r"D:\aria-seated\Results\loocv_metrics_results_w_bp_2.csv",
    "No BP": r"D:\aria-seated\Results\loocv_metrics_results_DT_no_bp.csv"
}

# Define columns
columns = ["MAE", "MSE", "RMSE", "R^2", "MAPE"]

# Iterate through both files
for label, file_path in file_paths.items():
    # Load CSV file
    df = pd.read_csv(file_path)

    # Select only the numerical columns
    df_numeric = df[columns]

    # Compute mean and median
    mean_values = df_numeric.mean()
    median_values = df_numeric.median()

    # Create a DataFrame for statistics
    summary_df = pd.DataFrame({"Mean": mean_values, "Median": median_values})

    # Display the table
    print(f"\nSummary Statistics ({label}):")
    print(summary_df)

    # Save the table as a CSV file
    summary_df.to_csv(f"metrics_summary_{label.replace(' ', '_')}.csv")

    # Plot Mean and Median as a Bar Chart
    plt.figure(figsize=(10, 5))
    summary_df.plot(kind='bar', figsize=(10, 6))
    plt.title(f"Mean and Median of Evaluation Metrics ({label})")
    plt.xlabel("Metrics")
    plt.ylabel("Values")
    plt.xticks(rotation=45)
    plt.legend(["Mean", "Median"])
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Save and show the bar chart
    plt.savefig(f"metrics_summary_{label.replace(' ', '_')}.png", dpi=300)
    plt.show()

    # Chunk the data into 4 parts (each with ~30 instances)
    chunk_size = 30
    num_chunks = (len(df_numeric) + chunk_size - 1) // chunk_size  # Calculate number of chunks

    # Plot multiple line charts in chunks of 30
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(df_numeric))  # Ensure last chunk captures all remaining values

        plt.figure(figsize=(12, 6))

        for col in columns:
            plt.plot(df_numeric.index[start_idx:end_idx], df_numeric[col][start_idx:end_idx], label=col)

        plt.title(f"Line Chart of Instances {start_idx} to {end_idx - 1} ({label})")
        plt.xlabel("Instance (Row Index)")
        plt.ylabel("Metric Values")
        plt.legend()
        plt.grid(True)

        # Save each chunk as a separate image
        plt.savefig(f"metrics_line_chart_{label.replace(' ', '_')}_{i + 1}.png", dpi=300)
        plt.show()
