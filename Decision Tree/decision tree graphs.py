import pandas as pd
import matplotlib.pyplot as plt

# Load your data
file_path = r"D:\aria-seated\output_files\decision_tree_results.csv"
df = pd.read_csv(file_path)

# Parameters for chunking
chunk_size = 30  # Number of rows per chunk
num_chunks = (len(df) + chunk_size - 1) // chunk_size  # Calculate total chunks

# Plotting metrics in chunks
for i in range(num_chunks):
    start_idx = i * chunk_size
    end_idx = min((i + 1) * chunk_size, len(df))
    chunk = df.iloc[start_idx:end_idx]

    # Prepare data for plotting
    x_labels = chunk.iloc[:, 0]  # First column (file names) for x-axis
    metrics = chunk.iloc[:, 1:]  # Remaining columns for metrics

    # Plot each metric for the chunk
    fig, ax = plt.subplots(figsize=(12, 6))

    for column in metrics.columns:
        ax.plot(x_labels, metrics[column], label=column, marker='o')

    # Adding plot details
    ax.set_title(f"Metrics for Files {start_idx + 1} to {end_idx}")
    ax.set_xlabel("Files")
    ax.set_ylabel("Metric Values")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.7)
    plt.xticks(rotation=45, ha="right")

    # Save the plot
    output_plot_path = rf"D:\aria-seated\output_files\metrics_chunk_{i + 1}.png"
    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=300)
    plt.close(fig)
    print(f"Plot saved: {output_plot_path}")

