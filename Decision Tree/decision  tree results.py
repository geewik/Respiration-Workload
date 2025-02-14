import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# Load decision tree results data
decision_tree_data = r"D:\aria-seated\output_files\decision_tree_results.csv"
read_data = pd.read_csv(decision_tree_data)

# Calculate median and mean
median_col = read_data.median(numeric_only=True)
mean_col = read_data.mean(numeric_only=True)

# Create and save the median table
fig, ax = plt.subplots(figsize=(6, len(median_col) * 0.5))
ax.axis('tight')
ax.axis('off')

median_table_data = median_col.reset_index()
median_table_data.columns = ['Column', 'Median']
median_table = ax.table(cellText=median_table_data.values, colLabels=median_table_data.columns, loc='center', cellLoc='center')

# Style the median table
median_table.auto_set_font_size(False)
median_table.set_fontsize(10)
median_table.auto_set_column_width([0, 1])

# Save the median table as an image
output_median_image_path = r"D:\aria-seated\output_files\dt_median_table.png"
plt.savefig(output_median_image_path, bbox_inches='tight', dpi=300)
plt.close(fig)  # Close the median figure

# Create and save the mean table
fig, ax = plt.subplots(figsize=(6, len(mean_col) * 0.5))
ax.axis('tight')
ax.axis('off')

mean_table_data = mean_col.reset_index()
mean_table_data.columns = ['Column', 'Mean']
mean_table = ax.table(cellText=mean_table_data.values, colLabels=mean_table_data.columns, loc='center', cellLoc='center')

# Style the mean table
mean_table.auto_set_font_size(False)
mean_table.set_fontsize(10)
mean_table.auto_set_column_width([0, 1])

# Save the mean table as an image
output_mean_image_path = r"D:\aria-seated\output_files\dt_mean_table.png"
plt.savefig(output_mean_image_path, bbox_inches='tight', dpi=300)
plt.close(fig)  # Close the mean figure

print(f"Median table saved as: {output_median_image_path}")
print(f"Mean table saved as: {output_mean_image_path}")

