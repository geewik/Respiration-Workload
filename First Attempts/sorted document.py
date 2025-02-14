import pandas as pd

# Load the dataset
file_path = r'D:\aria-seated\output_files\S12_L0_chunked_with_labels.csv'
data = pd.read_csv(file_path)

# Identify all columns labeled "timestamp_interp"
timestamp_columns = [col for col in data.columns if 'timestamp_interp' in col]

# Function to sort each BR group

def sort_group(group):
    for timestamp_col in timestamp_columns:
        # Sort the group by the current timestamp column in descending order
        sorted_indices = group[timestamp_col].sort_values(ascending=False).index
        group = group.loc[sorted_indices].reset_index(drop=True)
    return group

# Apply the sorting function to each BR group
# Explicitly exclude 'BR' after grouping and re-add it later
data_sorted = (
    data.groupby('BR', group_keys=False)
        .apply(lambda group: sort_group(group.loc[:, group.columns != 'BR']).assign(BR=group['BR'].values))
)

# Save the result to a new file
output_file = r"D:\aria-seated\output_files\sorted_by_timestamp_and_BR.csv"
data_sorted.to_csv(output_file, index=False)

print(f"Data sorted and saved to {output_file}")


