import pandas as pd
import glob
import os

# Path to labeled data folder
folder_path = r"C:\Users\sonu\Desktop\big data\Original Reddit Data\Labelled Data"

# Load all CSVs
all_files = glob.glob(os.path.join(folder_path, "*.csv"))

dfs = []
for file in all_files:
    print(f"Loading: {file}")
    df = pd.read_csv(file)
    dfs.append(df)

# Combine all into one DataFrame
final_df = pd.concat(dfs, ignore_index=True)

# Show sample data
print(final_df.head())
print("\nShape of final dataset:", final_df.shape)
print("\nColumns:", final_df.columns.tolist())
