import pandas as pd
import os

# Path where your CSV files are stored
data_path = r"C:\Users\sonu\Desktop\big data\Original Reddit Data\Labelled Data"

# List of CSV files to combine
files = ["LD DA 1.csv", "LD EL1.csv", "LD PF1.csv", "LD TS 1.csv"]

# Load and combine
df_list = []
for file in files:
    file_path = os.path.join(data_path, file)
    df = pd.read_csv(file_path)
    df["source_file"] = file  # Add column to track origin
    df_list.append(df)

combined_df = pd.concat(df_list, ignore_index=True)

# Save combined dataset
output_path = r"C:\Users\sonu\Desktop\big data\reddit_mental_health_project\combined_dataset.csv"
combined_df.to_csv(output_path, index=False)

print(f"✅ Combined dataset saved at: {output_path}")
print(combined_df.head())
