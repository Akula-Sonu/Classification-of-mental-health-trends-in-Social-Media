import os
import pandas as pd
import matplotlib.pyplot as plt
import re

# -------------------------
# 1. Define raw data path
# -------------------------
raw_data_root = r"C:\Users\sonu\Desktop\big_data\project\Original Reddit Data\raw data"

# -------------------------
# 2. Collect all CSV files
# -------------------------
all_files = []
for year in ["2019", "2020", "2021", "2022"]:
    year_path = os.path.join(raw_data_root, year)
    for month in os.listdir(year_path):
        month_path = os.path.join(year_path, month)
        if os.path.isdir(month_path):
            for fname in os.listdir(month_path):
                if fname.endswith(".csv"):
                    all_files.append(os.path.join(month_path, fname))

if not all_files:
    print("⚠️ No CSV files found. Check raw data folder.")
    exit()

print(f"✅ Found {len(all_files)} files")

# -------------------------
# 3. Extract year, month, label from filename
# -------------------------
records = []
for filepath in all_files:
    fname = os.path.basename(filepath).lower()
    match = re.match(r"(anx|dep|lone|mh|sw)([a-z]+)(\d{2})\.csv", fname)
    if match:
        label, month, year_suffix = match.groups()
        year = "20" + year_suffix
        records.append((year, month[:3], label))

df = pd.DataFrame(records, columns=["year", "month", "label"])

# -------------------------
# 4. Aggregate counts
# -------------------------
df["count"] = 1
monthly_counts = df.groupby(["year", "month", "label"]).sum().reset_index()

# Month order mapping
month_order = ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]
month_map = {m: i+1 for i, m in enumerate(month_order)}
monthly_counts["month_num"] = monthly_counts["month"].map(month_map)

# Sort correctly
monthly_counts = monthly_counts.sort_values(["year", "month_num"])

# -------------------------
# 5. Plot per year
# -------------------------
for year in ["2019","2020","2021","2022"]:
    subset = monthly_counts[monthly_counts["year"] == year]
    if subset.empty:
        continue

    plt.figure(figsize=(10,6))
    for label in subset["label"].unique():
        label_subset = subset[subset["label"] == label]
        plt.plot(label_subset["month_num"], label_subset["count"], marker="o", label=label)

    plt.xticks(range(1,13), month_order, rotation=45)
    plt.title(f"Monthly Trends in {year}")
    plt.xlabel("Month")
    plt.ylabel("Number of Posts")
    plt.legend()
    plt.tight_layout()
    out_file = f"monthly_trends_{year}.png"
    plt.savefig(out_file)
    plt.close()
    print(f"📊 Saved: {out_file}")

print("✅ Finished generating yearly monthly trend graphs")
