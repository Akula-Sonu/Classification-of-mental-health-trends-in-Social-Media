import os
import re
import pandas as pd
import matplotlib.pyplot as plt

# Root raw data folder
raw_data_dir = r"C:\Users\sonu\Desktop\big_data\project\Original Reddit Data\raw data"

# Month mapping (short → number)
month_map = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    "may": 5, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "oct": 10, "nov": 11, "dec": 12
}

# Regex for filenames like "anxapr19.csv"
pattern = re.compile(r"^(anx|dep|lone|mh|sw)([a-z]+)(\d{2})\.csv$")

records = []

# Walk through all folders
for root, dirs, files in os.walk(raw_data_dir):
    for file in files:
        match = pattern.match(file.lower())
        if match:
            label, month_abbr, year_abbr = match.groups()
            year = 2000 + int(year_abbr)  # e.g. 19 → 2019
            month_num = month_map.get(month_abbr, None)

            if month_num is None:
                print(f"⚠️ Unknown month in file: {file}")
                continue

            file_path = os.path.join(root, file)

            # Count number of rows (posts)
            try:
                row_count = sum(1 for _ in open(file_path, encoding="utf-8", errors="ignore")) - 1
            except Exception:
                row_count = 0

            records.append({
                "label": label,
                "year": year,
                "month_num": month_num,
                "count": row_count
            })

# Convert to DataFrame
df = pd.DataFrame(records)

if df.empty:
    print("⚠️ No files matched. Check regex or folder structure.")
    exit()

# Sort by date
df = df.sort_values(["year", "month_num"])

# -------------------------
# Plot: separate figure per year
# -------------------------
for year in sorted(df["year"].unique()):
    yearly_df = df[df["year"] == year]

    plt.figure(figsize=(12,6))
    for label in yearly_df["label"].unique():
        subset = yearly_df[yearly_df["label"] == label]
        plt.plot(
            subset["month_num"],
            subset["count"],
            marker="o",
            label=label
        )

    plt.title(f"Monthly Trends per Label ({year})")
    plt.xlabel("Month")
    plt.ylabel("Number of Posts")
    plt.xticks(ticks=range(1,13), labels=[
        "Jan","Feb","Mar","Apr","May","Jun",
        "Jul","Aug","Sep","Oct","Nov","Dec"
    ])
    plt.legend(title="Label")
    plt.tight_layout()
    out_path = f"monthly_trends_{year}.png"
    plt.savefig(out_path)
    plt.show()
    print(f"✅ Graph saved: {out_path}")
