from pyspark.sql import SparkSession
from pyspark.sql.functions import regexp_extract, col, lower, when, lit
import matplotlib.pyplot as plt
import pandas as pd
import os

# -------------------------
# 1. Initialize Spark
# -------------------------
spark = SparkSession.builder.appName("EDA Step 3 - Monthly Trends").getOrCreate()

# Path to fixed dataset
fixed_parquet = r"C:\Users\sonu\Desktop\big_data\project\processed_data\parquet_fixed"

# Load cleaned dataset
df = spark.read.parquet(fixed_parquet)
print("✅ Cleaned dataset loaded")
print("Number of rows:", df.count())

# Keep only valid labels
df = df.filter(col("label") != "unknown")

# -------------------------
# 2. Extract year and month from filename robustly
# -------------------------
df = df.withColumn("filename_lower", lower(col("filename")))

# Extract year (mandatory)
df = df.withColumn("year", regexp_extract("filename_lower", r'/(\d{4})/', 1))

# Extract month (optional)
df = df.withColumn("month", regexp_extract("filename_lower", r'/([a-z]{3})/', 1))

# Standardize month names
month_dict = {
    'jan':'Jan','feb':'Feb','mar':'Mar','apr':'Apr','may':'May','jun':'Jun',
    'jul':'Jul','aug':'Aug','sep':'Sep','oct':'Oct','nov':'Nov','dec':'Dec'
}

for k, v in month_dict.items():
    df = df.withColumn("month", when(col("month")==k, v).otherwise(col("month")))

# Fill missing months with 'Jan' as a fallback (you can change this strategy)
df = df.withColumn("month", when(col("month").isNull(), lit("Jan")).otherwise(col("month")))

# -------------------------
# 3. Aggregate counts per label per month
# -------------------------
monthly_trends = df.groupBy("year", "month", "label").count().orderBy("year", "month", "label")

# -------------------------
# 4. Save aggregated CSV
# -------------------------
output_csv = r"C:\Users\sonu\Desktop\big_data\project\processed_data\monthly_trends_csv"
if not os.path.exists(output_csv):
    os.makedirs(output_csv)

monthly_trends.write.mode("overwrite").csv(output_csv, header=True)
print(f"✅ Aggregated monthly trends saved to: {output_csv}")

# -------------------------
# 5. Convert all parts to Pandas
# -------------------------
all_parts = [os.path.join(output_csv, f) for f in os.listdir(output_csv) if f.endswith(".csv")]

dfs = []
for part in all_parts:
    try:
        df_part = pd.read_csv(part, encoding='utf-8')
    except UnicodeDecodeError:
        df_part = pd.read_csv(part, encoding='ISO-8859-1')
    dfs.append(df_part)

monthly_pd = pd.concat(dfs, ignore_index=True)

# Strip column names
monthly_pd.columns = [c.strip().replace('"','') for c in monthly_pd.columns]

# -------------------------
# 6. Convert month to datetime safely
# -------------------------
month_map = {'Jan':1,'Feb':2,'Mar':3,'Apr':4,'May':5,'Jun':6,
             'Jul':7,'Aug':8,'Sep':9,'Oct':10,'Nov':11,'Dec':12}

monthly_pd['month_num'] = monthly_pd['month'].map(month_map)
monthly_pd = monthly_pd[monthly_pd['month_num'].notnull()]  # remove still missing months

monthly_pd['year_month_dt'] = pd.to_datetime(monthly_pd['year'].astype(int).astype(str) + '-' + monthly_pd['month_num'].astype(int).astype(str))

monthly_pd = monthly_pd.sort_values('year_month_dt')
monthly_pd['month_year'] = monthly_pd['month'] + '-' + monthly_pd['year'].astype(str)

# -------------------------
# 7. Plot
# -------------------------
plt.figure(figsize=(16,6))
for label in monthly_pd['label'].unique():
    subset = monthly_pd[monthly_pd['label']==label]
    plt.plot(subset['year_month_dt'], subset['count'], marker='o', label=label)

plt.title('Monthly Trends per Label')
plt.xlabel('Month-Year')
plt.ylabel('Number of Posts')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig('eda_monthly_trends.png')
print("📊 Saved: eda_monthly_trends.png")

# -------------------------
# 8. Stop Spark
# -------------------------
spark.stop()
print("✅ EDA Step 3 finished")







from pyspark.sql import SparkSession
from pyspark.sql.functions import regexp_extract, col, upper, substring, concat, when
import matplotlib.pyplot as plt
import pandas as pd
import os

# -------------------------
# 1. Initialize Spark
# -------------------------
spark = SparkSession.builder.appName("EDA Step 3 - Monthly Trends").getOrCreate()

# Path to fixed dataset
fixed_parquet = r"C:\Users\sonu\Desktop\big_data\project\processed_data\parquet_fixed"

# Load cleaned dataset
df = spark.read.parquet(fixed_parquet)
print("✅ Cleaned dataset loaded")
print("Number of rows:", df.count())

# Keep only valid labels
df = df.filter(col("label") != "unknown")

# -------------------------
# 2. Extract year and month from filename
# -------------------------
# Assumes filenames like .../2020/apr/anxapr20.csv
df = df.withColumn("year", regexp_extract("filename", r'/(\d{4})/', 1))
df = df.withColumn("month", regexp_extract("filename", r'/([a-z]{3})/', 1))  # lowercase month

# Capitalize first letter of month
df = df.withColumn(
    "month",
    when(col("month").isNotNull(),
         concat(
             upper(substring(col("month"), 1, 1)),
             substring(col("month"), 2, 2)
         )
    )
)

# -------------------------
# 3. Aggregate counts per label per month
# -------------------------
monthly_trends = df.groupBy("year", "month", "label").count().orderBy("year", "month", "label")

# -------------------------
# 4. Save aggregated CSV
# -------------------------
output_csv = r"C:\Users\sonu\Desktop\big_data\project\processed_data\monthly_trends_csv"
if not os.path.exists(output_csv):
    os.makedirs(output_csv)

monthly_trends.write.mode("overwrite").csv(output_csv, header=True)
print(f"✅ Aggregated monthly trends saved to: {output_csv}")

# -------------------------
# 5. Convert all parts to Pandas for plotting
# -------------------------
all_parts = [os.path.join(output_csv, f) for f in os.listdir(output_csv) if f.endswith(".csv")]

dfs = []
for part in all_parts:
    try:
        df_part = pd.read_csv(part, encoding='utf-8')
    except UnicodeDecodeError:
        df_part = pd.read_csv(part, encoding='ISO-8859-1')
    dfs.append(df_part)

monthly_pd = pd.concat(dfs, ignore_index=True)

# Strip column names
monthly_pd.columns = [c.strip().replace('"','') for c in monthly_pd.columns]

# -------------------------
# 6. Map month to numeric for sorting
# -------------------------
month_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
month_map = {m:i+1 for i,m in enumerate(month_order)}
monthly_pd["month_num"] = monthly_pd["month"].map(month_map)

# Drop rows with NaN month
monthly_pd = monthly_pd.dropna(subset=["month_num"])

# Create datetime column for proper ordering
monthly_pd['year_month_dt'] = pd.to_datetime(monthly_pd['year'].astype(int).astype(str) + '-' + monthly_pd['month_num'].astype(int).astype(str))

# Sort by datetime
monthly_pd = monthly_pd.sort_values(['year_month_dt', 'label'])

# -------------------------
# 7. Plot monthly trends
# -------------------------
plt.figure(figsize=(15,6))
for label in monthly_pd["label"].unique():
    subset = monthly_pd[monthly_pd["label"] == label]
    plt.plot(subset['year_month_dt'], subset["count"], marker="o", label=label)

plt.title("Monthly Trends per Label (2019-2022)")
plt.xlabel("Month-Year")
plt.ylabel("Number of Posts")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig("eda_monthly_trends.png")
print("📊 Saved: eda_monthly_trends.png")

# -------------------------
# 8. Stop Spark
# -------------------------
spark.stop()
print("✅ EDA Step 3 finished")



# from pyspark.sql import SparkSession
# from pyspark.sql.functions import col, regexp_extract
# import pandas as pd
# import matplotlib.pyplot as plt
# import os
# from itertools import product

# # -------------------------
# # 1. Initialize Spark
# # -------------------------
# spark = SparkSession.builder.appName("EDA Step 3 - Monthly Trends").getOrCreate()

# # Path to fixed dataset
# fixed_parquet = r"C:\Users\sonu\Desktop\big_data\project\processed_data\parquet_fixed"

# # Load cleaned dataset
# df = spark.read.parquet(fixed_parquet)
# print("✅ Cleaned dataset loaded")
# print("Number of rows:", df.count())

# # Keep only valid labels
# df = df.filter(col("label") != "unknown")

# # -------------------------
# # 2. Extract year and month from filename
# # -------------------------
# # Filenames like: anxapr19, depjun20, etc.
# # We'll extract 3-letter month and last 2 digits of year
# df = df.withColumn("month", regexp_extract("filename", r'([a-z]{3})\d{2}', 1))
# df = df.withColumn("year_suffix", regexp_extract("filename", r'([a-z]{3})(\d{2})', 2))
# # Convert year suffix to full year
# df = df.withColumn("year", (col("year_suffix").cast("int") + 2000).cast("string"))

# # Capitalize month
# df = df.withColumn("month", col("month").substr(1,1).upper() + col("month").substr(2,2))

# # -------------------------
# # 3. Aggregate counts per label per month
# # -------------------------
# monthly_trends = df.groupBy("year", "month", "label").count().orderBy("year", "month", "label")

# # -------------------------
# # 4. Save aggregated CSV
# # -------------------------
# output_csv = r"C:\Users\sonu\Desktop\big_data\project\processed_data\monthly_trends_csv"
# if not os.path.exists(output_csv):
#     os.makedirs(output_csv)

# monthly_trends.write.mode("overwrite").csv(output_csv, header=True)
# print(f"✅ Aggregated monthly trends saved to: {output_csv}")

# # -------------------------
# # 5. Convert all CSV parts to Pandas
# # -------------------------
# all_parts = [os.path.join(output_csv, f) for f in os.listdir(output_csv) if f.endswith(".csv")]
# dfs = []
# for part in all_parts:
#     try:
#         df_part = pd.read_csv(part, encoding='utf-8')
#     except UnicodeDecodeError:
#         df_part = pd.read_csv(part, encoding='ISO-8859-1')
#     dfs.append(df_part)

# monthly_pd = pd.concat(dfs, ignore_index=True)
# monthly_pd.columns = [c.strip().replace('"','') for c in monthly_pd.columns]

# # -------------------------
# # 6. Fill missing months with 0
# # -------------------------
# years = monthly_pd['year'].unique()
# months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
# labels = monthly_pd['label'].unique()

# all_combinations = pd.DataFrame(list(product(years, months, labels)), columns=['year','month','label'])

# # Convert 'year' to string to match types
# monthly_pd['year'] = monthly_pd['year'].astype(str)
# all_combinations['year'] = all_combinations['year'].astype(str)

# monthly_pd = pd.merge(all_combinations, monthly_pd, on=['year','month','label'], how='left')
# monthly_pd["count"] = monthly_pd["count"].fillna(0)

# # -------------------------
# # 7. Sort month-year for plotting
# # -------------------------
# monthly_pd["month"] = pd.Categorical(monthly_pd["month"], categories=months, ordered=True)
# monthly_pd = monthly_pd.sort_values(["year","month"])
# monthly_pd["month_year"] = monthly_pd["month"].astype(str) + "-" + monthly_pd["year"].astype(str)

# # -------------------------
# # 8. Plot monthly trends
# # -------------------------
# plt.figure(figsize=(14,7))
# for label in monthly_pd["label"].unique():
#     subset = monthly_pd[monthly_pd["label"] == label]
#     plt.plot(subset["month_year"], subset["count"], marker="o", label=label)

# plt.title("Monthly Trends per Label (2019-2022)")
# plt.xlabel("Month-Year")
# plt.ylabel("Number of Posts")
# plt.xticks(rotation=45)
# plt.legend()
# plt.tight_layout()
# plt.savefig("eda_monthly_trends.png")
# print("📊 Saved: eda_monthly_trends.png")

# # -------------------------
# # 9. Stop Spark
# # -------------------------
# spark.stop()
# print("✅ EDA Step 3 finished")
