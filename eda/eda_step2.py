# eda_step2.py
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, length
import matplotlib.pyplot as plt
import pandas as pd

# Initialize Spark
spark = SparkSession.builder.appName("EDA Step 2").getOrCreate()

# Path to cleaned data
fixed_parquet = r"C:\Users\sonu\Desktop\big_data\project\processed_data\parquet_fixed"

# Load cleaned dataset
df = spark.read.parquet(fixed_parquet)
print("✅ Cleaned dataset loaded")
print("Number of rows:", df.count())

# Keep only valid labels (drop 'unknown')
df_valid = df.filter(col("label") != "unknown")

# -------------------------
# 1. Distribution of labels
# -------------------------
label_counts = df_valid.groupBy("label").count().orderBy(col("count").desc())
label_pd = label_counts.toPandas()

plt.figure(figsize=(8,6))
plt.bar(label_pd["label"], label_pd["count"], color="skyblue")
plt.title("Posts per Label")
plt.xlabel("Label")
plt.ylabel("Count")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("eda_label_distribution.png")
print("📊 Saved: eda_label_distribution.png")

# -------------------------
# 2. Text length per label
# -------------------------
df_textlen = df_valid.withColumn("text_length", length(col("selftext")))

# Sample 50k rows for plotting
df_sample = df_textlen.sample(withReplacement=False, fraction=0.01, seed=42).limit(50000)
sample_pd = df_sample.select("label", "text_length").toPandas()

plt.figure(figsize=(8,6))
sample_pd.boxplot(column="text_length", by="label", grid=False, rot=30)
plt.title("Distribution of Selftext Length by Label")
plt.suptitle("")
plt.xlabel("Label")
plt.ylabel("Text Length (characters)")
plt.tight_layout()
plt.savefig("eda_text_length.png")
print("📊 Saved: eda_text_length.png")

# -------------------------
# 3. (Optional) Yearly trends
# -------------------------
# Extract year from filename
from pyspark.sql.functions import regexp_extract

df_year = df_valid.withColumn("year", regexp_extract("filename", r'/(\d{4})/', 1))
year_counts = df_year.groupBy("year", "label").count().orderBy("year", "label")
year_pd = year_counts.toPandas()

if not year_pd.empty:
    plt.figure(figsize=(10,6))
    for label in year_pd["label"].unique():
        subset = year_pd[year_pd["label"] == label]
        plt.plot(subset["year"], subset["count"], marker="o", label=label)
    plt.title("Posts per Label by Year")
    plt.xlabel("Year")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig("eda_yearly_trends.png")
    print("📊 Saved: eda_yearly_trends.png")

# Stop Spark
spark.stop()
print("✅ EDA Step 2 finished")
