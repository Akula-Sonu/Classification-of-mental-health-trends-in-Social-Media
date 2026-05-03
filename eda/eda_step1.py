# from pyspark.sql import SparkSession
# from pyspark.sql.functions import col

# # Initialize Spark
# spark = SparkSession.builder.appName("EDA Step 1").getOrCreate()

# # Path to cleaned data
# cleaned_parquet = r"C:\Users\sonu\Desktop\big_data\project\processed_data\parquet_fixed"

# # Load cleaned dataset
# df = spark.read.parquet(cleaned_parquet)

# print("✅ Cleaned dataset loaded")
# print("Number of rows:", df.count())
# print("Columns:", df.columns)

# # Show sample rows
# df.show(5, truncate=False)

# # Count posts per label
# print("\n📊 Posts per label:")
# df.groupBy("label").count().orderBy(col("count").desc()).show()

# # Stop Spark
# spark.stop()

from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# Initialize Spark
spark = SparkSession.builder.appName("EDA Step 1").getOrCreate()

# Path to cleaned data
cleaned_parquet = r"C:\Users\sonu\Desktop\big_data\project\processed_data\parquet_fixed"

# Load cleaned dataset
df = spark.read.parquet(cleaned_parquet)

print("✅ Cleaned dataset loaded")
print("Number of rows:", df.count())
print("Columns:", df.columns)

# Show sample rows
df.show(5, truncate=False)

# Count posts per label
print("\n📊 Posts per label:")
df.groupBy("label").count().orderBy(col("count").desc()).show()

# Optional: ignore "unknown" and get only valid labels
df_valid = df.filter(col("label") != "unknown")
print("\n📊 Posts per valid label only:")
df_valid.groupBy("label").count().orderBy(col("count").desc()).show()

# Stop Spark
spark.stop()
