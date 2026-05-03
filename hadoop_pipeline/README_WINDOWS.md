# Reddit Mental Health Analysis — Hadoop + Local ML (Windows)

This project uses **Hadoop (HDFS + optional Streaming)** for storage/aggregation and **local Python (VS Code)** for ML.
You will:
1) Train a text classifier using your **Labelled Data** (LD*.csv).
2) Run predictions on **raw monthly CSVs**.
3) Aggregate counts (locally or via Hadoop Streaming).
4) Generate simple charts.

---

## 0) Prereqs
- Windows 10/11
- Hadoop installed and `hdfs`/`hadoop` on PATH
- Python 3.9–3.11 installed (`py --version`)
- VS Code

⚠️ Paths with spaces work if quoted. If you can, move your data to a no-spaces folder like `C:\\data\\reddit`.

---

## 1) Create a Python venv and install deps

```bat
cd %USERPROFILE%\\Downloads
mkdir reddit_mental_health_project
cd reddit_mental_health_project

REM Copy the files from the zip here (or download directly).

py -m venv venv
venv\\Scripts\\activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 2) Train the classifier on Labelled Data

**Labelled folder (example):**
`C:\\Users\\sonu\\Desktop\\big data\\Original Reddit Data\\Labelled Data`

```bat
venv\\Scripts\\activate

python train_classifier.py ^
  --label_dir "C:\\Users\\sonu\\Desktop\\big data\\Original Reddit Data\\Labelled Data" ^
  --model_out model_reddit.joblib ^
  --report_out model_report.txt
```

Check `model_report.txt` for accuracy/precision/recall, and `model_reddit.joblib` is your saved model.

---

## 3) Predict on a raw month (example: 2019/Apr depression file)

```bat
venv\\Scripts\\activate

python predict_local.py ^
  --input_csv "C:\\Users\\sonu\\Desktop\\big data\\Original Reddit Data\\raw data\\2019\\apr\\depapr19.csv" ^
  --model model_reddit.joblib ^
  --out_csv predictions_depapr19.csv
```

Repeat for other files (anxapr19.csv, loneapr19.csv, etc.).

---

## 4A) Aggregate counts locally

```bat
python aggregate_counts.py --inputs predictions_*.csv --out counts.tsv
python generate_charts.py --counts_tsv counts.tsv --out_dir charts
```

Charts saved in `charts\\`:
- `overall_distribution.png`
- `monthly_trend.png`

---

## 4B) (Optional) Store/aggregate in Hadoop

**Put raw files in HDFS:**
```bat
hdfs dfs -mkdir -p /reddit/raw/2019/apr
hdfs dfs -put "C:\\Users\\sonu\\Desktop\\big data\\Original Reddit Data\\raw data\\2019\\apr\\*.csv" /reddit/raw/2019/apr
```

**Put predictions in HDFS:**
```bat
hdfs dfs -mkdir -p /reddit/predictions
hdfs dfs -put predictions_*.csv /reddit/predictions
```

**Run Hadoop Streaming to aggregate counts by month+label:**
```bat
set STREAMING_JAR=%HADOOP_HOME%\\share\\hadoop\\tools\\lib\\hadoop-streaming*.jar

hadoop jar "%STREAMING_JAR%" ^
  -D mapreduce.job.name="reddit_monthly_label_counts" ^
  -files mapper_counts.py,reducer_counts.py ^
  -input /reddit/predictions ^
  -output /reddit/agg_counts ^
  -mapper "python mapper_counts.py" ^
  -reducer "python reducer_counts.py"
```

**Fetch results:**
```bat
hdfs dfs -cat /reddit/agg_counts/part-* > counts_from_hadoop.tsv
```

You can now visualize `counts_from_hadoop.tsv` with `generate_charts.py`:
```bat
python generate_charts.py --counts_tsv counts_from_hadoop.tsv --out_dir charts_hdfs
```

---

## 5) Tools chosen — What & Why

- **Hadoop (HDFS)** — Reliable storage for lots of monthly CSVs; easy to scale later.
- **Hadoop Streaming (optional)** — Lets us aggregate results across many files using simple Python mappers/reducers without installing Spark.
- **Python + scikit-learn (LogisticRegression + TF‑IDF)** —
  - Strong baseline for text classification on normal hardware.
  - Fast to train, interpretable, and easy to deploy.
  - Handles class imbalance via `class_weight="balanced"`.
- **pandas** — Convenient CSV IO + chunked processing so you can run on big files without running out of RAM.
- **matplotlib** — Simple, no-frills charts for your report.

> If you later add **Spark**, you can move preprocessing and inference to distributed PySpark. If you add a GPU, you can upgrade the model to **DistilBERT** via Hugging Face for higher accuracy.

---

## 6) Quick Troubleshooting

- If `hdfs` command not found → add `%HADOOP_HOME%\\bin` to PATH in System Environment Variables.
- If Hadoop path has spaces → quote the entire path.
- If Unicode errors on CSV read → switch encoding between `utf-8` and `latin-1` (the scripts already try both).
- If labelled files have different column names → ensure they include `title`, `selftext`, `Label` exactly (case-insensitive handled).

---

## 7) Next steps (optional enhancements)

- Add **stopword removal/lemmatization** in `TfidfVectorizer` with custom analyzer for small gains.
- Add **model calibration** (CalibratedClassifierCV) for probability outputs.
- Add **subreddit-level trend plots**.
