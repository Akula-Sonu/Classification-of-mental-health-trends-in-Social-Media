import argparse, os, glob, joblib, sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

def read_labelled_csvs(label_dir: str) -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(label_dir, "LD*.csv")))
    if not paths:
        print(f"[ERROR] No labelled CSVs found in {label_dir}. Expected names like LD*.csv")
        sys.exit(1)
    frames = []
    for p in paths:
        try:
            df = pd.read_csv(p, encoding="utf-8", engine="python")
        except UnicodeDecodeError:
            df = pd.read_csv(p, encoding="latin-1", engine="python")
        df.columns = [c.strip() for c in df.columns]
        # normalize column names
        colmap = {c.lower(): c for c in df.columns}
        # required columns (case-insensitive): title, selftext, Label/label
        # also keep subreddit, score if present
        title_col = next((c for c in df.columns if c.lower()=="title"), None)
        text_col = next((c for c in df.columns if c.lower()=="selftext"), None)
        label_col = next((c for c in df.columns if c.lower()=="label"), None)
        if title_col is None or text_col is None or label_col is None:
            print(f"[WARN] Skipping {p} due to missing columns. Found: {df.columns.tolist()}")
            continue
        df = df[[title_col, text_col, label_col] + [c for c in ["subreddit","score"] if c in df.columns]]
        df = df.rename(columns={title_col:"title", text_col:"selftext", label_col:"label"})
        frames.append(df)
    if not frames:
        print("[ERROR] No valid labelled CSVs with required columns were found.")
        sys.exit(1)
    out = pd.concat(frames, ignore_index=True)
    out["title"] = out["title"].fillna("")
    out["selftext"] = out["selftext"].fillna("")
    out["text"] = (out["title"].astype(str) + " " + out["selftext"].astype(str)).str.strip()
    out["label"] = out["label"].astype(str).str.strip()
    out = out[out["text"].str.len() > 3]
    out = out[out["label"] != ""]
    return out[["text","label"]]

def build_pipeline():
    tfidf = TfidfVectorizer(
        sublinear_tf=True,
        ngram_range=(1,2),
        max_df=0.95,
        min_df=3,
        max_features=200_000
    )
    clf = LogisticRegression(
        max_iter=400,
        n_jobs=None,
        class_weight="balanced",
        solver="saga",
        verbose=0
    )
    pipe = Pipeline([("tfidf", tfidf), ("clf", clf)])
    return pipe

def main():
    ap = argparse.ArgumentParser(description="Train text classifier on Reddit labelled data")
    ap.add_argument("--label_dir", required=True, help="Folder containing LD*.csv files")
    ap.add_argument("--model_out", default="reddit_text_model.joblib", help="Output path for model")
    ap.add_argument("--report_out", default="model_report.txt", help="Where to write evaluation report")
    args = ap.parse_args()

    df = read_labelled_csvs(args.label_dir)
    if len(df) < 50:
        print(f"[ERROR] Too few labelled rows ({len(df)}). Need at least 50.")
        sys.exit(1)

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds)
    with open(args.report_out, "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write(report)

    joblib.dump(pipe, args.model_out)
    print(f"[OK] Saved model to {args.model_out}")
    print(f"[OK] Wrote evaluation report to {args.report_out}")
    print(f"[INFO] Classes: {sorted(df['label'].unique().tolist())}")
    print(f"[INFO] Accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
