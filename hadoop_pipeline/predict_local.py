import argparse, os, sys, joblib, pandas as pd
from dateutil import parser as dateparser

def to_year_month(ts):
    if pd.isna(ts):
        return ""
    try:
        # try unix epoch if numeric
        if isinstance(ts, (int, float)) or (isinstance(ts, str) and ts.isdigit()):
            return pd.to_datetime(float(ts), unit="s", errors="coerce").strftime("%Y-%m")
        # else parse as ISO-ish string
        return pd.to_datetime(ts, errors="coerce").strftime("%Y-%m")
    except Exception:
        return ""

def main():
    ap = argparse.ArgumentParser(description="Predict labels for raw Reddit CSV using saved model (chunked)")
    ap.add_argument("--input_csv", required=True, help="Path to a raw CSV (e.g., depapr19.csv)")
    ap.add_argument("--model", required=True, help="Path to joblib model saved by train_classifier.py")
    ap.add_argument("--out_csv", default="predictions.csv", help="Where to save predictions CSV")
    ap.add_argument("--chunksize", type=int, default=10000, help="Chunk size for pandas read_csv")
    args = ap.parse_args()

    pipe = joblib.load(args.model)

    out_rows = []
    cols = None
    # attempt utf-8, fallback latin-1
    for enc in ["utf-8", "latin-1"]:
        try:
            it = pd.read_csv(args.input_csv, chunksize=args.chunksize, encoding=enc, engine="python")
            first_chunk = next(it)
            cols = [c.strip() for c in first_chunk.columns]
            # chain back generator
            it = pd.concat([first_chunk], ignore_index=True), it
            break
        except Exception:
            continue

    # re-open properly to iterate
    reader = pd.read_csv(args.input_csv, chunksize=args.chunksize, encoding=("utf-8" if cols else "latin-1"), engine="python")

    for chunk in reader:
        chunk.columns = [c.strip() for c in chunk.columns]
        # Normalize expected columns
        title_col = next((c for c in chunk.columns if c.lower()=="title"), None)
        text_col = next((c for c in chunk.columns if c.lower()=="selftext"), None)
        sub_col = next((c for c in chunk.columns if c.lower()=="subreddit"), None)
        score_col = next((c for c in chunk.columns if c.lower()=="score"), None)
        created_col = next((c for c in chunk.columns if c.lower()=="created_utc"), None)
        ts_col = next((c for c in chunk.columns if c.lower()=="timestamp"), None)

        if title_col is None or text_col is None:
            # skip chunk if can't build text
            continue

        chunk[title_col] = chunk[title_col].fillna("")
        chunk[text_col] = chunk[text_col].fillna("")
        text = (chunk[title_col].astype(str) + " " + chunk[text_col].astype(str)).str.strip()

        preds = pipe.predict(text)

        # month derivation
        ym = ""
        if ts_col and ts_col in chunk.columns:
            ym_series = chunk[ts_col].apply(to_year_month)
            ym = ym_series
        elif created_col and created_col in chunk.columns:
            ym_series = chunk[created_col].apply(to_year_month)
            ym = ym_series
        else:
            ym_series = pd.Series([""]*len(chunk))

        out = pd.DataFrame({
            "subreddit": chunk[sub_col] if sub_col in chunk.columns else "",
            "score": chunk[score_col] if score_col in chunk.columns else "",
            "title": chunk[title_col],
            "selftext": chunk[text_col],
            "year_month": ym_series,
            "predicted_label": preds
        })
        out_rows.append(out)

    if not out_rows:
        print("[ERROR] No output rows produced. Check input CSV columns.")
        sys.exit(1)

    final = pd.concat(out_rows, ignore_index=True)
    final.to_csv(args.out_csv, index=False, encoding="utf-8")
    print(f"[OK] Wrote predictions to {args.out_csv} (rows: {len(final)})")

if __name__ == "__main__":
    main()
