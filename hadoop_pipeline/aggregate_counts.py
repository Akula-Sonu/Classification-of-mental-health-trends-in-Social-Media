import argparse, glob, os, pandas as pd

def main():
    ap = argparse.ArgumentParser(description="Aggregate counts per year_month and predicted_label")
    ap.add_argument("--inputs", required=True, nargs="+", help="One or more predictions CSVs (supports glob via shell)")
    ap.add_argument("--out", default="counts.tsv", help="Output TSV path with aggregated counts")
    args = ap.parse_args()

    frames = []
    for pattern in args.inputs:
        for p in glob.glob(pattern):
            df = pd.read_csv(p)
            frames.append(df[["year_month","predicted_label"]])
    if not frames:
        print("[ERROR] No files matched.")
        return
    all_df = pd.concat(frames, ignore_index=True)
    agg = (all_df
           .dropna(subset=["predicted_label"])
           .groupby(["year_month","predicted_label"]).size()
           .reset_index(name="count")
           .sort_values(["year_month","predicted_label"]))
    agg.to_csv(args.out, sep="\t", index=False)
    print(f"[OK] Wrote {args.out} with {len(agg)} rows.")

if __name__ == "__main__":
    main()
