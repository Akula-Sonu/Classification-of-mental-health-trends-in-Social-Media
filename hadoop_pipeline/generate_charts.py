import argparse, pandas as pd
import matplotlib.pyplot as plt

# IMPORTANT: single-plot figures; default colors

def main():
    ap = argparse.ArgumentParser(description="Generate simple charts from counts.tsv")
    ap.add_argument("--counts_tsv", required=True, help="Path to counts.tsv (year_month, predicted_label, count)")
    ap.add_argument("--out_dir", default="charts", help="Directory to save charts")
    args = ap.parse_args()

    df = pd.read_csv(args.counts_tsv, sep="\t")
    os.makedirs(args.out_dir, exist_ok=True)

    # 1) Overall distribution by label
    dist = df.groupby("predicted_label")["count"].sum().sort_values(ascending=False)
    plt.figure()
    dist.plot(kind="bar")
    plt.title("Overall Distribution by Predicted Label")
    plt.xlabel("Label")
    plt.ylabel("Total Count")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "overall_distribution.png"))
    plt.close()

    # 2) Trend per label over months
    pivot = df.pivot_table(index="year_month", columns="predicted_label", values="count", aggfunc="sum").fillna(0)
    plt.figure()
    pivot.plot()
    plt.title("Monthly Trend by Label")
    plt.xlabel("Year-Month")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "monthly_trend.png"))
    plt.close()

    print(f"[OK] Saved charts to {args.out_dir}")

if __name__ == "__main__":
    import os
    main()
