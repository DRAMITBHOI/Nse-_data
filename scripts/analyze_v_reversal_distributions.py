import os
import json
import pandas as pd
import numpy as np

DATA_DIR = "data"
INPUT_CSV = os.path.join(DATA_DIR, "v_reversal_dataset.csv")
OUTPUT_JSON = os.path.join(DATA_DIR, "v_reversal_parameters_report.json")
OUTPUT_MD = os.path.join(DATA_DIR, "v_reversal_parameters_report.md")


def analyze():
    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} not found.")
        return

    df = pd.read_csv(INPUT_CSV)
    if len(df) == 0:
        print("No records found.")
        return

    winners = df[df["is_successful"] == 1]
    losers = df[df["is_successful"] == 0]
    base_wr = (len(winners) / len(df)) * 100.0

    features = [
        ("drop_depth_pct", "Capitulation Drop Depth (%)"),
        ("vol_surge", "Trigger Volume Surge (x)"),
        ("deliv_vol_surge", "Trigger Delivery Volume Surge (x)"),
        ("deliv_pct_surge", "Delivery % Surge vs 50d (x)"),
        ("washout_vol_ratio", "Volume at Lowest Low Bar (x)"),
        ("range_closeness", "Candle Range Closeness"),
        ("rs_stock_nifty", "60d Excess Return vs Nifty"),
        ("risk_pct", "Stop Loss Distance (%)")
    ]

    summary = []
    for col, label in features:
        summary.append({
            "key": col,
            "feature": label,
            "winner_median": round(float(winners[col].median()), 2),
            "loser_median": round(float(losers[col].median()), 2),
            "winner_q25_q75": f"{round(float(winners[col].quantile(0.25)), 2)} - {round(float(winners[col].quantile(0.75)), 2)}"
        })

    # Reclaim Horizon Performance
    reclaim_stats = []
    for col, name in [("reclaims_2d", "Prior 2-Day High"), ("reclaims_3d", "Prior 3-Day High"), ("reclaims_5d", "Prior 5-Day High")]:
        subset = df[df[col] == 1]
        w = subset[subset["is_successful"] == 1]
        reclaim_stats.append({
            "reclaim": name,
            "trades": len(subset),
            "win_rate_%": round(float((len(w) / len(subset)) * 100.0), 2) if len(subset) > 0 else 0.0
        })

    # Conviction Score Deciles
    df["conv_bucket"] = pd.qcut(df["conv_score"], q=4, labels=["Q1 (Lowest)", "Q2 (Mid-Low)", "Q3 (Mid-High)", "Q4 (Highest)"])
    conv_buckets = []
    for b_name, b_df in df.groupby("conv_bucket", observed=False):
        b_wins = b_df[b_df["is_successful"] == 1]
        conv_buckets.append({
            "bucket": str(b_name),
            "trades": len(b_df),
            "win_rate_%": round(float((len(b_wins) / len(b_df)) * 100.0), 2) if len(b_df) > 0 else 0.0,
            "avg_pnl_exit22": round(float(b_df["pnl_exit22"].mean()), 2)
        })

    # Exit Engine Comparison
    exits = [
        ("pnl_exit22", "Exit 2.2 Engine"),
        ("pnl_staged_2r", "Staged 2R + BE + 20 EMA")
    ]
    exit_benchmarks = []
    for col, name in exits:
        ret = df[col]
        w = ret[ret > 0]
        l = ret[ret <= 0]
        pf = (w.sum() / abs(l.sum())) if abs(l.sum()) > 0 else 99.0
        exp = ((len(w) / len(ret)) * w.mean()) + ((len(l) / len(ret)) * l.mean())
        exit_benchmarks.append({
            "model": name,
            "win_rate_%": round(float((len(w) / len(ret)) * 100.0), 2),
            "avg_return_%": round(float(ret.mean()), 2),
            "profit_factor": round(float(pf), 2),
            "expectancy_%": round(float(exp), 2)
        })

    # Output JSON
    report_data = {
        "total_setups": len(df),
        "baseline_win_rate": round(base_wr, 2),
        "feature_metrics": summary,
        "reclaim_comparisons": reclaim_stats,
        "conviction_score_buckets": conv_buckets,
        "exit_comparisons": exit_benchmarks
    }
    with open(OUTPUT_JSON, "w", encoding="utf-8") as fp:
        json.dump(report_data, fp, indent=2)

    # Output Markdown
    md = [
        "# V-Reversal Empirical Derivation Report",
        f"- **Total Reversal Candidates:** {len(df):,}",
        f"- **Baseline Win Rate:** {base_wr:.2f}%\n",
        "## 1. Feature Distributions: Winners vs. Losers",
        "| Feature | Winner Median | Loser Median | Winner Core Range (25th–75th) |",
        "| :--- | :--- | :--- | :--- |"
    ]
    for r in summary:
        md.append(f"| {r['feature']} | **{r['winner_median']}** | {r['loser_median']} | {r['winner_q25_q75']} |")

    md.extend([
        "\n## 2. Reclaim Horizon Comparison",
        "| Reclaim Trigger | Total Setups | Win Rate (%) |",
        "| :--- | :--- | :--- |"
    ])
    for rc in reclaim_stats:
        md.append(f"| {rc['reclaim']} | {rc['trades']:,} | **{rc['win_rate_%']}%** |")

    md.extend([
        "\n## 3. Conviction Score Quartiles vs. Performance",
        "| Conviction Quartile | Trades | Win Rate (%) | Avg Trade PnL (Exit 2.2) |",
        "| :--- | :--- | :--- | :--- |"
    ])
    for cb in conv_buckets:
        md.append(f"| {cb['bucket']} | {cb['trades']:,} | {cb['win_rate_%']}% | **{cb['avg_pnl_exit22']:+.2f}%** |")

    md.extend([
        "\n## 4. Exit Engine Comparison (V-Reversals)",
        "| Architecture | Win Rate (%) | Avg Return (%) | Profit Factor | Expectancy |",
        "| :--- | :--- | :--- | :--- | :--- |"
    ])
    for ex in exit_benchmarks:
        md.append(f"| {ex['model']} | {ex['win_rate_%']}% | {ex['avg_return_%']}% | **{ex['profit_factor']}** | **{ex['expectancy_%']}%** |")

    with open(OUTPUT_MD, "w", encoding="utf-8") as fp:
        fp.write("\n".join(md))

    print(f"📊 Analysis complete. Report generated at {OUTPUT_MD}")


if __name__ == "__main__":
    analyze()
