import os
import json
import pandas as pd
import numpy as np

DATA_DIR = "data"
INPUT_CSV = os.path.join(DATA_DIR, "breakout_dataset.csv")
OUTPUT_JSON = os.path.join(DATA_DIR, "optimal_parameters_report.json")
OUTPUT_MD = os.path.join(DATA_DIR, "optimal_parameters_report.md")


def analyze():
    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} not found.")
        return

    df = pd.read_csv(INPUT_CSV)
    if len(df) == 0:
        print("No breakout records found.")
        return

    winners = df[df["is_successful"] == 1]
    losers = df[df["is_successful"] == 0]

    baseline_win_rate = (len(winners) / len(df)) * 100.0

    features = [
        ("base_length", "Base Length (Bars)"),
        ("base_spread_pct", "Base Spread (%)"),
        ("vdu_ratio", "Volume Dry-Up (VDU) Ratio"),
        ("vcp_atr_ratio", "VCP ATR Ratio (5d/14d)"),
        ("vol_surge", "Trigger Volume Surge (x)"),
        ("deliv_vol_surge", "Delivery Volume Surge (x)"),
        ("deliv_pct_surge", "Delivery % Surge (x)"),
        ("accum_ratio", "Base Accumulation Ratio"),
        ("range_closeness", "Candle Range Closeness"),
        ("rs_stock_nifty", "60d Excess Return vs Nifty"),
        ("dist_to_120h_pct", "Ceiling Distance to 120d High (%)"),
        ("risk_pct", "Initial Stop Loss Distance (%)")
    ]

    feature_summary = []
    for col, label in features:
        w_median = winners[col].median()
        l_median = losers[col].median()
        w_q25 = winners[col].quantile(0.25)
        w_q75 = winners[col].quantile(0.75)
        feature_summary.append({
            "key": col,
            "feature": label,
            "winner_median": round(float(w_median), 2),
            "loser_median": round(float(l_median), 2),
            "winner_p25_p75": f"{round(float(w_q25), 2)} - {round(float(w_q75), 2)}"
        })

    # Exit Model Comparisons
    exit_models = [
        ("pnl_exit22", "Exit 2.2 Engine (Current Baseline)"),
        ("pnl_ema20", "Pure 20 EMA Close Trail"),
        ("pnl_ema10", "Pure 10 EMA Close Trail"),
        ("pnl_staged_2r", "Staged 2R Partial + Breakeven + 20 EMA")
    ]

    exit_comparison = []
    for col, name in exit_models:
        returns = df[col]
        win_trades = returns[returns > 0]
        loss_trades = returns[returns <= 0]
        avg_win = win_trades.mean() if len(win_trades) > 0 else 0
        avg_loss = abs(loss_trades.mean()) if len(loss_trades) > 0 else 1.0
        profit_factor = (win_trades.sum() / abs(loss_trades.sum())) if abs(loss_trades.sum()) > 0 else 99.0
        expectancy = ((len(win_trades) / len(returns)) * avg_win) - ((len(loss_trades) / len(returns)) * avg_loss)

        exit_comparison.append({
            "model_key": col,
            "model_name": name,
            "win_rate_%": round(float((len(win_trades) / len(returns)) * 100.0), 2),
            "average_return_%": round(float(returns.mean()), 2),
            "profit_factor": round(float(profit_factor), 2),
            "expectancy_pct": round(float(expectancy), 2),
            "max_drawdown_trade_%": round(float(returns.min()), 2)
        })

    # Save JSON Payload
    report_data = {
        "total_breakouts_analyzed": len(df),
        "baseline_win_rate_pct": round(baseline_win_rate, 2),
        "feature_metrics": feature_summary,
        "exit_model_benchmarks": exit_comparison
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as fp:
        json.dump(report_data, fp, indent=2)

    # Format Markdown Report
    md_content = [
        "# Swing Ultra Empirical Derivation Report",
        f"- **Total Breakouts Analyzed:** {len(df):,}",
        f"- **Baseline Win Rate:** {baseline_win_rate:.2f}%\n",
        "## 1. Feature Distributions: Winners vs. Losers",
        "| Feature | Winner Median | Loser Median | Winner Core Range (25th–75th) |",
        "| :--- | :--- | :--- | :--- |"
    ]
    for row in feature_summary:
        md_content.append(f"| {row['feature']} | **{row['winner_median']}** | {row['loser_median']} | {row['winner_p25_p75']} |")

    md_content.extend([
        "\n## 2. Exit Architecture Performance Comparison",
        "| Exit Strategy | Win Rate (%) | Avg Trade (%) | Profit Factor | Expectancy / Trade | Max Single Loss (%) |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |"
    ])
    for ex in exit_comparison:
        md_content.append(f"| {ex['model_name']} | {ex['win_rate_%']}% | {ex['average_return_%']}% | **{ex['profit_factor']}** | **{ex['expectancy_pct']}%** | {ex['max_drawdown_trade_%']}% |")

    with open(OUTPUT_MD, "w", encoding="utf-8") as fp:
        fp.write("\n".join(md_content))

    print(f"📊 Analysis complete. Report generated at {OUTPUT_MD}")


if __name__ == "__main__":
    analyze()
