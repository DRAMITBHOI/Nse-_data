import os
import json
import glob
import numpy as np
import pandas as pd
from datetime import datetime

DATA_DIR = "data"
RESULTS_JSON = os.path.join(DATA_DIR, "swing3_results.json")
RESULTS_HTML = os.path.join(DATA_DIR, "swing3_summary.html")

MIN_TURNOVER_CR = 2.0  # Liquidity floor: 50-day average turnover >= 2 Crore


def clean_and_prepare(raw_data):
    if not raw_data or not isinstance(raw_data, list):
        return []

    date_map = {}
    for r in raw_data:
        if not isinstance(r, dict):
            continue
        raw_t = str(r.get("time", "")).strip()[:10]
        if not raw_t:
            continue
        try:
            c = float(r.get("close", 0) or 0)
            if c <= 0:
                continue
            entry = {
                "time": raw_t,
                "open": float(r.get("open", c) or c),
                "high": float(r.get("high", c) or c),
                "low": float(r.get("low", c) or c),
                "close": c,
                "delivery_vol": float(r.get("delivery_vol", 0) or 0),
                "volume": float(r.get("volume", 0) or 0),
                "deliv_pct": float(r.get("deliv_pct", 0) or 0),
            }
            if raw_t not in date_map or entry["volume"] > date_map[raw_t]["volume"]:
                date_map[raw_t] = entry
        except Exception:
            continue

    clean = [date_map[k] for k in sorted(date_map.keys())]

    # Split / Corporate Action Multiplier Detection
    known_multipliers = [2.0, 5.0, 10.0, 1.5, 2.5, 3.0, 4.0]
    for i in range(len(clean) - 1, 0, -1):
        prev_c = clean[i - 1]["close"]
        curr_o = clean[i]["open"]
        if prev_c > 0 and curr_o > 0:
            ratio = prev_c / curr_o
            adj_factor = None
            if ratio >= 1.35:
                for k in known_multipliers:
                    if abs(ratio - k) / k < 0.15:
                        adj_factor = k
                        break
                if not adj_factor and 1.70 <= ratio <= 2.30:
                    adj_factor = 2.0
                elif not adj_factor and 4.30 <= ratio <= 5.50:
                    adj_factor = 5.0
                elif not adj_factor and 8.50 <= ratio <= 11.50:
                    adj_factor = 10.0
            if adj_factor:
                for j in range(0, i):
                    clean[j]["open"] = round(clean[j]["open"] / adj_factor, 2)
                    clean[j]["high"] = round(clean[j]["high"] / adj_factor, 2)
                    clean[j]["low"] = round(clean[j]["low"] / adj_factor, 2)
                    clean[j]["close"] = round(clean[j]["close"] / adj_factor, 2)
                    clean[j]["delivery_vol"] = clean[j]["delivery_vol"] * adj_factor
                    clean[j]["volume"] = clean[j]["volume"] * adj_factor

    # Volume & Delivery Sanity Fill
    running_vol = 50000.0
    for i in range(len(clean)):
        v = clean[i]["volume"]
        dv = clean[i]["delivery_vol"]
        pct = clean[i]["deliv_pct"]
        if v > 0:
            running_vol = 0.9 * running_vol + 0.1 * v
        else:
            clean[i]["volume"] = running_vol
            v = running_vol
        if dv <= 0:
            clean[i]["delivery_vol"] = v * (pct / 100.0 if pct > 0 else 0.50)
            clean[i]["deliv_pct"] = pct if pct > 0 else 50.0
        elif dv > v:
            clean[i]["delivery_vol"] = v
            clean[i]["deliv_pct"] = 100.0

    return clean


def backtest_single_stock(symbol, clean_data):
    if len(clean_data) < 60:
        return []

    df = pd.DataFrame(clean_data)
    df["gross_vol_sma20"] = df["volume"].rolling(20, min_periods=1).mean()
    df["deliv_sma"] = df["delivery_vol"].rolling(20, min_periods=1).mean()
    df["turnover_cr"] = (df["close"] * df["volume"]) / 1e7
    df["turnover_50d"] = df["turnover_cr"].rolling(50, min_periods=10).mean().fillna(0)
    df["deliv_pct_50d"] = df["deliv_pct"].rolling(50, min_periods=10).mean().fillna(0)
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()

    c_range = df["high"] - df["low"]
    df["range_closeness"] = np.where(c_range > 0, (df["close"] - df["low"]) / c_range, 0.0)

    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    volumes = df["volume"].values
    vol_avgs = df["gross_vol_sma20"].values
    deliv_vols = df["delivery_vol"].values
    deliv_avgs = df["deliv_sma"].values
    to_50d = df["turnover_50d"].values
    deliv_pcts = df["deliv_pct"].values
    deliv_pct_avgs = df["deliv_pct_50d"].values
    range_closes = df["range_closeness"].values
    ema20s = df["ema20"].values
    times = df["time"].values

    trades = []
    in_position = False
    entry_price = 0.0
    initial_stop = 0.0
    current_stop = 0.0
    r_unit = 0.0
    moved_to_be = False
    entry_idx = 0
    setup_name = ""

    for i in range(40, len(df)):
        c = closes[i]
        h = highs[i]
        l = lows[i]
        v = volumes[i]
        v_avg = vol_avgs[i]
        dv = deliv_vols[i]
        dv_avg = deliv_avgs[i]
        turnover = to_50d[i]
        dp = deliv_pcts[i]
        dp_avg = deliv_pct_avgs[i]
        rc = range_closes[i]
        ema = ema20s[i]

        if not in_position:
            if turnover < MIN_TURNOVER_CR:
                continue

            entry_triggered = False
            curr_setup = ""
            calc_entry = 0.0
            calc_sl = 0.0

            # -------------------------------------------------------------
            # ENGINE 1: DOWNTREND REVERSAL (V-SHAPE / SELLING CLIMAX)
            # -------------------------------------------------------------
            recent_20_high = highs[i - 20:i].max()
            is_steep_drop = (recent_20_high - l) / recent_20_high >= 0.18

            if is_steep_drop:
                recent_trough = lows[i - 5:i].min()
                prior_3d_high = highs[i - 4:i].max()

                c_reclaim = (c >= prior_3d_high) and (c > closes[i - 1])
                c_vol_rev = v >= (1.3 * v_avg)
                c_deliv_rev = (dv >= 1.25 * dv_avg) or (dp >= 1.15 * dp_avg if dp_avg > 0 else False)
                c_candle_rev = rc >= 0.55

                if c_reclaim and c_vol_rev and c_deliv_rev and c_candle_rev:
                    entry_triggered = True
                    curr_setup = "V-REVERSAL"
                    calc_entry = round(prior_3d_high, 2)
                    calc_sl = round(recent_trough * 0.995, 2)

            # -------------------------------------------------------------
            # ENGINE 2: MICRO-LAUNCHPAD BREAKOUT (HFCL / PAISALO STYLE)
            # -------------------------------------------------------------
            if not entry_triggered:
                valid_launchpad = False
                launchpad_high = 0.0
                launchpad_low = 0.0

                for shelf_len in range(10, 19):
                    s_high = highs[i - shelf_len:i].max()
                    s_low = lows[i - shelf_len:i].min()
                    if s_low > 0 and ((s_high - s_low) / s_low) <= 0.13:
                        valid_launchpad = True
                        launchpad_high = round(float(s_high), 2)
                        launchpad_low = round(float(s_low), 2)
                        break

                if valid_launchpad:
                    base_up_deliv = sum(
                        deliv_vols[k] for k in range(i - 15, i) if closes[k] >= closes[k - 1]
                    )
                    base_down_deliv = sum(
                        deliv_vols[k] for k in range(i - 15, i) if closes[k] < closes[k - 1]
                    )
                    cumulative_ratio = (
                        (base_up_deliv / base_down_deliv) if base_down_deliv > 0 else 1.5
                    )

                    c_bo = (c >= launchpad_high) and (h >= launchpad_high)
                    c_vol_bo = v >= (1.4 * v_avg)
                    c_deliv_bo = (dv >= 1.25 * dv_avg) or (dp >= 1.25 * dp_avg if dp_avg > 0 else False)
                    c_candle_bo = rc >= 0.65

                    if c_bo and c_vol_bo and c_deliv_bo and c_candle_bo and (cumulative_ratio >= 1.15):
                        entry_triggered = True
                        curr_setup = "LAUNCHPAD-BO"
                        calc_entry = launchpad_high
                        calc_sl = round(min(l, launchpad_low), 2)

            # -------------------------------------------------------------
            # EXECUTION
            # -------------------------------------------------------------
            if entry_triggered:
                r_dist = calc_entry - calc_sl
                if r_dist > 0.05 and (r_dist / calc_entry) <= 0.12:
                    in_position = True
                    entry_price = calc_entry
                    initial_stop = calc_sl
                    current_stop = initial_stop
                    r_unit = r_dist
                    moved_to_be = False
                    entry_idx = i
                    setup_name = curr_setup

        else:
            # 1. Breakeven at +1.5R
            if not moved_to_be and h >= (entry_price + 1.5 * r_unit):
                current_stop = max(current_stop, entry_price)
                moved_to_be = True

            # 2. Trail along 20 EMA
            if moved_to_be:
                current_stop = max(current_stop, round(float(ema), 2))

            # 3. Exit Condition
            if c < current_stop:
                exit_price = round(current_stop, 2)
                pnl_pts = exit_price - entry_price
                pnl_pct = round((pnl_pts / entry_price) * 100.0, 2)
                r_multiple = round(pnl_pts / r_unit, 2)

                trades.append({
                    "Symbol": symbol,
                    "Setup": setup_name,
                    "Entry Date": times[entry_idx],
                    "Exit Date": times[i],
                    "Duration (Days)": i - entry_idx,
                    "Entry Price": entry_price,
                    "Exit Price": exit_price,
                    "Initial Stop": initial_stop,
                    "R Unit": round(r_unit, 2),
                    "PnL %": pnl_pct,
                    "R Multiple": r_multiple,
                    "Outcome": "WIN" if pnl_pts > 0 else "LOSS",
                })

                in_position = False
                entry_price = 0.0
                current_stop = 0.0
                moved_to_be = False

    return trades


def run_full_backtest():
    json_files = glob.glob(os.path.join(DATA_DIR, "*.json"))
    excluded_files = {
        "nifty750.json",
        "nifty50.json",
        "nifty.json",
        "gap_margin_candidates.json",
        "swing3_results.json",
        "backtest_results.json",
    }
    target_files = [
        f for f in json_files if os.path.basename(f).lower() not in excluded_files
    ]

    print(f"🚀 Starting Swing 3.0 Backtest across {len(target_files)} symbols in '{DATA_DIR}'...")

    all_trades = []
    scanned_count = 0

    for path in sorted(target_files):
        sym = os.path.splitext(os.path.basename(path))[0].upper()
        try:
            with open(path, "r", encoding="utf-8") as fp:
                raw = json.load(fp)
            clean = clean_and_prepare(raw)
            if clean:
                sym_trades = backtest_single_stock(sym, clean)
                all_trades.extend(sym_trades)
                scanned_count += 1
        except Exception:
            continue

    if not all_trades:
        print("⚠️ No qualifying trades triggered.")
        return

    # Metrics
    df_trades = pd.DataFrame(all_trades)
    total_trades = len(df_trades)
    wins = df_trades[df_trades["Outcome"] == "WIN"]
    losses = df_trades[df_trades["Outcome"] == "LOSS"]
    win_rate = round((len(wins) / total_trades) * 100.0, 2)

    avg_win_pct = round(wins["PnL %"].mean(), 2) if not wins.empty else 0.0
    avg_loss_pct = round(losses["PnL %"].mean(), 2) if not losses.empty else 0.0
    avg_r = round(df_trades["R Multiple"].mean(), 2)
    max_r = round(df_trades["R Multiple"].max(), 2)
    avg_hold_days = round(df_trades["Duration (Days)"].mean(), 1)

    gross_win = wins["PnL %"].sum() if not wins.empty else 0.0
    gross_loss = abs(losses["PnL %"].sum()) if not losses.empty else 1.0
    profit_factor = round(gross_win / gross_loss, 2) if gross_loss > 0 else 999.0

    breakout_trades = df_trades[df_trades["Setup"] == "LAUNCHPAD-BO"]
    reversal_trades = df_trades[df_trades["Setup"] == "V-REVERSAL"]

    summary_metrics = {
        "Strategy": "Swing 3.0 (Launchpad Breakouts & V-Reversals)",
        "Generated At": datetime.now().strftime("%Y-%m-%d %H:%M:%S IST"),
        "Symbols Scanned": scanned_count,
        "Total Trades": total_trades,
        "Win Rate %": win_rate,
        "Profit Factor": profit_factor,
        "Average R": avg_r,
        "Max R": max_r,
        "Avg Win %": avg_win_pct,
        "Avg Loss %": avg_loss_pct,
        "Avg Duration (Days)": avg_hold_days,
        "Launchpad Breakouts": {
            "Count": len(breakout_trades),
            "Win Rate %": round((len(breakout_trades[breakout_trades['Outcome'] == 'WIN']) / len(breakout_trades)) * 100.0, 2) if not breakout_trades.empty else 0,
            "Avg R": round(breakout_trades["R Multiple"].mean(), 2) if not breakout_trades.empty else 0,
        },
        "V-Reversals": {
            "Count": len(reversal_trades),
            "Win Rate %": round((len(reversal_trades[reversal_trades['Outcome'] == 'WIN']) / len(reversal_trades)) * 100.0, 2) if not reversal_trades.empty else 0,
            "Avg R": round(reversal_trades["R Multiple"].mean(), 2) if not reversal_trades.empty else 0,
        },
    }

    # Save JSON Payload
    payload = {
        "Metrics": summary_metrics,
        "Recent Trades": all_trades[-150:],
    }
    with open(RESULTS_JSON, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)

    # Save HTML Dashboard
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <title>Swing 3.0 Backtest Summary</title>
      <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background-color: #0b0f19; color: #e2e8f0; padding: 24px; }}
        .card {{ background-color: #131b2e; border: 1px solid #1e293b; border-radius: 8px; padding: 18px; margin-bottom: 20px; }}
        .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 14px; margin-top: 12px; }}
        .metric-item {{ background-color: #0a0e17; padding: 12px; border-radius: 6px; border-left: 3px solid #38bdf8; }}
        .metric-val {{ font-size: 20px; font-weight: 800; color: #ffffff; margin-top: 4px; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 14px; }}
        th {{ background-color: #1e293b; color: #38bdf8; text-align: left; padding: 10px; }}
        td {{ padding: 10px; border-bottom: 1px solid #1e293b; }}
        tr:nth-child(even) {{ background-color: #0d1322; }}
        .win {{ color: #00E676; font-weight: bold; }}
        .loss {{ color: #FF5252; font-weight: bold; }}
      </style>
    </head>
    <body>
      <h2>⚡ Swing 3.0 Backtest Dashboard</h2>
      <div class="card">
        <h3>Performance Summary</h3>
        <div class="metric-grid">
          <div class="metric-item"><div>Total Trades</div><div class="metric-val">{total_trades}</div></div>
          <div class="metric-item"><div>Win Rate</div><div class="metric-val">{win_rate}%</div></div>
          <div class="metric-item"><div>Profit Factor</div><div class="metric-val">{profit_factor}</div></div>
          <div class="metric-item"><div>Average R</div><div class="metric-val">+{avg_r}R</div></div>
          <div class="metric-item"><div>Max R Multiple</div><div class="metric-val">+{max_r}R</div></div>
          <div class="metric-item"><div>Avg Hold Time</div><div class="metric-val">{avg_hold_days} Days</div></div>
        </div>
      </div>
      <div class="card">
        <h3>Sample Trigger Log (Last 50 Trades)</h3>
        <table>
          <thead>
            <tr>
              <th>Symbol</th><th>Setup</th><th>Entry Date</th><th>Exit Date</th><th>Hold Days</th>
              <th>Entry</th><th>Exit</th><th>PnL %</th><th>R Multiple</th><th>Outcome</th>
            </tr>
          </thead>
          <tbody>
            {''.join(f"<tr><td>{t['Symbol']}</td><td>{t['Setup']}</td><td>{t['Entry Date']}</td><td>{t['Exit Date']}</td><td>{t['Duration (Days)']}</td><td>₹{t['Entry Price']}</td><td>₹{t['Exit Price']}</td><td class='{'win' if t['PnL %'] > 0 else 'loss'}'>{t['PnL %']:+}%</td><td>{t['R Multiple']:+}R</td><td class='{'win' if t['Outcome'] == 'WIN' else 'loss'}'>{t['Outcome']}</td></tr>" for t in all_trades[-50:])}
          </tbody>
        </table>
      </div>
    </body>
    </html>
    """
    with open(RESULTS_HTML, "w", encoding="utf-8") as fp:
        fp.write(html_content)

    print(f"✅ Swing 3.0 Backtest complete. Scanned: {scanned_count} | Trades: {total_trades} | Win Rate: {win_rate}% | Profit Factor: {profit_factor}")
    print(f"📁 Results saved to: '{RESULTS_JSON}' and '{RESULTS_HTML}'.")


if __name__ == "__main__":
    run_full_backtest()
