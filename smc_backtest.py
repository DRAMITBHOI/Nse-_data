import os
import json
import glob
import pandas as pd
import numpy as np

DATA_DIR = "data"
OUTPUT_DIR = "backtest_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRADE_COLUMNS = [
    "ticker", "entry_date", "entry_price", "sl_price", 
    "tp_price", "exit_date", "exit_price", "target_pct", 
    "sl_pct", "outcome", "pnl_pct"
]

def load_stock_df(file_path):
    """Loads a ticker's local JSON file from data/{sym}.json into a clean DataFrame."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        
        if not raw:
            return None

        if isinstance(raw, list):
            df = pd.DataFrame(raw)
        elif isinstance(raw, dict):
            # If stored as {"data": [...]} or dict of series
            if "data" in raw and isinstance(raw["data"], list):
                df = pd.DataFrame(raw["data"])
            else:
                df = pd.DataFrame(raw)
        else:
            return None

        # Clean column names (handles 'date', 'CH_TIMESTAMP', 'open', 'close', etc.)
        col_map = {}
        for c in df.columns:
            clean = str(c).strip().lower().replace("_", "").replace(" ", "")
            if clean in ["date", "timestamp", "chtimestamp", "time", "t"]:
                col_map[c] = "Date"
            elif clean in ["open", "openprice", "chopenprice", "o"]:
                col_map[c] = "Open"
            elif clean in ["high", "highprice", "chhighprice", "h"]:
                col_map[c] = "High"
            elif clean in ["low", "lowprice", "chlowprice", "l"]:
                col_map[c] = "Low"
            elif clean in ["close", "closeprice", "chcloseprice", "c"]:
                col_map[c] = "Close"

        df = df.rename(columns=col_map)
        required = {"Date", "Open", "High", "Low", "Close"}
        if not required.issubset(set(df.columns)):
            return None

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date", "Open", "High", "Low", "Close"])
        df = df.sort_values("Date").reset_index(drop=True)

        for col in ["Open", "High", "Low", "Close"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna().reset_index(drop=True)
        # Backtest from 2021 onwards
        df = df[df["Date"] >= "2021-01-01"].reset_index(drop=True)
        return df if len(df) >= 40 else None
    except Exception:
        return None

def run_smc_backtest(df, ticker):
    trades = []
    n = len(df)
    
    # 3-bar swing pivots
    df['swing_high'] = (df['High'] > df['High'].shift(1)) & (df['High'] > df['High'].shift(-1))
    df['swing_low'] = (df['Low'] < df['Low'].shift(1)) & (df['Low'] < df['Low'].shift(-1))
    df['is_green'] = df['Close'] >= df['Open']

    i = 15
    while i < n - 15:
        # 1 & 2: Sweep of prior swing low (fake breakdown)
        prior_lows = df.loc[:i-2].loc[df['swing_low'], 'Low']
        if prior_lows.empty:
            i += 1
            continue
        key_low = prior_lows.iloc[-1]

        # Fake breakdown sweep breached and reclaimed
        sweep_window = df.loc[max(0, i-3):i]
        breached = (sweep_window['Low'] < key_low).any()
        reclaimed = df.loc[i, 'Close'] > key_low
        if not (breached and reclaimed):
            i += 1
            continue

        # 3: Prior swing high before the sweep
        prior_highs = df.loc[:i-1].loc[df['swing_high'], 'High']
        if prior_highs.empty:
            i += 1
            continue
        major_high = prior_highs.iloc[-1]

        # 4: CHoCH Expansion rally breaking swing high
        choch_idx = None
        for j in range(i + 1, min(i + 12, n)):
            if df.loc[j, 'Close'] > major_high:
                choch_idx = j
                break

        if choch_idx is None:
            i += 1
            continue

        high_idx = df.loc[i+1:min(choch_idx + 6, n - 1), 'High'].idxmax()
        displacement_high = df.loc[high_idx, 'High']

        # 5: Bullish FVG detection in displacement leg
        fvg_entry, fvg_sl = None, None
        for k in range(i + 1, high_idx):
            if k + 1 >= n:
                break
            c1_high = df.loc[k - 1, 'High']
            c3_low = df.loc[k + 1, 'Low']
            c1_low = df.loc[k - 1, 'Low']

            if c3_low > c1_high:
                fvg_mid = (c3_low + c1_high) / 2.0
                target_entry = (c1_high + fvg_mid) / 2.0
                stop_loss = c1_low

                # If candle 1 low is further than 5%, use FVG bottom buffer to protect the trade
                if (target_entry - stop_loss) / target_entry > 0.05:
                    stop_loss = c1_high * 0.99

                target_pct = (displacement_high - target_entry) / target_entry
                sl_pct = (target_entry - stop_loss) / target_entry

                # Rule 8: Target >= 10-15% and SL <= 5-6%
                if target_pct >= 0.10 and sl_pct <= 0.06:
                    fvg_entry = target_entry
                    fvg_sl = stop_loss
                    break

        if fvg_entry is None:
            i = choch_idx + 1
            continue

        # 6: Retracement to entry zone
        entry_idx = None
        for r in range(high_idx + 1, min(high_idx + 35, n)):
            if df.loc[r, 'Low'] <= fvg_entry:
                entry_idx = r
                break

        if entry_idx is None:
            i = high_idx + 1
            continue

        # Forward simulate outcome
        exit_price, exit_date, outcome = None, None, "OPEN"
        for f in range(entry_idx + 1, min(entry_idx + 65, n)):
            curr_low = df.loc[f, 'Low']
            curr_high = df.loc[f, 'High']

            if curr_low <= fvg_sl:
                outcome = "LOSS"
                exit_price = fvg_sl
                exit_date = str(df.loc[f, 'Date'])[:10]
                break
            elif curr_high >= displacement_high:
                outcome = "WIN"
                exit_price = displacement_high
                exit_date = str(df.loc[f, 'Date'])[:10]
                break

        pnl_pct = ((exit_price - fvg_entry) / fvg_entry * 100) if exit_price else 0.0

        trades.append({
            "ticker": ticker,
            "entry_date": str(df.loc[entry_idx, 'Date'])[:10],
            "entry_price": round(float(fvg_entry), 2),
            "sl_price": round(float(fvg_sl), 2),
            "tp_price": round(float(displacement_high), 2),
            "exit_date": exit_date,
            "exit_price": round(float(exit_price), 2) if exit_price else None,
            "target_pct": round(float(target_pct) * 100, 2),
            "sl_pct": round(float(sl_pct) * 100, 2),
            "outcome": outcome,
            "pnl_pct": round(float(pnl_pct), 2)
        })

        i = entry_idx + 6

    return trades

def main():
    # Discover all stock JSON files in data/
    stock_files = glob.glob(os.path.join(DATA_DIR, "*.json"))
    
    # Exclude metadata/report JSON files
    excluded = {"nifty750.json", "fundamentals.json", "backtest_hp3_report.json"}
    valid_files = [f for f in stock_files if os.path.basename(f) not in excluded]

    print(f"Discovered {len(valid_files)} stock JSON files in '{DATA_DIR}/'. Starting SMC backtest...")

    all_trades = []
    processed = 0
    for file_path in valid_files:
        ticker = os.path.basename(file_path).replace(".json", "")
        df = load_stock_df(file_path)
        if df is not None:
            t = run_smc_backtest(df, ticker)
            all_trades.extend(t)
            processed += 1

    trades_df = pd.DataFrame(all_trades, columns=TRADE_COLUMNS)
    trades_df.to_csv(f"{OUTPUT_DIR}/smc_trades.csv", index=False)

    metrics = {
        "stocks_evaluated": processed,
        "total_signals": int(len(trades_df)),
        "closed_trades": 0,
        "win_rate": 0,
        "net_cumulative_pnl": 0
    }

    if not trades_df.empty:
        closed = trades_df[trades_df['outcome'].isin(['WIN', 'LOSS'])]
        wins = closed[closed['outcome'] == 'WIN']
        losses = closed[closed['outcome'] == 'LOSS']
        if len(closed) > 0:
            metrics = {
                "stocks_evaluated": processed,
                "total_signals": int(len(trades_df)),
                "closed_trades": int(len(closed)),
                "win_rate": round(len(wins) / len(closed) * 100, 2),
                "avg_win_pct": round(wins['pnl_pct'].mean(), 2) if not wins.empty else 0,
                "avg_loss_pct": round(losses['pnl_pct'].mean(), 2) if not losses.empty else 0,
                "profit_factor": round(abs(wins['pnl_pct'].sum() / losses['pnl_pct'].sum()), 2) if (not losses.empty and losses['pnl_pct'].sum() != 0) else 0,
                "net_cumulative_pnl": round(closed['pnl_pct'].sum(), 2)
            }

    with open(f"{OUTPUT_DIR}/smc_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Backtest complete. Evaluated {processed} stocks, generated {len(all_trades)} trades.")
    print(json.dumps(metrics, indent=4))

if __name__ == "__main__":
    main()
