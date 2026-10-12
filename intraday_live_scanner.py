import os
import sys
import json
import time
import datetime
import urllib.request
import urllib.parse
import pandas as pd
import yfinance as yf

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

LIVE_BREAKOUTS_FILE = os.path.join(DATA_DIR, "live_intraday_breakouts.json")
MY_TRADES_FILE = os.path.join(DATA_DIR, "my_trades.json")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "").strip()

# Cash / ETF exclusion list
EXCLUDED_SYMBOLS = {"LIQUIDCASE", "LIQGRWBEES", "LIQUIDBEES", "GOLDBEES", "NIFTYBEES", "BANKBEES"}

def send_telegram_alert(message: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ Telegram token or chat ID missing. Alert skipped.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML"
    }
    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status == 200:
                print("📡 Telegram alert dispatched successfully (HTTP 200)")
            else:
                print(f"⚠️ Telegram returned status {resp.status}")
    except Exception as e:
        print(f"❌ Telegram API Error: {e}")

def get_market_session_elapsed():
    now = datetime.datetime.now()
    # Market open 09:15 AM
    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    
    if now < market_open:
        return 1
    elif now > market_close:
        return 375
    else:
        elapsed = int((now - market_open).total_seconds() / 60)
        return max(1, min(375, elapsed))

def load_comprehensive_targets():
    """
    Loads Launchpad Coils, V-Reversals, and multi-day pre-reclaim watchlists
    from both Swing3 (Nifty 750) and Run2 (Non-N750) pipelines.
    """
    targets = {}
    
    # 1. Primary scanner sources
    scanner_sources = [
        ("data/swing3_scanner_results.json", "NIFTY 750"),
        ("data/run2_scanner_results.json", "NON-N750"),
        ("data/swing3_run2_results.json", "NON-N750")
    ]
    
    for filepath, default_univ in scanner_sources:
        if not os.path.exists(filepath):
            continue
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                d = json.load(f)
            
            # Combine all candidate lists (Coiling, Watchlist, Fresh triggers)
            pools = []
            for k in ["Coiling", "Watchlist", "Pre_Breakouts", "Pre-Breakout Watchlist", "pre_breakouts"]:
                if k in d and isinstance(d[k], list):
                    pools.extend(d[k])

            for item in pools:
                sym = str(item.get("symbol", "")).strip().upper()
                if not sym or sym.endswith("BEES") or sym in EXCLUDED_SYMBOLS:
                    continue

                # Parse Pivot Ceiling
                pivot = item.get("pivot_ceiling") or item.get("pivot_target") or item.get("pivot") or 0.0
                pivot = float(pivot)
                if pivot <= 0:
                    continue

                stop_loss = float(item.get("stop_loss", pivot * 0.92))
                setup = item.get("setup_class") or item.get("setup") or item.get("type") or "LAUNCHPAD COIL"
                universe = item.get("universe", default_univ)

                targets[sym] = {
                    "symbol": sym,
                    "universe": universe,
                    "setup": setup,
                    "pivot_ceiling": pivot,
                    "stop_loss": stop_loss
                }
        except Exception as e:
            print(f"⚠️ Error reading {filepath}: {e}")

    # 2. Retain previously tracked watchlists from existing live state (multi-day continuity)
    if os.path.exists(LIVE_BREAKOUTS_FILE):
        try:
            with open(LIVE_BREAKOUTS_FILE, "r", encoding="utf-8") as f:
                prior_state = json.load(f)
            for item in prior_state.get("Coiling", []):
                sym = str(item.get("symbol", "")).strip().upper()
                if sym and sym not in targets and (not sym.endswith("BEES")) and (sym not in EXCLUDED_SYMBOLS):
                    targets[sym] = {
                        "symbol": sym,
                        "universe": item.get("universe", "NON-N750"),
                        "setup": item.get("setup", "LAUNCHPAD COIL"),
                        "pivot_ceiling": float(item.get("pivot_ceiling", 0.0)),
                        "stop_loss": float(item.get("stop_loss", 0.0))
                    }
        except Exception:
            pass

    print(f"📋 Loaded {len(targets)} total watchlisted setups for live monitoring.")
    return targets

def run_live_intraday_scan():
    targets = load_comprehensive_targets()
    if not targets:
        print("❌ No watchlisted setups found to scan.")
        return

    # Check previously dispatched alerts today to avoid notification spam
    prior_breakouts_map = {}
    if os.path.exists(LIVE_BREAKOUTS_FILE):
        try:
            with open(LIVE_BREAKOUTS_FILE, "r", encoding="utf-8") as f:
                prev_data = json.load(f)
                for b in prev_data.get("Breakouts_All_Day", []):
                    prior_breakouts_map[b["symbol"]] = b
        except Exception:
            pass

    elapsed_mins = get_market_session_elapsed()
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %I:%M:%S %p IST")
    tickers_list = [f"{s}.NS" for s in targets.keys()]

    print(f"⚡ Pulling live intraday quotes for {len(tickers_list)} candidates (Elapsed: {elapsed_mins}m / 375m)...")
    try:
        live_data = yf.download(
            tickers_list,
            period="25d",
            interval="1d",
            group_by="ticker",
            auto_adjust=False,
            progress=False,
            threads=True
        )
    except Exception as e:
        print(f"❌ Failed to fetch live data: {e}")
        return

    confirmed_breakouts = []
    coiling_list = []
    new_alerts_to_send = []

    for sym, meta in targets.items():
        yf_sym = f"{sym}.NS"
        try:
            df = live_data if len(tickers_list) == 1 else live_data[yf_sym]
            df = df.dropna(how="all").copy()
            if len(df) < 2:
                continue

            df.columns = [c.lower() for c in df.columns]
            
            # Historical daily volumes (excluding current forming bar)
            hist_vols = df["volume"].iloc[:-1].tail(20)
            sma20_vol = hist_vols.mean() if len(hist_vols) > 0 else 1.0

            # Today's live candle
            today = df.iloc[-1]
            ltp = float(today["close"])
            day_high = float(today["high"])
            day_vol = float(today["volume"])

            # Projected session volume pace calculation
            expected_fraction = max(0.01, min(1.0, elapsed_mins / 375.0))
            projected_vol = day_vol / expected_fraction
            vol_pace = round(projected_vol / sma20_vol, 2) if sma20_vol > 0 else 1.0

            pivot = meta["pivot_ceiling"]
            stop_loss = meta["stop_loss"]
            dist_pivot_pct = round(((ltp - pivot) / pivot) * 100.0, 2)
            risk_pct = round(((pivot - stop_loss) / pivot) * 100.0, 2) if pivot > stop_loss else 8.0

            # Core Trigger Mechanics
            holding_pivot = ltp >= pivot
            wick_touched_pivot = day_high >= pivot
            
            # Volume surge threshold: 1.25x for Nifty 750, 1.0x for Non-N750
            pace_threshold = 1.25 if meta["universe"] == "NIFTY 750" else 1.00
            is_volume_confirmed = vol_pace >= pace_threshold

            record = {
                "symbol": sym,
                "universe": meta["universe"],
                "setup": meta["setup"],
                "pivot_ceiling": pivot,
                "ltp": ltp,
                "day_high": day_high,
                "stop_loss": stop_loss,
                "risk_%": risk_pct,
                "dist_pivot_%": dist_pivot_pct,
                "proj_vol_pace": vol_pace
            }

            # ---------------- EVALUATION GATE ----------------
            if holding_pivot and is_volume_confirmed:
                # Authentic breakout holding above pivot with volume confirmation
                record["action"] = "CONFIRMED BREAKOUT"
                confirmed_breakouts.append(record)

                if sym not in prior_breakouts_map:
                    new_alerts_to_send.append(record)
                    prior_breakouts_map[sym] = record

            elif holding_pivot and not is_volume_confirmed:
                # Price above pivot but low volume pace (Early warning / Coiling hold)
                record["action"] = "LOW-VOL DRIFT (ABOVE PIVOT)"
                coiling_list.append(record)

            elif wick_touched_pivot and not holding_pivot:
                # Intraday wick failure: High touched pivot, but LTP fell back below
                record["action"] = "WICK REJECTED"
                coiling_list.append(record)

            else:
                # Pre-breakout consolidation / Coiling
                record["action"] = "COILING"
                coiling_list.append(record)

        except Exception:
            continue

    # Sort pre-breakout watchlists by proximity to pivot
    coiling_list.sort(key=lambda x: x["dist_pivot_%"], reverse=True)
    confirmed_breakouts.sort(key=lambda x: x["proj_vol_pace"], reverse=True)

    # Save to data/live_intraday_breakouts.json
    output_payload = {
        "Scan_Timestamp": now_str,
        "Elapsed_Minutes": elapsed_mins,
        "Breakouts_All_Day": confirmed_breakouts,
        "Coiling": coiling_list[:30]
    }

    with open(LIVE_BREAKOUTS_FILE, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    print(f"✅ Saved live scan: {len(confirmed_breakouts)} Breakouts, {len(coiling_list)} Watchlist items.")

    # ---------------- TELEGRAM DISPATCH ----------------
    # 1. Real-time alert when a new breakout triggers
    for alert in new_alerts_to_send:
        msg = (
            f"🚀 <b>CONFIRMED BREAKOUT TRIGGER</b>\n"
            f"• <b>Symbol</b>: {alert['symbol']} ({alert['universe']} | {alert['setup']})\n"
            f"• <b>Pivot Ceiling</b>: ₹{alert['pivot_ceiling']:,.2f}\n"
            f"• <b>Live LTP</b>: ₹{alert['ltp']:,.2f} ({alert['dist_pivot_%']:+.2f}% vs Pivot)\n"
            f"• <b>Day High</b>: ₹{alert['day_high']:,.2f}\n"
            f"• <b>Stop Loss</b>: ₹{alert['stop_loss']:,.2f} (Risk: {alert['risk_%']}%)\n"
            f"• <b>Volume Pace</b>: <b>{alert['proj_vol_pace']}x SMA20</b>\n"
            f"✅ <i>Holding above pivot with confirmed volume expansion.</i>"
        )
        send_telegram_alert(msg)
        time.sleep(1)

    # 2. Comprehensive Progress Summary Broadcast (3:15 PM Closing Window)
    if elapsed_mins >= 345:
        summary_msg = (
            f"📊 <b>EOD BREAKOUT EXECUTION MONITOR</b>\n"
            f"🕒 <i>Time: {now_str}</i>\n"
            f"<b>Active Confirmed Breakouts ({len(confirmed_breakouts)}):</b>\n"
        )
        if confirmed_breakouts:
            for b in confirmed_breakouts:
                summary_msg += (
                    f"🟢 <b>{b['symbol']}</b>: ₹{b['ltp']:,.2f} ({b['dist_pivot_%']:+.2f}%) | "
                    f"Pace: {b['proj_vol_pace']}x\n"
                )
        else:
            summary_msg += "<i>No stocks held above pivot with volume confirmation today.</i>\n"

        # List wicked-out stocks so you see rejections transparently
        wicked_stocks = [c for c in coiling_list if c["action"] == "WICK REJECTED"][:5]
        if wicked_stocks:
            summary_msg += "\n<b>⚠️ Upper Wick Rejections (Closed Below Pivot):</b>\n"
            for w in wicked_stocks:
                summary_msg += f"🔴 {w['symbol']}: High ₹{w['day_high']:,.2f} ➔ LTP ₹{w['ltp']:,.2f} ({w['dist_pivot_%']:+.2f}%)\n"

        send_telegram_alert(summary_msg)

if __name__ == "__main__":
    run_live_intraday_scan()
