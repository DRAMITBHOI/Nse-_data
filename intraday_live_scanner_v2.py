import os
import sys
import re
import json
import time
import datetime
import urllib.request
import urllib.parse
import pandas as pd
import requests

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

LIVE_BREAKOUTS_V2_FILE = os.path.join(DATA_DIR, "live_intraday_breakouts_v2.json")

IST = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
EXCLUDED_SYMBOLS = {"LIQUIDCASE", "LIQGRWBEES", "LIQUIDBEES", "GOLDBEES", "NIFTYBEES", "BANKBEES", "SILVERBEES"}

def get_ist_now():
    return datetime.datetime.now(IST)

def get_market_session_elapsed():
    now = get_ist_now()
    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    if now < market_open:
        return 1
    elif now > market_close:
        return 375
    else:
        elapsed = int((now - market_open).total_seconds() / 60)
        return max(1, min(375, elapsed))

# ==============================================================================
# IMPROVEMENT 2: Authentic U-Curve Time-Weighted Volume Pace Curve
# ==============================================================================
def get_u_curve_expected_volume_fraction(elapsed_mins):
    """
    Intraday Volume U-Distribution in Indian Equity Markets:
    - 0-75m   (09:15 - 10:30): Morning opening drive (~32% of total day's volume)
    - 75-225m (10:30 - 01:00): Midday consolidation/lull (+28% -> 60% cumulative)
    - 225-330m(01:00 - 02:45): Afternoon pre-positioning (+20% -> 80% cumulative)
    - 330-375m(02:45 - 03:30): Heavy closing institutional auction surge (+20% -> 100%)
    """
    if elapsed_mins <= 0:
        return 0.02
    elif elapsed_mins <= 75:
        return 0.05 + (elapsed_mins / 75.0) * 0.27
    elif elapsed_mins <= 225:
        return 0.32 + ((elapsed_mins - 75.0) / 150.0) * 0.28
    elif elapsed_mins <= 330:
        return 0.60 + ((elapsed_mins - 225.0) / 105.0) * 0.20
    else:
        return 0.80 + ((elapsed_mins - 330.0) / 45.0) * 0.20

class LivePriceClient:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9"
        })

    def fetch_live_quote(self, symbol):
        # 1. Primary: Google Finance
        try:
            url = f"https://www.google.com/finance/quote/{symbol}:NSE"
            r = self.session.get(url, timeout=5)
            if r.status_code == 200:
                html = r.text
                price_match = re.search(r'data-last-price="([\d\.]+)"', html)
                if price_match:
                    ltp = float(price_match.group(1))
                    day_high = ltp
                    day_low = ltp

                    range_match = re.search(r'Day range.*?₹([\d\.,]+)\s*-\s*₹([\d\.,]+)', html, re.DOTALL)
                    if range_match:
                        day_low = float(range_match.group(1).replace(",", ""))
                        day_high = float(range_match.group(2).replace(",", ""))

                    day_vol = 0.0
                    vol_match = re.search(r'Volume</div><div class="P6K39c">([\d\.,MKmk]+)</div>', html)
                    if vol_match:
                        v_str = vol_match.group(1).replace(",", "").upper()
                        if "M" in v_str:
                            day_vol = float(v_str.replace("M", "")) * 1_000_000
                        elif "K" in v_str:
                            day_vol = float(v_str.replace("K", "")) * 1_000
                        else:
                            day_vol = float(v_str)

                    if ltp > 0:
                        return {"ltp": ltp, "day_high": max(day_high, ltp), "day_low": min(day_low, ltp), "day_vol": day_vol}
        except Exception:
            pass

        # 2. Secondary: Yahoo Chart endpoint
        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}.NS?interval=1d&range=1d"
            r = self.session.get(url, timeout=5)
            if r.status_code == 200:
                data = r.json()
                meta = data.get("chart", {}).get("result", [{}])[0].get("meta", {})
                ltp = float(meta.get("regularMarketPrice", 0) or 0)
                day_high = float(meta.get("regularMarketDayHigh", ltp) or ltp)
                day_low = float(meta.get("regularMarketDayLow", ltp) or ltp)
                day_vol = float(meta.get("regularMarketVolume", 0) or 0)
                if ltp > 0:
                    return {"ltp": ltp, "day_high": max(day_high, ltp), "day_low": min(day_low, ltp), "day_vol": day_vol}
        except Exception:
            pass

        return None

# ==============================================================================
# IMPROVEMENT 3: Live Nifty 50 Market Breadth Gate
# ==============================================================================
def get_nifty50_market_sentiment(client: LivePriceClient):
    """
    Fetches live NIFTY 50 change %.
    If broad market is tumbling (down > 0.60%), we raise volume hurdles.
    """
    try:
        url = "https://www.google.com/finance/quote/NIFTY_50:INDEXNSE"
        r = client.session.get(url, timeout=5)
        if r.status_code == 200:
            match = re.search(r'data-last-price="([\d\.]+)"', r.text)
            prev_match = re.search(r'Previous close.*?₹?([\d\.,]+)', r.text, re.DOTALL)
            if match and prev_match:
                curr = float(match.group(1))
                prev = float(prev_match.group(1).replace(",", ""))
                chg_pct = round(((curr - prev) / prev) * 100.0, 2)
                return chg_pct
    except Exception:
        pass
    return 0.0

# ==============================================================================
# IMPROVEMENT 4: Shelf Tightness & Consolidation Quality Extraction
# ==============================================================================
def load_comprehensive_targets():
    targets = {}
    sources = [
        ("https://raw.githubusercontent.com/DRAMITBHOI/Nse-_data/main/data/swing_ultra750_results.json", "NIFTY 750"),
        ("https://raw.githubusercontent.com/DRAMITBHOI/Nse-_data/main/data/swing_ultraN750_results.json", "NON-N750")
    ]
    for url, default_univ in sources:
        try:
            cache_buster = f"?_ts={int(time.time() * 1000)}"
            r = requests.get(url + cache_buster, headers={"User-Agent": "Mozilla/5.0", "Cache-Control": "no-cache"}, timeout=12)
            if r.status_code == 200:
                d = r.json()
                items = list(d.get("Watchlist", []))
                fresh = d.get("Fresh_Triggers", [])
                if isinstance(fresh, list):
                    items.extend(fresh)
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    sym = str(item.get("symbol", "")).strip().upper()
                    if not sym or sym.endswith("BEES") or sym in EXCLUDED_SYMBOLS:
                        continue
                    pivot = float(item.get("pivot_ceiling") or item.get("pivot_target") or item.get("pivot") or 0.0)
                    if pivot <= 0:
                        continue
                    stop_loss = float(item.get("stop_loss", pivot * 0.92))
                    setup = item.get("setup_class") or item.get("type") or "LAUNCHPAD COIL"
                    universe = item.get("universe", default_univ)
                    
                    # Extract shelf tightness and volatility spread
                    spread_pct = float(item.get("spread_%", item.get("shelf_spread_%", 6.0)) or 6.0)

                    targets[sym] = {
                        "symbol": sym,
                        "universe": universe,
                        "setup": setup,
                        "pivot_ceiling": pivot,
                        "stop_loss": stop_loss,
                        "shelf_spread_%": spread_pct
                    }
        except Exception:
            pass

    if os.path.exists(LIVE_BREAKOUTS_V2_FILE):
        try:
            with open(LIVE_BREAKOUTS_V2_FILE, "r", encoding="utf-8") as f:
                prior_state = json.load(f)
            for item in prior_state.get("Coiling", []):
                sym = str(item.get("symbol", "")).strip().upper()
                if sym and sym not in targets and (not sym.endswith("BEES")) and (sym not in EXCLUDED_SYMBOLS):
                    pivot = float(item.get("pivot_ceiling", 0.0))
                    stop_loss = float(item.get("stop_loss", pivot * 0.92))
                    ltp = float(item.get("ltp", pivot)) if pd.notnull(item.get("ltp")) else pivot
                    if ltp >= stop_loss and pivot > 0:
                        targets[sym] = {
                            "symbol": sym,
                            "universe": item.get("universe", "NON-N750"),
                            "setup": item.get("setup", "LAUNCHPAD COIL"),
                            "pivot_ceiling": pivot,
                            "stop_loss": stop_loss,
                            "shelf_spread_%": float(item.get("shelf_spread_%", 6.0))
                        }
        except Exception:
            pass

    return targets

def run_live_intraday_scan_v2():
    targets = load_comprehensive_targets()
    if not targets:
        print("❌ No watchlisted setups found to scan.")
        return

    now_dt = get_ist_now()
    today_date = now_dt.strftime("%Y-%m-%d")
    now_str = now_dt.strftime("%Y-%m-%d %I:%M:%S %p IST")
    elapsed_mins = get_market_session_elapsed()

    # ==========================================================================
    # IMPROVEMENT 5: Multi-Tick Persistence Tracker
    # ==========================================================================
    prior_consecutive_hits = {}
    if os.path.exists(LIVE_BREAKOUTS_V2_FILE):
        try:
            with open(LIVE_BREAKOUTS_V2_FILE, "r", encoding="utf-8") as f:
                prev_file = json.load(f)
                if prev_file.get("Date") == today_date:
                    prior_consecutive_hits = prev_file.get("Consecutive_Hits", {})
        except Exception:
            pass

    live_client = LivePriceClient()
    nifty_chg = get_nifty50_market_sentiment(live_client)
    is_nifty_weak = nifty_chg < -0.50
    print(f"📊 Market Condition: Nifty 50 {nifty_chg:+.2f}% | Weak Market Filter: {'ON' if is_nifty_weak else 'OFF'}")

    expected_fraction = get_u_curve_expected_volume_fraction(elapsed_mins)

    confirmed_breakouts = []
    radar_candidates = []
    coiling_list = []
    current_consecutive_hits = {}

    for sym, meta in targets.items():
        pivot = float(meta["pivot_ceiling"])
        structural_sl = float(meta["stop_loss"])
        setup = meta["setup"]
        universe = meta["universe"]
        shelf_spread = float(meta.get("shelf_spread_%", 6.0))
        is_v_reversal = "V-REVERSAL" in setup

        quote = live_client.fetch_live_quote(sym)
        time.sleep(0.06)

        ltp = None
        day_high = None
        day_low = None
        day_vol = 0.0
        sma20_vol = 50000.0

        if quote:
            ltp = quote["ltp"]
            day_high = quote["day_high"]
            day_low = quote["day_low"]
            day_vol = quote["day_vol"]

        local_json = os.path.join(DATA_DIR, f"{sym}.json")
        if os.path.exists(local_json):
            try:
                with open(local_json, "r", encoding="utf-8") as fp:
                    raw = json.load(fp)
                if raw and isinstance(raw, list):
                    vols = [float(x.get("volume", 0) or 0) for x in raw[-20:]]
                    if vols:
                        sma20_vol = sum(vols) / len(vols)
                    if ltp is None or ltp <= 0:
                        last_c = float(raw[-1].get("close", pivot))
                        ltp = last_c
                        day_high = float(raw[-1].get("high", last_c))
                        day_low = float(raw[-1].get("low", last_c))
            except Exception:
                pass

        if ltp is None or pd.isna(ltp) or ltp <= 0:
            ltp = pivot
            day_high = pivot
            day_low = pivot

        if day_high is None or pd.isna(day_high) or day_high < ltp:
            day_high = ltp
        if day_low is None or pd.isna(day_low) or day_low > ltp:
            day_low = ltp

        sma20_vol = max(1.0, sma20_vol)
        projected_vol = day_vol / expected_fraction
        vol_pace = round(projected_vol / sma20_vol, 2) if sma20_vol > 0 else 1.0

        clearance_pct = round(((ltp - pivot) / pivot) * 100.0, 2)
        drop_from_high_pct = round(((day_high - ltp) / day_high) * 100.0, 2) if day_high > 0 else 0.0

        # Dynamic Stop Loss
        if is_v_reversal:
            effective_sl = max(structural_sl, round(day_low * 0.995, 2))
        else:
            effective_sl = structural_sl

        risk_pct = round(((ltp - effective_sl) / ltp) * 100.0, 2) if ltp > effective_sl else 8.0

        # ======================================================================
        # CRITERIA REFINEMENTS (Launchpad Coils vs. V-Reversals)
        # ======================================================================
        # 1. Base Volume Threshold
        if is_v_reversal:
            base_pace = 2.0 if universe == "NIFTY 750" else 1.75
            min_clearance = 0.40
            max_clearance = 3.50  # Prevent buying exhausted climax runs
        else:
            # Launchpad Coils: tight bases require less volume, loose bases require more
            base_pace = 1.30 if universe == "NIFTY 750" else 0.85
            if shelf_spread > 7.0:  # Improvement 4: Wide shelf penalty
                base_pace += 0.35
            
            # Improvement 1: Closing Auction Discrepancy Buffer
            min_clearance = 0.25 if elapsed_mins < 330 else 0.50
            max_clearance = 6.00

        # Improvement 3: Broad Market Drag Protection
        if is_nifty_weak:
            base_pace += 0.30

        is_volume_confirmed = bool(vol_pace >= base_pace)
        cleared_pivot = bool(clearance_pct >= min_clearance and clearance_pct <= max_clearance)
        
        # Improvement 1b: Drop from intraday high filter (reject late closing dump)
        max_allowed_drop = 1.25 if elapsed_mins >= 330 else 2.00
        no_closing_fade = bool(drop_from_high_pct <= max_allowed_drop)
        wick_touched_pivot = bool(day_high >= pivot)

        # Multi-Tick Consecutive Counter (Improvement 5)
        if cleared_pivot:
            hits = prior_consecutive_hits.get(sym, 0) + 1
        else:
            hits = 0
        current_consecutive_hits[sym] = hits

        # During radar hours, require at least 2 ticks to confirm
        multi_tick_confirmed = hits >= 2 if (elapsed_mins < 330 and elapsed_mins > 45) else (hits >= 1)

        record = {
            "symbol": sym,
            "universe": universe,
            "setup": setup,
            "pivot_ceiling": round(pivot, 2),
            "ltp": round(ltp, 2),
            "day_high": round(day_high, 2),
            "day_low": round(day_low, 2),
            "stop_loss": round(effective_sl, 2),
            "risk_%": risk_pct,
            "shelf_spread_%": shelf_spread,
            "clearance_%": clearance_pct,
            "drop_from_high_%": drop_from_high_pct,
            "proj_vol_pace": vol_pace,
            "consecutive_ticks": hits
        }

        # Window Classification
        if elapsed_mins >= 330:  # 02:45 PM – 03:30 PM: Closing Execution Window
            if cleared_pivot and is_volume_confirmed and no_closing_fade:
                record["action"] = "CONFIRMED BREAKOUT"
                confirmed_breakouts.append(record)
            elif wick_touched_pivot and not cleared_pivot:
                record["action"] = "WICK REJECTED"
                coiling_list.append(record)
            else:
                record["action"] = "COILING"
                coiling_list.append(record)
        else:  # 09:15 AM – 02:45 PM: Radar Window
            if cleared_pivot and is_volume_confirmed and no_closing_fade and multi_tick_confirmed:
                record["action"] = "RADAR PIVOT CLEARANCE"
                radar_candidates.append(record)
            elif wick_touched_pivot and not cleared_pivot:
                record["action"] = "WICK TESTING"
                coiling_list.append(record)
            else:
                record["action"] = "COILING"
                coiling_list.append(record)

    coiling_list.sort(key=lambda x: x["clearance_%"], reverse=True)
    confirmed_breakouts.sort(key=lambda x: x["proj_vol_pace"], reverse=True)
    radar_candidates.sort(key=lambda x: x["proj_vol_pace"], reverse=True)

    output_payload = {
        "Date": today_date,
        "Scan_Timestamp": now_str,
        "Elapsed_Minutes": elapsed_mins,
        "Nifty50_Intraday_%": nifty_chg,
        "Consecutive_Hits": current_consecutive_hits,
        "Breakouts_All_Day": confirmed_breakouts if elapsed_mins >= 330 else radar_candidates,
        "Coiling": coiling_list[:50]
    }

    with open(LIVE_BREAKOUTS_V2_FILE, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    print(f"✅ [Scanner V2 Silent] {len(confirmed_breakouts)} Confirmed Breakouts, {len(radar_candidates)} Radar, {len(coiling_list)} Coils.")

if __name__ == "__main__":
    run_live_intraday_scan_v2()
