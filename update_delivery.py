from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import sys
import urllib.request
import pandas as pd

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)


def fetch_mto_for_date(d_str):
  url = f"https://nsearchives.nseindia.com/archives/equities/mto/MTO_{d_str}.DAT"
  headers = {
      "User-Agent": (
          "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
      )
  }
  req = urllib.request.Request(url, headers=headers)
  date_iso = f"{d_str[4:]}-{d_str[2:4]}-{d_str[:2]}"
  day_data = {}

  try:
    with urllib.request.urlopen(req, timeout=8) as resp:
      lines = resp.read().decode("utf-8", errors="ignore").splitlines()
      for line in lines:
        parts = [p.strip() for p in line.split(",")]
        # Record Type 20 = Detailed security delivery position
        if len(parts) >= 7 and parts[0] == "20" and parts[3] in ["EQ", "BE"]:
          sym = parts[2]
          day_data[sym] = {
              "Date": date_iso,
              "Delivery_Volume": int(parts[5]),
              "Delivery_Pct": float(parts[6]),
          }
  except Exception:
    pass
  return day_data


def update_all_stocks(days_back=10):
  date_range = pd.date_range(end=pd.Timestamp.now(), periods=days_back, freq="B")
  date_strings = [d.strftime("%d%m%Y") for d in date_range]

  print(
      f"Fetching NSE archives for {len(date_strings)} business days...",
      flush=True,
  )
  daily_results = []

  with ThreadPoolExecutor(max_workers=10) as executor:
    futures = [
        executor.submit(fetch_mto_for_date, d_str) for d in date_strings
    ]
    for f in as_completed(futures):
      res = f.result()
      if res:
        daily_results.append(res)

  # Group daily results by stock symbol
  stock_updates = {}
  for day_dict in daily_results:
    for sym, rec in day_dict.items():
      if sym not in stock_updates:
        stock_updates[sym] = []
      stock_updates[sym].append(rec)

  print(
      f"Saving records for {len(stock_updates)} symbols into '{DATA_DIR}/'...",
      flush=True,
  )

  for sym, new_records in stock_updates.items():
    file_path = os.path.join(DATA_DIR, f"{sym}.json")
    existing_records = {}

    if os.path.exists(file_path):
      try:
        with open(file_path, "r") as fp:
          for item in json.load(fp):
            existing_records[item["Date"]] = item
      except Exception:
        pass

    for rec in new_records:
      existing_records[rec["Date"]] = rec

    sorted_records = sorted(existing_records.values(), key=lambda x: x["Date"])

    with open(file_path, "w") as fp:
      json.dump(sorted_records, fp, separators=(",", ":"))

  print("Sync complete!", flush=True)


if __name__ == "__main__":
  days = int(sys.argv[1]) if len(sys.argv) > 1 else 10
  update_all_stocks(days_back=days)
