# Swing Ultra Empirical Derivation Report
- **Total Breakouts Analyzed:** 38,994
- **Baseline Win Rate:** 57.45%

## 1. Feature Distributions: Winners vs. Losers
| Feature | Winner Median | Loser Median | Winner Core Range (25th–75th) |
| :--- | :--- | :--- | :--- |
| Base Length (Bars) | **8.0** | 8.0 | 8.0 - 8.0 |
| Base Spread (%) | **8.69** | 7.31 | 6.37 - 11.71 |
| Volume Dry-Up (VDU) Ratio | **0.41** | 0.44 | 0.3 - 0.52 |
| VCP ATR Ratio (5d/14d) | **1.08** | 1.02 | 0.94 - 1.23 |
| Trigger Volume Surge (x) | **1.72** | 1.31 | 1.1 - 2.89 |
| Delivery Volume Surge (x) | **1.6** | 1.28 | 1.06 - 2.49 |
| Delivery % Surge (x) | **0.9** | 0.95 | 0.72 - 1.06 |
| Base Accumulation Ratio | **2.09** | 2.03 | 1.26 - 3.68 |
| Candle Range Closeness | **0.8** | 0.78 | 0.66 - 0.88 |
| 60d Excess Return vs Nifty | **0.13** | 0.1 | 0.03 - 0.26 |
| Ceiling Distance to 120d High (%) | **5.43** | 5.55 | 0.13 - 13.07 |
| Initial Stop Loss Distance (%) | **8.0** | 6.82 | 5.99 - 10.49 |

## 2. Exit Architecture Performance Comparison
| Exit Strategy | Win Rate (%) | Avg Trade (%) | Profit Factor | Expectancy / Trade | Max Single Loss (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Exit 2.2 Engine (Current Baseline) | 46.19% | 3.68% | **2.73** | **3.68%** | -24.81% |
| Pure 20 EMA Close Trail | 45.98% | 3.42% | **3.0** | **3.42%** | -23.72% |
| Pure 10 EMA Close Trail | 49.28% | 2.51% | **2.95** | **2.51%** | -23.72% |
| Staged 2R Partial + Breakeven + 20 EMA | 52.09% | 4.0% | **2.14** | **4.0%** | -28.94% |