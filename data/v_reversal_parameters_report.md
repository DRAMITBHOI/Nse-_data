# V-Reversal Empirical Derivation Report
- **Total Reversal Candidates:** 14,859
- **Baseline Win Rate:** 48.93%

## 1. Feature Distributions: Winners vs. Losers
| Feature | Winner Median | Loser Median | Winner Core Range (25th–75th) |
| :--- | :--- | :--- | :--- |
| Capitulation Drop Depth (%) | **11.9** | 11.86 | 10.21 - 14.53 |
| Trigger Volume Surge (x) | **1.05** | 0.91 | 0.71 - 1.74 |
| Trigger Delivery Volume Surge (x) | **1.04** | 0.9 | 0.71 - 1.61 |
| Delivery % Surge vs 50d (x) | **0.94** | 0.95 | 0.75 - 1.11 |
| Volume at Lowest Low Bar (x) | **0.81** | 0.83 | 0.54 - 1.21 |
| Candle Range Closeness | **0.81** | 0.78 | 0.68 - 0.88 |
| 60d Excess Return vs Nifty | **0.05** | 0.02 | -0.06 - 0.2 |
| Stop Loss Distance (%) | **7.2** | 6.87 | 5.53 - 9.28 |

## 2. Reclaim Horizon Comparison
| Reclaim Trigger | Total Setups | Win Rate (%) |
| :--- | :--- | :--- |
| Prior 2-Day High | 14,859 | **48.93%** |
| Prior 3-Day High | 8,456 | **59.86%** |
| Prior 5-Day High | 5,363 | **62.39%** |

## 3. Conviction Score Quartiles vs. Performance
| Conviction Quartile | Trades | Win Rate (%) | Avg Trade PnL (Exit 2.2) |
| :--- | :--- | :--- | :--- |
| Q1 (Lowest) | 3,719 | 40.39% | **-0.35%** |
| Q2 (Mid-Low) | 3,714 | 45.72% | **+0.46%** |
| Q3 (Mid-High) | 3,716 | 49.3% | **+1.02%** |
| Q4 (Highest) | 3,710 | 60.35% | **+2.79%** |

## 4. Exit Engine Comparison (V-Reversals)
| Architecture | Win Rate (%) | Avg Return (%) | Profit Factor | Expectancy |
| :--- | :--- | :--- | :--- | :--- |
| Exit 2.2 Engine | 35.48% | 0.98% | **1.4** | **0.98%** |
| Staged 2R + BE + 20 EMA | 41.42% | 1.3% | **1.31** | **1.3%** |