# V-Reversal Empirical Derivation Report
- **Total Reversal Candidates:** 50,549
- **Baseline Win Rate:** 53.29%

## 1. Feature Distributions: Winners vs. Losers
| Feature | Winner Median | Loser Median | Winner Core Range (25th–75th) |
| :--- | :--- | :--- | :--- |
| Capitulation Drop Depth (%) | **12.48** | 12.45 | 10.48 - 15.73 |
| Trigger Volume Surge (x) | **1.11** | 0.89 | 0.69 - 1.95 |
| Trigger Delivery Volume Surge (x) | **1.11** | 0.9 | 0.72 - 1.81 |
| Delivery % Surge vs 50d (x) | **0.95** | 0.97 | 0.76 - 1.11 |
| Volume at Lowest Low Bar (x) | **0.73** | 0.72 | 0.47 - 1.14 |
| Candle Range Closeness | **0.79** | 0.75 | 0.65 - 0.88 |
| 60d Excess Return vs Nifty | **0.08** | 0.04 | -0.04 - 0.26 |
| Stop Loss Distance (%) | **7.92** | 7.44 | 6.05 - 10.28 |

## 2. Reclaim Horizon Comparison
| Reclaim Trigger | Total Setups | Win Rate (%) |
| :--- | :--- | :--- |
| Prior 2-Day High | 50,549 | **53.29%** |
| Prior 3-Day High | 29,625 | **64.28%** |
| Prior 5-Day High | 19,102 | **68.13%** |

## 3. Conviction Score Quartiles vs. Performance
| Conviction Quartile | Trades | Win Rate (%) | Avg Trade PnL (Exit 2.2) |
| :--- | :--- | :--- | :--- |
| Q1 (Lowest) | 12,660 | 43.34% | **-0.42%** |
| Q2 (Mid-Low) | 12,628 | 48.83% | **+0.41%** |
| Q3 (Mid-High) | 12,628 | 53.39% | **+0.88%** |
| Q4 (Highest) | 12,633 | 67.61% | **+3.07%** |

## 4. Exit Engine Comparison (V-Reversals)
| Architecture | Win Rate (%) | Avg Return (%) | Profit Factor | Expectancy |
| :--- | :--- | :--- | :--- | :--- |
| Exit 2.2 Engine | 34.14% | 0.98% | **1.35** | **0.98%** |
| Staged 2R + BE + 20 EMA | 39.77% | 0.97% | **1.2** | **0.97%** |