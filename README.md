# KPI Assignment — Dataset Quality Report

> **Course Assignment** | Data Quality KPIs for AI Training Datasets  
> **Submitted:** April 2026 | **Source:** Yahoo Finance API

---

## Repository Structure

```
nasdaq-kpi-assignment/
│
├── kpi_assignment.py        ← Main Python script (data collection + KPI computation)
├── datacard.html            ← Interactive Data Card (open in browser)
├── README.md                ← This file
│
└── kpi_output/              ← Generated after running the script
    ├── stock_data_all.csv       Combined dataset (all 5 tickers)
    ├── AAPL_data.csv
    ├── MSFT_data.csv
    ├── NVDA_data.csv
    ├── AMZN_data.csv
    ├── META_data.csv
    ├── kpi_results.json         Full KPI scores per ticker
    ├── dataset_meta.json        Collection metadata
    ├── 01_close_prices.png      Close price charts
    ├── 02_volume.png            Volume bar charts
    ├── 03_kpi_dashboard.png     KPI comparison dashboard
    ├── 04_ohlc_AAPL.png         OHLC candlestick (AAPL)
    └── 05_correlation.png       Correlation heatmap
```

---

## Company Selection (5 NASDAQ Tickers — Different Periods)

| Ticker | Company                  | Sector                  | Period                    | Duration   |
|--------|--------------------------|-------------------------|---------------------------|------------|
| AAPL   | Apple Inc.               | Technology              | Apr 26 2024 → Apr 26 2025 | 12 months  |
| MSFT   | Microsoft Corporation    | Technology              | Jul 1 2024 → Apr 26 2025  | ~9 months  |
| NVDA   | NVIDIA Corporation       | Semiconductors          | Oct 1 2024 → Apr 26 2025  | ~6 months  |
| AMZN   | Amazon.com Inc.          | Consumer Discretionary  | Jan 1 2025 → Apr 26 2025  | ~4 months  |
| META   | Meta Platforms Inc.      | Communication Services  | Feb 1 2025 → Apr 26 2025  | ~3 months  |

> Each company uses a **deliberately different time period** to introduce variability in dataset size, enabling richer KPI comparison.

---

## KPI Definitions

### (a) Completeness
> *What fraction of expected data is actually present?*

```
Row Completeness  = (Rows Collected / Expected Trading Days) × 100%
Cell Completeness = (1 − NaN ratio) × 100%
Overall           = Average of both
```

Expected trading days = business days in the period (excl. US market holidays).

### (b) Latency
> *How fresh is the data relative to the collection date?*

```
Calendar Lag   = Collection Date − Last Data Point Date
Business Lag   = count of business days in that gap

Rating:
  ≤1 biz day  → Excellent
  ≤3 biz days → Good
  ≤7 biz days → Acceptable
  >7 biz days → Poor
```

### (c) Accuracy
> *Are OHLCV values internally logically consistent?*

```
Checks per row:
  • High ≥ Low
  • High ≥ Open  AND  High ≥ Close
  • Low  ≤ Open  AND  Low  ≤ Close
  • Volume ≥ 0

Accuracy % = (rows passing ALL checks / total rows) × 100
```

### (d) Consistency
> *Are types, formats, and value distributions uniform?*

```
Start at 100. Deduct for:
  • Non-numeric columns              → −20 pts
  • Duplicate date indices           → −5 pts each (max −20)
  • Volume outliers (> μ + 3σ)      → proportional, max −20
  • |Daily return| > 20%            → proportional, max −20
```

---

## KPI Summary (Reference Values)

| Ticker | Completeness | Latency       | Accuracy | Consistency |
|--------|-------------|---------------|----------|-------------|
| AAPL   | ~99%        | 1 biz day  | 100%     | ~97/100     |
| MSFT   | ~99%        | 1 biz day  | 100%     | ~98/100     |
| NVDA   | ~98%        | 1 biz day  | 100%     | ~95/100     |
| AMZN   | ~98%        | 1 biz day  | 100%     | ~97/100     |
| META   | ~97%        | 1 biz day  | 100%     | ~97/100     |

---

## Data Source Details

| Field             | Value                                              |
|-------------------|----------------------------------------------------|
| API / Library     | `yfinance` (Python wrapper for Yahoo Finance)      |
| Data Type         | Historical End-of-Day (EOD) OHLCV                 |
| Adjustment        | `auto_adjust=True` (splits + dividends adjusted)  |
| Frequency         | Daily (trading days only)                          |
| Currency          | USD                                                |
| Exchange          | NASDAQ                                             |
| Access            | Free, unauthenticated HTTP                         |

---

## Conclusion

The Yahoo Finance API provides **high-quality, low-latency financial data** suitable for AI/ML training datasets. Across all four KPIs:

- **Completeness** was 97–99% (minor gaps from market holidays, not true missing data)
- **Latency** was universally Excellent (1 business day — the theoretical minimum for EOD feeds)
- **Accuracy** scored 100% — all OHLCV relationships are logically consistent
- **Consistency** scored 95–98/100 — all columns are correctly typed; minor volume outliers on earnings days are genuine market events, not data errors

The dataset is suitable for academic and research use in financial machine learning, time series forecasting, and AI training pipeline design.

---
<img width="970" height="572" alt="Screenshot 2026-04-26 at 15 35 42" src="https://github.com/user-attachments/assets/f466128c-d3d0-4d0e-a90e-814a4a377274" />

*Data Card: see [`datacard.html`](./index.html)*  
*License: Educational use only*
