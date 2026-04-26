"""
KPI Assignment - NASDAQ Stock Data Quality Assessment
======================================================
5 Companies | Yahoo Finance API | KPI: Completeness, Latency, Accuracy, Consistency
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import json
import os

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 1. COMPANY SELECTION  (5 NASDAQ tickers, each with a DIFFERENT time period)
# ─────────────────────────────────────────────────────────────────────────────
COMPANIES = [
    {"ticker": "AAPL",  "name": "Apple Inc.",            "sector": "Technology",
     "start": "2024-04-26", "end": "2025-04-26"},   # 1 full year

    {"ticker": "MSFT",  "name": "Microsoft Corporation", "sector": "Technology",
     "start": "2024-07-01", "end": "2025-04-26"},   # ~9 months

    {"ticker": "NVDA",  "name": "NVIDIA Corporation",    "sector": "Semiconductors",
     "start": "2024-10-01", "end": "2025-04-26"},   # ~6 months

    {"ticker": "AMZN",  "name": "Amazon.com Inc.",       "sector": "Consumer Discretionary",
     "start": "2025-01-01", "end": "2025-04-26"},   # ~4 months

    {"ticker": "META",  "name": "Meta Platforms Inc.",   "sector": "Communication Services",
     "start": "2025-02-01", "end": "2025-04-26"},   # ~3 months
]

COLLECTION_TIMESTAMP = datetime.now()

# ─────────────────────────────────────────────────────────────────────────────
# 2. DATA COLLECTION
# ─────────────────────────────────────────────────────────────────────────────

def collect_data(companies):
    """Download OHLCV data from Yahoo Finance for each company."""
    all_data = {}
    meta = []

    for c in companies:
        print(f"  Fetching {c['ticker']} ({c['name']}) …")
        ticker_obj = yf.Ticker(c["ticker"])
        df = ticker_obj.history(start=c["start"], end=c["end"], auto_adjust=True)

        if df.empty:
            print(f"    WARNING: No data returned for {c['ticker']}")
            continue

        df.index = pd.to_datetime(df.index).tz_localize(None)   # strip tz for clean CSV
        df = df[["Open", "High", "Low", "Close", "Volume"]]
        df["Ticker"] = c["ticker"]

        # Expected trading days (approximate weekday count)
        start_dt = pd.Timestamp(c["start"])
        end_dt   = pd.Timestamp(c["end"])
        expected_days = len(pd.bdate_range(start=start_dt, end=end_dt))

        all_data[c["ticker"]] = df
        meta.append({
            **c,
            "rows_collected": len(df),
            "expected_trading_days": expected_days,
            "first_date": str(df.index.min().date()),
            "last_date":  str(df.index.max().date()),
            "collection_ts": str(COLLECTION_TIMESTAMP),
        })

        print(f"    ✓ {len(df)} rows  |  {c['start']} → {c['end']}")

    return all_data, meta


# ─────────────────────────────────────────────────────────────────────────────
# 3. KPI CALCULATIONS
# ─────────────────────────────────────────────────────────────────────────────

def kpi_completeness(df, expected_days):
    """
    Completeness = (actual rows / expected trading days) × 100
    Also checks for NaN cells within returned rows.
    """
    row_completeness = round(len(df) / expected_days * 100, 2) if expected_days else 0
    nan_ratio        = round(df.isnull().mean().mean() * 100, 2)
    cell_completeness = round(100 - nan_ratio, 2)
    overall           = round((row_completeness + cell_completeness) / 2, 2)
    return {
        "row_completeness_pct":  row_completeness,
        "cell_completeness_pct": cell_completeness,
        "overall_completeness_pct": overall,
        "missing_rows": max(0, expected_days - len(df)),
        "nan_cells":    int(df.isnull().sum().sum()),
    }


def kpi_latency(meta_entry):
    """
    Latency = difference between the last data point date and today's date.
    Lower is better. For end-of-day data, 1 business day is the baseline.
    """
    last_date    = pd.Timestamp(meta_entry["last_date"])
    today        = pd.Timestamp(COLLECTION_TIMESTAMP.date())
    latency_days = (today - last_date).days
    business_lag = len(pd.bdate_range(start=last_date + timedelta(days=1), end=today))
    rating       = "Excellent" if business_lag <= 1 else "Good" if business_lag <= 3 else "Acceptable" if business_lag <= 7 else "Poor"
    return {
        "last_data_point":       str(last_date.date()),
        "collection_date":       str(today.date()),
        "calendar_lag_days":     latency_days,
        "business_day_lag":      business_lag,
        "latency_rating":        rating,
    }


def kpi_accuracy(df):
    """
    Accuracy checks internal logical consistency of OHLCV values:
      - High >= Low
      - High >= Open  and  High >= Close
      - Low  <= Open  and  Low  <= Close
      - Volume >= 0
    Score = fraction of rows passing ALL checks.
    """
    checks = (
        (df["High"] >= df["Low"])  &
        (df["High"] >= df["Open"]) &
        (df["High"] >= df["Close"]) &
        (df["Low"]  <= df["Open"]) &
        (df["Low"]  <= df["Close"]) &
        (df["Volume"] >= 0)
    )
    passed = int(checks.sum())
    total  = len(df)
    return {
        "rows_passing_all_checks": passed,
        "total_rows": total,
        "accuracy_pct": round(passed / total * 100, 2) if total else 0,
        "anomaly_rows": int((~checks).sum()),
    }


def kpi_consistency(df):
    """
    Consistency measures:
      - Dtype uniformity (all numeric cols should be float/int)
      - No duplicate index (dates)
      - Volume outliers: rows where Volume > mean + 3*std
      - Price outliers: rows where |daily_return| > 20%
    Score penalises each inconsistency type.
    """
    # dtype check
    expected_dtypes = {"Open": "float", "High": "float", "Low": "float",
                       "Close": "float", "Volume": "float"}
    dtype_ok = all(pd.api.types.is_numeric_dtype(df[c]) for c in ["Open","High","Low","Close","Volume"])

    # duplicates
    dup_count = int(df.index.duplicated().sum())

    # volume outliers
    vol_mean, vol_std = df["Volume"].mean(), df["Volume"].std()
    vol_outliers = int((df["Volume"] > vol_mean + 3 * vol_std).sum())

    # price outliers
    daily_ret = df["Close"].pct_change().abs()
    price_outliers = int((daily_ret > 0.20).sum())

    # Composite score (penalise each category)
    n = len(df)
    score = 100
    if not dtype_ok:   score -= 20
    if dup_count > 0:  score -= min(20, dup_count * 5)
    score -= min(20, vol_outliers / n * 100)
    score -= min(20, price_outliers / n * 100)
    score = max(0, round(score, 2))

    return {
        "dtype_uniformity": "Pass" if dtype_ok else "Fail",
        "duplicate_dates": dup_count,
        "volume_outliers": vol_outliers,
        "price_outliers_gt20pct": price_outliers,
        "consistency_score": score,
    }


def descriptive_stats(df):
    return df[["Open","High","Low","Close","Volume"]].describe().round(4).to_dict()


# ─────────────────────────────────────────────────────────────────────────────
# 4. VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def plot_all(all_data, results, output_dir="kpi_output"):
    os.makedirs(output_dir, exist_ok=True)
    palette = ["#2563EB","#16A34A","#DC2626","#D97706","#7C3AED"]

    # ── 4a. Close price per company ──────────────────────────────────────────
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    axes = axes.flatten()
    fig.suptitle("NASDAQ Share Prices – Close", fontsize=16, fontweight="bold", y=1.01)

    for i, (ticker, df) in enumerate(all_data.items()):
        ax = axes[i]
        ax.plot(df.index, df["Close"], color=palette[i], linewidth=1.5)
        ax.fill_between(df.index, df["Close"], alpha=0.12, color=palette[i])
        ax.set_title(ticker, fontweight="bold")
        ax.set_xlabel("Date"); ax.set_ylabel("Price (USD)")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(True, alpha=0.3)

    axes[-1].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/01_close_prices.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved 01_close_prices.png")

    # ── 4b. Volume bar charts ─────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    axes = axes.flatten()
    fig.suptitle("Trading Volume", fontsize=16, fontweight="bold")

    for i, (ticker, df) in enumerate(all_data.items()):
        ax = axes[i]
        ax.bar(df.index, df["Volume"] / 1e6, color=palette[i], alpha=0.7, width=1)
        ax.set_title(ticker, fontweight="bold")
        ax.set_xlabel("Date"); ax.set_ylabel("Volume (M shares)")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(True, alpha=0.3, axis="y")

    axes[-1].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/02_volume.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved 02_volume.png")

    # ── 4c. KPI Dashboard ─────────────────────────────────────────────────────
    tickers  = list(results.keys())
    comp_vals = [results[t]["completeness"]["overall_completeness_pct"] for t in tickers]
    acc_vals  = [results[t]["accuracy"]["accuracy_pct"] for t in tickers]
    cons_vals = [results[t]["consistency"]["consistency_score"] for t in tickers]
    lat_vals  = [results[t]["latency"]["business_day_lag"] for t in tickers]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle("KPI Dashboard – All 5 Companies", fontsize=15, fontweight="bold")

    def bar_kpi(ax, vals, title, unit, color):
        bars = ax.barh(tickers, vals, color=color, alpha=0.85)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel(unit)
        ax.bar_label(bars, fmt="%.1f", padding=3)
        ax.set_xlim(0, max(vals) * 1.2 if vals else 10)
        ax.grid(True, axis="x", alpha=0.3)

    bar_kpi(axes[0], comp_vals, "Completeness",  "%",      "#2563EB")
    bar_kpi(axes[1], acc_vals,  "Accuracy",       "%",      "#16A34A")
    bar_kpi(axes[2], cons_vals, "Consistency",    "score",  "#7C3AED")
    bar_kpi(axes[3], lat_vals,  "Latency\n(lower=better)", "biz days", "#DC2626")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/03_kpi_dashboard.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved 03_kpi_dashboard.png")

    # ── 4d. Candlestick-style OHLC for AAPL (first company) ──────────────────
    ticker = list(all_data.keys())[0]
    df_plot = all_data[ticker].tail(60).copy()
    fig, ax = plt.subplots(figsize=(14, 5))
    up   = df_plot[df_plot["Close"] >= df_plot["Open"]]
    down = df_plot[df_plot["Close"] <  df_plot["Open"]]
    w = 0.6
    ax.bar(up.index,   up["Close"]   - up["Open"],   w, bottom=up["Open"],   color="#16A34A", alpha=0.85)
    ax.bar(down.index, down["Close"] - down["Open"], w, bottom=down["Open"], color="#DC2626", alpha=0.85)
    ax.vlines(up.index,   up["Low"],   up["High"],   color="#16A34A", linewidth=0.8)
    ax.vlines(down.index, down["Low"], down["High"], color="#DC2626", linewidth=0.8)
    ax.set_title(f"{ticker} – Last 60 Trading Days (OHLC)", fontweight="bold")
    ax.set_ylabel("Price (USD)")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/04_ohlc_{ticker}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved 04_ohlc_{ticker}.png")

    # ── 4e. Correlation heatmap of Close prices ───────────────────────────────
    close_df = pd.DataFrame({t: df["Close"] for t, df in all_data.items()})
    corr = close_df.corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                ax=ax, linewidths=0.5, square=True)
    ax.set_title("Correlation of Close Prices", fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/05_correlation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved 05_correlation.png")


# ─────────────────────────────────────────────────────────────────────────────
# 5. SAVE DATA & RESULTS
# ─────────────────────────────────────────────────────────────────────────────

def save_outputs(all_data, results, meta, output_dir="kpi_output"):
    os.makedirs(output_dir, exist_ok=True)

    # Combined CSV
    combined = pd.concat(all_data.values())
    combined.to_csv(f"{output_dir}/stock_data_all.csv")
    print(f"  Saved stock_data_all.csv  ({len(combined)} rows)")

    # Per-ticker CSVs
    for ticker, df in all_data.items():
        df.to_csv(f"{output_dir}/{ticker}_data.csv")

    # KPI results JSON
    with open(f"{output_dir}/kpi_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("  Saved kpi_results.json")

    # Meta JSON
    with open(f"{output_dir}/dataset_meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print("  Saved dataset_meta.json")


# ─────────────────────────────────────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    OUTPUT = "kpi_output"
    print("\n══════════════════════════════════════════════")
    print("  KPI ASSIGNMENT – NASDAQ Dataset Collection  ")
    print("══════════════════════════════════════════════\n")

    print("► Step 1: Collecting data from Yahoo Finance …")
    all_data, meta = collect_data(COMPANIES)
    meta_by_ticker = {m["ticker"]: m for m in meta}

    print("\n► Step 2: Computing KPIs …")
    results = {}
    for c in COMPANIES:
        t = c["ticker"]
        if t not in all_data:
            continue
        df  = all_data[t]
        m   = meta_by_ticker[t]
        kpi = {
            "company":     c["name"],
            "ticker":      t,
            "sector":      c["sector"],
            "period":      f"{c['start']} → {c['end']}",
            "completeness": kpi_completeness(df, m["expected_trading_days"]),
            "latency":      kpi_latency(m),
            "accuracy":     kpi_accuracy(df),
            "consistency":  kpi_consistency(df),
            "descriptive":  descriptive_stats(df),
        }
        results[t] = kpi
        print(f"  {t}: completeness={kpi['completeness']['overall_completeness_pct']}% | "
              f"accuracy={kpi['accuracy']['accuracy_pct']}% | "
              f"consistency={kpi['consistency']['consistency_score']} | "
              f"latency={kpi['latency']['business_day_lag']} biz days")

    print("\n► Step 3: Generating visualisations …")
    plot_all(all_data, results, OUTPUT)

    print("\n► Step 4: Saving data & results …")
    save_outputs(all_data, results, meta, OUTPUT)

    print(f"\n✅  All outputs saved to ./{OUTPUT}/")
    print("    Upload to GitHub and paste kpi_results.json into your data card.\n")


if __name__ == "__main__":
    main()
