"""
download_data.py — Council of Alphas
Download 3 years of SOL/USDT 15m candles from Binance public REST API.

Uses /api/v3/klines directly (no ccxt) to get all 8 required columns:
open, high, low, close, volume, quote_volume, count, taker_buy_volume.

No API key required — this is public market data.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

# ── Constants ─────────────────────────────────────────────────────────────────
BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
SYMBOL = "SOLUSDT"
INTERVAL = "15m"
LIMIT = 1000                  # max candles per request
SLEEP_SECONDS = 0.2           # respect rate limits
YEARS_BACK = 3
MIN_EXPECTED_ROWS = 100_000   # 3 years of 15m ≈ 105,120 candles

# Output path (relative to project root)
OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = OUTPUT_DIR / "sol_usd_15m_3y.parquet"

# Binance kline array indices
# [0] open_time, [1] open, [2] high, [3] low, [4] close, [5] volume,
# [6] close_time, [7] quote_volume, [8] count, [9] taker_buy_volume,
# [10] taker_buy_quote_volume, [11] ignore
KEEP_INDICES = {
    "open_time": 0,
    "open": 1,
    "high": 2,
    "low": 3,
    "close": 4,
    "volume": 5,
    "quote_volume": 7,
    "count": 8,
    "taker_buy_volume": 9,
}


def _ms_now() -> int:
    """Current UTC time in milliseconds."""
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def _ms_years_ago(years: int) -> int:
    """UTC time N years ago in milliseconds."""
    dt = datetime.now(timezone.utc) - timedelta(days=years * 365)
    return int(dt.timestamp() * 1000)


def download_sol_usdt_15m() -> pd.DataFrame:
    """
    Download SOL/USDT 15m candles from Binance for the last 3 years.

    Returns
    -------
    pd.DataFrame
        Columns: open, high, low, close, volume, quote_volume, count, taker_buy_volume
        Index: open_time (UTC datetime)
    """
    start_ms = _ms_years_ago(YEARS_BACK)
    end_ms = _ms_now()
    cursor = start_ms

    all_rows: list[list] = []
    batch_num = 0

    print(f"Downloading {SYMBOL} {INTERVAL} candles from Binance...")
    print(f"  Start: {datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  End:   {datetime.fromtimestamp(end_ms / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print()

    while cursor < end_ms:
        params = {
            "symbol": SYMBOL,
            "interval": INTERVAL,
            "startTime": cursor,
            "limit": LIMIT,
        }

        resp = requests.get(BINANCE_KLINES_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if not data:
            break

        all_rows.extend(data)
        batch_num += 1

        first_ts = datetime.fromtimestamp(data[0][0] / 1000, tz=timezone.utc)
        last_ts = datetime.fromtimestamp(data[-1][0] / 1000, tz=timezone.utc)

        if batch_num % 20 == 0 or len(data) < LIMIT:
            print(
                f"  Batch {batch_num}: {len(data)} candles "
                f"({first_ts.strftime('%Y-%m-%d')} to {last_ts.strftime('%Y-%m-%d')}), "
                f"total so far: {len(all_rows):,}"
            )

        # Advance cursor past the last candle's open_time
        cursor = data[-1][0] + 1

        # Stop if we got fewer than LIMIT (reached end of available data)
        if len(data) < LIMIT:
            break

        time.sleep(SLEEP_SECONDS)

    print(f"\nDownload complete: {len(all_rows):,} candles in {batch_num} batches")

    if not all_rows:
        raise RuntimeError("No data returned from Binance. Check symbol and network.")

    # ── Build DataFrame ───────────────────────────────────────────────────────
    records = []
    for row in all_rows:
        records.append({col: row[idx] for col, idx in KEEP_INDICES.items()})

    df = pd.DataFrame(records)

    # Convert open_time from ms to UTC datetime, set as index
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.set_index("open_time")

    # Cast types
    float_cols = ["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_volume"]
    for col in float_cols:
        df[col] = df[col].astype(float)
    df["count"] = df["count"].astype(int)

    # Drop any duplicate timestamps (Binance pagination overlap edge case)
    df = df[~df.index.duplicated(keep="first")]

    # Sort by time
    df = df.sort_index()

    return df


def main() -> None:
    """Download, validate, and save to parquet."""
    df = download_sol_usdt_15m()

    # ── Validate ──────────────────────────────────────────────────────────────
    print(f"\nRows: {len(df):,}")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Index name: {df.index.name}")
    print(f"\nFirst 3 rows:")
    print(df.head(3).to_string())
    print(f"\nLast 3 rows:")
    print(df.tail(3).to_string())

    # ── Save ──────────────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, engine="pyarrow")
    print(f"\nSaved to {OUTPUT_PATH}")

    # ── Final assertion ───────────────────────────────────────────────────────
    if len(df) >= MIN_EXPECTED_ROWS:
        print(f"PASS: row count {len(df):,} >= {MIN_EXPECTED_ROWS:,}")
    else:
        print(f"WARNING: row count {len(df):,} < {MIN_EXPECTED_ROWS:,} — expected ~105,120 for 3 years of 15m data")
        print("The download may be incomplete. Check Binance data availability for SOL/USDT.")


if __name__ == "__main__":
    main()
