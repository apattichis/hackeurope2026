# Data Directory

## Required File
Place your Binance BTC-USD 15m parquet file here:
- btc_usd_15m_3y.parquet — 3 years of 15-minute candles

## Auto-Generated
- state_matrix.parquet — Created automatically on first pipeline run

## Expected Columns in Raw Parquet
- Index: open_time (UTC datetime)
- Columns: open, high, low, close, volume, 
           quote_volume, count, taker_buy_volume