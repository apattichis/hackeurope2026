# Data Directory

## Required File
Place your SOL-USD 1h candle parquet file here:
- `sol_usd_1h.parquet` - ~36k bars (Jan 2022 - Feb 2026)

## Auto-Generated
- `state_matrix_1h.parquet` - Created automatically on first pipeline run (21 columns)
- `results/` - Pipeline output directory (speciation, champions, ranked hybrids)

## Expected Columns in Raw Parquet
- Index: `open_time` (UTC datetime)
- Columns: `open`, `high`, `low`, `close`, `volume`, `quote_volume`, `count`, `taker_buy_volume`

## State Matrix Columns (21 total)
OHLCV (5) + quote_volume + count + taker_buy_volume + ATR_24 + session + trend_regime + vol_regime + tbm_label + tbm_long/short_pnl + tbm_long/short_exit_idx + tbm_long/short_duration + tbm_long/short_outcome
