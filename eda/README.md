# EDA Notebooks

## eda.ipynb - State Matrix Exploratory Data Analysis
Validates and profiles the pre-built state matrix (SOL/USD 1h candles, ~36k bars).

Sections:
- **A. Setup** - Load data, validate schema and index
- **B. Data Quality** - Missingness, sentinel values, numeric distributions
- **C. Regime Coverage** - Candle distribution across 24 micro-buckets (1D, 2D, 3D)
- **D. Regime Edge** - TBM label composition per bucket (bias, actionability)
- **E. Temporal Drift** - Monthly stability of regime distributions and edges
- **F. Validation** - Automated consistency checks + consolidated findings table

## winners_analysis.ipynb - Post-Run Strategy Analysis
Analyzes the surviving hybrid strategies after a full pipeline run (`python run_stage3.py`).

Sections:
1. **Head** - First trades from each hybrid
2. **Info** - DataFrame structure
3. **Portfolio Performance** - Equity, returns, drawdown, win rate (TP/SL), risk-reward
4. **Sharpe Ratio** - Per-trade and sqrt-N scaled Sharpe, zero-fee estimate
5. **Equity Curve** - Individual + overlay plots ($100k start, 0.5% risk, MEXC 0.04%)
6. **Holding Time** - Distribution of trade durations (1 bar = 1 hour)

Reads trade logs from `data/results/ranked/` - run Stage 3 first.
