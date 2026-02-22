"""
config.py — Council of Alphas
Single source of truth for all pipeline constants.
All other modules import from here. No magic numbers in code.
"""

# ── Data ──────────────────────────────────────────────────────────────────────
DATA_PATH = "data/sol_usd_1h.parquet"
STATE_MATRIX_PATH = "data/state_matrix_1h.parquet"

# ── State Matrix / Regime Detection ───────────────────────────────────────────
TREND_SMA_WINDOW = 50                # 50 bars = ~2 days on 1h
TREND_SLOPE_LOOKBACK = 3
TREND_SLOPE_THRESHOLD = 0.0005
ATR_WINDOW = 24                      # 24 bars = 24 hours on 1h
VOL_SMA_WINDOW = 20

# ── Triple Barrier (fixed system-wide — NOT on Strategy objects) ───────────────
TBM_WIN = 2.0                        # Take Profit = entry + 2.0 × ATR
TBM_LOSS = 1.0                       # Stop Loss   = entry - 1.0 × ATR
TBM_TIME_HORIZON = 24                # Max 24 bars (~24 hours on 1h)
TBM_TIE_BREAK = "stop_first"        # Worst-case on whipsaw

# ── Backtesting ────────────────────────────────────────────────────────────────
BACKTEST_FEE = 0.00075              # 0.075% taker fee
RISK_PER_TRADE = 0.005              # 0.5% risk per trade

# ── Diagnostics ────────────────────────────────────────────────────────────────
MIN_TRADES_SUFFICIENT_EVIDENCE = 30

# ── Speciation ─────────────────────────────────────────────────────────────────
MAX_STRATEGIES_PER_SPECIALIST = 3
MAX_GENERATION_ATTEMPTS = 3
STRATEGY_TIMEOUT_SECONDS = 60
MIN_INDICATORS_PER_PROMPT = 2
MAX_INDICATORS_PER_PROMPT = 4

SPECIALIST_FAMILIES = ["trend", "momentum", "volatility", "volume"]

FAMILY_INDICATORS = {
    "trend":      ["ema", "hma", "macd", "adx", "slope"],
    "momentum":   ["rsi", "cci", "roc", "mfi", "zscore"],
    "volatility": ["natr", "bollinger_bands", "keltner_channels", "choppiness_index"],
    "volume":     ["vwap", "obv", "cmf"],
}

# ── Niche Selection ────────────────────────────────────────────────────────────
MIN_FITNESS_THRESHOLD = -999.0       # Only filter truly broken strategies (-999)

# ── Fitness ─────────────────────────────────────────────────────────────────────
# Hard elimination: require at least this many total trades across tradable
# 3D buckets (granularity=3D, sufficient_evidence=True).
MIN_TOTAL_TRADES_TRADABLE_BUCKETS = 300

# ── UNVIABLE Thresholds ───────────────────────────────────────────────────────
UNVIABLE_GLOBAL_SHARPE = -5.0       # GLOBAL Sharpe below this → UNVIABLE
UNVIABLE_MAX_CONSEC_LOSSES = 20     # GLOBAL max consecutive losses above this → UNVIABLE

# ── Models ─────────────────────────────────────────────────────────────────────
SPECIALIST_MODEL = "claude-opus-4-6"
SPECIALIST_TEMPERATURE = 0

# ── Regime Labels ──────────────────────────────────────────────────────────────
SESSIONS = ["ASIA", "LONDON", "NY", "OTHER"]
TREND_REGIMES = ["UPTREND", "DOWNTREND", "CONSOLIDATION"]
VOL_REGIMES = ["HIGH_VOL", "LOW_VOL"]

# ── API Cost Tracking ─────────────────────────────────────────────────────
# Pricing per million tokens (USD) — Claude Opus 4.6 (Feb 2026)
OPUS_INPUT_COST_PER_M = 15.0
OPUS_OUTPUT_COST_PER_M = 75.0
