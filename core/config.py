"""
config.py — Council of Alphas
Single source of truth for all pipeline constants.
All other modules import from here. No magic numbers in code.
"""

# ── Data ──────────────────────────────────────────────────────────────────────
DATA_PATH = "data/sol_usd_15m_3y.parquet"
STATE_MATRIX_PATH = "data/state_matrix.parquet"

# ── State Matrix / Regime Detection ───────────────────────────────────────────
TREND_SMA_WINDOW = 50
TREND_SLOPE_LOOKBACK = 3
TREND_SLOPE_THRESHOLD = 0.0005      # ±0.0005 for 15m candles
ATR_WINDOW = 24                      # 24 bars = 6 hours on 15m
VOL_SMA_WINDOW = 20

# ── Triple Barrier (fixed system-wide — NOT on Strategy objects) ───────────────
TBM_WIN = 3.0                        # Take Profit = entry + 3.0 × ATR
TBM_LOSS = 1.5                       # Stop Loss   = entry - 1.5 × ATR
TBM_TIME_HORIZON = 32                # Max 32 bars (~8 hours)
TBM_TIE_BREAK = "stop_first"        # Worst-case on whipsaw

# ── Backtesting ────────────────────────────────────────────────────────────────
BACKTEST_FEE = 0.00075              # 0.075% Binance maker fee

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

# ── Scientist Loop ─────────────────────────────────────────────────────────────
MAX_SCIENTIST_ITERATIONS = 5
MIN_IMPROVEMENT_THRESHOLD = 0.05    # Early exit if 2 consecutive below this

# ── UNVIABLE Thresholds (checked before Critic) ────────────────────────────────
UNVIABLE_GLOBAL_SHARPE = -0.5       # GLOBAL Sharpe below this → UNVIABLE
UNVIABLE_MAX_CONSEC_LOSSES = 20     # GLOBAL max consecutive losses above this → UNVIABLE

# ── Models ─────────────────────────────────────────────────────────────────────
SONNET_MODEL = "claude-sonnet-4-6"
OPUS_MODEL = "claude-opus-4-6"
SPECIALIST_TEMPERATURE = 0
CRITIC_TEMPERATURE = 0
REFINER_TEMPERATURE = 0

# ── Regime Labels ──────────────────────────────────────────────────────────────
SESSIONS = ["ASIA", "LONDON", "NY", "OTHER"]
TREND_REGIMES = ["UPTREND", "DOWNTREND", "CONSOLIDATION"]
VOL_REGIMES = ["HIGH_VOL", "LOW_VOL"]
