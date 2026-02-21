# Council of Alphas — Claude Code Build Guide

**READ THIS FIRST. Then read MASTER_SPEC.md. Then read each pre-built file before writing anything.**

---

## Your Mission

Build the Council of Alphas pipeline. The core infrastructure modules are already written and tested. Your job is to wire them together into a complete working system.

**DO NOT rewrite the pre-built modules unless explicitly told to update parameters.**

---

## Step 0: Understand the Files You Already Have

Read these files IN THIS ORDER before writing a single line:

1. `MASTER_SPEC.md` — complete specification of every decision
2. `config.py` — all constants (read this before any other .py)
3. `state_builder.py` — needs parameter update (see Step 1)
4. `labeling.py` — ready, do not touch
5. `backtesting.py` — ready, do not touch
6. `diagnostics.py` — ready, do not touch
7. `whitelist_indicators.py` — ready, do not touch
8. `indicator_sampler.py` — needs cleanup (see Step 2)
9. `strategy_base.py` — needs update (see Step 3)

---

## Step 1: Update state_builder.py

Change ONLY these default parameter values in `StateMatrixBuilder.__init__`:

```python
# BEFORE:
trend_slope_threshold: float = 0.001
atr_window: int = 14

# AFTER:
trend_slope_threshold: float = 0.0005
atr_window: int = 24
```

Also update `tbm_win=2.0` and `tbm_loss=1.0` (already correct if using new file).

After building, the `_add_triple_barrier_targets` method should remain — labeling IS called from state_builder. The state matrix DOES contain TBM columns.

After build, drop intermediate columns before saving parquet:
- Drop: `sma_50`, `sma_50_slope_3`, `ATR_24`, `ATR_24_SMA_20`
- Keep: all OHLCV + regime labels + TBM columns

---

## Step 2: Update indicator_sampler.py

Remove these indicators from CATEGORIES (they are hidden from LLMs):
- From Trend: `sma`
- From Momentum: `stoch`, `williams_r`
- From Volatility: `atr`, `donchian_channels`

Final CATEGORIES should be:
```python
CATEGORIES = {
    "trend": ["ema", "hma", "macd", "adx", "slope"],
    "momentum": ["rsi", "cci", "roc", "mfi", "zscore"],
    "volatility": ["natr", "bollinger_bands", "keltner_channels", "choppiness_index"],
    "volume": ["vwap", "obv", "cmf"],
}
```

Remove the `generate_llm_prompts` method entirely. The sampler's only job is `sample_sets_for_category()`. Prompt building is handled by `prompt_builder.py`.

---

## Step 3: Update strategy_base.py

```python
from whitelist_indicators import Indicators
import pandas as pd

class Strategy(Indicators):
    """Base class for all LLM-generated strategies."""
    
    name: str = "unnamed"
    family: str = "unknown"       # trend / momentum / volatility / volume / hybrid
    description: str = ""         # one sentence: what does this strategy do
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Must return pd.Series with values:
          1  = long entry signal
         -1  = short entry signal  
          0  = no signal / flat
        Index must match data.index exactly.
        No lookahead bias permitted.
        """
        raise NotImplementedError("LLM or HybridBuilder implements this")
```

**IMPORTANT: Remove tbm_win and tbm_loss — these are now fixed system-wide in config.py**

---

## Step 4: Build config.py

```python
# config.py — Single source of truth for all constants

# Data
DATA_PATH = "data/sol_usd_15m_3y.parquet"
STATE_MATRIX_PATH = "data/state_matrix.parquet"

# State Matrix / Regime Detection
TREND_SMA_WINDOW = 50
TREND_SLOPE_LOOKBACK = 3
TREND_SLOPE_THRESHOLD = 0.0005
ATR_WINDOW = 24
VOL_SMA_WINDOW = 20

# Triple Barrier (fixed system-wide)
TBM_WIN = 2.0
TBM_LOSS = 1.0
TBM_TIME_HORIZON = 50
TBM_TIE_BREAK = "stop_first"

# Backtesting
BACKTEST_FEE = 0.00075

# Diagnostics
MIN_TRADES_SUFFICIENT_EVIDENCE = 30

# Speciation
MAX_STRATEGIES_PER_SPECIALIST = 3
MAX_GENERATION_ATTEMPTS = 3
STRATEGY_TIMEOUT_SECONDS = 60
MIN_INDICATORS = 2
MAX_INDICATORS = 4

# Fitness
# (no additional constants — formula is Global_Sharpe * ln(N) * Coverage)

# Niche Selection
MIN_FITNESS_THRESHOLD = 0.0  # must pass hard eliminations only

# Scientist
MAX_SCIENTIST_ITERATIONS = 5
MIN_IMPROVEMENT_THRESHOLD = 0.05

# Models
SONNET_MODEL = "claude-sonnet-4-6"
OPUS_MODEL = "claude-opus-4-6"
SPECIALIST_TEMPERATURE = 0
CRITIC_TEMPERATURE = 0
REFINER_TEMPERATURE = 0
```

---

## Step 5: Build fitness.py

Implement `compute_fitness(diagnostics_df: pd.DataFrame) -> float`

```python
import numpy as np
import pandas as pd
from config import MIN_TRADES_SUFFICIENT_EVIDENCE

def compute_fitness(diagnostics_df: pd.DataFrame) -> float:
    """
    Score = Global_Sharpe * ln(N) * Coverage
    
    Coverage = trade-weighted fraction of actual trades in profitable 3D buckets
    
    Returns -999 if hard elimination conditions are met.
    """
    # 1. Get GLOBAL row
    global_rows = diagnostics_df[diagnostics_df["granularity"] == "GLOBAL"]
    if len(global_rows) == 0:
        return -999.0
    
    global_row = global_rows.iloc[0]
    
    # 2. Hard eliminations
    if not global_row["sufficient_evidence"]:
        return -999.0
    if global_row["sharpe"] <= 0:
        return -999.0
    
    # 3. Get active 3D buckets
    active = diagnostics_df[
        (diagnostics_df["granularity"] == "3D") &
        (diagnostics_df["sufficient_evidence"] == True)
    ]
    
    if len(active) == 0:
        return -999.0
    
    # 4. Trade-weighted coverage
    profitable_trades = active[active["sharpe"] > 0]["trade_count"].sum()
    total_trades = active["trade_count"].sum()
    
    if total_trades == 0:
        return -999.0
    
    coverage = profitable_trades / total_trades
    
    # 5. Score
    global_sharpe = float(global_row["sharpe"])
    N = int(global_row["trade_count"])
    
    score = global_sharpe * np.log(N) * coverage
    return float(score)
```

---

## Step 6: Build prompt_builder.py

Builds the full specialist prompt by combining:
- Fixed template skeleton
- Family-specific guidance
- Randomly sampled indicator help text (filtered from whitelist_indicators.get_help_text())

Key function signature:
```python
def build_specialist_prompt(family: str, sampled_indicators: list) -> str:
    """
    family: one of 'trend', 'momentum', 'volatility', 'volume'
    sampled_indicators: list of indicator name strings from IndicatorSampler
    returns: complete prompt string ready for API call
    """
```

Filter `Indicators.get_help_text()` to only include lines for the sampled indicators.

Include all rules from MASTER_SPEC.md Section 5.5.
Include family guidance from MASTER_SPEC.md Section 5.6.

---

## Step 7: Build specialist_agent.py

Handles one strategy generation attempt for one specialist:

```python
async def generate_strategy(
    family: str,
    state_matrix: pd.DataFrame,
    attempt: int = 0,
    previous_error: str = None,
) -> tuple[Strategy, float, pd.DataFrame]:
    """
    Returns: (strategy_object, fitness_score, diagnostics_df)
    Raises: StrategyGenerationError after MAX_GENERATION_ATTEMPTS
    """
```

Pipeline per attempt:
1. Sample indicators via IndicatorSampler
2. Build prompt via PromptBuilder (inject previous_error if retry)
3. Call Claude Sonnet API
4. Validate code (4 checks from MASTER_SPEC Section 5.8)
5. Execute generate_signals with timeout
6. Run VectorizedBacktester
7. Run DiagnosticsEngine
8. compute_fitness()
9. Return result or retry with error

---

## Step 8: Build niche_selector.py

```python
def select_champions(
    all_results: dict[str, list[tuple]]
) -> dict[str, tuple]:
    """
    all_results: {family: [(strategy, score, diagnostics), ...]}
    returns: {family: (strategy, score, diagnostics)} for champions only
    Champions must have score > 0 (pass hard eliminations)
    """
```

---

## Step 9: Build hybrid_builder.py

Three methods, all pure Python. See MASTER_SPEC Section 10 for exact logic.

```python
class HybridBuilder:
    
    def build_regime_router(
        self,
        champions: dict,
        diagnostics: dict,
        champion_signals: dict,
        state_matrix: pd.DataFrame,
    ) -> Strategy:
        """
        For each of 24 (session, trend, vol) combinations:
        assign champion with highest sufficient_evidence Sharpe.
        Fallback to best GLOBAL Sharpe champion if no sufficient evidence.
        """
    
    def build_consensus_gate(
        self,
        champion_signals: dict,
        state_matrix: pd.DataFrame,
    ) -> Strategy:
        """
        Fire long if >= 3/4 champions say long.
        Fire short if >= 3/4 champions say short.
        """
    
    def build_weighted_combination(
        self,
        champions: dict,
        champion_signals: dict,
        fitness_scores: dict,
        state_matrix: pd.DataFrame,
    ) -> Strategy:
        """
        Weighted sum of signals, weights = fitness scores.
        np.sign of weighted sum = final signal.
        """
```

**All hybrids must be inline Strategy subclasses (not compositional).**
**family = "hybrid" for all three.**

---

## Step 10: Build critic_agent.py

```python
def run_critic(
    strategy_code: str,
    diagnostics_df: pd.DataFrame,
) -> dict:
    """
    Filters diagnostics to sufficient_evidence=True rows.
    Calls Claude Opus with Critic prompt from MASTER_SPEC Section 11.5.
    Parses structured response into dict:
    {
        'primary_failure': str,
        'root_cause': str,
        'surgical_fix': str,
        'expected_impact': str,
        'verdict': 'CONTINUE' | 'UNVIABLE'
    }
    """
```

Parse the structured output format exactly. If parsing fails → treat as UNVIABLE (safe default).

---

## Step 11: Build refiner_agent.py

```python
def run_refiner(
    strategy_code: str,
    surgical_fix: str,
) -> str:
    """
    Calls Claude Sonnet with Refiner prompt from MASTER_SPEC Section 11.6.
    Returns updated strategy code string.
    """
```

---

## Step 12: Build scientist.py

Orchestrates the full refinement loop for ONE hybrid strategy:

```python
def run_scientist(
    hybrid: Strategy,
    state_matrix: pd.DataFrame,
) -> tuple[Strategy, float, pd.DataFrame]:
    """
    Runs max MAX_SCIENTIST_ITERATIONS iterations.
    Implements early exit (2x improvement < MIN_IMPROVEMENT_THRESHOLD).
    Implements revert (if score degrades).
    Returns best (strategy, fitness_score, diagnostics_df).
    """
```

Track iteration history for UI Panel 4 (emit as log events).

---

## Step 13: Build orchestrator.py

Main pipeline controller. Implements the full flow from MASTER_SPEC Section 12.1.

Key methods:
```python
class Orchestrator:
    
    async def run(self) -> PipelineResult:
        """Full pipeline. Returns ranked final strategies."""
    
    def _load_or_build_state_matrix(self) -> pd.DataFrame:
        """Load from parquet if exists, else build and save."""
    
    async def _run_speciation(self, state_matrix) -> dict:
        """Parallel specialist generation."""
    
    def _run_niche_selection(self, all_results) -> dict:
        """Select champions."""
    
    def _run_hybrid_building(self, champions, state_matrix) -> list:
        """Build 3 hybrids."""
    
    def _run_scientist_loop(self, hybrids, state_matrix) -> list:
        """Refine each hybrid."""
    
    def _final_ranking(self, survivors, champions) -> list:
        """Rank survivors or fall back to best champion."""
```

Emit log events at every step for real-time UI consumption.

---

## Step 14: Build app.py (Streamlit — Andreas)

5 panels as defined in MASTER_SPEC Section 13.

Use `st.session_state` to preserve pipeline results across reruns.
Use Plotly for all charts (heatmap + PnL curves).

---

## Critical Rules

1. **Never import from files you haven't read** — read every pre-built file before using it
2. **Never rewrite core logic** in labeling.py, backtesting.py, diagnostics.py, whitelist_indicators.py
3. **All constants** come from config.py — no magic numbers in code
4. **TBM multipliers are NOT on Strategy objects** — they are in config.py only
5. **State Matrix is read-only** after build — never modify it in place
6. **Fitness formula is exactly:** `Global_Sharpe * ln(N) * Coverage` — no deviations
7. **Coverage is trade-weighted** — not bucket-count-weighted
8. **Critic receives only sufficient_evidence=True rows** — filter before sending
9. **HybridBuilder is pure Python** — no LLM calls
10. **All 3 hybrids use all 4 champions** — no 2-champion hybrids

---

## Repo Structure

```
council_of_alphas/
│
├── data/
│   ├── sol_usd_15m_3y.parquet          # Raw Binance data (you provide)
│   └── state_matrix.parquet            # Auto-generated on first run
│
├── docs/
│   ├── MASTER_SPEC.md                  # Complete specification
│   ├── ARCHITECTURE.mermaid            # Pipeline diagram
│   ├── BUILD_GUIDE.md                  # This file
│   └── council_of_alphas.pdf           # Full PDF (same as MASTER_SPEC)
│
├── pre_built/                          # DO NOT REWRITE CORE LOGIC
│   ├── state_builder.py                # Update parameters only
│   ├── labeling.py                     # Ready
│   ├── backtesting.py                  # Ready
│   ├── diagnostics.py                  # Ready
│   ├── whitelist_indicators.py         # Ready
│   ├── indicator_sampler.py            # Update categories + remove generate_llm_prompts
│   └── strategy_base.py               # Update: remove tbm params, add description
│
├── config.py                           # Build first
├── fitness.py                          # Step 5
├── prompt_builder.py                   # Step 6
├── specialist_agent.py                 # Step 7
├── niche_selector.py                   # Step 8
├── hybrid_builder.py                   # Step 9
├── critic_agent.py                     # Step 10
├── refiner_agent.py                    # Step 11
├── scientist.py                        # Step 12
├── orchestrator.py                     # Step 13
└── app.py                              # Step 14 (Andreas)
```

---

## Build Order (strict dependency order)

```
1. config.py                (no dependencies)
2. strategy_base.py         (depends on: whitelist_indicators)
3. indicator_sampler.py     (depends on: config)
4. fitness.py               (depends on: config)
5. prompt_builder.py        (depends on: config, indicator_sampler, whitelist_indicators)
6. specialist_agent.py      (depends on: config, prompt_builder, strategy_base, 
                                         backtesting, diagnostics, fitness)
7. niche_selector.py        (depends on: config, fitness)
8. hybrid_builder.py        (depends on: config, strategy_base, diagnostics)
9. critic_agent.py          (depends on: config)
10. refiner_agent.py        (depends on: config)
11. scientist.py            (depends on: config, backtesting, diagnostics, 
                                         fitness, critic_agent, refiner_agent)
12. orchestrator.py         (depends on: everything above)
13. app.py                  (depends on: orchestrator)
```
