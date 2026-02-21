# Council of Alphas — Master Specification
**HackEurope 2026 | Team: The Greeks (Andreas + Markos)**
**Version: Final | Date: February 21, 2026**

---

## 0. One-Line Summary

An evolutionary multi-agent framework that prevents mode collapse in LLM strategy generation through enforced specialist diversity, niche-preserving selection, deterministic hybrid construction, and evidence-locked diagnostic refinement.

---

## 1. Data

| Parameter | Value |
|---|---|
| Instrument | SOL-USD |
| Timeframe | 15-minute candles |
| Source | Binance (parquet file) |
| Date Range | 3 years |
| Index | `open_time` (UTC datetime, already set as index) |

**Columns kept from raw Binance data:**
- `open`, `high`, `low`, `close`, `volume`
- `quote_volume`, `count`, `taker_buy_volume`

---

## 2. Regime Detection

Three dimensions used exclusively as diagnostic buckets (NOT as entry conditions for strategies).

### 2.1 Session Regime
Based on UTC hour of candle:

| Label | UTC Hours |
|---|---|
| ASIA | 00:00 – 07:59 |
| LONDON | 08:00 – 12:59 |
| NY | 13:00 – 20:59 |
| OTHER | 21:00 – 23:59 |

### 2.2 Trend Regime
- Indicator: SMA(50) slope over 3-bar lookback (`pct_change(3)`)
- Threshold: ±0.0005

| Label | Condition |
|---|---|
| UPTREND | slope > +0.0005 |
| DOWNTREND | slope < -0.0005 |
| CONSOLIDATION | -0.0005 ≤ slope ≤ +0.0005 |

### 2.3 Volatility Regime
- ATR window: 24 bars (= 6 hours on 15m)
- Smoothing: SMA(20) of ATR

| Label | Condition |
|---|---|
| HIGH_VOL | ATR(24) > SMA20(ATR(24)) |
| LOW_VOL | ATR(24) ≤ SMA20(ATR(24)) |

### 2.4 Total Micro-Buckets
4 sessions × 3 trend states × 2 vol states = **24 micro-buckets**

---

## 3. Triple Barrier Labeling (TBM)

### 3.1 Parameters (Fixed System-Wide)
| Parameter | Value |
|---|---|
| Win multiplier | 2.0 × ATR |
| Loss multiplier | 1.0 × ATR |
| Time horizon | 50 bars (~12.5 hours) |
| ATR window | 24 bars (6 hours) |
| Tie-break | stop_first (worst-case) |

### 3.2 Label Values
| Label | Meaning |
|---|---|
| +1.0 | Long trade: TP hit (price went up) |
| -1.0 | Short trade: TP hit (price went down) |
| 0.0 | Timeout: neither barrier hit in time_horizon |
| NaN | Whipsaw: both long AND short hit in same bar (untradable, excluded from ML) |

### 3.3 Logic
- Long scan and short scan run **simultaneously** for every candle
- **Long takes priority** for oracle label: if long hits TP → +1, else check short → -1, else 0
- Whipsaw (both directions hit) → NaN label
- Every single candle in the dataset gets labeled

### 3.4 Output Columns (appended to State Matrix)
| Column | Description |
|---|---|
| `tbm_label` | Oracle label: +1, -1, 0, or NaN |
| `tbm_long_pnl` | Exact fractional return if long trade taken |
| `tbm_long_exit_idx` | Row index where long trade exits |
| `tbm_long_duration` | Candles the long trade was open |
| `tbm_short_pnl` | Exact fractional return if short trade taken |
| `tbm_short_exit_idx` | Row index where short trade exits |
| `tbm_short_duration` | Candles the short trade was open |

---

## 4. State Matrix

### 4.1 Definition
A single pre-computed pandas DataFrame saved as parquet. Built **once**, loaded on every subsequent run. Every downstream component reads from it — nothing is recomputed.

### 4.2 Complete Column Schema

| Column | Source | Type |
|---|---|---|
| `open` | Raw Binance | float |
| `high` | Raw Binance | float |
| `low` | Raw Binance | float |
| `close` | Raw Binance | float |
| `volume` | Raw Binance | float |
| `quote_volume` | Raw Binance | float |
| `count` | Raw Binance | float |
| `taker_buy_volume` | Raw Binance | float |
| `session` | StateMatrixBuilder | str: ASIA/LONDON/NY/OTHER |
| `trend_regime` | StateMatrixBuilder | str: UPTREND/DOWNTREND/CONSOLIDATION |
| `vol_regime` | StateMatrixBuilder | str: HIGH_VOL/LOW_VOL |
| `tbm_label` | core/labeling.py | float: +1/-1/0/NaN |
| `tbm_long_pnl` | core/labeling.py | float |
| `tbm_long_exit_idx` | core/labeling.py | int |
| `tbm_long_duration` | core/labeling.py | int |
| `tbm_short_pnl` | core/labeling.py | float |
| `tbm_short_exit_idx` | core/labeling.py | int |
| `tbm_short_duration` | core/labeling.py | int |

**Index:** `open_time` (UTC datetime)

### 4.3 Build Rules
- No intermediate calculation columns saved (sma_50, atr values etc. are dropped)
- If parquet file exists on disk → load it, skip build
- If parquet does not exist → build and save
- `force_rebuild=True` parameter to override cache

### 4.4 StateMatrixBuilder Updated Parameters
```python
StateMatrixBuilder(
    trend_sma_window=50,
    trend_slope_lookback=3,
    trend_slope_threshold=0.0005,  # updated from 0.001
    atr_window=24,                  # updated from 14
    vol_sma_window=20,
    tbm_win=2.0,                    # fixed system-wide
    tbm_loss=1.0,                   # fixed system-wide
    tbm_time_horizon=50,
    tbm_tie_break="stop_first",
)
```

---

## 5. Specialist Agents

### 5.1 Overview
4 LLM agents (Claude Sonnet, temp=0), each locked to a distinct strategy family through prompt constraints and a restricted indicator subset.

### 5.2 Strategy Families & Indicator Subsets

| Family | Allowed Indicators |
|---|---|
| Trend | ema, hma, macd, adx, slope |
| Momentum | rsi, cci, roc, mfi, zscore |
| Volatility | natr, bollinger_bands, keltner_channels, choppiness_index |
| Volume | vwap, obv, cmf |

### 5.3 Random Indicator Sampling
Before each strategy generation, `IndicatorSampler` randomly samples a subset of the family's allowed indicators:
- Min indicators: 2
- Max indicators: min(4, len(family_pool))
- Volume family cap: max 3 (only has 3 indicators)
- Sampled subset injected into prompt — LLM cannot use what it cannot see

### 5.4 Strategy Base Class
```python
class Strategy(Indicators):
    name: str = "unnamed"
    family: str = "unknown"          # trend/momentum/volatility/volume/hybrid
    description: str = ""            # one sentence: what does this strategy do
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Returns pd.Series of 1 (long), -1 (short), 0 (flat).
        Index must match data.index exactly.
        No lookahead bias permitted.
        """
        raise NotImplementedError
```

**Note: `tbm_win` and `tbm_loss` are REMOVED from strategy class — fixed system-wide at win=2.0, loss=1.0**

### 5.5 Specialist Prompt Template
```
You are a quantitative trading strategy specialist.
Your family: {FAMILY}

═══ STRICT RULES ═══
1. Your class MUST inherit from Strategy
2. You MUST set: name, family, description
3. generate_signals() MUST return a pd.Series of 1 (long), -1 (short), 0 (flat)
4. The Series index MUST match data.index exactly
5. You may ONLY use the indicators listed below
6. No other libraries, no raw pandas rolling logic, nothing else
7. No lookahead bias — no shift(-1), no future data
8. The strategy must generate both long AND short signals
9. Do not use regime columns (session, trend_regime, vol_regime) as inputs

═══ YOUR INDICATOR TOOLKIT ═══
{RANDOMLY_SAMPLED_INDICATOR_HELP_TEXT}

═══ DATA AVAILABLE ═══
The data DataFrame columns: open, high, low, close, volume
Index: open_time (UTC datetime)

═══ OUTPUT FORMAT ═══
Return ONLY the Python class code. No explanation. No markdown.
No imports (they are handled). Just the class.

═══ FAMILY GUIDANCE ═══
{FAMILY_SPECIFIC_GUIDANCE}

═══ EXAMPLE STRUCTURE ═══
class MyStrategy(Strategy):
    name = "your_strategy_name"
    family = "{FAMILY_LOWER}"
    description = "one sentence describing entry logic"

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=data.index)
        # your logic here using only the indicators above
        return signals
```

### 5.6 Family-Specific Guidance (injected per specialist)

**Trend:**
```
You are a TREND specialist. Your strategies must identify and follow
directional momentum. Think: when is price clearly moving in one direction
and how do you ride that move? Avoid mean-reversion logic.
```

**Momentum:**
```
You are a MOMENTUM specialist. Your strategies must identify overbought/
oversold extremes and momentum exhaustion. Think: when has price moved
too far too fast and is likely to reverse or accelerate?
```

**Volatility:**
```
You are a VOLATILITY specialist. Your strategies must use volatility
expansion and contraction to time entries. Think: when are price boundaries
being broken, when is a squeeze about to release?
```

**Volume:**
```
You are a VOLUME specialist. Your strategies must use order flow and
institutional participation to confirm price moves. Think: is volume
backing this price move or is it a trap?
```

### 5.7 Generation Rules
- PoC: 1-3 strategies per specialist (configurable)
- Score immediately after each generation (not batch)
- Max 3 code generation attempts per strategy (with error feedback injection)
- Parallel generation across specialists using `asyncio.gather(return_exceptions=True)`
- 60-second timeout per strategy execution

### 5.8 Code Validation Steps
1. Syntax check: `compile(code)`
2. Execution check: `strategy.generate_signals(sample_data)`
3. Return type check: valid pd.Series of 1/-1/0
4. Trade count check: must produce > 0 trades
If any check fails → retry with error message injected into prompt

---

## 6. Backtesting

### 6.1 Engine
Custom `VectorizedBacktester` — no VectorBT dependency. Numba-accelerated with pure Python fallback.

### 6.2 Parameters
| Parameter | Value |
|---|---|
| Fee | 0.00075 (0.075% per trade) |
| Capital lock | No overlapping trades allowed |
| TBM | Reads pre-computed columns from State Matrix |

### 6.3 Flow
```
State Matrix (with TBM columns)
+ Strategy signal column (1/-1/0)
→ VectorizedBacktester.run(df, signal_col)
→ Trade Log DataFrame
```

### 6.4 Trade Log Schema
| Column | Description |
|---|---|
| `entry_ts` | Entry timestamp |
| `exit_ts` | Exit timestamp |
| `entry_index` | Entry row index |
| `exit_index` | Exit row index |
| `duration` | Candles open |
| `side` | 1 (long) or -1 (short) |
| `net_trade_return` | PnL after fee |
| `session` | Regime at entry |
| `trend_regime` | Regime at entry |
| `vol_regime` | Regime at entry |

---

## 7. Diagnostics Engine

### 7.1 Overview
Hierarchical bucket system. Computes 5 metrics across all regime combinations.

### 7.2 Hierarchy
| Level | Description | Row Count |
|---|---|---|
| GLOBAL | All trades combined | 1 |
| 1D | By session only | 4 |
| 1D | By trend only | 3 |
| 1D | By vol only | 2 |
| 2D | Session × Trend | 12 |
| 2D | Session × Vol | 8 |
| 2D | Trend × Vol | 6 |
| 3D | Session × Trend × Vol | 24 |
| **TOTAL** | | **60** |

Note: 60 rows (not 48) because OTHER session adds extra rows.

### 7.3 Output Schema
| Column | Description |
|---|---|
| `granularity` | GLOBAL / 1D / 2D / 3D |
| `session` | Session value or ALL |
| `trend_regime` | Trend value or ALL |
| `vol_regime` | Vol value or ALL |
| `trade_count` | N trades in bucket |
| `win_rate` | % winning trades |
| `sharpe` | mean_return / std_return (ddof=0) |
| `max_consecutive_losses` | Longest losing streak |
| `sufficient_evidence` | True if trade_count ≥ 30 |

### 7.4 Sharpe Rules
- Forced to NaN if std = 0 or trade_count < 2
- Population std (ddof=0)

### 7.5 Critic Input Filter
Only rows where `sufficient_evidence = True` are passed to the Critic.

---

## 8. Fitness Function

### 8.1 Formula
```
Score = Global_Sharpe × ln(N) × Coverage
```

### 8.2 Component Definitions

**Global_Sharpe:** Sharpe from the GLOBAL row of diagnostics table.

**ln(N):** Natural log of total trade count from GLOBAL row. Rewards sample size with diminishing returns.

**Coverage (trade-weighted):**
```python
active = 3D buckets where sufficient_evidence == True
profitable_trades = sum(trade_count for buckets where sharpe > 0)
total_trades = sum(trade_count for all active buckets)
coverage = profitable_trades / total_trades
```
Measures what fraction of actual trades occur in profitable regimes. Trade-weighted so large buckets dominate correctly.

### 8.3 Hard Eliminations (return -999 before scoring)
```python
if global_row.sufficient_evidence == False: return -999
if isnan(global_row.sharpe): return -999
if len(active_3d_buckets) == 0: return -999
```
Note: Negative Sharpe is allowed through — strategies with Sharpe between -0.5 and 0 can compete and be improved by the Scientist loop. The is_unviable() gate (Sharpe < -0.5) catches truly hopeless strategies before wasting API calls.

### 8.4 Why This Formula Works
- **Global_Sharpe** gates: is this strategy actually profitable risk-adjusted?
- **ln(N)** gates: do we have enough evidence?
- **Coverage** gates: is it profitable across the regimes where it actually trades?
- Multiplication means all three must be positive — no compensating for weakness in one with strength in another
- UPTREND-only specialist scores well: Coverage ≈ 1.0 (all its active trades are profitable), Fragility = 0

### 8.5 Same formula used for both champions (Stage 3) and hybrids (Stage 5).

---

## 9. Niche Selection

### 9.1 Rules
- Select top 1 champion per family by fitness score
- Minimum threshold: score > 0 (must pass hard eliminations)
- If a family has no viable strategy → that family sends no champion
- Architect proceeds with however many champions survive (minimum 1)

### 9.2 Order
```
Specialists generate → Fitness scored → Niche selection →
HybridBuilder → Scientist → Final fitness → Ranking
```

---

## 10. HybridBuilder (Pure Python — No LLM)

### 10.1 Overview
Deterministic Python class. Takes up to 4 champions and their diagnostics. Produces exactly 3 hybrid strategies.

### 10.2 Hybrid 1 — Regime Router
**Logic:** For each of the 24 regime combinations, assign the champion with the highest Sharpe in that specific 3D bucket.

```python
# For each (session, trend, vol) combination:
# Find which champion has best Sharpe with sufficient_evidence=True
# Assign that champion's signal for that regime
# Fallback: if no champion has sufficient evidence → use champion with best GLOBAL Sharpe
```

**Signal generation:** At each candle, look up current regime → fire that regime's assigned champion signal.

### 10.3 Hybrid 2 — Consensus Gate
**Logic:** All champions vote. Trade fires only when 3 out of 4 agree on direction.

```python
votes = trend_signal + momentum_signal + vol_signal + volume_signal
# Range: -4 to +4
long_signal = votes >= 3
short_signal = votes <= -3
```

### 10.4 Hybrid 3 — Weighted Combination
**Logic:** Champions weighted by their fitness score. Weighted sum's sign determines direction.

```python
weights = [fitness_trend, fitness_momentum, fitness_vol, fitness_volume]
weighted_sum = sum(w * s for w, s in zip(weights, signals))
signal = np.sign(weighted_sum)
```

### 10.5 Code Structure
All hybrids are **inline** — one self-contained Strategy class. Parent logic is copied with clear section comments:
```python
class RegimeRouterHybrid(Strategy):
    name = "regime_router"
    family = "hybrid"
    description = "..."
    
    def generate_signals(self, data):
        # ── TREND CHAMPION LOGIC ──────────────
        # ... inlined trend logic ...
        
        # ── MOMENTUM CHAMPION LOGIC ───────────
        # ... inlined momentum logic ...
        
        # ── ROUTING TABLE ─────────────────────
        # ... routing logic ...
```

### 10.6 Degraded Champion Handling
If a champion has zero sufficient_evidence 3D buckets → use its global signal directly without regime-specific routing.

---

## 11. Scientist / Critic Loop

### 11.1 Loop Structure (per hybrid, independent)
```
1. BACKTEST → trade log
2. DIAGNOSTICS → 60-row bucket table
3. FITNESS CHECK → if -999: UNVIABLE, stop
4. CRITIC (Opus, temp=0) → structured diagnosis
5. IF UNVIABLE → discard hybrid
   IF CONTINUE → send to Refiner
6. REFINER (Sonnet, temp=0) → updated code (one change)
7. VALIDATION GATE → accept/revert/early-exit
8. REPEAT max 5 iterations
```

### 11.2 Iteration Rules
- Max iterations: **5**
- Early exit: 2 consecutive iterations with improvement < 0.05 Sharpe → stop, keep best version
- Revert: if new score < previous score → revert to previous version
- Guaranteed monotonic improvement: v_n ≥ v_{n-1} always

### 11.3 UNVIABLE Conditions
Declare UNVIABLE immediately if ANY of these are true:
- GLOBAL Sharpe < -0.5 (with sufficient_evidence=True)
- Zero 3D buckets have both sufficient_evidence=True AND Sharpe > 0
- GLOBAL max_consecutive_losses > 20

### 11.4 Fallback
If all 3 hybrids are declared UNVIABLE → fall back to best champion by fitness score.

### 11.5 Critic Prompt (Claude Opus, temp=0)
```
You are the Evidence-Locked Critic for the Council of Alphas framework.

═══ YOUR ROLE ═══
You diagnose why a trading strategy is underperforming by reading
its diagnostic bucket table and strategy code. You do not guess.
Every claim you make must cite an exact bucket and an exact number
from the table provided.

═══ INPUTS YOU ARE GIVEN ═══
1. STRATEGY CODE — read this only to identify what specific parameter
   or logic explains the failure pattern you find in the diagnostics.
2. DIAGNOSTIC TABLE — sufficient_evidence=True rows only. Columns:
   granularity, session, trend_regime, vol_regime, trade_count,
   win_rate, sharpe, max_consecutive_losses, sufficient_evidence.

═══ HOW TO DIAGNOSE ═══
Scan strictly in this order:
1. GLOBAL row — is overall Sharpe salvageable?
2. 1D slices — which single dimension is the primary drag?
3. 2D slices — which interaction is the core problem?
4. 3D buckets — identify the exact failing micro-regime(s).

═══ STRICT CONSTRAINTS ═══
- You may NOT suggest structural rewrites
- You may NOT change the strategy family or its indicators
- You may NOT invent metrics not present in the table
- One surgical fix only — a parameter value, a threshold,
  or a single condition change
- Every claim must cite: [bucket] | sharpe=[x] | n=[x]

═══ UNVIABLE CONDITIONS ═══
Declare UNVIABLE immediately if ANY of these are true:
- GLOBAL sharpe < -0.5 (with sufficient_evidence=True)
- Zero 3D buckets have both sufficient_evidence=True AND sharpe > 0
- GLOBAL max_consecutive_losses > 20

═══ OUTPUT FORMAT ═══
Respond in exactly this structure, nothing else:

PRIMARY_FAILURE: [exact bucket] | sharpe=[value] | n=[value]
ROOT_CAUSE: [one sentence citing the code]
SURGICAL_FIX: [exact code change — parameter name, old value, new value]
EXPECTED_IMPACT: [which metric improves and why]
VERDICT: CONTINUE | UNVIABLE
```

### 11.6 Refiner Prompt (Claude Sonnet, temp=0)
```
You are the Surgical Refiner for the Council of Alphas framework.

═══ YOUR ROLE ═══
You receive a trading strategy and one precise instruction
from the Critic. Your job is to apply exactly that fix
and nothing else.

═══ STRICT CONSTRAINTS ═══
- Apply ONE change only — exactly what the Critic specified
- Do NOT restructure the code
- Do NOT change indicators or strategy family
- Do NOT add new logic
- Do NOT remove existing logic unless explicitly instructed
- Return the complete updated class, nothing else

═══ INPUT ═══
STRATEGY CODE: {code}
CRITIC INSTRUCTION: {surgical_fix}

═══ OUTPUT ═══
Return ONLY the complete updated Python class.
No explanation. No markdown. No imports.
```

---

## 12. Orchestration

### 12.1 Complete Pipeline Flow
```
PIPELINE START
│
├── 1. LOAD DATA
│   └── Load Binance SOL-USD 15m parquet
│
├── 2. BUILD / LOAD STATE MATRIX
│   ├── If parquet exists → load
│   └── If not → StateMatrixBuilder.build() → save parquet
│       (includes regime tagging + TBM labeling)
│
├── 3. SPECIATION (parallel, asyncio.gather)
│   ├── 4 specialists run concurrently
│   ├── Each specialist (1-3 strategies):
│   │   ├── IndicatorSampler.sample() → random subset
│   │   ├── PromptBuilder.build() → full prompt
│   │   ├── Claude Sonnet API call → strategy code
│   │   ├── Code validation (3 attempts max, error feedback)
│   │   ├── strategy.generate_signals(state_matrix) → signal col
│   │   ├── VectorizedBacktester.run() → trade log
│   │   ├── DiagnosticsEngine.compute() → bucket table
│   │   └── compute_fitness() → score (immediate, not batched)
│   └── Collect all (strategy, score, diagnostics) per family
│
├── 4. NICHE SELECTION
│   ├── For each family: pick max fitness score
│   ├── If score ≤ 0 → family eliminated
│   └── Champions dict: {family: (strategy, score, diagnostics)}
│
├── 5. HYBRID BUILDING (pure Python, no LLM)
│   ├── HybridBuilder.build_regime_router()
│   ├── HybridBuilder.build_consensus_gate()
│   └── HybridBuilder.build_weighted_combination()
│
├── 6. SCIENTIST LOOP (parallel, all hybrids concurrently)
│   ├── For each hybrid:
│   │   ├── Backtest → diagnostics → fitness
│   │   ├── UNVIABLE check → discard if triggered
│   │   ├── Max 5 iterations:
│   │   │   ├── Critic (Opus) → diagnosis + verdict
│   │   │   ├── If UNVIABLE → stop
│   │   │   ├── Refiner (Sonnet) → updated code
│   │   │   ├── Rebacktest → new fitness
│   │   │   ├── If improvement < 0.05 twice → early exit
│   │   │   └── If degraded → revert
│   │   └── Store best version
│   └── Collect surviving hybrids
│
├── 7. FINAL RANKING
│   ├── Score all survivors with compute_fitness()
│   ├── If survivors > 0 → rank by score
│   └── If survivors == 0 → fall back to best champion
│
└── OUTPUT → Streamlit UI
```

### 12.2 Error Handling Tiers
| Stage | Failure Type | Action |
|---|---|---|
| Data Load | File not found | Fatal — stop pipeline |
| State Matrix | Build error | Fatal — stop pipeline |
| Speciation | All strategies fail | Fatal — stop pipeline |
| Speciation | One specialist fails | Warn, continue with 3 families |
| Niche Selection | < 2 champions | Warn, continue |
| Hybrid Building | One template fails | Skip that hybrid |
| Scientist | All hybrids UNVIABLE | Fall back to best champion |
| Ranking | No survivors | Return best champion |

### 12.3 Configuration Constants
```python
# core/config.py
MAX_STRATEGIES_PER_SPECIALIST = 3
MAX_GENERATION_ATTEMPTS = 3
STRATEGY_TIMEOUT_SECONDS = 60
MAX_SCIENTIST_ITERATIONS = 5
MIN_IMPROVEMENT_THRESHOLD = 0.05  # early exit if below this twice
TBM_WIN = 2.0
TBM_LOSS = 1.0
TBM_TIME_HORIZON = 50
TBM_ATR_WINDOW = 24
BACKTEST_FEE = 0.00075
MIN_TRADES_SUFFICIENT_EVIDENCE = 30
TREND_SLOPE_THRESHOLD = 0.0005
```

### 12.4 Parallelism
- Specialists: `asyncio.gather(return_exceptions=True)` — parallel API calls
- One specialist crash does not kill others
- Scientist loop: `asyncio.gather(return_exceptions=True)` — all 3 hybrids refined concurrently
- One hybrid crash does not kill others
- Inside each hybrid's loop: iterations run sequential (each depends on previous)

### 12.5 Logging
Every stage emits structured log events consumed by Streamlit UI in real time.

---

## 13. Output / UI (Streamlit — Andreas)

### 13.1 Panel 1 — Pipeline Status (Live)
Real-time log of every pipeline event. Shows the system is actively thinking.
```
✅ State Matrix loaded (105,247 candles)
⚡ Trend Specialist generating strategy 1/3...
✅ Trend Strategy 1: fitness=2.84, sharpe=0.41, trades=623
...
```

### 13.2 Panel 2 — Champion Leaderboard
| Column | Source |
|---|---|
| Family | strategy.family |
| Strategy Name | strategy.name |
| Fitness Score | compute_fitness() |
| Sharpe | GLOBAL diagnostics row |
| Win Rate | GLOBAL diagnostics row |
| Trades | GLOBAL diagnostics row |
| Coverage | fitness component |

### 13.3 Panel 3 — Diagnostics Heatmap (Plotly)
- One heatmap per strategy (champions + hybrids)
- Rows: Session × Trend combinations
- Columns: HIGH_VOL / LOW_VOL
- Color: Sharpe value (red → white → green)
- Grey: insufficient evidence buckets

### 13.4 Panel 4 — Scientist Loop Trace
Per hybrid: iteration history showing Sharpe improvement, Critic diagnosis summary, fix applied.

### 13.5 Panel 5 — Final Ranked Results
- **Lineage view:** family tree showing which champions fed which hybrid
- **Cumulative PnL chart (Plotly):** one line per surviving strategy, X=trade number, Y=cumulative return %
- **Diagnostics expandable** per strategy
- Fallback champion shown if no hybrids survived

---

## 14. Model Assignment Summary

| Role | Model | Temperature |
|---|---|---|
| Specialist Agents (×4) | Claude Sonnet | 0 |
| Architect | **Python only — no LLM** | N/A |
| Scientist Critic | Claude Opus | 0 |
| Scientist Refiner | Claude Sonnet | 0 |

---

## 15. Pre-Built Files (Provided — Do Not Rewrite Core Logic)

| File | Status | Notes |
|---|---|---|
| `core/state_builder.py` | Needs parameter updates | Update thresholds per Section 4.4 |
| `core/labeling.py` | Ready | New dual-direction version |
| `core/backtesting.py` | Ready | VectorizedBacktester |
| `core/diagnostics.py` | Ready | Includes sufficient_evidence |
| `core/whitelist_indicators.py` | Ready | Full indicator library |
| `pipeline/indicator_sampler.py` | Needs cleanup | Remove hidden indicators, replace prompt generation |
| `core/strategy_base.py` | Needs update | Remove tbm_win/tbm_loss, add description |

## 16. Files to Build

| File | Purpose |
|---|---|
| `core/config.py` | All constants in one place |
| `pipeline/prompt_builder.py` | Builds specialist prompts from template + sampled indicators |
| `pipeline/specialist_agent.py` | LLM call + code validation + retry logic |
| `pipeline/fitness.py` | compute_fitness() function |
| `pipeline/niche_selector.py` | Champion selection logic |
| `pipeline/hybrid_builder.py` | HybridBuilder class (all 3 templates) |
| `agents/critic_agent.py` | Opus Critic call + response parser |
| `agents/refiner_agent.py` | Sonnet Refiner call |
| `agents/scientist.py` | Full Scientist loop orchestration |
| `orchestrator.py` | Main pipeline controller |
| `app.py` | Streamlit UI |
