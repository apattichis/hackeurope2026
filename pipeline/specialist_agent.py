"""
specialist_agent.py — Council of Alphas
Async strategy generation with code validation, backtesting, and retry logic.

Each specialist is locked to one family (trend/momentum/volatility/volume)
with a randomly sampled indicator subset. Generates strategy code via Claude
Sonnet, validates it, backtests it, and returns (strategy, fitness, diagnostics).

Public API:
    generate_one_strategy()  — single strategy attempt with retry
    run_specialist()         — one family, multiple strategies
    run_all_specialists()    — all 4 families in parallel
"""

from __future__ import annotations

import sys
import asyncio
import logging
import inspect
from pathlib import Path

# ── Path setup for flat imports (matches existing codebase pattern) ───────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _p in [str(_PROJECT_ROOT / "core"), str(_PROJECT_ROOT / "pipeline")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pandas as pd
import anthropic
from dotenv import load_dotenv

from config import (
    MAX_STRATEGIES_PER_SPECIALIST,
    MAX_GENERATION_ATTEMPTS,
    STRATEGY_TIMEOUT_SECONDS,
    BACKTEST_FEE,
    RISK_PER_TRADE,
    MIN_TRADES_SUFFICIENT_EVIDENCE,
    SPECIALIST_MODEL,
    SPECIALIST_TEMPERATURE,
    SPECIALIST_FAMILIES,
)
from strategy_base import Strategy
from whitelist_indicators import Indicators
from backtesting import VectorizedBacktester
from diagnostics import DiagnosticsEngine
from fitness import compute_fitness
from indicator_sampler import IndicatorSampler
from prompt_builder import build_specialist_prompt

# Load .env for ANTHROPIC_API_KEY
load_dotenv(_PROJECT_ROOT / ".env")

logger = logging.getLogger("council.specialist")


# ── Exceptions ────────────────────────────────────────────────────────────────

class StrategyGenerationError(Exception):
    """Raised when strategy generation fails after all retry attempts."""
    pass


# ── Code Extraction ──────────────────────────────────────────────────────────

def _extract_code(raw_response: str) -> str:
    """
    Strip markdown fences and extract Python code from LLM response.

    Handles:
    - ```python ... ```
    - ``` ... ```
    - Clean code with no fences
    """
    text = raw_response.strip()

    # Remove markdown code fences
    if text.startswith("```python"):
        text = text[len("```python"):]
    elif text.startswith("```"):
        text = text[3:]

    if text.endswith("```"):
        text = text[:-3]

    return text.strip()


# ── Strategy Validation ──────────────────────────────────────────────────────

def _validate_and_instantiate(code: str) -> Strategy:
    """
    Compile, execute in sandbox, find Strategy subclass, and instantiate.

    Raises ValueError with descriptive message on any validation failure.
    """
    # Step 1: Syntax check
    try:
        compile(code, "<strategy>", "exec")
    except SyntaxError as e:
        raise ValueError(f"Syntax error on line {e.lineno}: {e.msg}") from e

    # Step 2: Execute in sandboxed namespace
    namespace = {
        "Strategy": Strategy,
        "Indicators": Indicators,
        "pd": pd,
        "np": np,
        "__builtins__": __builtins__,
    }

    try:
        exec(code, namespace)
    except Exception as e:
        raise ValueError(f"Execution error: {type(e).__name__}: {e}") from e

    # Step 3: Find Strategy subclasses (exclude base classes)
    strategy_classes = [
        v for k, v in namespace.items()
        if isinstance(v, type)
        and issubclass(v, Strategy)
        and v is not Strategy
        and v is not Indicators
    ]

    if not strategy_classes:
        raise ValueError(
            "No Strategy subclass found in generated code. "
            "The class must inherit from Strategy."
        )

    # Step 4: Instantiate the first (should be only) subclass
    cls = strategy_classes[0]
    try:
        strategy = cls()
    except Exception as e:
        raise ValueError(f"Instantiation error: {type(e).__name__}: {e}") from e

    return strategy


def _validate_signals(signals: pd.Series, data_len: int) -> None:
    """
    Validate generate_signals() output.

    Checks:
    1. Is a pd.Series
    2. Length matches input data
    3. Values are all in {-1, 0, 1}
    4. At least one non-zero signal (produces trades)
    """
    if not isinstance(signals, pd.Series):
        raise ValueError(
            f"generate_signals returned {type(signals).__name__}, expected pd.Series"
        )

    if len(signals) != data_len:
        raise ValueError(
            f"Signal length {len(signals)} != data length {data_len}"
        )

    # Allow both int and float representations of -1, 0, 1
    unique_vals = set(signals.dropna().unique())
    allowed = {-1, 0, 1, -1.0, 0.0, 1.0}
    invalid = unique_vals - allowed
    if invalid:
        raise ValueError(
            f"Invalid signal values: {invalid}. Must be -1, 0, or 1"
        )

    trade_count = int((signals.abs() > 0).sum())
    if trade_count == 0:
        raise ValueError("Strategy produced 0 trades (all signals are 0)")


# ── Core Generation Logic ────────────────────────────────────────────────────

async def generate_one_strategy(
    family: str,
    state_matrix: pd.DataFrame,
    sampled_indicators: list[str],
    attempt: int = 0,
    previous_error: str | None = None,
) -> tuple[Strategy, float, pd.DataFrame]:
    """
    Generate, validate, backtest, and score one strategy.

    Parameters
    ----------
    family : str
        One of: trend, momentum, volatility, volume
    state_matrix : pd.DataFrame
        Full state matrix (read-only, will be copied for signal attachment)
    sampled_indicators : list[str]
        Indicator names from IndicatorSampler
    attempt : int
        Current attempt number (0-indexed). Retries up to MAX_GENERATION_ATTEMPTS.
    previous_error : str | None
        Error message from previous failed attempt (injected into retry prompt)

    Returns
    -------
    tuple[Strategy, float, pd.DataFrame]
        (strategy_instance, fitness_score, diagnostics_df)

    Raises
    ------
    StrategyGenerationError
        After MAX_GENERATION_ATTEMPTS consecutive failures.
    """
    if attempt >= MAX_GENERATION_ATTEMPTS:
        raise StrategyGenerationError(
            f"Failed after {MAX_GENERATION_ATTEMPTS} attempts for {family}. "
            f"Last error: {previous_error}"
        )

    try:
        # 1. Build prompt
        prompt = build_specialist_prompt(family, sampled_indicators, previous_error)

        # 2. Call Claude Sonnet
        client = anthropic.AsyncAnthropic()
        response = await client.messages.create(
            model=SPECIALIST_MODEL,
            max_tokens=4096,
            temperature=SPECIALIST_TEMPERATURE,
            messages=[{"role": "user", "content": prompt}],
        )
        raw_code = response.content[0].text

        # 3. Extract code from response
        code = _extract_code(raw_code)

        # 4. Validate and instantiate
        strategy = _validate_and_instantiate(code)
        strategy._source_code = code  # store for Critic agent later

        # 5. Run generate_signals with timeout
        signals = await asyncio.wait_for(
            asyncio.to_thread(strategy.generate_signals, state_matrix),
            timeout=STRATEGY_TIMEOUT_SECONDS,
        )
        _validate_signals(signals, len(state_matrix))

        # 6. Backtest (add signals to a copy — state matrix is read-only)
        df = state_matrix.copy()
        df["_signal"] = signals.values
        backtester = VectorizedBacktester(fee=BACKTEST_FEE, risk_per_trade=RISK_PER_TRADE)
        trade_log = backtester.run(df, "_signal")

        # 7. Diagnostics
        engine = DiagnosticsEngine(
            min_trades=MIN_TRADES_SUFFICIENT_EVIDENCE,
            state_matrix=state_matrix,
        )
        diagnostics = engine.compute(trade_log)

        # 8. Fitness
        score = compute_fitness(diagnostics)

        # 9. Attach trade log to strategy for result saving
        strategy._trade_log = trade_log

        logger.info(
            f"[{family}] {strategy.name}: fitness={score:.3f}, "
            f"trades={len(trade_log)}, attempt={attempt + 1}"
        )

        return strategy, score, diagnostics

    except StrategyGenerationError:
        raise
    except asyncio.TimeoutError:
        error_msg = f"generate_signals timed out after {STRATEGY_TIMEOUT_SECONDS}s"
        logger.warning(f"[{family}] Attempt {attempt + 1} failed: {error_msg}")
        return await generate_one_strategy(
            family, state_matrix, sampled_indicators,
            attempt=attempt + 1,
            previous_error=error_msg,
        )
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        logger.warning(f"[{family}] Attempt {attempt + 1} failed: {error_msg}")
        return await generate_one_strategy(
            family, state_matrix, sampled_indicators,
            attempt=attempt + 1,
            previous_error=error_msg,
        )


async def run_specialist(
    family: str,
    state_matrix: pd.DataFrame,
    num_strategies: int = MAX_STRATEGIES_PER_SPECIALIST,
) -> list[tuple[Strategy, float, pd.DataFrame]]:
    """
    Generate multiple strategies for one specialist family.

    Samples unique indicator subsets and generates one strategy per subset.
    Failed strategies are logged and skipped (partial results returned).

    Parameters
    ----------
    family : str
        One of: trend, momentum, volatility, volume
    state_matrix : pd.DataFrame
        Full state matrix
    num_strategies : int
        Number of strategies to attempt

    Returns
    -------
    list[tuple[Strategy, float, pd.DataFrame]]
        List of (strategy, fitness_score, diagnostics_df) for successful generations.
        May be empty if all attempts fail.
    """
    sampler = IndicatorSampler()
    indicator_sets = sampler.sample_unique_sets(family, num_strategies)

    logger.info(
        f"[{family}] Generating {len(indicator_sets)} strategies "
        f"(requested {num_strategies})"
    )

    results: list[tuple[Strategy, float, pd.DataFrame]] = []
    for i, indicators in enumerate(indicator_sets):
        try:
            result = await generate_one_strategy(
                family, state_matrix, indicators
            )
            results.append(result)
            strategy, score, _ = result
            logger.info(
                f"[{family}] Strategy {i + 1}/{len(indicator_sets)}: "
                f"{strategy.name} fitness={score:.3f}"
            )
        except StrategyGenerationError as e:
            logger.warning(
                f"[{family}] Strategy {i + 1}/{len(indicator_sets)}: "
                f"FAILED after all retries - {e}"
            )

    logger.info(
        f"[{family}] Complete: {len(results)}/{len(indicator_sets)} strategies viable"
    )
    return results


async def run_all_specialists(
    state_matrix: pd.DataFrame,
) -> dict[str, list[tuple[Strategy, float, pd.DataFrame]]]:
    """
    Run all 4 specialist families in parallel via asyncio.gather.

    One specialist crash does not kill others (return_exceptions=True).

    Returns
    -------
    dict[str, list[tuple]]
        {family: [(strategy, score, diagnostics), ...]}
        Failed families map to empty lists.
    """
    logger.info("Starting speciation: 4 specialists in parallel")

    tasks = [
        run_specialist(family, state_matrix)
        for family in SPECIALIST_FAMILIES
    ]

    raw_results = await asyncio.gather(*tasks, return_exceptions=True)

    all_results: dict[str, list[tuple[Strategy, float, pd.DataFrame]]] = {}
    for family, result in zip(SPECIALIST_FAMILIES, raw_results):
        if isinstance(result, BaseException):
            logger.error(f"[{family}] Specialist crashed: {type(result).__name__}: {result}")
            all_results[family] = []
        else:
            all_results[family] = result

    total = sum(len(v) for v in all_results.values())
    logger.info(f"Speciation complete: {total} total strategies across all families")

    return all_results


# ── Smoke Test ────────────────────────────────────────────────────────────────

def _build_dummy_state_matrix(n: int = 500) -> pd.DataFrame:
    """Build a realistic dummy state matrix for testing without real data."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=n, freq="15min", tz="utc")

    # Random walk price series
    close = 100.0 + np.cumsum(np.random.randn(n) * 0.3)
    close = np.maximum(close, 10.0)  # keep positive

    # TBM exit indices: each trade exits 5-15 bars later, clamped to valid range
    exit_offsets = np.random.randint(5, 16, n)
    exit_indices = np.clip(np.arange(n) + exit_offsets, 0, n - 1)

    df = pd.DataFrame({
        "open": close + np.random.randn(n) * 0.1,
        "high": close + np.abs(np.random.randn(n)) * 0.5,
        "low": close - np.abs(np.random.randn(n)) * 0.5,
        "close": close,
        "volume": np.random.uniform(1000, 5000, n),
        "session": np.random.choice(["ASIA", "LONDON", "NY", "OTHER"], n),
        "trend_regime": np.random.choice(
            ["UPTREND", "DOWNTREND", "CONSOLIDATION"], n
        ),
        "vol_regime": np.random.choice(["HIGH_VOL", "LOW_VOL"], n),
        "tbm_label": np.random.choice([1.0, -1.0, 0.0], n),
        "tbm_long_pnl": np.random.uniform(-0.02, 0.04, n),
        "tbm_long_exit_idx": exit_indices,
        "tbm_long_duration": exit_offsets,
        "tbm_short_pnl": np.random.uniform(-0.02, 0.04, n),
        "tbm_short_exit_idx": exit_indices,
        "tbm_short_duration": exit_offsets,
    }, index=dates)
    df.index.name = "open_time"

    return df


# Hardcoded valid strategy for smoke testing (no API call needed)
_TEST_STRATEGY_CODE = '''\
class TestEMACrossover(Strategy):
    name = "test_ema_crossover"
    family = "trend"
    description = "EMA 10/30 crossover for smoke testing"

    def generate_signals(self, data):
        signals = pd.Series(0, index=data.index)
        fast = self.ema(data, period=10)
        slow = self.ema(data, period=30)
        signals[fast > slow] = 1
        signals[fast < slow] = -1
        return signals
'''


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print("[SMOKE TEST] specialist_agent.py\n")

    state_matrix = _build_dummy_state_matrix(500)
    print(f"  Dummy state matrix: {state_matrix.shape[0]} rows, "
          f"{state_matrix.shape[1]} cols")

    # ── Test 1: Code extraction ───────────────────────────────────────────
    fenced = f"```python\n{_TEST_STRATEGY_CODE}\n```"
    extracted = _extract_code(fenced)
    assert "class TestEMACrossover" in extracted
    print("  Code extraction (markdown fences): OK")

    clean_extracted = _extract_code(_TEST_STRATEGY_CODE)
    assert "class TestEMACrossover" in clean_extracted
    print("  Code extraction (no fences): OK")

    # ── Test 2: Validation + instantiation ────────────────────────────────
    strategy = _validate_and_instantiate(extracted)
    assert strategy.name == "test_ema_crossover"
    assert strategy.family == "trend"
    assert strategy.description != ""
    print(f"  Validation: OK (class={type(strategy).__name__}, "
          f"name={strategy.name})")

    # ── Test 3: Signal generation ─────────────────────────────────────────
    signals = strategy.generate_signals(state_matrix)
    _validate_signals(signals, len(state_matrix))
    longs = int((signals == 1).sum())
    shorts = int((signals == -1).sum())
    flat = int((signals == 0).sum())
    print(f"  Signals: OK (len={len(signals)}, "
          f"longs={longs}, shorts={shorts}, flat={flat})")

    # ── Test 4: Backtest ──────────────────────────────────────────────────
    df = state_matrix.copy()
    df["_signal"] = signals.values
    backtester = VectorizedBacktester(fee=BACKTEST_FEE, risk_per_trade=RISK_PER_TRADE)
    trade_log = backtester.run(df, "_signal")
    print(f"  Backtest: OK ({len(trade_log)} trades)")

    # ── Test 5: Diagnostics ───────────────────────────────────────────────
    engine = DiagnosticsEngine(
        min_trades=MIN_TRADES_SUFFICIENT_EVIDENCE,
        state_matrix=state_matrix,
    )
    diagnostics = engine.compute(trade_log)
    expected_granularities = {"GLOBAL", "1D", "2D", "3D"}
    actual_granularities = set(diagnostics["granularity"].unique())
    assert expected_granularities == actual_granularities, (
        f"Expected {expected_granularities}, got {actual_granularities}"
    )
    print(f"  Diagnostics: OK ({len(diagnostics)} rows, "
          f"granularities={sorted(actual_granularities)})")

    # ── Test 6: Fitness ───────────────────────────────────────────────────
    score = compute_fitness(diagnostics)
    print(f"  Fitness: {score:.4f}")

    # ── Test 7: Bad code handling ─────────────────────────────────────────
    try:
        _validate_and_instantiate("this is not valid python {{{")
        assert False, "Should have raised"
    except ValueError as e:
        print(f"  Bad syntax rejection: OK ({e})")

    try:
        _validate_and_instantiate("x = 42  # no Strategy subclass")
        assert False, "Should have raised"
    except ValueError as e:
        print(f"  Missing class rejection: OK")

    try:
        no_trade_code = '''\
class NoTradeStrategy(Strategy):
    name = "no_trade"
    family = "trend"
    description = "Always flat"
    def generate_signals(self, data):
        return pd.Series(0, index=data.index)
'''
        strat = _validate_and_instantiate(no_trade_code)
        sigs = strat.generate_signals(state_matrix)
        _validate_signals(sigs, len(state_matrix))
        assert False, "Should have raised"
    except ValueError as e:
        print(f"  Zero-trade rejection: OK")

    print("\nPASS")
