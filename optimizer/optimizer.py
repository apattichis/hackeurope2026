"""
optimizer.py - Council of Alphas
Deterministic 2D regime filter for hybrid strategies.

Replaces the LLM-based Scientist/Critic/Refiner loop with a single-pass
mathematical filter:
  1. Evaluate hybrid as-is (get full diagnostics)
  2. Extract 2D regime buckets from diagnostics
  3. Mark each 2D bucket as tradable (sharpe > 0 AND sufficient_evidence)
  4. For each bar: tradable = ALL 3 parent 2D buckets are tradable
  5. Zero signals where any parent 2D bucket is not tradable
  6. Re-evaluate with filtered signals
  7. Accept if fitness improved, else keep original (monotonic guarantee)

No LLM calls. Deterministic. One pass.

Public API:
    run_optimizer()       - one hybrid through the filter
    run_all_optimizers()  - all hybrids in parallel (async)
"""

from __future__ import annotations

import sys
import asyncio
import logging
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _p in [str(_PROJECT_ROOT / "core"), str(_PROJECT_ROOT / "pipeline"),
           str(_PROJECT_ROOT / "optimizer")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pandas as pd

from config import (
    BACKTEST_FEE,
    RISK_PER_TRADE,
    MIN_TRADES_SUFFICIENT_EVIDENCE,
    STRATEGY_TIMEOUT_SECONDS,
)
from strategy_base import Strategy
from backtesting import VectorizedBacktester
from diagnostics import DiagnosticsEngine
from fitness import compute_fitness, is_unviable
from specialist_agent import _validate_signals

logger = logging.getLogger("council.optimizer")


# ── Evaluation Helper ────────────────────────────────────────────────────

def _evaluate_strategy(
    strategy: Strategy,
    state_matrix: pd.DataFrame,
) -> tuple[float, pd.DataFrame]:
    """
    Run the full evaluation pipeline on a strategy.
    Pipeline: generate_signals -> validate -> backtest -> diagnostics -> fitness
    """
    signals = strategy.generate_signals(state_matrix)
    _validate_signals(signals, len(state_matrix))

    df = state_matrix.copy()
    df["_signal"] = signals.values
    backtester = VectorizedBacktester(fee=BACKTEST_FEE, risk_per_trade=RISK_PER_TRADE)
    trade_log = backtester.run(df, "_signal")

    engine = DiagnosticsEngine(
        min_trades=MIN_TRADES_SUFFICIENT_EVIDENCE,
        state_matrix=state_matrix,
    )
    diagnostics = engine.compute(trade_log)
    fitness = compute_fitness(diagnostics)

    strategy._trade_log = trade_log
    return fitness, diagnostics


# ── 2D Regime Filter ─────────────────────────────────────────────────────

def _extract_2d_tradability(
    diagnostics: pd.DataFrame,
) -> tuple[dict, dict, dict, list[str]]:
    """
    Extract tradable/non-tradable 2D buckets from diagnostics.

    A 2D bucket is tradable if: sharpe > 0 AND sufficient_evidence == True

    Returns
    -------
    (st_tradable, sv_tradable, tv_tradable, filtered_buckets)
        st_tradable: {(session, trend_regime): bool}
        sv_tradable: {(session, vol_regime): bool}
        tv_tradable: {(trend_regime, vol_regime): bool}
        filtered_buckets: list of string descriptions of non-tradable buckets
    """
    rows_2d = diagnostics[diagnostics["granularity"] == "2D"]

    st_tradable: dict[tuple, bool] = {}
    sv_tradable: dict[tuple, bool] = {}
    tv_tradable: dict[tuple, bool] = {}
    filtered: list[str] = []

    for _, row in rows_2d.iterrows():
        tradable = (
            bool(row["sufficient_evidence"])
            and not pd.isna(row["sharpe"])
            and float(row["sharpe"]) > 0
        )

        s = row["session"]
        t = row["trend_regime"]
        v = row["vol_regime"]

        if v == "ALL" and s != "ALL" and t != "ALL":
            # session x trend_regime
            st_tradable[(s, t)] = tradable
            if not tradable:
                sharpe_str = f"{row['sharpe']:.3f}" if not pd.isna(row["sharpe"]) else "NaN"
                filtered.append(
                    f"(session={s}, trend={t}): sharpe={sharpe_str}, "
                    f"sufficient={row['sufficient_evidence']}"
                )
        elif t == "ALL" and s != "ALL" and v != "ALL":
            # session x vol_regime
            sv_tradable[(s, v)] = tradable
            if not tradable:
                sharpe_str = f"{row['sharpe']:.3f}" if not pd.isna(row["sharpe"]) else "NaN"
                filtered.append(
                    f"(session={s}, vol={v}): sharpe={sharpe_str}, "
                    f"sufficient={row['sufficient_evidence']}"
                )
        elif s == "ALL" and t != "ALL" and v != "ALL":
            # trend_regime x vol_regime
            tv_tradable[(t, v)] = tradable
            if not tradable:
                sharpe_str = f"{row['sharpe']:.3f}" if not pd.isna(row["sharpe"]) else "NaN"
                filtered.append(
                    f"(trend={t}, vol={v}): sharpe={sharpe_str}, "
                    f"sufficient={row['sufficient_evidence']}"
                )

    return st_tradable, sv_tradable, tv_tradable, filtered


def _build_2d_mask(
    st_tradable: dict,
    sv_tradable: dict,
    tv_tradable: dict,
    state_matrix: pd.DataFrame,
) -> pd.Series:
    """
    Build per-bar boolean mask: True = tradable (keep signal).

    A bar is tradable if ALL 3 parent 2D buckets are tradable.
    Missing buckets default to False (conservative).
    """
    sessions = state_matrix["session"].values
    trends = state_matrix["trend_regime"].values
    vols = state_matrix["vol_regime"].values

    n = len(state_matrix)
    st_ok = np.array([st_tradable.get((sessions[i], trends[i]), False) for i in range(n)])
    sv_ok = np.array([sv_tradable.get((sessions[i], vols[i]), False) for i in range(n)])
    tv_ok = np.array([tv_tradable.get((trends[i], vols[i]), False) for i in range(n)])

    mask = st_ok & sv_ok & tv_ok
    return pd.Series(mask, index=state_matrix.index)


# ── FilteredHybrid ────────────────────────────────────────────────────────

class FilteredHybrid(Strategy):
    """Wrapper that applies a 2D regime filter to an existing hybrid."""

    def __init__(
        self,
        original: Strategy,
        st_tradable: dict,
        sv_tradable: dict,
        tv_tradable: dict,
        filtered_buckets: list[str],
    ):
        self.name = original.name
        self.family = original.family
        self.description = original.description
        self._original = original
        self._st_tradable = st_tradable
        self._sv_tradable = sv_tradable
        self._tv_tradable = tv_tradable
        self._filtered_buckets = filtered_buckets

        # Forward champion strategies from original
        if hasattr(original, "_champion_strategies"):
            self._champion_strategies = original._champion_strategies

        # Build source code for result saving
        self._source_code = self._build_source_code(original)

    def generate_signals(self, data):
        signals = self._original.generate_signals(data)

        sessions = data["session"].values
        trends = data["trend_regime"].values
        vols = data["vol_regime"].values

        n = len(data)
        st_ok = np.array([self._st_tradable.get((sessions[i], trends[i]), False) for i in range(n)])
        sv_ok = np.array([self._sv_tradable.get((sessions[i], vols[i]), False) for i in range(n)])
        tv_ok = np.array([self._tv_tradable.get((trends[i], vols[i]), False) for i in range(n)])

        mask = st_ok & sv_ok & tv_ok

        filtered = signals.copy()
        filtered.values[~mask] = 0
        return filtered

    def _build_source_code(self, original: Strategy) -> str:
        original_code = getattr(original, "_source_code", "# (no source code)")

        if self._filtered_buckets:
            filtered_lines = "\n".join(
                f"#   {b}" for b in self._filtered_buckets
            )
        else:
            filtered_lines = "#   (none)"

        return (
            f"# -- Original Hybrid --\n"
            f"{original_code}\n\n"
            f"# -- 2D Regime Filter Applied --\n"
            f"# Filtered (non-tradable) 2D buckets:\n"
            f"{filtered_lines}\n"
            f"#\n"
            f"# Bars in these regime combos have signals zeroed out.\n"
            f"# Filter: tradable = sharpe > 0 AND sufficient_evidence == True\n"
            f"# Gate: ALL 3 parent 2D buckets must be tradable\n"
        )


# ── Single Hybrid Optimizer ──────────────────────────────────────────────

def run_optimizer(
    hybrid: Strategy,
    state_matrix: pd.DataFrame,
    champion_strategies: dict[str, Strategy],
) -> tuple[Strategy, float, pd.DataFrame] | None:
    """
    Apply the 2D regime filter to one hybrid strategy.

    Returns (strategy, fitness, diagnostics) or None if UNVIABLE.
    Monotonic guarantee: returned fitness >= input fitness.
    """
    name = hybrid.name

    # 1. Initial evaluation
    try:
        fitness_0, diagnostics_0 = _evaluate_strategy(hybrid, state_matrix)
    except Exception as e:
        logger.error(f"[{name}] Initial evaluation failed: {type(e).__name__}: {e}")
        return None

    logger.info(f"[{name}] Initial fitness: {fitness_0:.3f}")

    # 2. UNVIABLE gate
    if fitness_0 == -999.0:
        logger.warning(f"[{name}] UNVIABLE: hard elimination (fitness=-999)")
        return None

    unviable, reason = is_unviable(diagnostics_0)
    if unviable:
        logger.warning(f"[{name}] UNVIABLE: {reason}")
        return None

    # 3. Extract 2D tradability
    st_trad, sv_trad, tv_trad, filtered_buckets = _extract_2d_tradability(
        diagnostics_0
    )

    if not filtered_buckets:
        logger.info(f"[{name}] All 2D buckets tradable - no filtering needed")
        return (hybrid, fitness_0, diagnostics_0)

    logger.info(
        f"[{name}] Filtering {len(filtered_buckets)} non-tradable 2D buckets:"
    )
    for b in filtered_buckets:
        logger.info(f"[{name}]   {b}")

    # 4. Check if mask filters any bars
    mask = _build_2d_mask(st_trad, sv_trad, tv_trad, state_matrix)
    bars_filtered = int((~mask).sum())
    bars_total = len(mask)
    logger.info(
        f"[{name}] Mask: {bars_filtered}/{bars_total} bars filtered "
        f"({bars_filtered / bars_total * 100:.1f}%)"
    )

    if bars_filtered == 0:
        logger.info(f"[{name}] No bars affected by filter - keeping original")
        return (hybrid, fitness_0, diagnostics_0)

    # 5. Build FilteredHybrid
    filtered_hybrid = FilteredHybrid(
        original=hybrid,
        st_tradable=st_trad,
        sv_tradable=sv_trad,
        tv_tradable=tv_trad,
        filtered_buckets=filtered_buckets,
    )

    # 6. Evaluate filtered version
    try:
        fitness_1, diagnostics_1 = _evaluate_strategy(filtered_hybrid, state_matrix)
    except Exception as e:
        logger.error(
            f"[{name}] Filtered evaluation failed: {type(e).__name__}: {e}"
        )
        return (hybrid, fitness_0, diagnostics_0)

    logger.info(
        f"[{name}] Filtered fitness: {fitness_1:.3f} (original: {fitness_0:.3f})"
    )

    # 7. Monotonic guarantee: keep better version
    if fitness_1 > fitness_0:
        improvement = fitness_1 - fitness_0
        logger.info(
            f"[{name}] IMPROVED by {improvement:.3f} -> using filtered version"
        )
        return (filtered_hybrid, fitness_1, diagnostics_1)
    else:
        logger.info(f"[{name}] No improvement -> keeping original")
        return (hybrid, fitness_0, diagnostics_0)


# ── Parallel Execution ───────────────────────────────────────────────────

async def run_all_optimizers(
    hybrids: list[Strategy],
    state_matrix: pd.DataFrame,
    champion_strategies: dict[str, Strategy],
) -> list[tuple[Strategy, float, pd.DataFrame]]:
    """
    Run all hybrids through the 2D regime filter in parallel.

    Returns list of (strategy, fitness, diagnostics) for surviving hybrids.
    UNVIABLE hybrids are discarded.
    """
    logger.info(f"Starting optimizer: {len(hybrids)} hybrids")

    tasks = [
        asyncio.wait_for(
            asyncio.to_thread(
                run_optimizer, hybrid, state_matrix, champion_strategies
            ),
            timeout=STRATEGY_TIMEOUT_SECONDS * 3,
        )
        for hybrid in hybrids
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    survivors: list[tuple[Strategy, float, pd.DataFrame]] = []
    for hybrid, result in zip(hybrids, results):
        if isinstance(result, BaseException):
            logger.error(
                f"[{hybrid.name}] Optimizer crashed: "
                f"{type(result).__name__}: {result}"
            )
        elif result is None:
            logger.warning(f"[{hybrid.name}] Discarded (UNVIABLE)")
        else:
            survivors.append(result)

    logger.info(
        f"Optimizer complete: "
        f"{len(survivors)}/{len(hybrids)} hybrids survived"
    )
    return survivors


__all__ = ["run_optimizer", "run_all_optimizers"]


# ── Smoke Test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print("[SMOKE TEST] optimizer.py\n")

    from specialist_agent import (
        _build_dummy_state_matrix,
        _TEST_STRATEGY_CODE,
        _validate_and_instantiate,
    )

    from config import ATR_WINDOW

    state_matrix = _build_dummy_state_matrix(500)

    # Add columns required by backtester/diagnostics but missing from dummy
    close = state_matrix["close"].values
    atr_col = f"ATR_{ATR_WINDOW}"
    if atr_col not in state_matrix.columns:
        state_matrix[atr_col] = np.abs(np.random.randn(len(state_matrix))) * 0.5 + 0.5
    if "tbm_long_outcome" not in state_matrix.columns:
        state_matrix["tbm_long_outcome"] = np.random.choice(
            ["TP", "SL", "TIMEOUT"], len(state_matrix)
        )
    if "tbm_short_outcome" not in state_matrix.columns:
        state_matrix["tbm_short_outcome"] = np.random.choice(
            ["TP", "SL", "TIMEOUT"], len(state_matrix)
        )

    print(
        f"  State matrix: {state_matrix.shape[0]} rows, "
        f"{state_matrix.shape[1]} cols"
    )

    # ── Test 1: _evaluate_strategy end-to-end ────────────────────────────
    print("\n=== Test 1: _evaluate_strategy ===")
    strategy = _validate_and_instantiate(_TEST_STRATEGY_CODE)
    strategy._source_code = _TEST_STRATEGY_CODE

    fitness, diagnostics = _evaluate_strategy(strategy, state_matrix)
    print(f"  Fitness: {fitness:.4f}")
    print(f"  Diagnostics: {len(diagnostics)} rows")
    assert isinstance(fitness, float)
    assert isinstance(diagnostics, pd.DataFrame)
    assert len(diagnostics) > 0

    expected_grans = {"GLOBAL", "1D", "2D", "3D"}
    actual_grans = set(diagnostics["granularity"].unique())
    assert expected_grans == actual_grans, (
        f"Expected {expected_grans}, got {actual_grans}"
    )
    print("  OK")

    # ── Test 2: _extract_2d_tradability ──────────────────────────────────
    print("\n=== Test 2: _extract_2d_tradability ===")
    st, sv, tv, filtered = _extract_2d_tradability(diagnostics)

    print(f"  session x trend buckets: {len(st)}")
    print(f"  session x vol buckets:   {len(sv)}")
    print(f"  trend x vol buckets:     {len(tv)}")
    print(f"  Non-tradable buckets:    {len(filtered)}")

    assert len(st) > 0, "Expected session x trend 2D buckets"
    assert len(sv) > 0, "Expected session x vol 2D buckets"
    assert len(tv) > 0, "Expected trend x vol 2D buckets"

    for d in [st, sv, tv]:
        for v in d.values():
            assert isinstance(v, (bool, np.bool_)), f"Expected bool, got {type(v)}"
    print("  OK")

    # ── Test 3: _build_2d_mask ───────────────────────────────────────────
    print("\n=== Test 3: _build_2d_mask ===")
    mask = _build_2d_mask(st, sv, tv, state_matrix)

    assert len(mask) == len(state_matrix)
    assert mask.dtype == bool
    tradable_count = int(mask.sum())
    filtered_count = len(mask) - tradable_count
    print(f"  Tradable bars: {tradable_count}/{len(mask)}")
    print(f"  Filtered bars: {filtered_count}/{len(mask)}")
    print("  OK")

    # ── Test 4: FilteredHybrid wrapping ──────────────────────────────────
    print("\n=== Test 4: FilteredHybrid ===")
    wrapped = FilteredHybrid(
        original=strategy,
        st_tradable=st,
        sv_tradable=sv,
        tv_tradable=tv,
        filtered_buckets=filtered,
    )

    assert wrapped.name == strategy.name
    assert wrapped.family == strategy.family
    assert hasattr(wrapped, "_source_code")
    assert "2D Regime Filter" in wrapped._source_code

    original_signals = strategy.generate_signals(state_matrix)
    filtered_signals = wrapped.generate_signals(state_matrix)

    assert len(filtered_signals) == len(state_matrix)
    original_active = int((original_signals != 0).sum())
    filtered_active = int((filtered_signals != 0).sum())
    assert filtered_active <= original_active, (
        f"Filtered ({filtered_active}) should have <= signals "
        f"than original ({original_active})"
    )
    print(f"  Original active signals: {original_active}")
    print(f"  Filtered active signals: {filtered_active}")
    print(f"  Signals removed: {original_active - filtered_active}")
    print("  OK")

    # ── Test 5: UNVIABLE detection ───────────────────────────────────────
    print("\n=== Test 5: UNVIABLE detection ===")
    bad_diag = pd.DataFrame([{
        "granularity": "GLOBAL",
        "session": "ALL",
        "trend_regime": "ALL",
        "vol_regime": "ALL",
        "trade_count": 500,
        "win_rate": 0.30,
        "sharpe": -6.0,
        "max_consecutive_losses": 5,
        "sufficient_evidence": True,
    }])

    unviable_flag, unviable_reason = is_unviable(bad_diag)
    assert unviable_flag is True
    assert "sharpe" in unviable_reason.lower()
    print(f"  UNVIABLE detected: {unviable_reason}")
    print("  OK")

    # ── Test 6: Monotonic guarantee ──────────────────────────────────────
    print("\n=== Test 6: Monotonic guarantee ===")
    result = run_optimizer(strategy, state_matrix, {})
    if result is not None:
        _, result_fitness, _ = result
        print(f"  Original fitness: {fitness:.4f}")
        print(f"  Optimizer result: {result_fitness:.4f}")
        assert result_fitness >= fitness, (
            f"Monotonic violation: {result_fitness:.4f} < {fitness:.4f}"
        )
        print("  Monotonic guarantee: OK")
    else:
        print("  Strategy was UNVIABLE (acceptable)")
    print("  OK")

    print("\nPASS")
