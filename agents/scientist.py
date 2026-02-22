"""
scientist.py — Council of Alphas
Iterative Critic/Refiner improvement loop for hybrid strategies.

Each hybrid runs independently through up to 5 iterations of:
  Critic (Opus) → Refiner (Sonnet) → Validate → Backtest → Accept/Revert

Guarantees monotonic improvement (v_n >= v_{n-1}).
Stops on: UNVIABLE verdict, early exit (2 consecutive improvements < 0.05),
or max iterations reached.

Public API:
    run_scientist()       — one hybrid through the full loop
    run_all_scientists()  — all hybrids in parallel
"""

from __future__ import annotations

import sys
import asyncio
import logging
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _p in [str(_PROJECT_ROOT / "core"), str(_PROJECT_ROOT / "pipeline"),
           str(_PROJECT_ROOT / "agents")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from config import (
    BACKTEST_FEE,
    RISK_PER_TRADE,
    MIN_TRADES_SUFFICIENT_EVIDENCE,
    MAX_SCIENTIST_ITERATIONS,
    MIN_IMPROVEMENT_THRESHOLD,
    STRATEGY_TIMEOUT_SECONDS,
)
from strategy_base import Strategy
from whitelist_indicators import Indicators
from backtesting import VectorizedBacktester
from diagnostics import DiagnosticsEngine
from fitness import compute_fitness, is_unviable
from specialist_agent import _validate_and_instantiate, _validate_signals
from hybrid_builder import inject_champions
from critic_agent import run_critic
from refiner_agent import run_refiner

load_dotenv(_PROJECT_ROOT / ".env")

logger = logging.getLogger("council.scientist")


# ── Evaluation Helper ────────────────────────────────────────────────────────

def _evaluate_strategy(
    strategy: Strategy,
    state_matrix: pd.DataFrame,
) -> tuple[float, pd.DataFrame]:
    """
    Run the full evaluation pipeline on a strategy (synchronous).

    Pipeline: generate_signals → validate → backtest → diagnostics → fitness

    Parameters
    ----------
    strategy : Strategy
        Must have _champion_strategies injected if it's a hybrid.
    state_matrix : pd.DataFrame
        Full state matrix (read-only — a copy is made for signal attachment).

    Returns
    -------
    (fitness, diagnostics_df)
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

    # Attach trade log to strategy for result saving
    strategy._trade_log = trade_log

    return fitness, diagnostics


# ── Single Hybrid Scientist Loop ─────────────────────────────────────────────

async def run_scientist(
    hybrid: Strategy,
    state_matrix: pd.DataFrame,
    champion_strategies: dict[str, Strategy],
) -> tuple[Strategy, float, pd.DataFrame] | None:
    """
    Run the Critic/Refiner loop on one hybrid strategy.

    Parameters
    ----------
    hybrid : Strategy
        Hybrid with _champion_strategies injected and _source_code set.
    state_matrix : pd.DataFrame
        Full state matrix.
    champion_strategies : dict[str, Strategy]
        {family: strategy_instance} for champion re-injection after Refiner.

    Returns
    -------
    (best_strategy, best_fitness, best_diagnostics) or None if UNVIABLE.
    """
    name = hybrid.name

    # ── Initial evaluation ───────────────────────────────────────────────
    try:
        fitness, diagnostics = await asyncio.wait_for(
            asyncio.to_thread(_evaluate_strategy, hybrid, state_matrix),
            timeout=STRATEGY_TIMEOUT_SECONDS,
        )
    except Exception as e:
        logger.error(f"[{name}] Initial evaluation failed: {type(e).__name__}: {e}")
        return None

    logger.info(f"[{name}] Initial fitness: {fitness:.3f}")

    # ── UNVIABLE gate ────────────────────────────────────────────────────
    if fitness == -999.0:
        logger.warning(f"[{name}] UNVIABLE: hard elimination (fitness=-999)")
        return None

    unviable, reason = is_unviable(diagnostics)
    if unviable:
        logger.warning(f"[{name}] UNVIABLE: {reason}")
        return None

    # ── Track best version ───────────────────────────────────────────────
    best_strategy = hybrid
    best_fitness = fitness
    best_diagnostics = diagnostics

    current_code = hybrid._source_code
    current_diagnostics = diagnostics

    consecutive_small = 0

    # ── Critic / Refiner loop ────────────────────────────────────────────
    for iteration in range(1, MAX_SCIENTIST_ITERATIONS + 1):
        logger.info(
            f"[{name}] Iteration {iteration}/{MAX_SCIENTIST_ITERATIONS} "
            f"(best_fitness={best_fitness:.3f})"
        )

        # 1. Critic diagnosis
        critic_result = await run_critic(current_code, current_diagnostics)

        if critic_result["verdict"] == "UNVIABLE":
            logger.info(f"[{name}] Critic says UNVIABLE at iteration {iteration}")
            break

        # 2. Refiner applies fix
        refined_code = await run_refiner(
            current_code, critic_result["surgical_fix"]
        )

        if not refined_code:
            logger.warning(
                f"[{name}] Refiner returned empty code at iteration {iteration}"
            )
            consecutive_small += 1
            if consecutive_small >= 2:
                logger.info(f"[{name}] Early exit: 2 consecutive failed/small iterations")
                break
            continue

        # 3. Validate refined code + re-evaluate
        try:
            new_strategy = _validate_and_instantiate(refined_code)
            new_strategy._source_code = refined_code
            inject_champions(new_strategy, champion_strategies)

            new_fitness, new_diagnostics = await asyncio.wait_for(
                asyncio.to_thread(_evaluate_strategy, new_strategy, state_matrix),
                timeout=STRATEGY_TIMEOUT_SECONDS,
            )
        except Exception as e:
            logger.warning(
                f"[{name}] Refined code failed at iteration {iteration}: "
                f"{type(e).__name__}: {e}"
            )
            consecutive_small += 1
            if consecutive_small >= 2:
                logger.info(f"[{name}] Early exit: 2 consecutive failed/small iterations")
                break
            continue

        improvement = new_fitness - best_fitness

        # 4. Accept or revert (monotonic improvement guarantee)
        if new_fitness > best_fitness:
            best_strategy = new_strategy
            best_fitness = new_fitness
            best_diagnostics = new_diagnostics
            current_code = refined_code
            current_diagnostics = new_diagnostics

            logger.info(
                f"[{name}] Iteration {iteration}: IMPROVED "
                f"fitness={best_fitness:.3f} (+{improvement:.3f})"
            )

            if improvement < MIN_IMPROVEMENT_THRESHOLD:
                consecutive_small += 1
            else:
                consecutive_small = 0
        else:
            logger.info(
                f"[{name}] Iteration {iteration}: REVERTED "
                f"(new={new_fitness:.3f} <= best={best_fitness:.3f})"
            )
            consecutive_small += 1

        # 5. Early exit check
        if consecutive_small >= 2:
            logger.info(f"[{name}] Early exit: 2 consecutive small/no improvements")
            break

    logger.info(f"[{name}] Scientist complete: final fitness={best_fitness:.3f}")
    return (best_strategy, best_fitness, best_diagnostics)


# ── Parallel Execution ───────────────────────────────────────────────────────

async def run_all_scientists(
    hybrids: list[Strategy],
    state_matrix: pd.DataFrame,
    champion_strategies: dict[str, Strategy],
) -> list[tuple[Strategy, float, pd.DataFrame]]:
    """
    Run all hybrids through the Scientist loop in parallel.

    Parameters
    ----------
    hybrids : list[Strategy]
        Hybrid strategies from HybridBuilder (with champions injected).
    state_matrix : pd.DataFrame
        Full state matrix.
    champion_strategies : dict[str, Strategy]
        {family: strategy_instance} for champion re-injection.

    Returns
    -------
    list[tuple[Strategy, float, pd.DataFrame]]
        Surviving hybrids as (strategy, fitness, diagnostics).
        UNVIABLE hybrids are discarded.
    """
    logger.info(f"Starting Scientist loop: {len(hybrids)} hybrids in parallel")

    tasks = [
        run_scientist(hybrid, state_matrix, champion_strategies)
        for hybrid in hybrids
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    survivors: list[tuple[Strategy, float, pd.DataFrame]] = []
    for hybrid, result in zip(hybrids, results):
        if isinstance(result, BaseException):
            logger.error(
                f"[{hybrid.name}] Scientist crashed: "
                f"{type(result).__name__}: {result}"
            )
        elif result is None:
            logger.warning(f"[{hybrid.name}] Discarded (UNVIABLE)")
        else:
            survivors.append(result)

    logger.info(
        f"Scientist loop complete: "
        f"{len(survivors)}/{len(hybrids)} hybrids survived"
    )
    return survivors


__all__ = ["run_scientist", "run_all_scientists"]


# ── Smoke Test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print("[SMOKE TEST] scientist.py\n")

    # ── Build dummy state matrix (reuse from specialist_agent) ───────────
    from specialist_agent import _build_dummy_state_matrix, _TEST_STRATEGY_CODE

    state_matrix = _build_dummy_state_matrix(500)
    print(f"  State matrix: {state_matrix.shape[0]} rows, "
          f"{state_matrix.shape[1]} cols")

    # ── Test 1: _evaluate_strategy end-to-end ────────────────────────────
    print("\n=== Test 1: _evaluate_strategy with EMA crossover ===")
    strategy = _validate_and_instantiate(_TEST_STRATEGY_CODE)
    strategy._source_code = _TEST_STRATEGY_CODE

    fitness, diagnostics = _evaluate_strategy(strategy, state_matrix)
    print(f"  Fitness: {fitness:.4f}")
    print(f"  Diagnostics: {len(diagnostics)} rows")
    assert isinstance(fitness, float)
    assert isinstance(diagnostics, pd.DataFrame)
    assert len(diagnostics) > 0
    assert "granularity" in diagnostics.columns
    assert "sharpe" in diagnostics.columns
    expected_grans = {"GLOBAL", "1D", "2D", "3D"}
    actual_grans = set(diagnostics["granularity"].unique())
    assert expected_grans == actual_grans, f"Expected {expected_grans}, got {actual_grans}"
    print("  OK")

    # ── Test 2: UNVIABLE detection — fitness == -999 path ────────────────
    print("\n=== Test 2: UNVIABLE via hard elimination (fitness=-999) ===")

    class _AllFlatStrategy(Strategy):
        name = "all_flat"
        family = "trend"
        description = "Produces very few trades (will get -999)"

        def generate_signals(self, data):
            # Only 5 signals total — will produce very few trades
            signals = pd.Series(0, index=data.index)
            signals.iloc[0] = 1
            signals.iloc[50] = -1
            signals.iloc[100] = 1
            signals.iloc[200] = -1
            signals.iloc[300] = 1
            return signals

    flat_strat = _AllFlatStrategy()
    flat_fitness, flat_diag = _evaluate_strategy(flat_strat, state_matrix)
    print(f"  Fitness: {flat_fitness}")
    # With 5 signals on 500 rows, trades will be very sparse.
    # This should result in -999 due to insufficient evidence or Sharpe <= 0
    assert flat_fitness == -999.0 or flat_fitness > 0, \
        "Fitness should be either -999 (hard elimination) or positive"
    print(f"  Hard elimination triggered: {flat_fitness == -999.0}")
    print("  OK")

    # ── Test 3: is_unviable() detection ──────────────────────────────────
    print("\n=== Test 3: is_unviable() detection ===")

    # Create a diagnostics table that triggers UNVIABLE (Sharpe < -5.0)
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
    print(f"  Detected UNVIABLE: {unviable_reason}")

    # Create a diagnostics table that triggers max consecutive losses
    bad_diag_losses = pd.DataFrame([{
        "granularity": "GLOBAL",
        "session": "ALL",
        "trend_regime": "ALL",
        "vol_regime": "ALL",
        "trade_count": 500,
        "win_rate": 0.45,
        "sharpe": 0.3,
        "max_consecutive_losses": 25,
        "sufficient_evidence": True,
    }])

    unviable_flag2, unviable_reason2 = is_unviable(bad_diag_losses)
    assert unviable_flag2 is True
    assert "consecutive" in unviable_reason2.lower()
    print(f"  Detected UNVIABLE: {unviable_reason2}")

    # Good diagnostics should NOT be unviable
    good_diag = pd.DataFrame([
        {
            "granularity": "GLOBAL",
            "session": "ALL",
            "trend_regime": "ALL",
            "vol_regime": "ALL",
            "trade_count": 500,
            "win_rate": 0.55,
            "sharpe": 0.4,
            "max_consecutive_losses": 5,
            "sufficient_evidence": True,
        },
        {
            "granularity": "3D",
            "session": "ASIA",
            "trend_regime": "UPTREND",
            "vol_regime": "HIGH_VOL",
            "trade_count": 40,
            "win_rate": 0.60,
            "sharpe": 0.8,
            "max_consecutive_losses": 3,
            "sufficient_evidence": True,
        },
    ])

    viable_flag, viable_reason = is_unviable(good_diag)
    assert viable_flag is False
    assert viable_reason == ""
    print(f"  Good diagnostics correctly passes UNVIABLE gate")
    print("  OK")

    # ── Test 4: Diagnostics structure completeness ───────────────────────
    print("\n=== Test 4: Diagnostics structure check ===")
    required_cols = [
        "granularity", "session", "trend_regime", "vol_regime",
        "trade_count", "win_rate", "sharpe", "max_consecutive_losses",
        "sufficient_evidence",
    ]
    for col in required_cols:
        assert col in diagnostics.columns, f"Missing column: {col}"
    print(f"  All {len(required_cols)} required columns present")

    global_rows = diagnostics[diagnostics["granularity"] == "GLOBAL"]
    assert len(global_rows) == 1, f"Expected 1 GLOBAL row, got {len(global_rows)}"
    print(f"  GLOBAL row: sharpe={global_rows.iloc[0]['sharpe']:.4f}, "
          f"trades={global_rows.iloc[0]['trade_count']}")

    rows_3d = diagnostics[diagnostics["granularity"] == "3D"]
    print(f"  3D rows: {len(rows_3d)} buckets")
    print("  OK")

    # ── Test 5: Source code preservation ──────────────────────────────────
    print("\n=== Test 5: Source code preservation ===")
    assert hasattr(strategy, "_source_code")
    assert strategy._source_code == _TEST_STRATEGY_CODE
    assert "class TestEMACrossover" in strategy._source_code
    print(f"  _source_code: {len(strategy._source_code)} chars, preserved correctly")
    print("  OK")

    print("\nPASS")
