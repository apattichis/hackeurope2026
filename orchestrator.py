"""
orchestrator.py — Council of Alphas
Main pipeline controller.

Implements the full flow from MASTER_SPEC Section 12.1:
  Load Data → State Matrix → Speciation → Niche Selection →
  Hybrid Building → Optimizer → Final Ranking

Error handling follows Section 12.2 tiers:
  Fatal: data load, state matrix, zero champions
  Warn/continue: one specialist fails, < 2 champions
  Fallback: all hybrids UNVIABLE → best champion

Public API:
    Orchestrator.run()  — async, runs full pipeline
    PipelineResult      — dataclass with all outputs for UI
"""

from __future__ import annotations

import sys
import asyncio
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
for _p in [str(_PROJECT_ROOT / "core"), str(_PROJECT_ROOT / "pipeline"),
           str(_PROJECT_ROOT / "agents")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from config import DATA_PATH, STATE_MATRIX_PATH
from fitness import compute_fitness
from state_builder import StateMatrixBuilder
from strategy_base import Strategy
from specialist_agent import run_all_specialists, get_api_usage, reset_api_usage
from niche_selector import select_champions
from hybrid_builder import HybridBuilder
from optimizer import run_all_optimizers

load_dotenv(_PROJECT_ROOT / ".env")

logger = logging.getLogger("council.orchestrator")


# ── Pipeline Result ──────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    """Complete pipeline output for UI consumption."""
    ranked: list[tuple]           # [(strategy, fitness, diagnostics), ...] best first
    champions: dict[str, tuple]   # {family: (strategy, score, diagnostics)}
    hybrids_built: int
    hybrids_survived: int
    fallback_used: bool
    elapsed_seconds: float
    all_speciation: dict | None = None  # {family: [(strategy, score, diag), ...]}
    api_usage: dict | None = None       # {input_tokens, output_tokens, api_calls, estimated_cost_usd}


# ── Orchestrator ─────────────────────────────────────────────────────────────

class Orchestrator:
    """
    Main pipeline controller.

    Runs the full Council of Alphas pipeline:
    1. Load data
    2. Build/load state matrix
    3. Speciation (4 parallel specialists)
    4. Niche selection (1 champion per family)
    5. Hybrid building (3 deterministic templates)
    6. Optimizer (deterministic 2D regime filter)
    7. Final ranking
    """

    def __init__(
        self,
        data_path: str = DATA_PATH,
        state_matrix_path: str = STATE_MATRIX_PATH,
    ) -> None:
        self.data_path = Path(_PROJECT_ROOT) / data_path
        self.state_matrix_path = Path(_PROJECT_ROOT) / state_matrix_path

    async def run(self, state_matrix=None) -> PipelineResult:
        """Run the full pipeline. Returns PipelineResult."""
        start = time.time()
        reset_api_usage()

        if state_matrix is None:
            # ── 1. Load data ─────────────────────────────────────────────
            raw_df = self._load_data()

            # ── 2. Build or load state matrix ────────────────────────────
            state_matrix = self._load_or_build_state_matrix(raw_df)

        # ── 3. Speciation ────────────────────────────────────────────────
        all_results = await self._run_speciation(state_matrix)

        # ── 4. Niche selection ───────────────────────────────────────────
        champions = self._run_niche_selection(all_results)

        # ── 5-7. Hybrid building + Scientist + Ranking ───────────────────
        fallback_used = False
        hybrids_built = 0
        hybrids_survived = 0

        if len(champions) == 1:
            # 1 champion → skip hybrids, return champion directly
            logger.warning(
                "Only 1 champion survived - skipping hybrid building. "
                "Returning champion directly."
            )
            family = list(champions.keys())[0]
            strategy, score, diagnostics = champions[family]
            ranked = [(strategy, score, diagnostics)]
            fallback_used = True

        else:
            # 5. Build hybrids
            hybrids = self._run_hybrid_building(champions, state_matrix)
            hybrids_built = len(hybrids)

            if hybrids_built == 0:
                logger.warning("No hybrids built - falling back to best champion")
                ranked = self._fallback_to_best_champion(champions)
                fallback_used = True
            else:
                # 6. Optimizer (2D regime filter)
                champion_strategies = {
                    f: champ[0] for f, champ in champions.items()
                }
                survivors = await self._run_optimizer(
                    hybrids, state_matrix, champion_strategies
                )
                hybrids_survived = len(survivors)

                if hybrids_survived == 0:
                    logger.warning(
                        "All hybrids UNVIABLE - falling back to best champion"
                    )
                    ranked = self._fallback_to_best_champion(champions)
                    fallback_used = True
                else:
                    # 7. Final ranking
                    ranked = self._final_ranking(survivors)

        elapsed = time.time() - start
        api_usage = get_api_usage()
        logger.info(
            f"Pipeline complete in {elapsed:.1f}s: "
            f"{len(ranked)} strategies ranked, "
            f"fallback={'yes' if fallback_used else 'no'}"
        )
        logger.info(
            f"API usage: {api_usage['api_calls']} calls, "
            f"{api_usage['input_tokens']:,} input + {api_usage['output_tokens']:,} output tokens, "
            f"estimated cost: ${api_usage['estimated_cost_usd']:.4f}"
        )

        return PipelineResult(
            ranked=ranked,
            champions=champions,
            hybrids_built=hybrids_built,
            hybrids_survived=hybrids_survived,
            fallback_used=fallback_used,
            elapsed_seconds=elapsed,
            all_speciation=all_results,
            api_usage=api_usage,
        )

    # ── Stage Implementations ────────────────────────────────────────────

    def _load_data(self) -> pd.DataFrame:
        """Load raw OHLCV data from parquet. Fatal if missing."""
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Data file not found: {self.data_path}. "
                f"Run data/download_data.py first."
            )

        df = pd.read_parquet(self.data_path)
        logger.info(f"Loaded data: {len(df):,} candles from {self.data_path.name}")
        return df

    def _load_or_build_state_matrix(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Load state matrix from parquet if exists, else build and save."""
        required_cols = {"tbm_long_outcome", "tbm_short_outcome"}
        if self.state_matrix_path.exists():
            state_matrix = pd.read_parquet(self.state_matrix_path)
            missing = required_cols - set(state_matrix.columns)
            if not missing:
                logger.info(
                    f"Loaded state matrix: {len(state_matrix):,} rows "
                    f"from {self.state_matrix_path.name}"
                )
                return state_matrix
            logger.warning(
                f"State matrix missing required TBM outcome columns {sorted(missing)}. "
                "Rebuilding state matrix from raw data."
            )

        logger.info("Building state matrix from raw data...")
        builder = StateMatrixBuilder()
        state_matrix = builder.build(raw_df)

        self.state_matrix_path.parent.mkdir(parents=True, exist_ok=True)
        state_matrix.to_parquet(self.state_matrix_path, engine="pyarrow")
        logger.info(
            f"State matrix built and saved: {len(state_matrix):,} rows "
            f"to {self.state_matrix_path.name}"
        )
        return state_matrix

    async def _run_speciation(
        self, state_matrix: pd.DataFrame,
    ) -> dict[str, list[tuple]]:
        """Run all 4 specialists in parallel. Fatal if zero strategies."""
        logger.info("Starting speciation...")
        all_results = await run_all_specialists(state_matrix)

        total = sum(len(v) for v in all_results.values())
        if total == 0:
            raise RuntimeError(
                "FATAL: All specialists failed. Zero strategies generated."
            )

        logger.info(
            f"Speciation complete: {total} strategies "
            f"across {len(all_results)} families"
        )
        return all_results

    def _run_niche_selection(
        self, all_results: dict[str, list[tuple]],
    ) -> dict[str, tuple]:
        """Select one champion per family. Fatal if zero champions."""
        logger.info("Running niche selection...")
        champions = select_champions(all_results)

        if len(champions) == 0:
            raise RuntimeError(
                "FATAL: No families produced viable champions. "
                "All strategies failed hard eliminations."
            )

        logger.info(
            f"Niche selection: {len(champions)} champion(s) "
            f"({', '.join(champions.keys())})"
        )
        return champions

    def _run_hybrid_building(
        self,
        champions: dict[str, tuple],
        state_matrix: pd.DataFrame,
    ) -> list[Strategy]:
        """Build 3 hybrid strategies from champions."""
        logger.info(f"Building hybrids from {len(champions)} champions...")
        builder = HybridBuilder(champions, state_matrix)
        hybrids = builder.build_all()
        logger.info(f"Built {len(hybrids)} hybrids")
        return hybrids

    async def _run_optimizer(
        self,
        hybrids: list[Strategy],
        state_matrix: pd.DataFrame,
        champion_strategies: dict[str, Strategy],
    ) -> list[tuple[Strategy, float, pd.DataFrame]]:
        """Run 2D regime filter on all hybrids in parallel."""
        logger.info(f"Starting optimizer: {len(hybrids)} hybrids...")
        survivors = await run_all_optimizers(
            hybrids, state_matrix, champion_strategies
        )
        logger.info(f"Optimizer complete: {len(survivors)} hybrids survived")
        return survivors

    def _final_ranking(
        self,
        survivors: list[tuple[Strategy, float, pd.DataFrame]],
    ) -> list[tuple[Strategy, float, pd.DataFrame]]:
        """Rank survivors by fitness score (descending)."""
        ranked = sorted(survivors, key=lambda x: x[1], reverse=True)
        for i, (strat, score, _) in enumerate(ranked):
            logger.info(f"  #{i + 1}: {strat.name} (fitness={score:.3f})")
        return ranked

    def _fallback_to_best_champion(
        self,
        champions: dict[str, tuple],
    ) -> list[tuple[Strategy, float, pd.DataFrame]]:
        """Fall back to best champion by fitness score."""
        best_family = max(champions, key=lambda f: champions[f][1])
        strategy, score, diagnostics = champions[best_family]
        logger.info(
            f"Fallback: best champion = {strategy.name} "
            f"(family={best_family}, fitness={score:.3f})"
        )
        return [(strategy, score, diagnostics)]


# ── Results Saver ───────────────────────────────────────────────────────────

def save_pipeline_results(
    result: PipelineResult,
    all_speciation: dict[str, list[tuple]] | None = None,
    results_dir: str | Path = "data/results",
) -> Path:
    """
    Save complete pipeline results to disk for analysis.

    Saves:
    - summary.json: top-level pipeline stats + per-strategy summary
    - speciation/: all strategies from speciation (diagnostics + trade logs)
    - champions/: champion strategies (diagnostics + trade logs)
    - ranked/: final ranked strategies (diagnostics + trade logs)

    Parameters
    ----------
    result : PipelineResult
        Full pipeline output.
    all_speciation : dict[str, list[tuple]] | None
        Raw speciation results: {family: [(strategy, score, diagnostics), ...]}
    results_dir : str | Path
        Base directory for results (relative to project root).

    Returns
    -------
    Path to results directory.
    """
    base = Path(_PROJECT_ROOT) / results_dir
    base.mkdir(parents=True, exist_ok=True)

    # ── Helper: extract strategy summary dict ───────────────────────────
    def _strat_summary(strategy, score, diagnostics):
        global_rows = diagnostics[diagnostics["granularity"] == "GLOBAL"]
        g = global_rows.iloc[0] if len(global_rows) > 0 else {}
        summary = {
            "name": getattr(strategy, "name", "unknown"),
            "family": getattr(strategy, "family", "unknown"),
            "description": getattr(strategy, "description", ""),
            "fitness": float(score),
            "sharpe": float(g["sharpe"]) if isinstance(g, pd.Series) and not pd.isna(g.get("sharpe")) else None,
            "win_rate": float(g["win_rate"]) if isinstance(g, pd.Series) else None,
            "trades": int(g["trade_count"]) if isinstance(g, pd.Series) else 0,
            "max_consec_losses": int(g["max_consecutive_losses"]) if isinstance(g, pd.Series) else None,
        }
        # 3D bucket summary
        active_3d = diagnostics[
            (diagnostics["granularity"] == "3D")
            & (diagnostics["sufficient_evidence"] == True)
        ]
        profitable_3d = active_3d[active_3d["sharpe"] > 0]
        summary["profitable_3d"] = len(profitable_3d)
        summary["total_3d_sufficient"] = len(active_3d)
        summary["total_3d"] = len(diagnostics[diagnostics["granularity"] == "3D"])
        return summary

    # ── Helper: save strategy artifacts ─────────────────────────────────
    def _save_strategy_artifacts(strategy, score, diagnostics, folder: Path):
        folder.mkdir(parents=True, exist_ok=True)

        # Diagnostics table (full 60 rows)
        diagnostics.to_csv(folder / "diagnostics.csv", index=False)

        # Trade log if available
        trade_log = getattr(strategy, "_trade_log", None)
        if trade_log is not None and isinstance(trade_log, pd.DataFrame) and len(trade_log) > 0:
            trade_log.to_csv(folder / "trade_log.csv", index=False)

        # Strategy source code
        source = getattr(strategy, "_source_code", None)
        if source:
            (folder / "strategy_code.py").write_text(source, encoding="utf-8")

        # Summary JSON
        summary = _strat_summary(strategy, score, diagnostics)
        (folder / "summary.json").write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8"
        )

    # ── 1. Save speciation results ──────────────────────────────────────
    if all_speciation:
        spec_dir = base / "speciation"
        spec_summaries = {}
        for family, strats in all_speciation.items():
            spec_summaries[family] = []
            for i, (strategy, score, diagnostics) in enumerate(strats):
                name = getattr(strategy, "name", f"{family}_{i}")
                folder = spec_dir / family / name
                _save_strategy_artifacts(strategy, score, diagnostics, folder)
                spec_summaries[family].append(_strat_summary(strategy, score, diagnostics))
        (spec_dir / "speciation_summary.json").write_text(
            json.dumps(spec_summaries, indent=2, default=str), encoding="utf-8"
        )
        logger.info(f"Saved speciation results to {spec_dir}")

    # ── 2. Save champion results ────────────────────────────────────────
    champ_dir = base / "champions"
    champ_summaries = {}
    for family, (strategy, score, diagnostics) in result.champions.items():
        folder = champ_dir / family
        _save_strategy_artifacts(strategy, score, diagnostics, folder)
        champ_summaries[family] = _strat_summary(strategy, score, diagnostics)
    (champ_dir / "champions_summary.json").write_text(
        json.dumps(champ_summaries, indent=2, default=str), encoding="utf-8"
    )
    logger.info(f"Saved champion results to {champ_dir}")

    # ── 3. Save ranked (final) results ──────────────────────────────────
    ranked_dir = base / "ranked"
    ranked_summaries = []
    for rank, (strategy, score, diagnostics) in enumerate(result.ranked, 1):
        name = getattr(strategy, "name", f"rank_{rank}")
        folder = ranked_dir / f"{rank}_{name}"
        _save_strategy_artifacts(strategy, score, diagnostics, folder)
        summary = _strat_summary(strategy, score, diagnostics)
        summary["rank"] = rank
        ranked_summaries.append(summary)
    (ranked_dir / "ranked_summary.json").write_text(
        json.dumps(ranked_summaries, indent=2, default=str), encoding="utf-8"
    )
    logger.info(f"Saved ranked results to {ranked_dir}")

    # ── 4. Top-level summary ────────────────────────────────────────────
    top_summary = {
        "elapsed_seconds": result.elapsed_seconds,
        "hybrids_built": result.hybrids_built,
        "hybrids_survived": result.hybrids_survived,
        "fallback_used": result.fallback_used,
        "num_champions": len(result.champions),
        "num_ranked": len(result.ranked),
        "api_usage": result.api_usage,
        "champions": champ_summaries,
        "ranked": ranked_summaries,
    }
    if all_speciation:
        top_summary["speciation"] = {
            family: len(strats) for family, strats in all_speciation.items()
        }
    (base / "summary.json").write_text(
        json.dumps(top_summary, indent=2, default=str), encoding="utf-8"
    )
    logger.info(f"Full results saved to {base}")

    return base


__all__ = ["Orchestrator", "PipelineResult", "save_pipeline_results"]


# ── Smoke Test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print("[SMOKE TEST] orchestrator.py\n")

    # ── Test 1: Instantiation ────────────────────────────────────────────
    print("=== Test 1: Instantiation ===")
    orch = Orchestrator()
    assert orch.data_path.name == "sol_usd_1h.parquet"
    assert orch.state_matrix_path.name == "state_matrix_1h.parquet"
    print(f"  data_path: {orch.data_path}")
    print(f"  state_matrix_path: {orch.state_matrix_path}")
    print("  OK")

    # ── Test 2: Data loading ─────────────────────────────────────────────
    print("\n=== Test 2: Data loading ===")
    if orch.data_path.exists():
        raw_df = orch._load_data()
        print(f"  Loaded {len(raw_df):,} rows")
        assert len(raw_df) > 100_000
        required_cols = ["open", "high", "low", "close", "volume"]
        for col in required_cols:
            assert col in raw_df.columns, f"Missing column: {col}"
        print(f"  Columns: {raw_df.columns.tolist()}")
        print("  OK")
    else:
        print("  SKIPPED (parquet not found)")

    # ── Test 3: Final ranking ────────────────────────────────────────────
    print("\n=== Test 3: Final ranking ===")

    class _FakeStrat:
        def __init__(self, name, family):
            self.name = name
            self.family = family

    fake_survivors = [
        (_FakeStrat("hybrid_a", "hybrid"), 2.5, "diag_a"),
        (_FakeStrat("hybrid_b", "hybrid"), 3.1, "diag_b"),
        (_FakeStrat("hybrid_c", "hybrid"), 1.8, "diag_c"),
    ]
    ranked = orch._final_ranking(fake_survivors)
    assert ranked[0][1] == 3.1, "Best should be first"
    assert ranked[1][1] == 2.5
    assert ranked[2][1] == 1.8
    assert ranked[0][0].name == "hybrid_b"
    print(f"  Ranked: {[f'{r[0].name}={r[1]}' for r in ranked]}")
    print("  OK")

    # ── Test 4: Fallback logic ───────────────────────────────────────────
    print("\n=== Test 4: Fallback to best champion ===")
    fake_champions = {
        "trend": (_FakeStrat("trend_ema", "trend"), 3.10, "diag_t"),
        "momentum": (_FakeStrat("mom_rsi", "momentum"), 1.92, "diag_m"),
        "volatility": (_FakeStrat("vol_bb", "volatility"), 0.45, "diag_v"),
    }
    fallback = orch._fallback_to_best_champion(fake_champions)
    assert len(fallback) == 1
    assert fallback[0][0].name == "trend_ema"
    assert fallback[0][1] == 3.10
    print(f"  Fallback champion: {fallback[0][0].name} (fitness={fallback[0][1]})")
    print("  OK")

    # ── Test 5: PipelineResult structure ─────────────────────────────────
    print("\n=== Test 5: PipelineResult structure ===")
    result = PipelineResult(
        ranked=ranked,
        champions=fake_champions,
        hybrids_built=3,
        hybrids_survived=2,
        fallback_used=False,
        elapsed_seconds=42.5,
    )
    assert len(result.ranked) == 3
    assert len(result.champions) == 3
    assert result.hybrids_built == 3
    assert result.hybrids_survived == 2
    assert result.fallback_used is False
    assert result.elapsed_seconds == 42.5
    print(f"  ranked: {len(result.ranked)} strategies")
    print(f"  champions: {len(result.champions)} families")
    print(f"  hybrids_built: {result.hybrids_built}")
    print(f"  hybrids_survived: {result.hybrids_survived}")
    print(f"  fallback_used: {result.fallback_used}")
    print(f"  elapsed: {result.elapsed_seconds}s")
    print("  OK")

    # ── Test 6: Error handling — missing data ────────────────────────────
    print("\n=== Test 6: Missing data error ===")
    bad_orch = Orchestrator(data_path="data/nonexistent.parquet")
    try:
        bad_orch._load_data()
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError as e:
        assert "not found" in str(e).lower()
        print(f"  FileNotFoundError raised correctly")
    print("  OK")

    print("\nPASS")
