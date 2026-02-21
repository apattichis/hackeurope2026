"""
niche_selector.py — Council of Alphas
Select one champion per family by maximum fitness score.

Families where no strategy passed hard eliminations (score > 0)
are eliminated entirely. The pipeline continues with however
many champions survive (minimum 1 for hybrid building).

Public API:
    select_champions()  — pick best per family, discard losers
"""

from __future__ import annotations

import sys
import logging
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _p in [str(_PROJECT_ROOT / "core"), str(_PROJECT_ROOT / "pipeline")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import MIN_FITNESS_THRESHOLD

logger = logging.getLogger("council.niche")


def select_champions(
    all_results: dict[str, list[tuple]],
) -> dict[str, tuple]:
    """
    Select one champion per family by maximum fitness score.

    Parameters
    ----------
    all_results : dict[str, list[tuple]]
        {family: [(strategy, score, diagnostics), ...]}
        Output from run_all_specialists().

    Returns
    -------
    dict[str, tuple]
        {family: (strategy, score, diagnostics)} for surviving champions.
        Families with no viable strategies (score <= 0) are excluded.
    """
    champions: dict[str, tuple] = {}

    for family, results in all_results.items():
        if not results:
            logger.warning(f"[{family}] No strategies generated — family eliminated")
            continue

        # Filter to strategies that passed hard eliminations
        viable = [(s, score, diag) for s, score, diag in results if score > MIN_FITNESS_THRESHOLD]

        if not viable:
            scores = [score for _, score, _ in results]
            logger.warning(
                f"[{family}] All {len(results)} strategies failed hard eliminations "
                f"(scores: {[f'{s:.3f}' for s in scores]}) — family eliminated"
            )
            continue

        # Pick the best by fitness score
        champion = max(viable, key=lambda x: x[1])
        strategy, score, diagnostics = champion
        champions[family] = champion

        logger.info(
            f"[{family}] Champion: {strategy.name} "
            f"(fitness={score:.3f}, "
            f"beat {len(viable) - 1} other viable strategies)"
        )

    logger.info(
        f"Niche selection complete: {len(champions)}/{len(all_results)} families survived"
    )

    return champions


__all__ = ["select_champions"]


# ── Smoke Test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print("[SMOKE TEST] niche_selector.py\n")

    # Create dummy strategy objects with known scores
    class _FakeStrategy:
        def __init__(self, name: str, family: str):
            self.name = name
            self.family = family
            self.description = "fake"

    # Simulate results from run_all_specialists
    all_results = {
        "trend": [
            (_FakeStrategy("trend_ema", "trend"), 2.84, "diag_placeholder"),
            (_FakeStrategy("trend_macd", "trend"), 1.50, "diag_placeholder"),
            (_FakeStrategy("trend_hma", "trend"), 3.10, "diag_placeholder"),
        ],
        "momentum": [
            (_FakeStrategy("mom_rsi", "momentum"), 1.92, "diag_placeholder"),
            (_FakeStrategy("mom_cci", "momentum"), 0.80, "diag_placeholder"),
        ],
        "volatility": [
            # All failed hard eliminations
            (_FakeStrategy("vol_bb", "volatility"), -999.0, "diag_placeholder"),
            (_FakeStrategy("vol_kc", "volatility"), -999.0, "diag_placeholder"),
        ],
        "volume": [
            (_FakeStrategy("vol_vwap", "volume"), 0.45, "diag_placeholder"),
        ],
    }

    champions = select_champions(all_results)

    # ── Verify results ────────────────────────────────────────────────────
    print()

    # Check trend: best is trend_hma at 3.10
    assert "trend" in champions
    assert champions["trend"][0].name == "trend_hma"
    assert champions["trend"][1] == 3.10
    print(f"  trend: champion={champions['trend'][0].name}, "
          f"score={champions['trend'][1]:.2f} (correct)")

    # Check momentum: best is mom_rsi at 1.92
    assert "momentum" in champions
    assert champions["momentum"][0].name == "mom_rsi"
    assert champions["momentum"][1] == 1.92
    print(f"  momentum: champion={champions['momentum'][0].name}, "
          f"score={champions['momentum'][1]:.2f} (correct)")

    # Check volatility: eliminated (all -999)
    assert "volatility" not in champions
    print(f"  volatility: ELIMINATED (correct)")

    # Check volume: vol_vwap at 0.45
    assert "volume" in champions
    assert champions["volume"][0].name == "vol_vwap"
    assert champions["volume"][1] == 0.45
    print(f"  volume: champion={champions['volume'][0].name}, "
          f"score={champions['volume'][1]:.2f} (correct)")

    assert len(champions) == 3
    print(f"\n  Champions selected: {len(champions)}/4 families")

    # ── Edge case: all families eliminated ────────────────────────────────
    empty_results = {
        "trend": [(_FakeStrategy("t", "trend"), -999.0, "d")],
        "momentum": [],
        "volatility": [(_FakeStrategy("v", "volatility"), -999.0, "d")],
        "volume": [],
    }
    empty_champions = select_champions(empty_results)
    assert len(empty_champions) == 0
    print(f"  Edge case (all eliminated): {len(empty_champions)} champions (correct)")

    # ── Edge case: empty input ────────────────────────────────────────────
    none_champions = select_champions({})
    assert len(none_champions) == 0
    print(f"  Edge case (empty input): {len(none_champions)} champions (correct)")

    print("\nPASS")
