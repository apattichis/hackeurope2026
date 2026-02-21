"""
hybrid_builder.py — Council of Alphas
Pure Python. Zero LLM calls.

Builds 3 hybrid strategies from niche-selected champions:
  1. Regime Router   — best champion per 3D regime bucket
  2. Consensus Gate  — fire when N-1 champions agree on direction
  3. Weighted Combo  — fitness-weighted signal sum, np.sign for direction

All produce inline Strategy subclasses with combination parameters
as CLASS ATTRIBUTES (modifiable by the Scientist's Refiner).

Public API:
    HybridBuilder     — build all 3 hybrids
    inject_champions() — re-inject champion refs after Refiner modifies code
"""

from __future__ import annotations

import sys
import logging
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _p in [str(_PROJECT_ROOT / "core"), str(_PROJECT_ROOT / "pipeline")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pandas as pd

from config import SESSIONS, TREND_REGIMES, VOL_REGIMES
from strategy_base import Strategy

logger = logging.getLogger("council.hybrid")


# ── Champion Injection (used by Scientist after Refiner modifies code) ────────

def inject_champions(hybrid: Strategy, champion_strategies: dict) -> None:
    """
    Inject champion strategy references into a hybrid after instantiation.
    Must be called before generate_signals() on any hybrid.

    Parameters
    ----------
    hybrid : Strategy
        A hybrid Strategy instance (from build or from exec of refined code)
    champion_strategies : dict[str, Strategy]
        {family: strategy_instance} for all surviving champions
    """
    hybrid._champion_strategies = champion_strategies


# ── HybridBuilder ─────────────────────────────────────────────────────────────

class HybridBuilder:
    """
    Deterministic builder for hybrid strategies.
    Takes champion strategies + diagnostics, produces 3 hybrid Strategy instances.

    Pure Python — no LLM calls.
    """

    def __init__(
        self,
        champions: dict[str, tuple],
        state_matrix: pd.DataFrame,
    ) -> None:
        """
        Parameters
        ----------
        champions : dict[str, tuple]
            {family: (strategy, score, diagnostics)} from niche selection
        state_matrix : pd.DataFrame
            Full state matrix (read-only, used only for regime column reference)
        """
        self.champions = champions
        self.state_matrix = state_matrix

        # Extract for convenience
        self._strategies = {f: champ[0] for f, champ in champions.items()}
        self._scores = {f: champ[1] for f, champ in champions.items()}
        self._diagnostics = {f: champ[2] for f, champ in champions.items()}

    def build_all(self) -> list[Strategy]:
        """
        Build all 3 hybrids.

        Returns list of Strategy instances. Failed builds are logged and skipped.
        """
        builders = [
            ("regime_router", self.build_regime_router),
            ("consensus_gate", self.build_consensus_gate),
            ("weighted_combination", self.build_weighted_combination),
        ]

        hybrids: list[Strategy] = []
        for name, builder_fn in builders:
            try:
                hybrid = builder_fn()
                hybrids.append(hybrid)
                logger.info(f"Built {name}: OK")
            except Exception as e:
                logger.error(f"Failed to build {name}: {type(e).__name__}: {e}")

        return hybrids

    # ── Routing Table Construction ────────────────────────────────────────

    def _build_routing_table(self) -> tuple[dict[tuple, str], str]:
        """
        For each (session, trend, vol) combo, find which champion has the
        best Sharpe with sufficient_evidence=True in that 3D bucket.

        Fallback: champion with best GLOBAL Sharpe.

        Returns
        -------
        (routing_table, fallback_family)
            routing_table: {(session, trend, vol): family_name}
            fallback_family: family with best global Sharpe
        """
        # 1. Find fallback: champion with best GLOBAL Sharpe
        best_global_family = None
        best_global_sharpe = -float("inf")
        for family, diag in self._diagnostics.items():
            global_rows = diag[diag["granularity"] == "GLOBAL"]
            if len(global_rows) == 0:
                continue
            sharpe = global_rows.iloc[0]["sharpe"]
            if not pd.isna(sharpe) and sharpe > best_global_sharpe:
                best_global_sharpe = sharpe
                best_global_family = family

        # If no champion has a valid global Sharpe, use the highest fitness
        if best_global_family is None:
            best_global_family = max(self._scores, key=self._scores.get)

        # 2. Build routing table
        routing: dict[tuple, str] = {}
        for session in SESSIONS:
            for trend in TREND_REGIMES:
                for vol in VOL_REGIMES:
                    best_family = None
                    best_sharpe = -float("inf")

                    for family, diag in self._diagnostics.items():
                        bucket = diag[
                            (diag["granularity"] == "3D")
                            & (diag["session"] == session)
                            & (diag["trend_regime"] == trend)
                            & (diag["vol_regime"] == vol)
                            & (diag["sufficient_evidence"] == True)
                        ]
                        if len(bucket) > 0:
                            sharpe = bucket.iloc[0]["sharpe"]
                            if not pd.isna(sharpe) and sharpe > best_sharpe:
                                best_sharpe = sharpe
                                best_family = family

                    # Fallback if no champion has sufficient evidence here
                    routing[(session, trend, vol)] = (
                        best_family if best_family is not None
                        else best_global_family
                    )

        return routing, best_global_family

    # ── Hybrid 1: Regime Router ───────────────────────────────────────────

    def build_regime_router(self) -> Strategy:
        """
        For each of 24 regime buckets, assign the champion with the
        best sufficient-evidence Sharpe. Fallback to best global Sharpe.
        """
        routing_table, fallback = self._build_routing_table()
        champion_strategies = dict(self._strategies)

        class RegimeRouterHybrid(Strategy):
            name = "regime_router"
            family = "hybrid"
            description = "Routes to best champion per regime bucket based on 3D Sharpe"

            # ── Combination parameters (modifiable by Refiner) ────────
            ROUTING = dict(routing_table)
            FALLBACK = fallback

            def generate_signals(self, data):
                # ── Compute all champion signals ──────────────────────
                champ_sigs = {}
                for fam, strat in self._champion_strategies.items():
                    champ_sigs[fam] = strat.generate_signals(data)

                # ── Route per regime bucket (vectorized) ──────────────
                signals = pd.Series(0, index=data.index)
                for key, assigned in self.ROUTING.items():
                    if assigned not in champ_sigs:
                        assigned = self.FALLBACK
                    if assigned not in champ_sigs:
                        continue
                    session, trend, vol = key
                    mask = (
                        (data["session"] == session)
                        & (data["trend_regime"] == trend)
                        & (data["vol_regime"] == vol)
                    )
                    signals[mask] = champ_sigs[assigned][mask]

                return signals

        instance = RegimeRouterHybrid()
        instance._champion_strategies = champion_strategies
        instance._source_code = self._router_source(routing_table, fallback)

        logger.info(
            f"Regime Router: {len(routing_table)} routing entries, "
            f"fallback={fallback}"
        )
        return instance

    # ── Hybrid 2: Consensus Gate ──────────────────────────────────────────

    def build_consensus_gate(self) -> Strategy:
        """
        All champions vote. Fire when threshold champions agree on direction.
        Threshold adapts: 4->3, 3->2, 2->2 (unanimous).
        """
        champion_strategies = dict(self._strategies)
        n = len(champion_strategies)
        threshold = max(n - 1, 2)  # 4->3, 3->2, 2->2

        class ConsensusGateHybrid(Strategy):
            name = "consensus_gate"
            family = "hybrid"
            description = f"Fires when {threshold}/{n} champions agree on direction"

            # ── Combination parameters (modifiable by Refiner) ────────
            THRESHOLD = threshold

            def generate_signals(self, data):
                # ── Compute all champion signals ──────────────────────
                champ_sigs = {}
                for fam, strat in self._champion_strategies.items():
                    champ_sigs[fam] = strat.generate_signals(data)

                # ── Sum votes ─────────────────────────────────────────
                votes = pd.Series(0.0, index=data.index)
                for sig in champ_sigs.values():
                    votes += sig

                # ── Apply threshold ───────────────────────────────────
                signals = pd.Series(0, index=data.index)
                signals[votes >= self.THRESHOLD] = 1
                signals[votes <= -self.THRESHOLD] = -1

                return signals

        instance = ConsensusGateHybrid()
        instance._champion_strategies = champion_strategies
        instance._source_code = self._consensus_source(threshold, n)

        logger.info(f"Consensus Gate: threshold={threshold}/{n} champions")
        return instance

    # ── Hybrid 3: Weighted Combination ────────────────────────────────────

    def build_weighted_combination(self) -> Strategy:
        """
        Champions weighted by their fitness score.
        Weighted sum's sign determines direction.
        """
        champion_strategies = dict(self._strategies)
        weights = dict(self._scores)

        class WeightedCombinationHybrid(Strategy):
            name = "weighted_combination"
            family = "hybrid"
            description = "Fitness-weighted signal sum, np.sign for direction"

            # ── Combination parameters (modifiable by Refiner) ────────
            WEIGHTS = dict(weights)

            def generate_signals(self, data):
                # ── Compute all champion signals ──────────────────────
                champ_sigs = {}
                for fam, strat in self._champion_strategies.items():
                    champ_sigs[fam] = strat.generate_signals(data)

                # ── Weighted sum ──────────────────────────────────────
                weighted_sum = pd.Series(0.0, index=data.index)
                for fam, sig in champ_sigs.items():
                    weighted_sum += self.WEIGHTS[fam] * sig

                # ── Sign → direction ──────────────────────────────────
                signals = pd.Series(
                    np.sign(weighted_sum).astype(int), index=data.index
                )
                return signals

        instance = WeightedCombinationHybrid()
        instance._champion_strategies = champion_strategies
        instance._source_code = self._weighted_source(weights)

        logger.info(
            f"Weighted Combination: weights="
            f"{{{', '.join(f'{f}: {w:.3f}' for f, w in weights.items())}}}"
        )
        return instance

    # ── Source Code Generation (for Critic / Refiner) ─────────────────────

    def _router_source(self, routing: dict, fallback: str) -> str:
        routing_lines = []
        for key in sorted(routing.keys()):
            fam = routing[key]
            routing_lines.append(f"        {key!r}: {fam!r},")

        return (
            'class RegimeRouterHybrid(Strategy):\n'
            '    name = "regime_router"\n'
            '    family = "hybrid"\n'
            '    description = "Routes to best champion per regime bucket"\n'
            '\n'
            '    ROUTING = {\n'
            + "\n".join(routing_lines) + "\n"
            '    }\n'
            f'    FALLBACK = {fallback!r}\n'
            '\n'
            '    def generate_signals(self, data):\n'
            '        champ_sigs = {}\n'
            '        for fam, strat in self._champion_strategies.items():\n'
            '            champ_sigs[fam] = strat.generate_signals(data)\n'
            '        signals = pd.Series(0, index=data.index)\n'
            '        for key, assigned in self.ROUTING.items():\n'
            '            if assigned not in champ_sigs:\n'
            '                assigned = self.FALLBACK\n'
            '            if assigned not in champ_sigs:\n'
            '                continue\n'
            '            session, trend, vol = key\n'
            '            mask = (\n'
            '                (data["session"] == session)\n'
            '                & (data["trend_regime"] == trend)\n'
            '                & (data["vol_regime"] == vol)\n'
            '            )\n'
            '            signals[mask] = champ_sigs[assigned][mask]\n'
            '        return signals\n'
        )

    def _consensus_source(self, threshold: int, n: int) -> str:
        return (
            'class ConsensusGateHybrid(Strategy):\n'
            '    name = "consensus_gate"\n'
            '    family = "hybrid"\n'
            f'    description = "Fires when {threshold}/{n} champions agree"\n'
            '\n'
            f'    THRESHOLD = {threshold}\n'
            '\n'
            '    def generate_signals(self, data):\n'
            '        champ_sigs = {}\n'
            '        for fam, strat in self._champion_strategies.items():\n'
            '            champ_sigs[fam] = strat.generate_signals(data)\n'
            '        votes = pd.Series(0.0, index=data.index)\n'
            '        for sig in champ_sigs.values():\n'
            '            votes += sig\n'
            '        signals = pd.Series(0, index=data.index)\n'
            f'        signals[votes >= self.THRESHOLD] = 1\n'
            f'        signals[votes <= -self.THRESHOLD] = -1\n'
            '        return signals\n'
        )

    def _weighted_source(self, weights: dict) -> str:
        weight_repr = "{\n"
        for f, w in weights.items():
            weight_repr += f"        {f!r}: {w:.4f},\n"
        weight_repr += "    }"

        return (
            'class WeightedCombinationHybrid(Strategy):\n'
            '    name = "weighted_combination"\n'
            '    family = "hybrid"\n'
            '    description = "Fitness-weighted signal sum, np.sign for direction"\n'
            '\n'
            f'    WEIGHTS = {weight_repr}\n'
            '\n'
            '    def generate_signals(self, data):\n'
            '        champ_sigs = {}\n'
            '        for fam, strat in self._champion_strategies.items():\n'
            '            champ_sigs[fam] = strat.generate_signals(data)\n'
            '        weighted_sum = pd.Series(0.0, index=data.index)\n'
            '        for fam, sig in champ_sigs.items():\n'
            '            weighted_sum += self.WEIGHTS[fam] * sig\n'
            '        signals = pd.Series(np.sign(weighted_sum).astype(int), index=data.index)\n'
            '        return signals\n'
        )


__all__ = ["HybridBuilder", "inject_champions"]


# ── Smoke Test ────────────────────────────────────────────────────────────────

def _make_fake_diagnostics(
    family: str,
    global_sharpe: float = 0.5,
    n_trades: int = 200,
) -> pd.DataFrame:
    """
    Build a fake diagnostics table with GLOBAL + 3D rows.
    Each family gets different per-bucket Sharpe values so the
    routing table assigns different champions to different buckets.
    """
    rows = []

    # GLOBAL row
    rows.append({
        "granularity": "GLOBAL",
        "session": "ALL",
        "trend_regime": "ALL",
        "vol_regime": "ALL",
        "trade_count": n_trades,
        "win_rate": 0.55,
        "sharpe": global_sharpe,
        "max_consecutive_losses": 5,
        "sufficient_evidence": True,
    })

    # 3D rows for every regime combo
    for session in SESSIONS:
        for trend in TREND_REGIMES:
            for vol in VOL_REGIMES:
                # Deterministic but different per family+bucket
                seed_val = hash((family, session, trend, vol)) % 100
                bucket_sharpe = global_sharpe + (seed_val - 50) / 100.0

                rows.append({
                    "granularity": "3D",
                    "session": session,
                    "trend_regime": trend,
                    "vol_regime": vol,
                    "trade_count": 40,
                    "win_rate": 0.55,
                    "sharpe": bucket_sharpe,
                    "max_consecutive_losses": 3,
                    "sufficient_evidence": True,
                })

    return pd.DataFrame(rows)


class _ConstantStrategy(Strategy):
    """Test champion that returns a fixed signal for every candle."""

    def __init__(self, name_: str, family_: str, signal_value: int):
        self.name = name_
        self.family = family_
        self.description = f"Always returns {signal_value}"
        self._signal_value = signal_value

    def generate_signals(self, data):
        return pd.Series(self._signal_value, index=data.index)


class _PatternStrategy(Strategy):
    """Test champion: first half long, second half short."""

    def __init__(self, name_: str, family_: str):
        self.name = name_
        self.family = family_
        self.description = "First half long, second half short"

    def generate_signals(self, data):
        n = len(data)
        signals = pd.Series(0, index=data.index)
        signals.iloc[: n // 2] = 1
        signals.iloc[n // 2:] = -1
        return signals


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print("[SMOKE TEST] hybrid_builder.py\n")

    # ── Build dummy state matrix ──────────────────────────────────────────
    n = 1000
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=n, freq="15min", tz="utc")

    state_matrix = pd.DataFrame({
        "open": np.random.uniform(90, 110, n),
        "high": np.random.uniform(95, 115, n),
        "low": np.random.uniform(85, 105, n),
        "close": np.random.uniform(90, 110, n),
        "volume": np.random.uniform(1000, 5000, n),
        "session": np.tile(SESSIONS, n // len(SESSIONS) + 1)[:n],
        "trend_regime": np.tile(TREND_REGIMES, n // len(TREND_REGIMES) + 1)[:n],
        "vol_regime": np.tile(VOL_REGIMES, n // len(VOL_REGIMES) + 1)[:n],
    }, index=dates)
    state_matrix.index.name = "open_time"

    print(f"  State matrix: {n} rows")

    # ── Create 4 champion strategies ──────────────────────────────────────
    trend_champ = _ConstantStrategy("trend_ema", "trend", 1)
    mom_champ = _ConstantStrategy("mom_rsi", "momentum", -1)
    vol_champ = _PatternStrategy("vol_bb", "volatility")
    volume_champ = _ConstantStrategy("vol_vwap", "volume", 1)

    champions = {
        "trend": (
            trend_champ, 3.10,
            _make_fake_diagnostics("trend", global_sharpe=0.60),
        ),
        "momentum": (
            mom_champ, 1.92,
            _make_fake_diagnostics("momentum", global_sharpe=0.45),
        ),
        "volatility": (
            vol_champ, 0.85,
            _make_fake_diagnostics("volatility", global_sharpe=0.30),
        ),
        "volume": (
            volume_champ, 0.45,
            _make_fake_diagnostics("volume", global_sharpe=0.20),
        ),
    }

    print(f"  Champions: {list(champions.keys())}")

    # ── Build all 3 hybrids ───────────────────────────────────────────────
    builder = HybridBuilder(champions, state_matrix)
    hybrids = builder.build_all()

    assert len(hybrids) == 3, f"Expected 3 hybrids, got {len(hybrids)}"
    print(f"\n  Built {len(hybrids)} hybrids:")
    for h in hybrids:
        print(f"    - {h.name} (family={h.family})")

    # ── Test 1: All hybrids produce valid signals ─────────────────────────
    print("\n=== Signal generation ===")
    for h in hybrids:
        signals = h.generate_signals(state_matrix)
        assert isinstance(signals, pd.Series), f"{h.name}: not a Series"
        assert len(signals) == n, f"{h.name}: wrong length"
        unique = set(signals.unique())
        assert unique <= {-1, 0, 1}, f"{h.name}: invalid values {unique}"
        longs = int((signals == 1).sum())
        shorts = int((signals == -1).sum())
        flat = int((signals == 0).sum())
        print(f"  {h.name}: longs={longs}, shorts={shorts}, flat={flat}")

    # ── Test 2: Consensus Gate logic ──────────────────────────────────────
    print("\n=== Consensus Gate verification ===")
    consensus = hybrids[1]
    assert consensus.name == "consensus_gate"
    threshold = consensus.THRESHOLD
    print(f"  Threshold: {threshold}/4")

    # Champions: trend=1, momentum=-1, volatility=pattern, volume=1
    # At row 0 (first half): votes = 1 + (-1) + 1 + 1 = 2
    # At row 500+ (second half): votes = 1 + (-1) + (-1) + 1 = 0
    # Neither reaches threshold=3, so all signals should be 0
    sigs = consensus.generate_signals(state_matrix)
    # Actually trend=1, mom=-1, vol=first half 1/second half -1, volume=1
    # First half votes: 1 + (-1) + 1 + 1 = 2 < 3 → 0
    # Second half votes: 1 + (-1) + (-1) + 1 = 0 < 3 → 0
    assert (sigs == 0).all(), "Expected all flat (no 3/4 agreement)"
    print(f"  With mixed champions (max vote=2): all flat (correct, need {threshold})")

    # Now test with 3 agreeing champions
    champions_agree = {
        "trend": (
            _ConstantStrategy("t", "trend", 1), 3.0,
            _make_fake_diagnostics("trend"),
        ),
        "momentum": (
            _ConstantStrategy("m", "momentum", 1), 2.0,
            _make_fake_diagnostics("momentum"),
        ),
        "volatility": (
            _ConstantStrategy("v", "volatility", 1), 1.0,
            _make_fake_diagnostics("volatility"),
        ),
        "volume": (
            _ConstantStrategy("vol", "volume", -1), 0.5,
            _make_fake_diagnostics("volume"),
        ),
    }
    builder2 = HybridBuilder(champions_agree, state_matrix)
    consensus2 = builder2.build_consensus_gate()
    sigs2 = consensus2.generate_signals(state_matrix)
    # votes = 1 + 1 + 1 + (-1) = 2 < 3 → still flat
    assert (sigs2 == 0).all(), "Expected flat (votes=2 < threshold=3)"
    print(f"  With 3 long + 1 short (votes=2): all flat (correct)")

    # 4 agreeing → should fire
    champions_all_long = {
        f: (_ConstantStrategy(f, f, 1), 1.0, _make_fake_diagnostics(f))
        for f in ["trend", "momentum", "volatility", "volume"]
    }
    builder3 = HybridBuilder(champions_all_long, state_matrix)
    consensus3 = builder3.build_consensus_gate()
    sigs3 = consensus3.generate_signals(state_matrix)
    # votes = 1 + 1 + 1 + 1 = 4 >= 3 → long
    assert (sigs3 == 1).all(), "Expected all long (votes=4 >= 3)"
    print(f"  With 4 long (votes=4): all long (correct)")

    # ── Test 3: Weighted Combination logic ────────────────────────────────
    print("\n=== Weighted Combination verification ===")
    weighted = hybrids[2]
    assert weighted.name == "weighted_combination"
    print(f"  Weights: {weighted.WEIGHTS}")

    # trend(1)*3.10 + momentum(-1)*1.92 + volatility(pattern)*0.85 + volume(1)*0.45
    # First half: 3.10 - 1.92 + 0.85 + 0.45 = 2.48 → sign = +1
    # Second half: 3.10 - 1.92 - 0.85 + 0.45 = 0.78 → sign = +1
    wsigs = weighted.generate_signals(state_matrix)
    first_half = wsigs.iloc[:n // 2]
    second_half = wsigs.iloc[n // 2:]
    print(f"  First half (trend dominates): all long = {(first_half == 1).all()}")
    print(f"  Second half (trend still dominates): all long = {(second_half == 1).all()}")

    # ── Test 4: Regime Router has correct routing table ───────────────────
    print("\n=== Regime Router verification ===")
    router = hybrids[0]
    assert router.name == "regime_router"
    routing = router.ROUTING
    print(f"  Routing entries: {len(routing)}")
    assert len(routing) == 24, f"Expected 24 entries, got {len(routing)}"

    # Check that different families are assigned to different buckets
    assigned_families = set(routing.values())
    print(f"  Families used: {sorted(assigned_families)}")

    # Verify routing produces valid signals
    rsigs = router.generate_signals(state_matrix)
    print(f"  Signals: longs={int((rsigs == 1).sum())}, "
          f"shorts={int((rsigs == -1).sum())}, flat={int((rsigs == 0).sum())}")

    # ── Test 5: inject_champions works ────────────────────────────────────
    print("\n=== inject_champions() ===")
    # Simulate what Scientist does: create fresh instance of same type + inject
    CLS = type(consensus)
    fresh_consensus = CLS()
    inject_champions(fresh_consensus, {
        f: champ[0] for f, champ in champions.items()
    })
    fresh_sigs = fresh_consensus.generate_signals(state_matrix)
    assert len(fresh_sigs) == n
    print(f"  inject_champions + generate_signals: OK")

    # ── Test 6: Source code is readable ────────────────────────────────────
    print("\n=== Source code generation ===")
    for h in hybrids:
        code = h._source_code
        assert "class " in code, f"{h.name}: no class in source"
        assert "generate_signals" in code, f"{h.name}: no generate_signals"
        lines = code.strip().split("\n")
        print(f"  {h.name}: {len(lines)} lines")

    # ── Test 7: Degraded case — 2 champions ───────────────────────────────
    print("\n=== Degraded: 2 champions ===")
    champions_2 = {
        "trend": champions["trend"],
        "momentum": champions["momentum"],
    }
    builder_2 = HybridBuilder(champions_2, state_matrix)
    hybrids_2 = builder_2.build_all()
    assert len(hybrids_2) == 3
    consensus_2 = hybrids_2[1]
    assert consensus_2.THRESHOLD == 2, f"Expected threshold=2, got {consensus_2.THRESHOLD}"
    print(f"  Consensus threshold: {consensus_2.THRESHOLD}/2 (unanimous, correct)")
    print(f"  All 3 hybrids built with 2 champions: OK")

    # ── Test 8: Degraded case — 3 champions ───────────────────────────────
    print("\n=== Degraded: 3 champions ===")
    champions_3 = {k: v for k, v in list(champions.items())[:3]}
    builder_3 = HybridBuilder(champions_3, state_matrix)
    hybrids_3 = builder_3.build_all()
    consensus_3 = hybrids_3[1]
    assert consensus_3.THRESHOLD == 2, f"Expected threshold=2, got {consensus_3.THRESHOLD}"
    print(f"  Consensus threshold: {consensus_3.THRESHOLD}/3 (correct)")
    print(f"  All 3 hybrids built with 3 champions: OK")

    print("\nPASS")
