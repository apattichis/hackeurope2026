"""
fitness.py — Council of Alphas
Composite fitness scoring for all strategies (champions and hybrids).

Formula: Score = Global_Sharpe * ln(N) * Coverage

Where Coverage is TRADE-WEIGHTED:
  coverage = sum(trade_count for profitable 3D sufficient_evidence buckets)
           / sum(trade_count for all 3D sufficient_evidence buckets)

Hard eliminations return -999.0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import MIN_TRADES_SUFFICIENT_EVIDENCE


def compute_fitness(diagnostics_df: pd.DataFrame) -> float:
    """
    Compute composite fitness score from DiagnosticsEngine output.

    Parameters
    ----------
    diagnostics_df : pd.DataFrame
        Full output from DiagnosticsEngine.compute() — all 60 rows.

    Returns
    -------
    float
        Fitness score. Returns -999.0 if hard elimination conditions are met.

    Hard Elimination Conditions (return -999.0):
    - GLOBAL row has sufficient_evidence == False
    - GLOBAL Sharpe is NaN
    - No valid 3D buckets with sufficient_evidence == True
    """

    # ── 1. Get GLOBAL row ────────────────────────────────────────────────────
    global_rows = diagnostics_df[diagnostics_df["granularity"] == "GLOBAL"]
    if len(global_rows) == 0:
        return -999.0

    global_row = global_rows.iloc[0]

    # ── 2. Hard eliminations ─────────────────────────────────────────────────
    if not global_row["sufficient_evidence"]:
        return -999.0

    if pd.isna(global_row["sharpe"]):
        return -999.0

    # ── 3. Get 3D sufficient_evidence buckets ─────────────────────────────────
    active = diagnostics_df[
        (diagnostics_df["granularity"] == "3D") &
        (diagnostics_df["sufficient_evidence"] == True)
    ].copy()

    if len(active) == 0:
        return -999.0

    # ── 4. Trade-weighted coverage ────────────────────────────────────────────
    total_active_trades = active["trade_count"].sum()
    if total_active_trades == 0:
        return -999.0

    profitable = active[active["sharpe"] > 0]
    profitable_trades = profitable["trade_count"].sum()

    coverage = float(profitable_trades) / float(total_active_trades)

    # ── 5. Final score ────────────────────────────────────────────────────────
    global_sharpe = float(global_row["sharpe"])
    N = int(global_row["trade_count"])

    if N < 2:
        return -999.0

    score = global_sharpe * np.log(N) * coverage
    return float(score)


def is_unviable(diagnostics_df: pd.DataFrame) -> tuple[bool, str]:
    """
    Check UNVIABLE conditions before running the Critic.
    Returns (is_unviable: bool, reason: str).

    UNVIABLE if any of:
    - GLOBAL Sharpe < -0.5 (with sufficient evidence)
    - Zero 3D buckets with sufficient_evidence=True AND Sharpe > 0
    - GLOBAL max_consecutive_losses > 20
    """
    global_rows = diagnostics_df[diagnostics_df["granularity"] == "GLOBAL"]
    if len(global_rows) == 0:
        return True, "No GLOBAL row found in diagnostics"

    global_row = global_rows.iloc[0]

    # Check 1: deeply negative Sharpe
    if (
        global_row["sufficient_evidence"]
        and not pd.isna(global_row["sharpe"])
        and global_row["sharpe"] < -0.5
    ):
        return True, f"GLOBAL sharpe={global_row['sharpe']:.3f} < -0.5 threshold"

    # Check 2: no profitable 3D buckets
    active_3d = diagnostics_df[
        (diagnostics_df["granularity"] == "3D") &
        (diagnostics_df["sufficient_evidence"] == True)
    ]
    profitable_3d = active_3d[active_3d["sharpe"] > 0]
    if len(active_3d) > 0 and len(profitable_3d) == 0:
        return True, "Zero 3D buckets with sufficient_evidence=True and Sharpe > 0"

    # Check 3: catastrophic loss streak
    if (
        global_row["sufficient_evidence"]
        and global_row["max_consecutive_losses"] > 20
    ):
        return (
            True,
            f"GLOBAL max_consecutive_losses={global_row['max_consecutive_losses']} > 20",
        )

    return False, ""


__all__ = ["compute_fitness", "is_unviable"]
