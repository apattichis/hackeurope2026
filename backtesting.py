from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

try:
    from numba import njit
    _NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover
    _NUMBA_AVAILABLE = False
    njit = None  # type: ignore[assignment]


def _tbm_lookup_py(
    signal: np.ndarray,
    l_pnl: np.ndarray,
    l_exit: np.ndarray,
    s_pnl: np.ndarray,
    s_exit: np.ndarray,
    fee: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(signal)
    
    # Pre-allocate arrays
    entry_indices = np.full(n, -1, dtype=np.int64)
    exit_indices = np.full(n, -1, dtype=np.int64)
    pnls = np.full(n, np.nan, dtype=np.float64)
    sides = np.full(n, 0, dtype=np.int8)

    trade_count = 0
    next_free = 0

    for i in range(n):
        if i < next_free:
            continue

        sig = signal[i]
        if sig == 0 or np.isnan(sig):
            continue

        if sig == 1:
            cur_exit = l_exit[i]
            cur_pnl = l_pnl[i]
        elif sig == -1:
            cur_exit = s_exit[i]
            cur_pnl = s_pnl[i]
        else:
            continue

        # Skip incomplete trades at the edge of the dataset
        if cur_exit < 0 or np.isnan(cur_pnl):
            continue

        entry_indices[trade_count] = i
        exit_indices[trade_count] = cur_exit
        pnls[trade_count] = cur_pnl - fee
        sides[trade_count] = sig
        
        trade_count += 1
        # Capital Lock: No overlaps allowed
        next_free = cur_exit + 1

    return (
        entry_indices[:trade_count],
        exit_indices[:trade_count],
        pnls[:trade_count],
        sides[:trade_count]
    )


if _NUMBA_AVAILABLE:
    @njit(cache=True)  # type: ignore[misc]
    def _tbm_lookup_nb(
        signal: np.ndarray,
        l_pnl: np.ndarray,
        l_exit: np.ndarray,
        s_pnl: np.ndarray,
        s_exit: np.ndarray,
        fee: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n = signal.shape[0]
        
        entry_indices = np.empty(n, dtype=np.int64)
        exit_indices = np.empty(n, dtype=np.int64)
        pnls = np.empty(n, dtype=np.float64)
        sides = np.empty(n, dtype=np.int8)

        trade_count = 0
        next_free = 0

        for i in range(n):
            if i < next_free:
                continue

            sig = signal[i]
            if sig == 0 or np.isnan(sig):
                continue

            if sig == 1:
                cur_exit = l_exit[i]
                cur_pnl = l_pnl[i]
            elif sig == -1:
                cur_exit = s_exit[i]
                cur_pnl = s_pnl[i]
            else:
                continue

            if cur_exit < 0 or np.isnan(cur_pnl):
                continue

            entry_indices[trade_count] = i
            exit_indices[trade_count] = cur_exit
            pnls[trade_count] = cur_pnl - fee
            sides[trade_count] = sig
            
            trade_count += 1
            next_free = cur_exit + 1

        return (
            entry_indices[:trade_count],
            exit_indices[:trade_count],
            pnls[:trade_count],
            sides[:trade_count]
        )


class VectorizedBacktester:
    """Consumes LLM strategy signals and dual-directional TBM outputs to create a trade log."""
    
    def __init__(self, fee: float = 0.0) -> None:
        self.fee = float(fee)

    def run(self, df: pd.DataFrame, signal_col: str) -> pd.DataFrame:
        """
        Executes the strategy signals and returns a Trade Log DataFrame ready for DiagnosticsEngine.
        """
        # 1. Strict Input Validation for Diagnostics
        req_regimes = ["session", "trend_regime", "vol_regime"]
        for col in req_regimes:
            if col not in df.columns:
                raise KeyError(f"Missing required regime column: {col}. Run StateMatrixBuilder first.")
        
        if signal_col not in df.columns:
            raise KeyError(f"Signal column '{signal_col}' not found in DataFrame.")

        # 2. Strict Input Validation for the pre-computed TBM timelines
        req_tbm = ["tbm_long_pnl", "tbm_long_exit_idx", "tbm_short_pnl", "tbm_short_exit_idx"]
        for col in req_tbm:
            if col not in df.columns:
                raise KeyError(f"Missing TBM column: {col}. Ensure the new labeling.py was applied.")

        # 3. Extract Numpy Arrays for speed
        signal_arr = df[signal_col].to_numpy(dtype=np.float64, copy=False)
        l_pnl_arr = df["tbm_long_pnl"].to_numpy(dtype=np.float64, copy=False)
        l_exit_arr = df["tbm_long_exit_idx"].to_numpy(dtype=np.int64, copy=False)
        s_pnl_arr = df["tbm_short_pnl"].to_numpy(dtype=np.float64, copy=False)
        s_exit_arr = df["tbm_short_exit_idx"].to_numpy(dtype=np.int64, copy=False)

        # 4. Run Vectorized Lookup
        if _NUMBA_AVAILABLE:
            entries, exits, pnls, sides = _tbm_lookup_nb( # type: ignore[misc]
                signal_arr, l_pnl_arr, l_exit_arr, s_pnl_arr, s_exit_arr, self.fee
            )
        else:
            entries, exits, pnls, sides = _tbm_lookup_py(
                signal_arr, l_pnl_arr, l_exit_arr, s_pnl_arr, s_exit_arr, self.fee
            )

        # 5. Build Diagnostics-Compliant Trade Log
        if len(entries) == 0:
            return pd.DataFrame(columns=[
                "entry_ts", "exit_ts", "entry_index", "exit_index", "duration",
                "side", "net_trade_return", "session", "trend_regime", "vol_regime"
            ])

        timestamps = df.index.to_series(name="ts").reset_index(drop=True)
        
        trades = pd.DataFrame({
            "entry_index": entries,
            "exit_index": exits,
            "side": sides,
            "net_trade_return": pnls,  # Alias matched exactly for DiagnosticsEngine
        })
        
        trades["duration"] = trades["exit_index"] - trades["entry_index"]
        trades["entry_ts"] = trades["entry_index"].map(timestamps)
        trades["exit_ts"] = trades["exit_index"].map(timestamps)
        
        # Map Market Regimes precisely from the entry candle
        trades["session"] = trades["entry_index"].map(df["session"].reset_index(drop=True))
        trades["trend_regime"] = trades["entry_index"].map(df["trend_regime"].reset_index(drop=True))
        trades["vol_regime"] = trades["entry_index"].map(df["vol_regime"].reset_index(drop=True))
        
        # Reorder columns for readability
        return trades[[
            "entry_ts", "exit_ts", "entry_index", "exit_index", "duration",
            "side", "net_trade_return", "session", "trend_regime", "vol_regime"
        ]]

__all__ = ["VectorizedBacktester"]