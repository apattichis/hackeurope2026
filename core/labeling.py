from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

try:
    from numba import njit

    _NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - fallback path when numba is not installed
    _NUMBA_AVAILABLE = False
    njit = None  # type: ignore[assignment]


_EPS = 1e-12


def _resolve_col(df: pd.DataFrame, preferred: str, fallback: str) -> str:
    if preferred in df.columns:
        return preferred
    if fallback in df.columns:
        return fallback
    raise KeyError(f"Missing required column: '{preferred}' (or fallback '{fallback}')")


def _atr_series(
    df: pd.DataFrame,
    high_col: str,
    low_col: str,
    close_col: str,
    window: int,
) -> pd.Series:
    prev_close = df[close_col].shift(1)
    tr = pd.concat(
        [
            df[high_col] - df[low_col],
            (df[high_col] - prev_close).abs(),
            (df[low_col] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window=window, min_periods=window).mean()


def _tbm_core_py(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    atr_val: np.ndarray,
    win: float,
    loss: float,
    time_horizon: int,
    stop_first: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = close.shape[0]
    
    # Pre-allocate Oracle label
    labels = np.zeros(n, dtype=np.float64)  
    
    # Pre-allocate completely independent Long and Short timelines
    l_pnl = np.full(n, np.nan, dtype=np.float64)
    l_exit = np.full(n, -1, dtype=np.int64)
    l_dur = np.full(n, -1, dtype=np.int32)
    
    s_pnl = np.full(n, np.nan, dtype=np.float64)
    s_exit = np.full(n, -1, dtype=np.int64)
    s_dur = np.full(n, -1, dtype=np.int32)

    for idx in range(n):
        atr_now = atr_val[idx]
        entry = close[idx]
        max_forward = min(time_horizon, n - 1 - idx)

        if max_forward <= 0 or np.isnan(atr_now) or np.isnan(entry) or abs(entry) < _EPS:
            labels[idx] = np.nan
            continue

        long_tp_price = entry + win * atr_now
        long_sl_price = entry - loss * atr_now
        short_tp_price = entry - win * atr_now
        short_sl_price = entry + loss * atr_now

        # Tracking statuses: 0=Pending, 1=Win, -1=Loss, 2=Whipsaw
        long_status = 0
        short_status = 0
        cur_l_dur = max_forward
        cur_s_dur = max_forward

        for i in range(1, max_forward + 1):
            h = high[idx + i]
            l = low[idx + i]

            # Evaluate Long position
            if long_status == 0:
                hit_l_tp = h >= long_tp_price
                hit_l_sl = l <= long_sl_price
                if hit_l_tp and hit_l_sl:
                    long_status = 2  
                    cur_l_dur = i
                elif hit_l_tp:
                    long_status = 1
                    cur_l_dur = i
                elif hit_l_sl:
                    long_status = -1
                    cur_l_dur = i

            # Evaluate Short position
            if short_status == 0:
                hit_s_tp = l <= short_tp_price
                hit_s_sl = h >= short_sl_price
                if hit_s_tp and hit_s_sl:
                    short_status = 2  
                    cur_s_dur = i
                elif hit_s_tp:
                    short_status = 1
                    cur_s_dur = i
                elif hit_s_sl:
                    short_status = -1
                    cur_s_dur = i

            # Stop scanning only if BOTH directions are resolved
            if long_status != 0 and short_status != 0:
                break

        # 1. Resolve exact Long PnL (Worst-case execution applied on whipsaw)
        if long_status == 2 or long_status == -1:
            cur_l_pnl = (long_sl_price - entry) / entry
        elif long_status == 1:
            cur_l_pnl = (long_tp_price - entry) / entry
        else: # Timeout
            cur_l_pnl = (close[idx + max_forward] - entry) / entry

        # 2. Resolve exact Short PnL (Worst-case execution applied on whipsaw)
        if short_status == 2 or short_status == -1:
            cur_s_pnl = (entry - short_sl_price) / entry
        elif short_status == 1:
            cur_s_pnl = (entry - short_tp_price) / entry
        else: # Timeout
            cur_s_pnl = (entry - close[idx + max_forward]) / entry

        # 3. Resolve Oracle ML Label (Untradable Whipsaws become NaN)
        if long_status == 2 or short_status == 2:
            cur_label = np.nan
        elif long_status == 1:
            cur_label = 1.0
        elif short_status == 1:
            cur_label = -1.0
        else:
            cur_label = 0.0

        # Store outputs
        labels[idx] = cur_label
        
        l_pnl[idx] = cur_l_pnl
        l_exit[idx] = idx + cur_l_dur
        l_dur[idx] = cur_l_dur
        
        s_pnl[idx] = cur_s_pnl
        s_exit[idx] = idx + cur_s_dur
        s_dur[idx] = cur_s_dur

    return labels, l_pnl, l_exit, l_dur, s_pnl, s_exit, s_dur


if _NUMBA_AVAILABLE:

    @njit(cache=True)  # type: ignore[misc]
    def _tbm_core_nb(
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        atr_val: np.ndarray,
        win: float,
        loss: float,
        time_horizon: int,
        stop_first: bool,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n = close.shape[0]
        
        labels = np.zeros(n, dtype=np.float64)
        l_pnl = np.empty(n, dtype=np.float64)
        l_exit = np.empty(n, dtype=np.int64)
        l_dur = np.empty(n, dtype=np.int32)
        s_pnl = np.empty(n, dtype=np.float64)
        s_exit = np.empty(n, dtype=np.int64)
        s_dur = np.empty(n, dtype=np.int32)

        for k in range(n):
            l_pnl[k] = np.nan
            l_exit[k] = -1
            l_dur[k] = -1
            s_pnl[k] = np.nan
            s_exit[k] = -1
            s_dur[k] = -1

        for idx in range(n):
            atr_now = atr_val[idx]
            entry = close[idx]
            rem = n - 1 - idx
            max_forward = time_horizon if time_horizon < rem else rem

            if max_forward <= 0:
                labels[idx] = np.nan
                continue
            if np.isnan(atr_now) or np.isnan(entry) or np.abs(entry) < _EPS:
                labels[idx] = np.nan
                continue

            long_tp_price = entry + win * atr_now
            long_sl_price = entry - loss * atr_now
            short_tp_price = entry - win * atr_now
            short_sl_price = entry + loss * atr_now

            long_status = 0
            short_status = 0
            cur_l_dur = max_forward
            cur_s_dur = max_forward

            for i in range(1, max_forward + 1):
                h = high[idx + i]
                l = low[idx + i]

                if long_status == 0:
                    hit_l_tp = h >= long_tp_price
                    hit_l_sl = l <= long_sl_price
                    if hit_l_tp and hit_l_sl:
                        long_status = 2
                        cur_l_dur = i
                    elif hit_l_tp:
                        long_status = 1
                        cur_l_dur = i
                    elif hit_l_sl:
                        long_status = -1
                        cur_l_dur = i

                if short_status == 0:
                    hit_s_tp = l <= short_tp_price
                    hit_s_sl = h >= short_sl_price
                    if hit_s_tp and hit_s_sl:
                        short_status = 2
                        cur_s_dur = i
                    elif hit_s_tp:
                        short_status = 1
                        cur_s_dur = i
                    elif hit_s_sl:
                        short_status = -1
                        cur_s_dur = i

                if long_status != 0 and short_status != 0:
                    break

            if long_status == 2 or long_status == -1:
                cur_l_pnl = (long_sl_price - entry) / entry
            elif long_status == 1:
                cur_l_pnl = (long_tp_price - entry) / entry
            else:
                cur_l_pnl = (close[idx + max_forward] - entry) / entry
                
            if short_status == 2 or short_status == -1:
                cur_s_pnl = (entry - short_sl_price) / entry
            elif short_status == 1:
                cur_s_pnl = (entry - short_tp_price) / entry
            else:
                cur_s_pnl = (entry - close[idx + max_forward]) / entry

            if long_status == 2 or short_status == 2:
                cur_label = np.nan
            elif long_status == 1:
                cur_label = 1.0
            elif short_status == 1:
                cur_label = -1.0
            else:
                cur_label = 0.0

            labels[idx] = cur_label
            
            l_pnl[idx] = cur_l_pnl
            l_exit[idx] = idx + cur_l_dur
            l_dur[idx] = cur_l_dur
            
            s_pnl[idx] = cur_s_pnl
            s_exit[idx] = idx + cur_s_dur
            s_dur[idx] = cur_s_dur

        return labels, l_pnl, l_exit, l_dur, s_pnl, s_exit, s_dur


def apply_triple_barrier(
    df: pd.DataFrame,
    win: float,
    loss: float,
    time_horizon: int,
    atr_window: int,
    tie_break: str = "stop_first",  
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
) -> pd.DataFrame:
    if tie_break not in {"stop_first", "profit_first"}:
        raise ValueError("tie_break must be one of {'stop_first', 'profit_first'}")
    if time_horizon <= 0:
        raise ValueError("time_horizon must be > 0")
    if atr_window <= 0:
        raise ValueError("atr_window must be > 0")

    out = df.copy()
    high_use = _resolve_col(out, high_col, "High")
    low_use = _resolve_col(out, low_col, "Low")
    close_use = _resolve_col(out, close_col, "Close")

    atr_col = f"ATR_{atr_window}"
    out[atr_col] = _atr_series(
        out,
        high_col=high_use,
        low_col=low_use,
        close_col=close_use,
        window=atr_window,
    )

    close_arr = out[close_use].to_numpy(dtype=np.float64, copy=False)
    high_arr = out[high_use].to_numpy(dtype=np.float64, copy=False)
    low_arr = out[low_use].to_numpy(dtype=np.float64, copy=False)
    atr_arr = out[atr_col].to_numpy(dtype=np.float64, copy=False)
    stop_first = tie_break == "stop_first"

    if _NUMBA_AVAILABLE:
        labels, l_pnl, l_exit, l_dur, s_pnl, s_exit, s_dur = _tbm_core_nb(  # type: ignore[misc]
            close_arr, high_arr, low_arr, atr_arr,
            float(win), float(loss), int(time_horizon), stop_first
        )
    else:
        labels, l_pnl, l_exit, l_dur, s_pnl, s_exit, s_dur = _tbm_core_py(
            close_arr, high_arr, low_arr, atr_arr,
            float(win), float(loss), int(time_horizon), stop_first
        )

    # Output Oracle Target
    out["tbm_label"] = labels.astype(np.float64)
    
    # Output Fully Resolved Long Path
    out["tbm_long_pnl"] = l_pnl.astype(np.float64)
    out["tbm_long_exit_idx"] = l_exit.astype(np.int64)
    out["tbm_long_duration"] = l_dur.astype(np.int32)
    
    # Output Fully Resolved Short Path
    out["tbm_short_pnl"] = s_pnl.astype(np.float64)
    out["tbm_short_exit_idx"] = s_exit.astype(np.int64)
    out["tbm_short_duration"] = s_dur.astype(np.int32)
    
    return out


__all__ = ["apply_triple_barrier"]