from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from config import ATR_WINDOW, BACKTEST_FEE, TBM_LOSS

try:
    from numba import njit
    _NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover
    _NUMBA_AVAILABLE = False
    njit = None  # type: ignore[assignment]


def _tbm_portfolio_lookup_py(
    signal: np.ndarray,
    l_pnl: np.ndarray,
    l_exit: np.ndarray,
    s_pnl: np.ndarray,
    s_exit: np.ndarray,
    close: np.ndarray,
    atr: np.ndarray,
    tbm_loss: float,
    fee: float,
    initial_capital: float,
    risk_per_trade: float,
    max_leverage: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(signal)
    
    # Pre-allocate arrays
    entry_indices = np.full(n, -1, dtype=np.int64)
    exit_indices = np.full(n, -1, dtype=np.int64)
    account_pnls = np.full(n, np.nan, dtype=np.float64)  # The % return on the ACCOUNT
    capital_curve = np.full(n, np.nan, dtype=np.float64) # The actual $ value
    sides = np.full(n, 0, dtype=np.int8)

    trade_count = 0
    next_free = 0
    current_capital = initial_capital

    for i in range(n):
        if i < next_free:
            continue

        sig = signal[i]
        if sig == 0 or np.isnan(sig):
            continue

        if sig == 1:
            cur_exit = l_exit[i]
            asset_pnl = l_pnl[i]
        elif sig == -1:
            cur_exit = s_exit[i]
            asset_pnl = s_pnl[i]
        else:
            continue

        # Skip incomplete trades at the edge of the dataset
        if cur_exit < 0 or np.isnan(asset_pnl):
            continue

        # --- DYNAMIC POSITION SIZING MATH ---
        entry_price = close[i]
        current_atr = atr[i]
        if (
            not np.isfinite(entry_price)
            or not np.isfinite(current_atr)
            or entry_price <= 0.0
            or current_atr <= 0.0
        ):
            continue
        
        # How far away is the stop loss in pure percentage terms?
        stop_loss_pct = (tbm_loss * current_atr) / entry_price
        if not np.isfinite(stop_loss_pct) or stop_loss_pct <= 0.0:
            continue
        
        # Calculate leverage needed to risk exactly `risk_per_trade` (e.g., 1%)
        leverage = risk_per_trade / stop_loss_pct
        if not np.isfinite(leverage) or leverage <= 0.0:
            continue
        leverage = min(leverage, max_leverage)
        
        # Calculate the actual return on the portfolio
        portfolio_return = leverage * (asset_pnl - fee)
        if not np.isfinite(portfolio_return):
            continue
        
        # Update compounding capital
        current_capital *= (1.0 + portfolio_return)

        # Log the trade
        entry_indices[trade_count] = i
        exit_indices[trade_count] = cur_exit
        account_pnls[trade_count] = portfolio_return
        capital_curve[trade_count] = current_capital
        sides[trade_count] = sig
        
        trade_count += 1
        # Capital Lock: No overlaps allowed
        next_free = cur_exit + 1

    return (
        entry_indices[:trade_count],
        exit_indices[:trade_count],
        account_pnls[:trade_count],
        capital_curve[:trade_count],
        sides[:trade_count]
    )


if _NUMBA_AVAILABLE:
    @njit(cache=True)  # type: ignore[misc]
    def _tbm_portfolio_lookup_nb(
        signal: np.ndarray,
        l_pnl: np.ndarray,
        l_exit: np.ndarray,
        s_pnl: np.ndarray,
        s_exit: np.ndarray,
        close: np.ndarray,
        atr: np.ndarray,
        tbm_loss: float,
        fee: float,
        initial_capital: float,
        risk_per_trade: float,
        max_leverage: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n = signal.shape[0]
        
        entry_indices = np.empty(n, dtype=np.int64)
        exit_indices = np.empty(n, dtype=np.int64)
        account_pnls = np.empty(n, dtype=np.float64)
        capital_curve = np.empty(n, dtype=np.float64)
        sides = np.empty(n, dtype=np.int8)

        trade_count = 0
        next_free = 0
        current_capital = initial_capital

        for i in range(n):
            if i < next_free:
                continue

            sig = signal[i]
            if sig == 0 or np.isnan(sig):
                continue

            if sig == 1:
                cur_exit = l_exit[i]
                asset_pnl = l_pnl[i]
            elif sig == -1:
                cur_exit = s_exit[i]
                asset_pnl = s_pnl[i]
            else:
                continue

            if cur_exit < 0 or np.isnan(asset_pnl):
                continue

            entry_price = close[i]
            current_atr = atr[i]
            if (
                not np.isfinite(entry_price)
                or not np.isfinite(current_atr)
                or entry_price <= 0.0
                or current_atr <= 0.0
            ):
                continue
            
            stop_loss_pct = (tbm_loss * current_atr) / entry_price
            if not np.isfinite(stop_loss_pct) or stop_loss_pct <= 0.0:
                continue

            leverage = risk_per_trade / stop_loss_pct
            if not np.isfinite(leverage) or leverage <= 0.0:
                continue
            leverage = min(leverage, max_leverage)

            portfolio_return = leverage * (asset_pnl - fee)
            if not np.isfinite(portfolio_return):
                continue
            
            current_capital *= (1.0 + portfolio_return)

            entry_indices[trade_count] = i
            exit_indices[trade_count] = cur_exit
            account_pnls[trade_count] = portfolio_return
            capital_curve[trade_count] = current_capital
            sides[trade_count] = sig
            
            trade_count += 1
            next_free = cur_exit + 1

        return (
            entry_indices[:trade_count],
            exit_indices[:trade_count],
            account_pnls[:trade_count],
            capital_curve[:trade_count],
            sides[:trade_count]
        )


class VectorizedBacktester:
    """Executes trades using Fixed Fractional Position Sizing (e.g. 1% risk per trade)."""
    
    def __init__(
        self, 
        initial_capital: float = 100000.0, 
        risk_per_trade: float = 0.01, 
        tbm_loss: float = TBM_LOSS,
        atr_window: int = ATR_WINDOW,
        fee: float = BACKTEST_FEE,
        max_leverage: float = 20.0,
    ) -> None:
        self.initial_capital = float(initial_capital)
        self.risk_per_trade = float(risk_per_trade)
        self.tbm_loss = float(tbm_loss)
        self.atr_window = int(atr_window)
        self.fee = float(fee)
        self.max_leverage = float(max_leverage)

    def run(self, df: pd.DataFrame, signal_col: str) -> pd.DataFrame:
        """
        Executes the strategy signals and returns a Trade Log DataFrame ready for DiagnosticsEngine.
        """
        # 1. Strict Input Validation for Diagnostics & Sizing
        req_regimes = ["session", "trend_regime", "vol_regime", "close", f"ATR_{self.atr_window}"]
        for col in req_regimes:
            if col not in df.columns:
                raise KeyError(f"Missing required column: {col}. Run StateMatrixBuilder first.")
        
        if signal_col not in df.columns:
            raise KeyError(f"Signal column '{signal_col}' not found in DataFrame.")

        req_tbm = ["tbm_long_pnl", "tbm_long_exit_idx", "tbm_short_pnl", "tbm_short_exit_idx"]
        for col in req_tbm:
            if col not in df.columns:
                raise KeyError(f"Missing TBM column: {col}. Ensure labeling.py was applied.")

        # 2. Extract Numpy Arrays for speed
        signal_arr = df[signal_col].to_numpy(dtype=np.float64, copy=False)
        l_pnl_arr = df["tbm_long_pnl"].to_numpy(dtype=np.float64, copy=False)
        l_exit_arr = df["tbm_long_exit_idx"].to_numpy(dtype=np.int64, copy=False)
        s_pnl_arr = df["tbm_short_pnl"].to_numpy(dtype=np.float64, copy=False)
        s_exit_arr = df["tbm_short_exit_idx"].to_numpy(dtype=np.int64, copy=False)
        close_arr = df["close"].to_numpy(dtype=np.float64, copy=False)
        atr_arr = df[f"ATR_{self.atr_window}"].to_numpy(dtype=np.float64, copy=False)

        # 3. Run Vectorized Lookup
        if _NUMBA_AVAILABLE:
            entries, exits, acc_pnls, capital, sides = _tbm_portfolio_lookup_nb( # type: ignore[misc]
                signal_arr, l_pnl_arr, l_exit_arr, s_pnl_arr, s_exit_arr, 
                close_arr, atr_arr, self.tbm_loss, self.fee, 
                self.initial_capital, self.risk_per_trade, self.max_leverage
            )
        else:
            entries, exits, acc_pnls, capital, sides = _tbm_portfolio_lookup_py(
                signal_arr, l_pnl_arr, l_exit_arr, s_pnl_arr, s_exit_arr, 
                close_arr, atr_arr, self.tbm_loss, self.fee, 
                self.initial_capital, self.risk_per_trade, self.max_leverage
            )

        # 4. Build Diagnostics-Compliant Trade Log
        if len(entries) == 0:
            return pd.DataFrame(columns=[
                "entry_ts", "exit_ts", "entry_index", "exit_index", "duration",
                "side", "net_trade_return", "account_balance", "session", "trend_regime", "vol_regime"
            ])

        timestamps = df.index.to_series(name="ts").reset_index(drop=True)
        
        trades = pd.DataFrame({
            "entry_index": entries,
            "exit_index": exits,
            "side": sides,
            "net_trade_return": acc_pnls,  # This now perfectly maps to the portfolio % return!
            "account_balance": capital     # Track the compounded $ balance
        })
        
        trades["duration"] = trades["exit_index"] - trades["entry_index"]
        trades["entry_ts"] = trades["entry_index"].map(timestamps)
        trades["exit_ts"] = trades["exit_index"].map(timestamps)
        
        trades["session"] = trades["entry_index"].map(df["session"].reset_index(drop=True))
        trades["trend_regime"] = trades["entry_index"].map(df["trend_regime"].reset_index(drop=True))
        trades["vol_regime"] = trades["entry_index"].map(df["vol_regime"].reset_index(drop=True))
        
        return trades[[
            "entry_ts", "exit_ts", "entry_index", "exit_index", "duration",
            "side", "net_trade_return", "account_balance", "session", "trend_regime", "vol_regime"
        ]]

__all__ = ["VectorizedBacktester"]
