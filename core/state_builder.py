from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from config import (
    ATR_WINDOW,
    TBM_LOSS,
    TBM_TIME_HORIZON,
    TBM_TIE_BREAK,
    TBM_WIN,
    TREND_SLOPE_LOOKBACK,
    TREND_SLOPE_THRESHOLD,
    TREND_SMA_WINDOW,
    VOL_SMA_WINDOW,
)
from labeling import _atr_series, _resolve_col, apply_triple_barrier


class StateMatrixBuilder:
    """Build a master state matrix from raw 15m OHLCV data."""

    def __init__(
        self,
        open_col: str = "open",
        high_col: str = "high",
        low_col: str = "low",
        close_col: str = "close",
        volume_col: str = "volume",
        datetime_col: Optional[str] = None,
        trend_sma_window: int = TREND_SMA_WINDOW,
        trend_slope_lookback: int = TREND_SLOPE_LOOKBACK,
        trend_slope_threshold: float = TREND_SLOPE_THRESHOLD,
        atr_window: int = ATR_WINDOW,
        vol_sma_window: int = VOL_SMA_WINDOW,
        tbm_win: float = TBM_WIN,
        tbm_loss: float = TBM_LOSS,
        tbm_time_horizon: int = TBM_TIME_HORIZON,
        tbm_tie_break: str = TBM_TIE_BREAK,
    ) -> None:
        if tbm_tie_break not in {"stop_first", "profit_first"}:
            raise ValueError("tbm_tie_break must be one of {'stop_first', 'profit_first'}")
        if trend_sma_window <= 0 or trend_slope_lookback <= 0:
            raise ValueError("trend windows must be > 0")
        if atr_window <= 0 or vol_sma_window <= 0:
            raise ValueError("vol windows must be > 0")
        if tbm_time_horizon <= 0:
            raise ValueError("tbm_time_horizon must be > 0")

        self.open_col = open_col
        self.high_col = high_col
        self.low_col = low_col
        self.close_col = close_col
        self.volume_col = volume_col
        self.datetime_col = datetime_col

        self.trend_sma_window = int(trend_sma_window)
        self.trend_slope_lookback = int(trend_slope_lookback)
        self.trend_slope_threshold = float(trend_slope_threshold)
        self.atr_window = int(atr_window)
        self.vol_sma_window = int(vol_sma_window)

        self.tbm_win = float(tbm_win)
        self.tbm_loss = float(tbm_loss)
        self.tbm_time_horizon = int(tbm_time_horizon)
        self.tbm_tie_break = tbm_tie_break

        self._state_matrix: Optional[pd.DataFrame] = None
        self._col_cache: Optional[dict] = None

    def build(self, raw_df: pd.DataFrame, force_rebuild: bool = False) -> pd.DataFrame:
        """Build the state matrix exactly once per builder instance unless force_rebuild=True."""
        if self._state_matrix is not None and not force_rebuild:
            return self._state_matrix.copy()

        df = self._prepare_input(raw_df)
        df = self._add_session(df)
        df = self._add_trend_regime(df)
        df = self._add_vol_regime(df)
        df = self._add_triple_barrier_targets(df)
        df = self._finalize_output(df)

        self._state_matrix = df
        return df.copy()

    def get_state_matrix(self) -> pd.DataFrame:
        if self._state_matrix is None:
            raise RuntimeError("State matrix is not built yet. Call build(raw_df) first.")
        return self._state_matrix.copy()

    def _prepare_input(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        if raw_df is None or len(raw_df) == 0:
            raise ValueError("raw_df is empty")

        df = raw_df.copy()
        self._col_cache = self._resolve_price_columns(df)

        if not isinstance(df.index, pd.DatetimeIndex):
            dt_col = self.datetime_col
            if dt_col is None:
                for c in ("datetime", "timestamp", "date", "time"):
                    if c in df.columns:
                        dt_col = c
                        break
            if dt_col is None:
                raise ValueError(
                    "Input must have a DatetimeIndex or a datetime column "
                    "(e.g. datetime/timestamp/date/time)."
                )
            dt = pd.to_datetime(df[dt_col], utc=True, errors="coerce")
            df = df.loc[dt.notna()].copy()
            df.index = dt[dt.notna()]
            if dt_col in df.columns:
                df = df.drop(columns=[dt_col])
        else:
            idx = pd.to_datetime(df.index, utc=True, errors="coerce")
            df = df.loc[idx.notna()].copy()
            df.index = idx[idx.notna()]

        df = df.sort_index()
        df.index.name = "open_time"

        # Ensure numeric OHLCV for downstream calculations.
        for key in ("open", "high", "low", "close", "volume"):
            col = self._col_cache[key]
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if col != key:
                df[key] = df[col]

        # Normalize optional market microstructure columns when present.
        optional_map = {
            "quote_volume": ("quote_volume", "Quote volume"),
            "count": ("count", "Number of trades"),
            "taker_buy_volume": ("taker_buy_volume", "Taker buy base asset volume"),
        }
        for canonical, aliases in optional_map.items():
            for candidate in aliases:
                if candidate in df.columns:
                    df[canonical] = pd.to_numeric(df[candidate], errors="coerce")
                    break

        return df

    def _resolve_price_columns(self, df: pd.DataFrame) -> dict:
        return {
            "open": _resolve_col(df, self.open_col, "Open"),
            "high": _resolve_col(df, self.high_col, "High"),
            "low": _resolve_col(df, self.low_col, "Low"),
            "close": _resolve_col(df, self.close_col, "Close"),
            "volume": _resolve_col(df, self.volume_col, "Volume"),
        }

    def _add_session(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        hours = out.index.hour
        out["session"] = np.select(
            [
                (hours >= 0) & (hours <= 7),
                (hours >= 8) & (hours <= 12),
                (hours >= 13) & (hours <= 20),
            ],
            ["ASIA", "LONDON", "NY"],
            default="OTHER",
        )
        return out

    def _add_trend_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        close = out[self._col_cache["close"]]
        out["sma_50"] = close.rolling(self.trend_sma_window, min_periods=self.trend_sma_window).mean()
        out["sma_50_slope_3"] = out["sma_50"].pct_change(self.trend_slope_lookback)

        out["trend_regime"] = np.select(
            [
                out["sma_50_slope_3"] > self.trend_slope_threshold,
                out["sma_50_slope_3"] < -self.trend_slope_threshold,
            ],
            ["UPTREND", "DOWNTREND"],
            default="CONSOLIDATION",
        )
        return out

    def _add_vol_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        atr_col = f"ATR_{self.atr_window}"
        out[atr_col] = _atr_series(
            out,
            high_col=self._col_cache["high"],
            low_col=self._col_cache["low"],
            close_col=self._col_cache["close"],
            window=self.atr_window,
        )
        out[f"{atr_col}_SMA_{self.vol_sma_window}"] = out[atr_col].rolling(
            self.vol_sma_window,
            min_periods=self.vol_sma_window,
        ).mean()
        out["vol_regime"] = np.where(
            out[atr_col] > out[f"{atr_col}_SMA_{self.vol_sma_window}"],
            "HIGH_VOL",
            "LOW_VOL",
        )
        return out

    def _add_triple_barrier_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        return apply_triple_barrier(
            df,
            self.tbm_win,
            self.tbm_loss,
            self.tbm_time_horizon,
            self.atr_window,
            self.tbm_tie_break,
            self._col_cache["high"],
            self._col_cache["low"],
            self._col_cache["close"],
        )

    def _finalize_output(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        atr_col = f"ATR_{self.atr_window}"
        tmp_cols = [
            "sma_50",
            "sma_50_slope_3",
            atr_col,
            f"{atr_col}_SMA_{self.vol_sma_window}",
        ]
        out = out.drop(columns=[c for c in tmp_cols if c in out.columns], errors="ignore")

        ordered = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_volume",
            "count",
            "taker_buy_volume",
            "session",
            "trend_regime",
            "vol_regime",
            "tbm_label",
            "tbm_long_pnl",
            "tbm_long_exit_idx",
            "tbm_long_duration",
            "tbm_short_pnl",
            "tbm_short_exit_idx",
            "tbm_short_duration",
        ]
        existing = [c for c in ordered if c in out.columns]
        return out[existing]


__all__ = ["StateMatrixBuilder"]
