from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


class DiagnosticsEngine:
    """Lightweight hierarchical diagnostics for strategy trade performance."""

    DIMENSIONS: Tuple[str, str, str] = ("session", "trend_regime", "vol_regime")
    HIERARCHY: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
        ("GLOBAL", ()),
        ("1D", ("session",)),
        ("1D", ("trend_regime",)),
        ("1D", ("vol_regime",)),
        ("2D", ("session", "trend_regime")),
        ("2D", ("session", "vol_regime")),
        ("2D", ("trend_regime", "vol_regime")),
        ("3D", ("session", "trend_regime", "vol_regime")),
    )
    OUTPUT_COLUMNS: List[str] = [
        "granularity",
        "session",
        "trend_regime",
        "vol_regime",
        "trade_count",
        "win_rate",
        "sharpe",
        "max_consecutive_losses",
        "sufficient_evidence",
    ]

    def __init__(
        self,
        return_col: Optional[str] = None,
        win_col: Optional[str] = None,
        order_col: Optional[str] = None,
        min_trades: int = 30,
    ) -> None:
        self.return_col = return_col
        self.win_col = win_col
        self.order_col = order_col
        self.min_trades = int(min_trades)

    def compute(self, trades: pd.DataFrame) -> pd.DataFrame:
        """Return hierarchical diagnostics table for the provided trades DataFrame."""
        return self.compute_performance_cube(trades)

    def compute_performance_cube(self, trades: pd.DataFrame) -> pd.DataFrame:
        """Compute GLOBAL/1D/2D/3D summary with the required metrics only."""
        base = self._prepare_trades(trades)

        frames: List[pd.DataFrame] = []
        for granularity, group_keys in self.HIERARCHY:
            frame = self._compute_slice(base, granularity=granularity, group_keys=group_keys)
            frames.append(frame)

        out = pd.concat(frames, ignore_index=True)
        return out[self.OUTPUT_COLUMNS]

    def _prepare_trades(self, trades: pd.DataFrame) -> pd.DataFrame:
        if trades is None:
            raise ValueError("trades DataFrame is required")

        df = trades.copy()
        for col in self.DIMENSIONS:
            if col not in df.columns:
                df[col] = "ALL"
            df[col] = df[col].fillna("ALL").astype(str)

        ret_col = self._resolve_return_col(df)
        df["__ret__"] = pd.to_numeric(df[ret_col], errors="coerce")

        if self.win_col and self.win_col in df.columns:
            df["__win__"] = df[self.win_col].astype(bool)
        else:
            df["__win__"] = df["__ret__"] > 0.0
        df["__loss__"] = df["__ret__"] < 0.0

        order_col = self._resolve_order_col(df)
        if order_col is not None:
            df[order_col] = pd.to_datetime(df[order_col], utc=True, errors="coerce")
            if df[order_col].notna().any():
                df = df.sort_values(order_col).reset_index(drop=True)

        return df

    def _compute_slice(
        self,
        df: pd.DataFrame,
        granularity: str,
        group_keys: Sequence[str],
    ) -> pd.DataFrame:
        if len(df) == 0:
            row = {
                "granularity": granularity,
                "session": "ALL",
                "trend_regime": "ALL",
                "vol_regime": "ALL",
                "trade_count": 0,
                "win_rate": np.nan,
                "sharpe": np.nan,
                "max_consecutive_losses": 0,
                "sufficient_evidence": False,
            }
            return pd.DataFrame([row], columns=self.OUTPUT_COLUMNS)

        if len(group_keys) == 0:
            grouped = df.groupby(np.zeros(len(df), dtype=int), sort=False)
            agg = self._aggregate_group(grouped)
            agg = agg.reset_index(drop=True)
            for dim in self.DIMENSIONS:
                agg[dim] = "ALL"
        else:
            grouped = df.groupby(list(group_keys), dropna=False, sort=False)
            agg = self._aggregate_group(grouped).reset_index()
            for dim in self.DIMENSIONS:
                if dim not in group_keys:
                    agg[dim] = "ALL"

        agg["granularity"] = granularity
        agg["sufficient_evidence"] = agg["trade_count"] >= self.min_trades
        return agg[self.OUTPUT_COLUMNS]

    def _aggregate_group(self, grouped: pd.core.groupby.DataFrameGroupBy) -> pd.DataFrame:
        stats = grouped.agg(
            trade_count=("__ret__", "size"),
            win_rate=("__win__", "mean"),
            ret_mean=("__ret__", "mean"),
            ret_std=("__ret__", lambda s: s.std(ddof=0)),
        )

        with np.errstate(divide="ignore", invalid="ignore"):
            sharpe = stats["ret_mean"] / stats["ret_std"]
        sharpe[(stats["ret_std"] <= 0) | (stats["trade_count"] < 2)] = np.nan
        stats["sharpe"] = sharpe

        loss_streaks = grouped["__loss__"].apply(self._max_consecutive_true).rename("max_consecutive_losses")
        out = stats.join(loss_streaks)
        return out.drop(columns=["ret_mean", "ret_std"])

    @staticmethod
    def _max_consecutive_true(values: Iterable[bool]) -> int:
        arr = np.asarray(list(values), dtype=np.int8)
        if arr.size == 0:
            return 0
        max_run = 0
        run = 0
        for v in arr:
            if v == 1:
                run += 1
                if run > max_run:
                    max_run = run
            else:
                run = 0
        return int(max_run)

    def _resolve_return_col(self, df: pd.DataFrame) -> str:
        if self.return_col and self.return_col in df.columns:
            return self.return_col

        for candidate in ("net_trade_return", "trade_return", "pnl_pct", "pnl", "return"):
            if candidate in df.columns:
                return candidate
        raise ValueError(
            "No return column found. Provide one of: net_trade_return, trade_return, pnl_pct, pnl, return "
            "or pass return_col explicitly."
        )

    def _resolve_order_col(self, df: pd.DataFrame) -> Optional[str]:
        if self.order_col and self.order_col in df.columns:
            return self.order_col

        for candidate in ("entry_ts", "datetime", "entry_date", "timestamp"):
            if candidate in df.columns:
                return candidate
        return None


__all__ = ["DiagnosticsEngine"]