import numpy as np
import pandas as pd
from typing import Optional, Tuple, Union


class Indicators:
    """
    Whitelisted technical indicators mixin for LLM-generated strategy classes.

    All methods are implemented using only pandas/numpy and avoid look-ahead bias
    by using rolling/expanding windows over historical data.
    """

    _EPS = 1e-12

    @staticmethod
    def _resolve_col(data: pd.DataFrame, col: str) -> str:
        if col in data.columns:
            return col
        lower_map = {c.lower(): c for c in data.columns}
        key = col.lower()
        if key in lower_map:
            return lower_map[key]
        raise KeyError(f"Column '{col}' not found in DataFrame.")

    def _series(self, data: pd.DataFrame, col: str) -> pd.Series:
        return data[self._resolve_col(data, col)].astype(float)

    def _true_range(
        self,
        data: pd.DataFrame,
        high_col: str = "high",
        low_col: str = "low",
        close_col: str = "close",
    ) -> pd.Series:
        high = self._series(data, high_col)
        low = self._series(data, low_col)
        close = self._series(data, close_col)
        prev_close = close.shift(1)

        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        tr.name = "tr"
        return tr

    @staticmethod
    def _wilder_smooth(series: pd.Series, period: int) -> pd.Series:
        return series.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    @staticmethod
    def _wma(series: pd.Series, period: int) -> pd.Series:
        weights = np.arange(1, period + 1, dtype=float)
        denom = weights.sum()
        return series.rolling(period, min_periods=period).apply(
            lambda x: float(np.dot(x, weights) / denom),
            raw=True,
        )

    # -------------------------
    # Trend
    # -------------------------
    def sma(self, data: pd.DataFrame, period: int = 20, price_col: str = "close") -> pd.Series:
        """Simple moving average."""
        close = self._series(data, price_col)
        out = close.rolling(period, min_periods=period).mean()
        out.name = f"sma_{period}"
        return out

    def ema(self, data: pd.DataFrame, period: int = 20, price_col: str = "close") -> pd.Series:
        """Exponential moving average."""
        close = self._series(data, price_col)
        out = close.ewm(span=period, adjust=False, min_periods=period).mean()
        out.name = f"ema_{period}"
        return out

    def hma(self, data: pd.DataFrame, period: int = 20, price_col: str = "close") -> pd.Series:
        """Hull moving average."""
        if period < 2:
            raise ValueError("period must be >= 2")

        close = self._series(data, price_col)
        half = max(period // 2, 1)
        sqrt_n = max(int(np.sqrt(period)), 1)

        wma_half = self._wma(close, half)
        wma_full = self._wma(close, period)
        raw = 2.0 * wma_half - wma_full
        out = self._wma(raw, sqrt_n)
        out.name = f"hma_{period}"
        return out

    def macd(
        self,
        data: pd.DataFrame,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
        price_col: str = "close",
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD line, signal line, histogram."""
        close = self._series(data, price_col)
        ema_fast = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
        ema_slow = close.ewm(span=slow, adjust=False, min_periods=slow).mean()

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
        hist = macd_line - signal_line

        macd_line.name = f"macd_{fast}_{slow}_{signal}"
        signal_line.name = f"macd_signal_{fast}_{slow}_{signal}"
        hist.name = f"macd_hist_{fast}_{slow}_{signal}"
        return macd_line, signal_line, hist

    def adx(
        self,
        data: pd.DataFrame,
        period: int = 14,
        high_col: str = "high",
        low_col: str = "low",
        close_col: str = "close",
    ) -> pd.Series:
        """Average Directional Index (Wilder smoothing)."""
        high = self._series(data, high_col)
        low = self._series(data, low_col)

        up_move = high.diff()
        down_move = -low.diff()

        plus_dm = pd.Series(
            np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
            index=data.index,
        )
        minus_dm = pd.Series(
            np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
            index=data.index,
        )

        tr = self._true_range(data, high_col=high_col, low_col=low_col, close_col=close_col)
        atr = self._wilder_smooth(tr, period)

        plus_di = 100.0 * self._wilder_smooth(plus_dm, period) / (atr + self._EPS)
        minus_di = 100.0 * self._wilder_smooth(minus_dm, period) / (atr + self._EPS)

        dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di + self._EPS)
        out = self._wilder_smooth(dx, period)
        out.name = f"adx_{period}"
        return out

    def slope(
        self,
        data: Union[pd.DataFrame, pd.Series],
        period: int = 20,
        col: str = "close",
    ) -> pd.Series:
        """Rolling linear-regression slope of a series (vectorized convolution approach)."""
        if period < 2:
            raise ValueError("period must be >= 2")

        if isinstance(data, pd.Series):
            series = data.astype(float)
        else:
            series = self._series(data, col)

        x = np.arange(period, dtype=float)
        x_centered = x - x.mean()
        denom = float((x_centered**2).sum())

        y = series.to_numpy(dtype=float)
        y_filled = np.nan_to_num(y, nan=0.0)
        numer = np.convolve(y_filled, x_centered[::-1], mode="valid")

        out_arr = np.full(len(series), np.nan, dtype=float)
        out_arr[period - 1 :] = numer / (denom + self._EPS)

        valid = series.rolling(period, min_periods=period).count() == period
        out_arr[~valid.to_numpy()] = np.nan

        out = pd.Series(out_arr, index=series.index, name=f"slope_{period}")
        return out

    # -------------------------
    # Momentum
    # -------------------------
    def rsi(self, data: pd.DataFrame, period: int = 14, price_col: str = "close") -> pd.Series:
        """Relative Strength Index (Wilder method)."""
        close = self._series(data, price_col)
        delta = close.diff()

        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)

        avg_gain = self._wilder_smooth(gain, period)
        avg_loss = self._wilder_smooth(loss, period)

        rs = avg_gain / (avg_loss + self._EPS)
        out = 100.0 - (100.0 / (1.0 + rs))
        out.name = f"rsi_{period}"
        return out

    def stoch(
        self,
        data: pd.DataFrame,
        k_period: int = 14,
        d_period: int = 3,
        high_col: str = "high",
        low_col: str = "low",
        close_col: str = "close",
    ) -> Tuple[pd.Series, pd.Series]:
        """Stochastic oscillator (%K, %D)."""
        high = self._series(data, high_col)
        low = self._series(data, low_col)
        close = self._series(data, close_col)

        ll = low.rolling(k_period, min_periods=k_period).min()
        hh = high.rolling(k_period, min_periods=k_period).max()

        k = 100.0 * (close - ll) / (hh - ll + self._EPS)
        d = k.rolling(d_period, min_periods=d_period).mean()

        k.name = f"stoch_k_{k_period}"
        d.name = f"stoch_d_{k_period}_{d_period}"
        return k, d

    def cci(
        self,
        data: pd.DataFrame,
        period: int = 20,
        high_col: str = "high",
        low_col: str = "low",
        close_col: str = "close",
    ) -> pd.Series:
        """Commodity Channel Index."""
        high = self._series(data, high_col)
        low = self._series(data, low_col)
        close = self._series(data, close_col)

        tp = (high + low + close) / 3.0
        ma = tp.rolling(period, min_periods=period).mean()
        md = tp.rolling(period, min_periods=period).apply(
            lambda x: float(np.mean(np.abs(x - x.mean()))),
            raw=True,
        )

        out = (tp - ma) / (0.015 * (md + self._EPS))
        out.name = f"cci_{period}"
        return out

    def roc(self, data: pd.DataFrame, period: int = 12, price_col: str = "close") -> pd.Series:
        """Rate of Change in percent."""
        close = self._series(data, price_col)
        out = (close / close.shift(period) - 1.0) * 100.0
        out.name = f"roc_{period}"
        return out

    def williams_r(
        self,
        data: pd.DataFrame,
        period: int = 14,
        high_col: str = "high",
        low_col: str = "low",
        close_col: str = "close",
    ) -> pd.Series:
        """Williams %R oscillator."""
        high = self._series(data, high_col)
        low = self._series(data, low_col)
        close = self._series(data, close_col)

        hh = high.rolling(period, min_periods=period).max()
        ll = low.rolling(period, min_periods=period).min()

        out = -100.0 * (hh - close) / (hh - ll + self._EPS)
        out.name = f"williams_r_{period}"
        return out

    def mfi(
        self,
        data: pd.DataFrame,
        period: int = 14,
        high_col: str = "high",
        low_col: str = "low",
        close_col: str = "close",
        volume_col: str = "volume",
    ) -> pd.Series:
        """Money Flow Index."""
        high = self._series(data, high_col)
        low = self._series(data, low_col)
        close = self._series(data, close_col)
        volume = self._series(data, volume_col)

        tp = (high + low + close) / 3.0
        rmf = tp * volume

        direction = tp.diff()
        pos_mf = pd.Series(np.where(direction > 0, rmf, 0.0), index=data.index)
        neg_mf = pd.Series(np.where(direction < 0, rmf, 0.0), index=data.index)

        pos_sum = pos_mf.rolling(period, min_periods=period).sum()
        neg_sum = neg_mf.rolling(period, min_periods=period).sum()

        mfr = pos_sum / (neg_sum + self._EPS)
        out = 100.0 - (100.0 / (1.0 + mfr))
        out.name = f"mfi_{period}"
        return out
        
    def zscore(
        self,
        data: Union[pd.DataFrame, pd.Series],
        period: int = 20,
        col: str = "close",
    ) -> pd.Series:
        """Rolling z-score."""
        if isinstance(data, pd.Series):
            series = data.astype(float)
        else:
            series = self._series(data, col)

        mean = series.rolling(period, min_periods=period).mean()
        std = series.rolling(period, min_periods=period).std(ddof=0)
        out = (series - mean) / (std + self._EPS)
        out.name = f"zscore_{period}"
        return out

    # -------------------------
    # Volatility
    # -------------------------
    def atr(
        self,
        data: pd.DataFrame,
        period: int = 14,
        high_col: str = "high",
        low_col: str = "low",
        close_col: str = "close",
    ) -> pd.Series:
        """Average True Range (Wilder smoothing)."""
        tr = self._true_range(data, high_col=high_col, low_col=low_col, close_col=close_col)
        out = self._wilder_smooth(tr, period)
        out.name = f"atr_{period}"
        return out

    def natr(
        self,
        data: pd.DataFrame,
        period: int = 14,
        high_col: str = "high",
        low_col: str = "low",
        close_col: str = "close",
    ) -> pd.Series:
        """Normalized Average True Range (%)."""
        atr_val = self.atr(data, period, high_col, low_col, close_col)
        close = self._series(data, close_col)
        out = (atr_val / (close + self._EPS)) * 100.0
        out.name = f"natr_{period}"
        return out

    def bollinger_bands(
        self,
        data: pd.DataFrame,
        period: int = 20,
        std_mult: float = 2.0,
        price_col: str = "close",
    ) -> Tuple[pd.Series, pd.Series]:
        """Bollinger Bands (upper, lower)."""
        close = self._series(data, price_col)
        ma = close.rolling(period, min_periods=period).mean()
        sd = close.rolling(period, min_periods=period).std(ddof=0)

        upper = ma + std_mult * sd
        lower = ma - std_mult * sd
        upper.name = f"bb_upper_{period}_{std_mult}"
        lower.name = f"bb_lower_{period}_{std_mult}"
        return upper, lower

    def keltner_channels(
        self,
        data: pd.DataFrame,
        period: int = 20,
        atr_period: int = 10,
        atr_mult: float = 2.0,
        price_col: str = "close",
        high_col: str = "high",
        low_col: str = "low",
    ) -> Tuple[pd.Series, pd.Series]:
        """Keltner Channels (upper, lower)."""
        center = self.ema(data, period=period, price_col=price_col)
        atr_val = self.atr(data, period=atr_period, high_col=high_col, low_col=low_col, close_col=price_col)

        upper = center + atr_mult * atr_val
        lower = center - atr_mult * atr_val
        upper.name = f"kc_upper_{period}_{atr_period}_{atr_mult}"
        lower.name = f"kc_lower_{period}_{atr_period}_{atr_mult}"
        return upper, lower

    def donchian_channels(
        self,
        data: pd.DataFrame,
        period: int = 20,
        high_col: str = "high",
        low_col: str = "low",
    ) -> Tuple[pd.Series, pd.Series]:
        """Donchian Channels (upper, lower)."""
        high = self._series(data, high_col)
        low = self._series(data, low_col)

        upper = high.rolling(period, min_periods=period).max()
        lower = low.rolling(period, min_periods=period).min()
        upper.name = f"donchian_upper_{period}"
        lower.name = f"donchian_lower_{period}"
        return upper, lower

    def choppiness_index(
        self,
        data: pd.DataFrame,
        period: int = 14,
        high_col: str = "high",
        low_col: str = "low",
        close_col: str = "close",
    ) -> pd.Series:
        """Choppiness Index (0-100 regime measure)."""
        tr = self._true_range(data, high_col=high_col, low_col=low_col, close_col=close_col)
        tr_sum = tr.rolling(period, min_periods=period).sum()

        high = self._series(data, high_col)
        low = self._series(data, low_col)
        hh = high.rolling(period, min_periods=period).max()
        ll = low.rolling(period, min_periods=period).min()

        out = 100.0 * np.log10((tr_sum + self._EPS) / (hh - ll + self._EPS)) / np.log10(period)
        out.name = f"chop_{period}"
        return out

    # -------------------------
    # Volume
    # -------------------------
    def vwap(
        self,
        data: pd.DataFrame,
        period: int = 96,
        high_col: str = "high",
        low_col: str = "low",
        close_col: str = "close",
        volume_col: str = "volume",
    ) -> pd.Series:
        """Volume Weighted Average Price (Rolling 24h default for 15m data)."""
        high = self._series(data, high_col)
        low = self._series(data, low_col)
        close = self._series(data, close_col)
        volume = self._series(data, volume_col)

        tp = (high + low + close) / 3.0
        pv = tp * volume

        num = pv.rolling(period, min_periods=period).sum()
        den = volume.rolling(period, min_periods=period).sum()
        out = num / (den + self._EPS)
        out.name = f"vwap_{period}"
        return out

    def obv(self, data: pd.DataFrame, close_col: str = "close", volume_col: str = "volume") -> pd.Series:
        """On-Balance Volume."""
        close = self._series(data, close_col)
        volume = self._series(data, volume_col)

        direction = np.sign(close.diff().fillna(0.0))
        out = (direction * volume).cumsum()
        out.name = "obv"
        return out

    def cmf(
        self,
        data: pd.DataFrame,
        period: int = 20,
        high_col: str = "high",
        low_col: str = "low",
        close_col: str = "close",
        volume_col: str = "volume",
    ) -> pd.Series:
        """Chaikin Money Flow."""
        high = self._series(data, high_col)
        low = self._series(data, low_col)
        close = self._series(data, close_col)
        volume = self._series(data, volume_col)

        mfm = ((close - low) - (high - close)) / (high - low + self._EPS)
        mfv = mfm * volume

        out = mfv.rolling(period, min_periods=period).sum() / (
            volume.rolling(period, min_periods=period).sum() + self._EPS
        )
        out.name = f"cmf_{period}"
        return out


    @classmethod
    def get_help_text(cls) -> str:
        """Return method reference text for prompt injection and LLM guidance."""
        return (
            "Available indicator methods on Indicators mixin:\n\n"
            "Trend\n"
            "- ema(data, period=20, price_col='close'): Exponential moving average (industry standard).\n"
            "- hma(data, period=20, price_col='close'): Hull moving average (hyper-responsive, low lag).\n"
            "- macd(data, fast=12, slow=26, signal=9, price_col='close') -> (macd, signal, hist): MACD tuple.\n"
            "- adx(data, period=14, high_col='high', low_col='low', close_col='close'): Wilder ADX regime/trend strength.\n"
            "- slope(data_or_series, period=20, col='close'): Rolling linear-regression slope.\n\n"
            "Momentum\n"
            "- rsi(data, period=14, price_col='close'): Relative Strength Index.\n"
            "- cci(data, period=20, high_col='high', low_col='low', close_col='close'): Commodity Channel Index.\n"
            "- roc(data, period=12, price_col='close'): Rate of Change (%).\n"
            "- mfi(data, period=14, high_col='high', low_col='low', close_col='close', volume_col='volume'): Money Flow Index (volume-weighted RSI).\n"
            "- zscore(data_or_series, period=20, col='close'): Rolling z-score (Statistical oscillator for extreme overbought/oversold).\n\n"
            "Volatility\n"
            "- natr(data, period=14, high_col='high', low_col='low', close_col='close'): Normalized ATR (Volatility as a % of price. Use this over raw ATR).\n"
            "- bollinger_bands(data, period=20, std_mult=2.0, price_col='close') -> (upper, lower): Bollinger bands (Standard Deviation).\n"
            "- keltner_channels(data, period=20, atr_period=10, atr_mult=2.0, price_col='close', high_col='high', low_col='low') -> (upper, lower): Keltner channels (ATR-based).\n"
            "- choppiness_index(data, period=14, high_col='high', low_col='low', close_col='close'): 0-100 choppiness regime score.\n\n"
            "Volume\n"
            "- vwap(data, period=96, high_col='high', low_col='low', close_col='close', volume_col='volume'): Rolling VWAP (defaults to 24h on 15m data).\n"
            "- obv(data, close_col='close', volume_col='volume'): On-Balance Volume.\n"
            "- cmf(data, period=20, high_col='high', low_col='low', close_col='close', volume_col='volume'): Chaikin Money Flow.\n"
        )
