"""
strategy_base.py — Council of Alphas
Base class for ALL LLM-generated and hybrid strategies.

IMPORTANT:
- tbm_win and tbm_loss are NOT here — fixed system-wide in config.py
- LLM generates a full class inheriting from Strategy
- HybridBuilder also produces Strategy subclasses
"""

from __future__ import annotations
import pandas as pd
from whitelist_indicators import Indicators


class Strategy(Indicators):
    """
    Base strategy class. 

    LLM-generated specialists and Python-built hybrids both inherit from this.
    All whitelisted indicators are available via the Indicators mixin.
    """

    name: str = "unnamed"
    family: str = "unknown"     # trend | momentum | volatility | volume | hybrid
    description: str = ""       # One sentence: what does this strategy do?

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals for every candle in data.

        Returns
        -------
        pd.Series
            Values must be one of:
              1  = long entry signal
             -1  = short entry signal
              0  = no signal / flat
            Index must match data.index exactly.
            No lookahead bias permitted (no shift(-1), no future data).
        """
        raise NotImplementedError("Implemented by LLM specialist or HybridBuilder")
