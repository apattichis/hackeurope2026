"""
indicator_sampler.py — Council of Alphas
Randomly samples bounded subsets of whitelisted indicators per specialist family.
Prevents intra-specialist loss of diversity — LLM cannot use what it cannot see.

IMPORTANT: This class is a SAMPLING ENGINE only.
Prompt construction is handled by prompt_builder.py.
"""

from __future__ import annotations

import random
from typing import Dict, List, Tuple

from config import (
    FAMILY_INDICATORS,
    MIN_INDICATORS_PER_PROMPT,
    MAX_INDICATORS_PER_PROMPT,
)


class IndicatorSampler:
    """
    Samples bounded subsets of whitelisted indicators per specialist family.

    Hidden indicators (excluded from all LLM prompts):
      - sma, stoch, williams_r, atr, donchian_channels
    These exist in whitelist_indicators.py but are intentionally invisible to specialists.
    """

    def __init__(self, seed: int | None = None) -> None:
        """
        Parameters
        ----------
        seed : int | None
            Fixed seed for reproducible runs. None = random each time.
        """
        self.seed = seed
        self.rng = random.Random(seed)

    def sample(
        self,
        family: str,
        min_indicators: int = MIN_INDICATORS_PER_PROMPT,
        max_indicators: int = MAX_INDICATORS_PER_PROMPT,
    ) -> List[str]:
        """
        Sample a random subset of indicators for the given family.

        Parameters
        ----------
        family : str
            One of: trend, momentum, volatility, volume
        min_indicators : int
            Minimum number of indicators to sample
        max_indicators : int
            Maximum number of indicators to sample

        Returns
        -------
        List[str]
            List of indicator name strings ready for prompt injection
        """
        if family not in FAMILY_INDICATORS:
            raise ValueError(
                f"Unknown family: '{family}'. Must be one of {list(FAMILY_INDICATORS.keys())}"
            )

        pool = FAMILY_INDICATORS[family]
        actual_max = min(max_indicators, len(pool))
        actual_min = min(min_indicators, actual_max)

        n = self.rng.randint(actual_min, actual_max)
        return self.rng.sample(pool, n)

    def sample_unique_sets(
        self,
        family: str,
        num_sets: int,
        min_indicators: int = MIN_INDICATORS_PER_PROMPT,
        max_indicators: int = MAX_INDICATORS_PER_PROMPT,
    ) -> List[Tuple[str, ...]]:
        """
        Generate multiple unique indicator subsets for a family.
        Used when generating multiple strategies per specialist.

        Returns
        -------
        List[Tuple[str, ...]]
            List of unique indicator tuples (sorted for deduplication)
        """
        if family not in FAMILY_INDICATORS:
            raise ValueError(f"Unknown family: '{family}'")

        pool = FAMILY_INDICATORS[family]
        actual_max = min(max_indicators, len(pool))
        actual_min = min(min_indicators, actual_max)

        unique_sets: set = set()
        max_attempts = num_sets * 20

        for _ in range(max_attempts):
            if len(unique_sets) >= num_sets:
                break
            n = self.rng.randint(actual_min, actual_max)
            sampled = tuple(sorted(self.rng.sample(pool, n)))
            unique_sets.add(sampled)

        return [list(s) for s in unique_sets]


__all__ = ["IndicatorSampler"]
