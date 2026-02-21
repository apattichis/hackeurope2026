"""
prompt_builder.py — Council of Alphas
Builds complete specialist prompts by combining:
- Fixed template skeleton
- Family-specific guidance
- Randomly sampled indicator help text (filtered from whitelist)
"""

from __future__ import annotations

from whitelist_indicators import Indicators
from config import FAMILY_INDICATORS

# ── Family Guidance Strings ────────────────────────────────────────────────────

FAMILY_GUIDANCE = {
    "trend": (
        "You are a TREND specialist. Your strategies must identify and follow "
        "directional momentum. Think: when is price clearly moving in one direction "
        "and how do you ride that move? Avoid mean-reversion logic."
    ),
    "momentum": (
        "You are a MOMENTUM specialist. Your strategies must identify overbought/"
        "oversold extremes and momentum exhaustion. Think: when has price moved "
        "too far too fast and is likely to reverse or accelerate?"
    ),
    "volatility": (
        "You are a VOLATILITY specialist. Your strategies must use volatility "
        "expansion and contraction to time entries. Think: when are price boundaries "
        "being broken, when is a squeeze about to release?"
    ),
    "volume": (
        "You are a VOLUME specialist. Your strategies must use order flow and "
        "institutional participation to confirm price moves. Think: is volume "
        "backing this price move or is it a trap?"
    ),
}

# ── Indicator Help Text Filtering ─────────────────────────────────────────────

def _get_filtered_help_text(sampled_indicators: list[str]) -> str:
    """
    Filter Indicators.get_help_text() to only include lines
    for the sampled indicator names.
    """
    full_help = Indicators.get_help_text()
    lines = full_help.split("\n")
    
    filtered_lines = []
    for line in lines:
        # Keep section headers (Trend, Momentum, etc.)
        if not line.startswith("- "):
            filtered_lines.append(line)
            continue
        # Keep only lines for sampled indicators
        for indicator in sampled_indicators:
            # Match indicator name at start of line description
            if line.startswith(f"- {indicator}("):
                filtered_lines.append(line)
                break
    
    # Clean up empty consecutive lines
    result = "\n".join(filtered_lines).strip()
    return result


# ── Main Prompt Builder ───────────────────────────────────────────────────────

def build_specialist_prompt(
    family: str,
    sampled_indicators: list[str],
    previous_error: str | None = None,
) -> str:
    """
    Build complete specialist prompt.

    Parameters
    ----------
    family : str
        One of: trend, momentum, volatility, volume
    sampled_indicators : list[str]
        Indicator names randomly sampled by IndicatorSampler
    previous_error : str | None
        If retrying after a failed attempt, inject the error here

    Returns
    -------
    str
        Complete prompt string ready for API call
    """
    if family not in FAMILY_GUIDANCE:
        raise ValueError(f"Unknown family: {family}. Must be one of {list(FAMILY_GUIDANCE.keys())}")

    indicator_help = _get_filtered_help_text(sampled_indicators)
    guidance = FAMILY_GUIDANCE[family]

    error_section = ""
    if previous_error:
        error_section = f"""
═══ PREVIOUS ATTEMPT FAILED ═══
Your previous code produced this error:
{previous_error}

Fix this specific issue in your new attempt.
"""

    prompt = f"""You are a quantitative trading strategy specialist.
Your family: {family.upper()}

═══ STRICT RULES ═══
1. Your class MUST inherit from Strategy
2. You MUST set: name, family, description
3. generate_signals() MUST return a pd.Series of 1 (long), -1 (short), 0 (flat)
4. The Series index MUST match data.index exactly
5. You may ONLY use the indicators listed below — no other libraries,
   no raw pandas rolling logic, no TA-lib, nothing else
6. No lookahead bias — no shift(-1), no future data access
7. The strategy must generate both long AND short signals
8. Do not use regime columns (session, trend_regime, vol_regime) as inputs

═══ YOUR INDICATOR TOOLKIT ═══
{indicator_help}

═══ DATA AVAILABLE ═══
The data DataFrame columns: open, high, low, close, volume
Index: open_time (UTC datetime)
{error_section}
═══ FAMILY GUIDANCE ═══
{guidance}

═══ OUTPUT FORMAT ═══
Return ONLY the Python class code. No explanation. No markdown.
No imports (they are handled). Just the class definition.

═══ EXAMPLE STRUCTURE ═══
class MyStrategy(Strategy):
    name = "your_strategy_name"
    family = "{family}"
    description = "one sentence describing entry logic"

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=data.index)
        # your logic here using only the indicators listed above
        return signals
"""
    return prompt.strip()


__all__ = ["build_specialist_prompt"]
