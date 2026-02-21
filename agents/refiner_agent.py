"""
refiner_agent.py — Council of Alphas
Surgical code modification via Claude Sonnet.

The Refiner receives a strategy's source code and exactly one
surgical fix instruction from the Critic. It applies the fix
and returns the complete updated class. Nothing more.

Validation (compile, exec, backtest) happens in the Scientist loop,
not here.

Public API:
    run_refiner()  — async, calls Sonnet, returns modified code string
"""

from __future__ import annotations

import re
import sys
import logging
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _p in [str(_PROJECT_ROOT / "core"), str(_PROJECT_ROOT / "pipeline"),
           str(_PROJECT_ROOT / "agents")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import anthropic
from dotenv import load_dotenv

from config import SONNET_MODEL, REFINER_TEMPERATURE

load_dotenv(_PROJECT_ROOT / ".env")

logger = logging.getLogger("council.refiner")


# ── Refiner Prompt (verbatim from MASTER_SPEC Section 11.6) ──────────────────

REFINER_SYSTEM_PROMPT = """\
You are the Surgical Refiner for the Council of Alphas framework.

═══ YOUR ROLE ═══
You receive a trading strategy and one precise instruction
from the Critic. Your job is to apply exactly that fix
and nothing else.

═══ STRICT CONSTRAINTS ═══
- Apply ONE change only — exactly what the Critic specified
- Do NOT restructure the code
- Do NOT change indicators or strategy family
- Do NOT add new logic
- Do NOT remove existing logic unless explicitly instructed
- Return the complete updated class, nothing else

═══ OUTPUT ═══
Return ONLY the complete updated Python class.
No explanation. No markdown. No imports."""


# ── Code Extraction ──────────────────────────────────────────────────────────

def _extract_refined_code(raw_response: str) -> str:
    """
    Extract clean Python code from the Refiner's response.

    Despite the prompt saying "no markdown", LLMs sometimes wrap
    output in ```python ... ``` fences. This strips them.

    Returns empty string if response is empty or whitespace-only.
    """
    text = raw_response.strip()

    if not text:
        return ""

    # Strip markdown fences: ```python ... ``` or ``` ... ```
    fence_pattern = r"^```(?:python)?\s*\n(.*?)\n```\s*$"
    match = re.match(fence_pattern, text, re.DOTALL)
    if match:
        text = match.group(1).strip()

    return text


# ── Public API ────────────────────────────────────────────────────────────────

async def run_refiner(
    strategy_code: str,
    surgical_fix: str,
) -> str:
    """
    Apply one surgical fix to a strategy via Claude Sonnet.

    Parameters
    ----------
    strategy_code : str
        Full Python source code of the strategy to modify.
    surgical_fix : str
        The Critic's SURGICAL_FIX instruction (one-line fix description).

    Returns
    -------
    str
        Modified Python class code. Empty string on API failure.
    """
    user_message = (
        f"STRATEGY CODE:\n{strategy_code}\n\n"
        f"CRITIC INSTRUCTION:\n{surgical_fix}"
    )

    try:
        client = anthropic.AsyncAnthropic()
        response = await client.messages.create(
            model=SONNET_MODEL,
            max_tokens=4096,
            temperature=REFINER_TEMPERATURE,
            system=REFINER_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        raw_text = response.content[0].text
        logger.debug(f"Refiner raw response:\n{raw_text}")

    except Exception as e:
        logger.error(f"Refiner API call failed: {type(e).__name__}: {e}")
        return ""

    refined_code = _extract_refined_code(raw_text)

    if not refined_code:
        logger.warning("Refiner returned empty or unparseable response")
        return ""

    logger.info(f"Refiner produced {len(refined_code)} chars of modified code")
    return refined_code


__all__ = ["run_refiner"]


# ── Smoke Test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print("[SMOKE TEST] refiner_agent.py\n")

    # ── Test 1: Clean code (no fences) ───────────────────────────────────
    print("=== Test 1: Clean code (no markdown fences) ===")
    clean_code = """\
class MyStrategy(Strategy):
    name = "test"
    family = "trend"
    description = "test strategy"

    def generate_signals(self, data):
        return pd.Series(0, index=data.index)"""

    extracted = _extract_refined_code(clean_code)
    assert extracted == clean_code
    assert "class MyStrategy" in extracted
    print(f"  Extracted {len(extracted)} chars")
    print("  OK")

    # ── Test 2: Code wrapped in ```python ... ``` ────────────────────────
    print("\n=== Test 2: Code wrapped in ```python fences ===")
    fenced_code = f"```python\n{clean_code}\n```"
    extracted2 = _extract_refined_code(fenced_code)
    assert extracted2 == clean_code
    assert "```" not in extracted2
    print(f"  Stripped fences, extracted {len(extracted2)} chars")
    print("  OK")

    # ── Test 3: Code wrapped in ``` ... ``` (no language tag) ────────────
    print("\n=== Test 3: Code wrapped in ``` fences (no language tag) ===")
    bare_fenced = f"```\n{clean_code}\n```"
    extracted3 = _extract_refined_code(bare_fenced)
    assert extracted3 == clean_code
    assert "```" not in extracted3
    print(f"  Stripped bare fences, extracted {len(extracted3)} chars")
    print("  OK")

    # ── Test 4: Empty response ───────────────────────────────────────────
    print("\n=== Test 4: Empty response ===")
    assert _extract_refined_code("") == ""
    assert _extract_refined_code("   ") == ""
    assert _extract_refined_code("\n\n") == ""
    print("  All empty variants return empty string")
    print("  OK")

    # ── Test 5: Whitespace-padded response ───────────────────────────────
    print("\n=== Test 5: Whitespace-padded response ===")
    padded = f"\n\n  {clean_code}  \n\n"
    extracted5 = _extract_refined_code(padded)
    assert "class MyStrategy" in extracted5
    # Should be stripped of leading/trailing whitespace
    assert not extracted5.startswith("\n")
    assert not extracted5.endswith("\n")
    print(f"  Stripped padding, extracted {len(extracted5)} chars")
    print("  OK")

    print("\nPASS")
