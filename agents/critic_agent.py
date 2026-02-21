"""
critic_agent.py — Council of Alphas
Evidence-locked diagnosis via Claude Opus.

The Critic reads a strategy's diagnostic bucket table and source code,
identifies the primary failing regime, and prescribes exactly one
surgical fix. Every claim must cite an exact bucket and number.

Public API:
    run_critic()  — async, calls Opus, returns parsed diagnosis dict
"""

from __future__ import annotations

import sys
import logging
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _p in [str(_PROJECT_ROOT / "core"), str(_PROJECT_ROOT / "pipeline"),
           str(_PROJECT_ROOT / "agents")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd
import anthropic
from dotenv import load_dotenv

from config import OPUS_MODEL, CRITIC_TEMPERATURE

load_dotenv(_PROJECT_ROOT / ".env")

logger = logging.getLogger("council.critic")


# ── Critic Prompt (verbatim from MASTER_SPEC Section 11.5) ───────────────────

CRITIC_SYSTEM_PROMPT = """\
You are the Evidence-Locked Critic for the Council of Alphas framework.

═══ YOUR ROLE ═══
You diagnose why a trading strategy is underperforming by reading
its diagnostic bucket table and strategy code. You do not guess.
Every claim you make must cite an exact bucket and an exact number
from the table provided.

═══ INPUTS YOU ARE GIVEN ═══
1. STRATEGY CODE — read this only to identify what specific parameter
   or logic explains the failure pattern you find in the diagnostics.
2. DIAGNOSTIC TABLE — sufficient_evidence=True rows only. Columns:
   granularity, session, trend_regime, vol_regime, trade_count,
   win_rate, sharpe, max_consecutive_losses, sufficient_evidence.

═══ HOW TO DIAGNOSE ═══
Scan strictly in this order:
1. GLOBAL row — is overall Sharpe salvageable?
2. 1D slices — which single dimension is the primary drag?
3. 2D slices — which interaction is the core problem?
4. 3D buckets — identify the exact failing micro-regime(s).

═══ STRICT CONSTRAINTS ═══
- You may NOT suggest structural rewrites
- You may NOT change the strategy family or its indicators
- You may NOT invent metrics not present in the table
- One surgical fix only — a parameter value, a threshold,
  or a single condition change
- Every claim must cite: [bucket] | sharpe=[x] | n=[x]

═══ UNVIABLE CONDITIONS ═══
Declare UNVIABLE immediately if ANY of these are true:
- GLOBAL sharpe < -0.5 (with sufficient_evidence=True)
- Zero 3D buckets have both sufficient_evidence=True AND sharpe > 0
- GLOBAL max_consecutive_losses > 20

═══ OUTPUT FORMAT ═══
Respond in exactly this structure, nothing else:

PRIMARY_FAILURE: [exact bucket] | sharpe=[value] | n=[value]
ROOT_CAUSE: [one sentence citing the code]
SURGICAL_FIX: [exact code change — parameter name, old value, new value]
EXPECTED_IMPACT: [which metric improves and why]
VERDICT: CONTINUE | UNVIABLE"""


# ── Diagnostics Formatting ────────────────────────────────────────────────────

def _format_diagnostics_table(diagnostics_df: pd.DataFrame) -> str:
    """
    Filter to sufficient_evidence=True rows and format as aligned text table.
    """
    filtered = diagnostics_df[diagnostics_df["sufficient_evidence"] == True].copy()

    if len(filtered) == 0:
        return "(No rows with sufficient_evidence=True)"

    # Select columns in the order the Critic expects
    cols = [
        "granularity", "session", "trend_regime", "vol_regime",
        "trade_count", "win_rate", "sharpe", "max_consecutive_losses",
    ]
    display = filtered[cols].copy()

    # Format numeric columns
    display["win_rate"] = display["win_rate"].apply(lambda x: f"{x:.4f}")
    display["sharpe"] = display["sharpe"].apply(
        lambda x: f"{x:.4f}" if pd.notna(x) else "NaN"
    )
    display["trade_count"] = display["trade_count"].astype(int)
    display["max_consecutive_losses"] = display["max_consecutive_losses"].astype(int)

    return display.to_string(index=False)


# ── Response Parsing ──────────────────────────────────────────────────────────

def _parse_critic_response(raw_response: str) -> dict:
    """
    Parse the Critic's structured 5-field response.

    Returns dict with keys:
        primary_failure, root_cause, surgical_fix, expected_impact, verdict

    If any field is missing, verdict defaults to UNVIABLE (safe fallback).
    """
    fields = {
        "primary_failure": "PRIMARY_FAILURE:",
        "root_cause": "ROOT_CAUSE:",
        "surgical_fix": "SURGICAL_FIX:",
        "expected_impact": "EXPECTED_IMPACT:",
        "verdict": "VERDICT:",
    }

    result = {}
    for key, prefix in fields.items():
        result[key] = ""
        for line in raw_response.strip().split("\n"):
            line_stripped = line.strip()
            if line_stripped.upper().startswith(prefix.upper()):
                # Extract value after the prefix
                value = line_stripped[len(prefix):].strip()
                result[key] = value
                break

    # Normalize verdict
    verdict_upper = result["verdict"].upper().strip()
    if verdict_upper in ("CONTINUE", "UNVIABLE"):
        result["verdict"] = verdict_upper
    else:
        # Safe default if verdict is malformed or missing
        logger.warning(
            f"Critic returned unclear verdict: {result['verdict']!r}, "
            f"defaulting to UNVIABLE"
        )
        result["verdict"] = "UNVIABLE"

    # If any critical field is empty, default to UNVIABLE
    if not result["primary_failure"] or not result["surgical_fix"]:
        logger.warning(
            "Critic response missing primary_failure or surgical_fix, "
            "defaulting to UNVIABLE"
        )
        result["verdict"] = "UNVIABLE"

    return result


# ── Public API ────────────────────────────────────────────────────────────────

async def run_critic(
    strategy_code: str,
    diagnostics_df: pd.DataFrame,
) -> dict:
    """
    Run the Evidence-Locked Critic on a strategy.

    Filters diagnostics to sufficient_evidence=True rows, sends to
    Claude Opus with the Critic prompt, and parses the structured response.

    Parameters
    ----------
    strategy_code : str
        Full Python source code of the strategy being evaluated.
    diagnostics_df : pd.DataFrame
        Full 60-row diagnostics output from DiagnosticsEngine.

    Returns
    -------
    dict
        {
            'primary_failure': str,
            'root_cause': str,
            'surgical_fix': str,
            'expected_impact': str,
            'verdict': 'CONTINUE' | 'UNVIABLE'
        }
    """
    # Format inputs
    table_text = _format_diagnostics_table(diagnostics_df)

    user_message = (
        f"═══ STRATEGY CODE ═══\n"
        f"{strategy_code}\n\n"
        f"═══ DIAGNOSTIC TABLE (sufficient_evidence=True only) ═══\n"
        f"{table_text}"
    )

    # Call Opus
    try:
        client = anthropic.AsyncAnthropic()
        response = await client.messages.create(
            model=OPUS_MODEL,
            max_tokens=1024,
            temperature=CRITIC_TEMPERATURE,
            system=CRITIC_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        raw_text = response.content[0].text
        logger.debug(f"Critic raw response:\n{raw_text}")

    except Exception as e:
        logger.error(f"Critic API call failed: {type(e).__name__}: {e}")
        return {
            "primary_failure": "",
            "root_cause": f"Critic API error: {e}",
            "surgical_fix": "",
            "expected_impact": "",
            "verdict": "UNVIABLE",
        }

    # Parse response
    result = _parse_critic_response(raw_text)

    logger.info(
        f"Critic verdict: {result['verdict']} | "
        f"fix: {result['surgical_fix'][:80]}"
    )

    return result


__all__ = ["run_critic"]


# ── Smoke Test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print("[SMOKE TEST] critic_agent.py\n")

    # ── Test 1: Parse well-formed CONTINUE response ───────────────────────
    print("=== Test 1: Well-formed CONTINUE response ===")
    good_response = """\
PRIMARY_FAILURE: [ASIA|DOWNTREND|HIGH_VOL] | sharpe=-0.42 | n=87
ROOT_CAUSE: RSI oversold threshold at 30 triggers too many false entries in high-volatility Asian session
SURGICAL_FIX: Change RSI oversold threshold from 30 to 25 in generate_signals()
EXPECTED_IMPACT: ASIA|DOWNTREND|HIGH_VOL sharpe improves from -0.42 toward 0 by filtering weak entries
VERDICT: CONTINUE"""

    parsed = _parse_critic_response(good_response)
    assert parsed["verdict"] == "CONTINUE"
    assert "ASIA" in parsed["primary_failure"]
    assert "sharpe=-0.42" in parsed["primary_failure"]
    assert "30" in parsed["surgical_fix"] and "25" in parsed["surgical_fix"]
    assert parsed["root_cause"] != ""
    assert parsed["expected_impact"] != ""
    print(f"  primary_failure: {parsed['primary_failure']}")
    print(f"  root_cause:      {parsed['root_cause'][:70]}...")
    print(f"  surgical_fix:    {parsed['surgical_fix']}")
    print(f"  expected_impact: {parsed['expected_impact'][:70]}...")
    print(f"  verdict:         {parsed['verdict']}")
    print("  OK")

    # ── Test 2: Parse well-formed UNVIABLE response ───────────────────────
    print("\n=== Test 2: Well-formed UNVIABLE response ===")
    unviable_response = """\
PRIMARY_FAILURE: [GLOBAL] | sharpe=-0.72 | n=1423
ROOT_CAUSE: Strategy is fundamentally unprofitable across all regimes
SURGICAL_FIX: N/A - strategy is unviable
EXPECTED_IMPACT: N/A
VERDICT: UNVIABLE"""

    parsed2 = _parse_critic_response(unviable_response)
    assert parsed2["verdict"] == "UNVIABLE"
    assert "GLOBAL" in parsed2["primary_failure"]
    print(f"  verdict: {parsed2['verdict']}")
    print("  OK")

    # ── Test 3: Malformed response → safe default UNVIABLE ────────────────
    print("\n=== Test 3: Malformed response ===")
    malformed = "I think the strategy needs a complete rewrite. The RSI is broken."
    parsed3 = _parse_critic_response(malformed)
    assert parsed3["verdict"] == "UNVIABLE"
    print(f"  verdict: {parsed3['verdict']} (safe default)")
    print("  OK")

    # ── Test 4: Partial response (missing fields) ─────────────────────────
    print("\n=== Test 4: Partial response (missing surgical_fix) ===")
    partial = """\
PRIMARY_FAILURE: [NY|UPTREND|LOW_VOL] | sharpe=-0.15 | n=42
ROOT_CAUSE: EMA period too short
VERDICT: CONTINUE"""

    parsed4 = _parse_critic_response(partial)
    assert parsed4["verdict"] == "UNVIABLE", "Missing surgical_fix should force UNVIABLE"
    print(f"  verdict: {parsed4['verdict']} (forced UNVIABLE due to missing fix)")
    print("  OK")

    # ── Test 5: Diagnostics table formatting ──────────────────────────────
    print("\n=== Test 5: Diagnostics table formatting ===")
    import numpy as np
    fake_diag = pd.DataFrame([
        {"granularity": "GLOBAL", "session": "ALL", "trend_regime": "ALL",
         "vol_regime": "ALL", "trade_count": 500, "win_rate": 0.55,
         "sharpe": 0.42, "max_consecutive_losses": 5,
         "sufficient_evidence": True},
        {"granularity": "3D", "session": "ASIA", "trend_regime": "UPTREND",
         "vol_regime": "HIGH_VOL", "trade_count": 45, "win_rate": 0.60,
         "sharpe": 0.85, "max_consecutive_losses": 3,
         "sufficient_evidence": True},
        {"granularity": "3D", "session": "NY", "trend_regime": "DOWNTREND",
         "vol_regime": "LOW_VOL", "trade_count": 10, "win_rate": 0.40,
         "sharpe": -0.30, "max_consecutive_losses": 4,
         "sufficient_evidence": False},  # should be filtered out
    ])

    table_text = _format_diagnostics_table(fake_diag)
    assert "GLOBAL" in table_text
    assert "ASIA" in table_text
    # The NY row has sufficient_evidence=False, should NOT appear
    assert "NY" not in table_text, "NY row (insufficient evidence) should be filtered"
    print(f"  Table output:\n{table_text}")
    print("  Filtered insufficient evidence rows: OK")

    # ── Test 6: Empty diagnostics (all insufficient) ──────────────────────
    print("\n=== Test 6: Empty diagnostics (all insufficient) ===")
    empty_diag = pd.DataFrame([
        {"granularity": "GLOBAL", "session": "ALL", "trend_regime": "ALL",
         "vol_regime": "ALL", "trade_count": 5, "win_rate": 0.40,
         "sharpe": -0.10, "max_consecutive_losses": 3,
         "sufficient_evidence": False},
    ])
    empty_table = _format_diagnostics_table(empty_diag)
    assert "No rows" in empty_table
    print(f"  Output: {empty_table}")
    print("  OK")

    print("\nPASS")
