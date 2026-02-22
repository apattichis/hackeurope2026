"""
generate_data_js.py — Reads pipeline results and writes ui/src/data.js
Run from the hackeurope2026 root directory:
    python generate_data_js.py
"""

import json
import os
import re
import math
import pandas as pd
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(ROOT, "data", "results")
OUT_PATH = os.path.join(ROOT, "ui", "src", "data.js")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def fmt(val, decimals=4):
    """Round a float to `decimals` places."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return 0.0
    return round(float(val), decimals)


def pct(val, decimals=2):
    """Convert a 0-1 ratio to a percentage, rounded."""
    return round(float(val) * 100, decimals)


DISPLAY_NAMES = {
    "consensus_gate": "Consensus Gate",
    "regime_router": "Regime Router",
    "weighted_combination": "Weighted Combination",
}

# ---------------------------------------------------------------------------
# 1. config
# ---------------------------------------------------------------------------
summary = load_json(os.path.join(RESULTS, "summary.json"))

# Read actual bar count from state matrix
state_matrix_path = os.path.join(ROOT, "data", "state_matrix_1h.parquet")
if os.path.exists(state_matrix_path):
    sm = pd.read_parquet(state_matrix_path)
    bars = len(sm)
else:
    bars = 36311

config = {
    "asset": "SOL/USD",
    "timeframe": "1h",
    "dateRange": "Jan 2022 - Feb 2026",
    "bars": bars,
    "initialCapital": 100000,
    "riskPerTrade": 0.005,
    "fee": 0.00075,
    "tbmWin": 2.0,
    "tbmLoss": 1.0,
    "tbmHorizon": 24,
    "model": "Claude Opus",
    "runtime": fmt(summary["elapsed_seconds"], 1),
}

# ---------------------------------------------------------------------------
# 2. familyIndicators
# ---------------------------------------------------------------------------
familyIndicators = {
    "trend": ["ema", "hma", "macd", "adx", "slope"],
    "momentum": ["rsi", "cci", "roc", "mfi", "zscore"],
    "volatility": ["natr", "bollinger_bands", "keltner_channels", "choppiness_index"],
    "volume": ["vwap", "obv", "cmf"],
}

# ---------------------------------------------------------------------------
# 3. speciation
# ---------------------------------------------------------------------------
spec_summary = load_json(os.path.join(RESULTS, "speciation", "speciation_summary.json"))
champ_summary = load_json(os.path.join(RESULTS, "champions", "champions_summary.json"))

# Build set of champion (name, trades) tuples for unique identification
# (volatility family has 3 strategies with the same name but different trade counts)
champion_keys = set()
for fam, champ in champ_summary.items():
    champion_keys.add((champ["name"], champ["trades"]))

speciation = {}
for family, strategies in spec_summary.items():
    speciation[family] = []
    for s in strategies:
        speciation[family].append({
            "name": s["name"],
            "family": s["family"],
            "description": s["description"],
            "fitness": fmt(s["fitness"]),
            "sharpe": fmt(s["sharpe"]),
            "winRate": pct(s["win_rate"]),
            "trades": s["trades"],
            "isChampion": (s["name"], s["trades"]) in champion_keys,
        })

# ---------------------------------------------------------------------------
# 4. champions
# ---------------------------------------------------------------------------
# Map family -> indicators extracted from strategy code
CHAMPION_INDICATORS = {
    "trend": ["EMA(12,26,50)", "ADX(14)", "Slope(10,25)"],
    "momentum": ["RSI(7,14,21)", "ROC(5,12,24)"],
    "volatility": ["BB(20,2.0)", "KC(20,1.5)", "KC(20,2.5)", "Choppiness(14)", "NATR(7,14)"],
    "volume": ["OBV", "VWAP(24)", "VWAP(48)"],
}

champions = []
for family in ["trend", "momentum", "volatility", "volume"]:
    c = champ_summary[family]
    champions.append({
        "name": c["name"],
        "family": family,
        "description": c["description"],
        "fitness": fmt(c["fitness"]),
        "sharpe": fmt(c["sharpe"]),
        "winRate": pct(c["win_rate"]),
        "trades": c["trades"],
        "indicators": CHAMPION_INDICATORS[family],
    })

# ---------------------------------------------------------------------------
# 5. hybrids
# ---------------------------------------------------------------------------
ranked_summary = load_json(os.path.join(RESULTS, "ranked", "ranked_summary.json"))

HYBRID_DIRS = {
    "consensus_gate": "1_consensus_gate",
    "regime_router": "2_regime_router",
    "weighted_combination": "3_weighted_combination",
}

hybrids = []
for h in ranked_summary:
    name = h["name"]
    hdir = os.path.join(RESULTS, "ranked", HYBRID_DIRS[name])

    # Read trade log
    trade_log = pd.read_csv(os.path.join(hdir, "trade_log.csv"))

    # Parse entry_ts as datetime
    trade_log["entry_ts"] = pd.to_datetime(trade_log["entry_ts"], utc=True)
    trade_log["exit_ts"] = pd.to_datetime(trade_log["exit_ts"], utc=True)

    # Compute equity curve: group by date (of exit), take LAST account_balance
    trade_log["date"] = trade_log["exit_ts"].dt.strftime("%Y-%m-%d")
    eq_grouped = trade_log.groupby("date")["account_balance"].last().reset_index()
    equity_curve = [
        {"date": row["date"], "balance": fmt(row["account_balance"], 2)}
        for _, row in eq_grouped.iterrows()
    ]

    # Compute drawdown from EVERY trade (per-trade granularity, matches notebook)
    peak = trade_log["account_balance"].cummax()
    dd_per_trade = (trade_log["account_balance"] - peak) / peak * 100  # percentage

    # Daily drawdown for the chart: take worst intra-day drawdown
    trade_log["_dd"] = dd_per_trade.values
    dd_daily = trade_log.groupby("date")["_dd"].min()
    drawdown = [
        {"date": d, "dd": fmt(v, 2)}
        for d, v in dd_daily.items()
    ]

    # Max drawdown (per-trade, not daily)
    max_drawdown = fmt(dd_per_trade.min(), 2)

    # Total return
    final_balance = trade_log["account_balance"].iloc[-1]
    total_return = fmt((final_balance / 100000 - 1) * 100, 2)

    # Average hold hours (duration column = bars, 1 bar = 1 hour)
    avg_hold_hours = fmt(trade_log["duration"].mean(), 1)

    # Longs / shorts
    longs = int((trade_log["side"] == 1).sum())
    shorts = int((trade_log["side"] == -1).sum())

    # Annualized Sharpe from daily returns (ALL calendar days, matching notebook)
    daily_returns = (
        trade_log.set_index("entry_ts")["net_trade_return"]
        .resample("1D").sum().fillna(0)
    )
    if len(daily_returns) > 1 and daily_returns.std() > 0:
        ann_sharpe = fmt(daily_returns.mean() / daily_returns.std() * math.sqrt(365), 4)
    else:
        ann_sharpe = 0.0

    # Diagnostics
    diag_df = pd.read_csv(os.path.join(hdir, "diagnostics.csv"))
    diagnostics = []
    for _, row in diag_df.iterrows():
        diagnostics.append({
            "granularity": row["granularity"],
            "session": row["session"],
            "trendRegime": row["trend_regime"],
            "volRegime": row["vol_regime"],
            "tradeCount": int(row["trade_count"]),
            "winRate": pct(row["win_rate"]),
            "sharpe": fmt(row["sharpe"]),
            "sufficientEvidence": bool(row["sufficient_evidence"]),
        })

    # Filtered buckets from strategy_code.py
    code = read_file(os.path.join(hdir, "strategy_code.py"))
    filtered_buckets = []
    if name != "weighted_combination":
        bucket_pattern = re.compile(r"#\s+\((.+?)\):\s+sharpe=")
        in_section = False
        for line in code.split("\n"):
            if "Filtered (non-tradable) 2D buckets:" in line:
                in_section = True
                continue
            if in_section:
                m = bucket_pattern.search(line)
                if m:
                    filtered_buckets.append(m.group(1))
                elif line.strip() == "#" or not line.strip().startswith("#"):
                    in_section = False

    # Strategy source code
    strategy_code = code  # already read above for filtered_buckets parsing

    hybrids.append({
        "rank": h["rank"],
        "name": name,
        "displayName": DISPLAY_NAMES[name],
        "description": h["description"],
        "fitness": fmt(h["fitness"]),
        "sharpe": fmt(h["sharpe"]),
        "winRate": pct(h["win_rate"]),
        "trades": h["trades"],
        "maxConsecLosses": h["max_consec_losses"],
        "annSharpe": ann_sharpe,
        "totalReturn": total_return,
        "maxDrawdown": max_drawdown,
        "avgHoldHours": avg_hold_hours,
        "longs": longs,
        "shorts": shorts,
        "strategyCode": strategy_code,
        "equityCurve": equity_curve,
        "drawdown": drawdown,
        "diagnostics": diagnostics,
        "filteredBuckets": filtered_buckets,
    })

# Sort by rank
hybrids.sort(key=lambda x: x["rank"])

# ---------------------------------------------------------------------------
# 6. routingTable
# ---------------------------------------------------------------------------
router_code = read_file(os.path.join(RESULTS, "ranked", "2_regime_router", "strategy_code.py"))

routing_pattern = re.compile(
    r"\('(\w+)',\s*'(\w+)',\s*'(\w+)'\):\s*'(\w+)'"
)
routing_table = []
for m in routing_pattern.finditer(router_code):
    routing_table.append({
        "session": m.group(1),
        "trend": m.group(2),
        "vol": m.group(3),
        "champion": m.group(4),
    })

# ---------------------------------------------------------------------------
# 7. correlation
# ---------------------------------------------------------------------------
# Compute daily PnL correlation for hybrids with sharpe > 0
positive_hybrids = [h for h in hybrids if h["sharpe"] > 0]
positive_names = [h["displayName"] for h in positive_hybrids]

if len(positive_hybrids) >= 2:
    # Build daily returns for each positive hybrid (ALL calendar days, consistent with Sharpe)
    daily_pnl = {}
    for h in positive_hybrids:
        hdir = os.path.join(RESULTS, "ranked", HYBRID_DIRS[h["name"]])
        tl = pd.read_csv(os.path.join(hdir, "trade_log.csv"))
        tl["entry_ts"] = pd.to_datetime(tl["entry_ts"], utc=True)
        daily_pnl[h["displayName"]] = (
            tl.set_index("entry_ts")["net_trade_return"]
            .resample("1D").sum().fillna(0)
        )

    # Align to common date range, fill missing with 0
    df_daily = pd.DataFrame(daily_pnl).fillna(0)

    # Compute Pearson correlation matrix
    corr_matrix = df_daily.corr().values.tolist()
    # Round
    corr_matrix = [[fmt(v, 4) for v in row] for row in corr_matrix]
else:
    corr_matrix = [[1.0]]

correlation = {
    "strategies": positive_names,
    "matrix": corr_matrix,
}

# ---------------------------------------------------------------------------
# Write data.js
# ---------------------------------------------------------------------------

def to_js_export(name, obj):
    """Convert a Python object to a JS export const statement."""
    json_str = json.dumps(obj, indent=2, ensure_ascii=False)
    return f"export const {name} = {json_str};\n"


with open(OUT_PATH, "w", encoding="utf-8") as f:
    f.write("// Auto-generated by generate_data_js.py — DO NOT EDIT MANUALLY\n\n")
    f.write(to_js_export("config", config))
    f.write("\n")
    f.write(to_js_export("familyIndicators", familyIndicators))
    f.write("\n")
    f.write(to_js_export("speciation", speciation))
    f.write("\n")
    f.write(to_js_export("champions", champions))
    f.write("\n")
    f.write(to_js_export("hybrids", hybrids))
    f.write("\n")
    f.write(to_js_export("routingTable", routing_table))
    f.write("\n")
    f.write(to_js_export("correlation", correlation))

print(f"Written {OUT_PATH}")
print(f"  config: {len(config)} keys")
print(f"  familyIndicators: {len(familyIndicators)} families")
print(f"  speciation: {sum(len(v) for v in speciation.values())} strategies across {len(speciation)} families")
print(f"  champions: {len(champions)}")
print(f"  hybrids: {len(hybrids)}")
print(f"  routingTable: {len(routing_table)} entries")
print(f"  correlation: {len(correlation['strategies'])} strategies")
