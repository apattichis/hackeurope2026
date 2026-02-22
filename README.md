# Council of Alphas

<p align="center">
  <img src="img/council_of_alphas.png" alt="Council of Alphas" width="700">
</p>

**HackEurope 2026 | Team: The Greeks (Andreas + Markos)**

**Live Dashboard**: [council-of-alphas.vercel.app](https://council-of-alphas.vercel.app)

An evolutionary multi-agent trading framework that prevents mode collapse in LLM strategy generation through enforced specialist diversity, niche-preserving selection, deterministic hybrid construction, and regime-aware filtering.

### Why "Council of Alphas"?

- **"Council"** - 4 specialist agents each contribute their domain expertise, then their outputs are merged through hybrid templates and filtered. A council of specialists whose collective intelligence produces the final result.
- **"Alphas"** - The surviving strategies that generate financial alpha (excess returns). The whole pipeline is an evolutionary funnel that produces alphas.

## Quick Start

1. **Read** `docs/MASTER_SPEC.md` - every locked decision
2. **Read** `core/config.py` - all constants
3. **Run** `python run_stage3.py` - full pipeline

## Pipeline Overview

```
12 strategies (3/family) -> 4 champions (1/family) -> 3 hybrids -> 2D regime filter -> ranked survivors
```

- **Data**: SOL-USD 1h candles (~36k bars, Jan 2022 - Feb 2026)
- **Families**: trend, momentum, volatility, volume
- **Hybrids**: Regime Router, Consensus Gate, Weighted Combination
- **Backtesting**: $100k initial, 0.5% risk/trade, 0.075% fee, TBM 2.0/1.0/24
- **Model**: Claude Opus (specialists only)

## Repo Structure

```
hackeurope2026/
|
├── .env                          # API keys (ANTHROPIC_API_KEY)
├── .gitignore
├── README.md
├── requirements.txt
├── orchestrator.py               # Main pipeline controller
├── run_stage3.py                 # CLI entry point
|
├── core/
|   ├── config.py                 # All constants (single source of truth)
|   ├── strategy_base.py          # Strategy base class (extends Indicators)
|   ├── whitelist_indicators.py   # Full indicator library
|   ├── labeling.py               # Triple Barrier Method labeling
|   ├── backtesting.py            # VectorizedBacktester (numba-accelerated)
|   ├── diagnostics.py            # Hierarchical Sharpe/WR across regime buckets
|   └── state_builder.py          # Builds state matrix (OHLCV + regimes + TBM)
|
├── pipeline/
|   ├── indicator_sampler.py      # Random indicator subset per strategy
|   ├── fitness.py                # Fitness = Global_Sharpe * ln(N) * Coverage
|   ├── prompt_builder.py         # Specialist prompt construction
|   ├── specialist_agent.py       # LLM strategy generation + validation + retry
|   ├── niche_selector.py         # Champion selection (top 1 per family)
|   └── hybrid_builder.py         # 3 hybrid templates (pure Python)
|
├── agents/
|   └── optimizer.py              # Deterministic 2D regime filter
|
├── ui/                           # React + Vite dashboard
|   └── src/
|       └── components/           # Leaderboard, LineageTree, TearSheet, etc.
|
├── img/
|   └── council_of_alphas.png     # Project banner
|
├── eda/
|   ├── eda.ipynb                 # State matrix EDA (regime coverage, TBM edge)
|   └── winners_analysis.ipynb    # Post-run analysis of winning strategies
|
├── data/
|   ├── sol_usd_1h.parquet        # Raw 1h candle data
|   ├── state_matrix_1h.parquet   # Pre-built state matrix (21 columns)
|   └── results/                  # Pipeline output (speciation, champions, ranked)
|
└── docs/
    ├── MASTER_SPEC.md
    ├── ARCHITECTURE.md
    └── RESEARCH.md
```

## 10 Rules That Must Never Be Broken

1. All constants come from `core/config.py` - no magic numbers anywhere
2. `tbm_win` and `tbm_loss` are NOT on Strategy objects - core/config.py only
3. State Matrix is read-only after build - never modify in place
4. Fitness formula is exactly: `Global_Sharpe * ln(N) * Coverage` - no deviations
5. Coverage is trade-weighted (by trade_count), not bucket-count-weighted
6. Optimizer keeps signals only where `sufficient_evidence=True AND sharpe > 0`
7. HybridBuilder is pure Python - zero LLM calls
8. All 3 hybrids use all surviving champions - no partial champion hybrids
9. Inline hybrid code only - no compositional/import-based hybrids
10. Specialists run parallel (`asyncio.gather`), optimizer runs parallel per hybrid

## The Greeks

Named after the Greeks in quantitative finance (delta, gamma, theta, vega) - and because we actually are Greek.

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/apattichis">
        <img src="https://github.com/apattichis.png" width="100" style="border-radius:50%"><br>
        <sub><b>Andreas Pattichis</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/MarkosMarkides">
        <img src="https://github.com/MarkosMarkides.png" width="100" style="border-radius:50%"><br>
        <sub><b>Markos Markides</b></sub>
      </a>
    </td>
  </tr>
</table>
