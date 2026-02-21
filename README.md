# Council of Alphas

## READ THIS FIRST

You are building the Council of Alphas — an evolutionary multi-agent trading framework.

## Your First 3 Actions

1. **Read** `docs/MASTER_SPEC.md` — every locked decision is in here
2. **Read** `docs/BUILD_GUIDE.md` — exact step-by-step instructions
3. **Read** `core/config.py` — all constants before touching any other file

## What Already Exists (DO NOT rewrite core logic)

| File | Status | Action |
|---|---|---|
| `core/whitelist_indicators.py` | ✅ Ready | Do not touch |
| `core/labeling.py` | ✅ Ready | Do not touch |
| `core/backtesting.py` | ✅ Ready | Do not touch |
| `core/diagnostics.py` | ✅ Ready | Do not touch |
| `core/state_builder.py` | ✅ Parameters updated | Do not touch |
| `pipeline/indicator_sampler.py` | ✅ Ready | Do not touch |
| `core/strategy_base.py` | ✅ Ready | Do not touch |
| `core/config.py` | ✅ Ready | Do not touch |
| `pipeline/fitness.py` | ✅ Ready | Do not touch |
| `pipeline/prompt_builder.py` | ✅ Ready | Do not touch |

## What You Need to Build

```
pipeline/specialist_agent.py   → async strategy generation + validation + retry
pipeline/niche_selector.py     → champion selection per family
pipeline/hybrid_builder.py     → HybridBuilder (3 deterministic templates, pure Python)
agents/critic_agent.py         → Claude Opus evidence-locked diagnosis
agents/refiner_agent.py        → Claude Sonnet surgical fix application
agents/scientist.py            → full refinement loop (max 5 iterations)
orchestrator.py                → main pipeline controller
app.py                         → Streamlit UI (Andreas)
```

## Repo Structure

```
council_of_alphas/
│
├── .env
├── .gitignore
├── README.md
├── requirements.txt
├── orchestrator.py               ← to build
├── app.py                        ← to build
│
├── core/
│   ├── __init__.py
│   ├── config.py
│   ├── strategy_base.py
│   ├── whitelist_indicators.py
│   ├── labeling.py
│   ├── backtesting.py
│   ├── diagnostics.py
│   └── state_builder.py
│
├── pipeline/
│   ├── __init__.py
│   ├── indicator_sampler.py
│   ├── fitness.py
│   ├── prompt_builder.py
│   ├── niche_selector.py
│   ├── hybrid_builder.py
│   └── specialist_agent.py       ← to build
│
├── agents/
│   ├── __init__.py
│   ├── critic_agent.py           ← to build
│   ├── refiner_agent.py          ← to build
│   └── scientist.py              ← to build
│
├── eda/
│   └── eda.ipynb                    ← state matrix EDA (regime coverage, TBM edge, drift)
│
├── data/
│   ├── README.md
│   ├── sol_usd_15m_3y.parquet
│   └── state_matrix.parquet
│
└── docs/
    ├── Council_of_Alphas_SPEC.pdf
    ├── MASTER_SPEC.md
    ├── BUILD_GUIDE.md
    ├── ARCHITECTURE.md
    └── RESEARCH.md
```

## 10 Rules That Must Never Be Broken

1. All constants come from `core/config.py` — no magic numbers anywhere
2. `tbm_win` and `tbm_loss` are NOT on Strategy objects — core/config.py only
3. State Matrix is read-only after build — never modify in place
4. Fitness formula is exactly: `Global_Sharpe * ln(N) * Coverage` — no deviations
5. Coverage is trade-weighted (by trade_count), not bucket-count-weighted
6. Critic receives ONLY `sufficient_evidence=True` rows
7. HybridBuilder is pure Python — zero LLM calls
8. All 3 hybrids use all 4 champions — no 2-champion hybrids
9. Inline hybrid code only — no compositional/import-based hybrids
10. Specialists run parallel (`asyncio.gather`), Scientist loop runs sequential
