# Council of Alphas

## READ THIS FIRST

You are building the Council of Alphas — an evolutionary multi-agent trading framework.

## Your First 3 Actions

1. **Read** `docs/MASTER_SPEC.md` — every locked decision is in here
2. **Read** `docs/BUILD_GUIDE.md` — exact step-by-step instructions
3. **Read** `config.py` — all constants before touching any other file

## What Already Exists (DO NOT rewrite core logic)

| File | Status | Action |
|---|---|---|
| `whitelist_indicators.py` | ✅ Ready | Do not touch |
| `labeling.py` | ✅ Ready | Do not touch |
| `backtesting.py` | ✅ Ready | Do not touch |
| `diagnostics.py` | ✅ Ready | Do not touch |
| `state_builder.py` | ✅ Parameters updated | Do not touch |
| `indicator_sampler.py` | ✅ Ready | Do not touch |
| `strategy_base.py` | ✅ Ready | Do not touch |
| `config.py` | ✅ Ready | Do not touch |
| `fitness.py` | ✅ Ready | Do not touch |
| `prompt_builder.py` | ✅ Ready | Do not touch |

## What You Need to Build

```
specialist_agent.py   → async strategy generation + validation + retry
niche_selector.py     → champion selection per family
hybrid_builder.py     → HybridBuilder (3 deterministic templates, pure Python)
critic_agent.py       → Claude Opus evidence-locked diagnosis
refiner_agent.py      → Claude Sonnet surgical fix application
scientist.py          → full refinement loop (max 5 iterations)
orchestrator.py       → main pipeline controller
app.py                → Streamlit UI (Andreas)
```

## Repo Structure

```
council_of_alphas/
│
├── .env
├── .gitignore
├── README.md
├── requirements.txt
│
├── config.py
├── strategy_base.py
├── whitelist_indicators.py
├── labeling.py
├── backtesting.py
├── diagnostics.py
├── state_builder.py
├── indicator_sampler.py
├── fitness.py
├── prompt_builder.py
│
├── specialist_agent.py           ← to build
├── niche_selector.py             ← to build
├── hybrid_builder.py             ← to build
├── critic_agent.py               ← to build
├── refiner_agent.py              ← to build
├── scientist.py                  ← to build
├── orchestrator.py               ← to build
├── app.py                        ← to build
│
├── data/
│   ├── README.md
│   ├── btc_usd_15m_3y.parquet
│   └── state_matrix.parquet
│
└── docs/
    ├── Council_of_Alphas_SPEC.pdf
    ├── MASTER_SPEC.md
    ├── BUILD_GUIDE.md
    └── ARCHITECTURE.md
```

## 10 Rules That Must Never Be Broken

1. All constants come from `config.py` — no magic numbers anywhere
2. `tbm_win` and `tbm_loss` are NOT on Strategy objects — config.py only
3. State Matrix is read-only after build — never modify in place
4. Fitness formula is exactly: `Global_Sharpe * ln(N) * Coverage` — no deviations
5. Coverage is trade-weighted (by trade_count), not bucket-count-weighted
6. Critic receives ONLY `sufficient_evidence=True` rows
7. HybridBuilder is pure Python — zero LLM calls
8. All 3 hybrids use all 4 champions — no 2-champion hybrids
9. Inline hybrid code only — no compositional/import-based hybrids
10. Specialists run parallel (`asyncio.gather`), Scientist loop runs sequential
