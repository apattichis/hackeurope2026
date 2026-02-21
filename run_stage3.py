"""
Stage 3: Full pipeline run with comprehensive result saving.

Runs the complete Council of Alphas pipeline and saves all results to data/results/:
- speciation/: all 12 strategies (3 per family x 4 families)
- champions/: 4 champion strategies
- ranked/: final ranked strategies (hybrids after Scientist)
- summary.json: top-level overview

Each strategy folder contains:
- summary.json: key metrics
- diagnostics.csv: full 60-row diagnostics table
- trade_log.csv: every trade with entry/exit/pnl/regime
- strategy_code.py: source code
"""

import sys
import asyncio
import logging
import shutil
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent
for p in [str(PROJECT_ROOT / "core"), str(PROJECT_ROOT / "pipeline"),
          str(PROJECT_ROOT / "agents")]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Force UTF-8 for Windows
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from orchestrator import Orchestrator, save_pipeline_results

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)


async def main():
    print("=" * 60)
    print("STAGE 3: Full Pipeline + Result Saving")
    print("=" * 60)

    # Clean previous results
    results_dir = PROJECT_ROOT / "data" / "results"
    if results_dir.exists():
        shutil.rmtree(results_dir)
        print(f"\nCleaned previous results: {results_dir}")

    # Run pipeline
    orch = Orchestrator()
    result = await orch.run()

    # Save everything
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)

    saved_to = save_pipeline_results(
        result,
        all_speciation=result.all_speciation,
        results_dir="data/results",
    )

    # Print summary
    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)

    print(f"\nElapsed: {result.elapsed_seconds:.1f}s")
    print(f"Fallback used: {result.fallback_used}")
    print(f"Hybrids built: {result.hybrids_built}")
    print(f"Hybrids survived: {result.hybrids_survived}")

    # Speciation summary
    if result.all_speciation:
        print(f"\n--- Speciation ({sum(len(v) for v in result.all_speciation.values())} strategies) ---")
        for family, strats in result.all_speciation.items():
            print(f"\n  {family.upper()} ({len(strats)} strategies):")
            for strategy, score, diag in strats:
                g = diag[diag["granularity"] == "GLOBAL"].iloc[0]
                active_3d = diag[(diag["granularity"] == "3D") & (diag["sufficient_evidence"] == True)]
                prof_3d = active_3d[active_3d["sharpe"] > 0]
                print(
                    f"    {strategy.name}: "
                    f"fitness={score:.4f}, "
                    f"sharpe={g['sharpe']:.4f}, "
                    f"WR={g['win_rate']:.3f}, "
                    f"trades={int(g['trade_count'])}, "
                    f"profitable_3D={len(prof_3d)}/{len(active_3d)}"
                )

    # Champions
    print(f"\n--- Champions ({len(result.champions)}) ---")
    for family, (strategy, score, diag) in result.champions.items():
        g = diag[diag["granularity"] == "GLOBAL"].iloc[0]
        active_3d = diag[(diag["granularity"] == "3D") & (diag["sufficient_evidence"] == True)]
        prof_3d = active_3d[active_3d["sharpe"] > 0]
        trade_log = getattr(strategy, "_trade_log", None)
        n_trades_log = len(trade_log) if trade_log is not None else "N/A"
        print(
            f"  {family}: {strategy.name} | "
            f"fitness={score:.4f} | "
            f"sharpe={g['sharpe']:.4f} | "
            f"WR={g['win_rate']:.3f} | "
            f"trades={int(g['trade_count'])} | "
            f"max_consec_loss={int(g['max_consecutive_losses'])} | "
            f"profitable_3D={len(prof_3d)}/{len(active_3d)}"
        )

    # Ranked
    print(f"\n--- Final Ranking ({len(result.ranked)}) ---")
    for rank, (strategy, score, diag) in enumerate(result.ranked, 1):
        g = diag[diag["granularity"] == "GLOBAL"].iloc[0]
        active_3d = diag[(diag["granularity"] == "3D") & (diag["sufficient_evidence"] == True)]
        prof_3d = active_3d[active_3d["sharpe"] > 0]
        print(
            f"  #{rank}: {strategy.name} ({strategy.family}) | "
            f"fitness={score:.4f} | "
            f"sharpe={g['sharpe']:.4f} | "
            f"WR={g['win_rate']:.3f} | "
            f"trades={int(g['trade_count'])} | "
            f"max_consec_loss={int(g['max_consecutive_losses'])} | "
            f"profitable_3D={len(prof_3d)}/{len(active_3d)}"
        )

    # Files saved
    print(f"\n--- Results saved to: {saved_to} ---")
    for p in sorted(saved_to.rglob("*")):
        if p.is_file():
            size_kb = p.stat().st_size / 1024
            rel = p.relative_to(saved_to)
            print(f"  {rel} ({size_kb:.1f} KB)")

    print("\nDONE")


if __name__ == "__main__":
    asyncio.run(main())
