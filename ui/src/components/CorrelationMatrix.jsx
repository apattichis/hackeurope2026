import { correlation, hybrids } from '../data.js';

function getCellStyle(value, isHovered) {
  const base = isHovered ? 0.15 : 0;

  if (value >= 0.99) {
    return {
      backgroundColor: isHovered ? 'rgb(37, 99, 235)' : 'rgb(30, 64, 175)',
      color: '#fff',
    };
  }
  if (value >= 0.3 && value < 0.5) {
    const alpha = isHovered ? 0.55 : 0.4;
    return {
      backgroundColor: `rgba(59, 130, 246, ${alpha})`,
      color: '#e2e8f0',
    };
  }
  if (value >= 0 && value < 0.3) {
    const alpha = isHovered ? 0.32 : 0.2;
    return {
      backgroundColor: `rgba(96, 165, 250, ${alpha})`,
      color: '#e2e8f0',
    };
  }
  // Negative
  const alpha = isHovered ? 0.45 : 0.3;
  return {
    backgroundColor: `rgba(244, 63, 94, ${alpha})`,
    color: '#fecdd3',
  };
}

function getInterpretation(value) {
  if (value >= 0.7) return 'Strong positive correlation';
  if (value >= 0.5) return 'Moderate-high positive correlation';
  if (value >= 0.3) return 'Moderate positive correlation';
  if (value >= 0.1) return 'Weak positive correlation';
  if (value >= -0.1) return 'Near-zero correlation';
  if (value >= -0.3) return 'Weak negative correlation';
  return 'Strong negative correlation';
}

import React, { useState } from 'react';

export default function CorrelationMatrix() {
  const [hoveredCell, setHoveredCell] = useState(null);

  const strategies = correlation.strategies;
  const matrix = correlation.matrix;

  const profitableHybrids = hybrids
    .filter((h) => h.sharpe > 0)
    .sort((a, b) => a.rank - b.rank);

  const offDiagonalValue = matrix[0][1];

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 p-6">
      {/* Header */}
      <div className="mb-8">
        <h2 className="text-2xl font-bold text-slate-100 tracking-tight">
          PnL Correlation Matrix
        </h2>
        <p className="mt-1 text-sm text-slate-400">
          Daily return correlation between profitable hybrids (Sharpe &gt; 0)
        </p>
      </div>

      {/* Main content: matrix + analysis */}
      <div className="flex flex-col lg:flex-row gap-6 mb-6">
        {/* Correlation Matrix Visual */}
        <div className="flex-shrink-0">
          <div className="bg-slate-900 border border-slate-700 rounded-xl p-6">
            <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-widest mb-5">
              Correlation Matrix
            </h3>

            {/* Grid */}
            <div className="flex">
              {/* Top-left spacer + row labels column */}
              <div className="flex flex-col">
                {/* Spacer for header row */}
                <div className="h-16 w-36" />
                {strategies.map((rowLabel) => (
                  <div
                    key={rowLabel}
                    className="w-36 h-20 flex items-center justify-end pr-3"
                  >
                    <span
                      className="text-xs text-slate-400 text-right leading-tight"
                      style={{ maxWidth: '8rem' }}
                    >
                      {rowLabel}
                    </span>
                  </div>
                ))}
              </div>

              {/* Columns */}
              <div className="flex flex-col">
                {/* Column headers */}
                <div className="flex">
                  {strategies.map((colLabel) => (
                    <div
                      key={colLabel}
                      className="w-32 h-16 flex items-end justify-center pb-2 px-1"
                    >
                      <span className="text-xs text-slate-400 text-center leading-tight">
                        {colLabel}
                      </span>
                    </div>
                  ))}
                </div>

                {/* Rows of cells */}
                {matrix.map((row, rowIdx) => (
                  <div key={rowIdx} className="flex">
                    {row.map((value, colIdx) => {
                      const isHovered =
                        hoveredCell &&
                        hoveredCell.row === rowIdx &&
                        hoveredCell.col === colIdx;
                      const cellStyle = getCellStyle(value, isHovered);

                      return (
                        <div
                          key={colIdx}
                          className="w-32 h-20 flex flex-col items-center justify-center rounded-lg m-0.5 cursor-default transition-all duration-150 border border-slate-700/50"
                          style={cellStyle}
                          onMouseEnter={() =>
                            setHoveredCell({ row: rowIdx, col: colIdx })
                          }
                          onMouseLeave={() => setHoveredCell(null)}
                        >
                          <span className="font-mono text-lg font-semibold">
                            {value.toFixed(2)}
                          </span>
                          {rowIdx === colIdx && (
                            <span className="text-xs text-blue-200 mt-0.5 opacity-70">
                              self
                            </span>
                          )}
                        </div>
                      );
                    })}
                  </div>
                ))}
              </div>
            </div>

            {/* Color legend */}
            <div className="mt-5 flex items-center gap-4 flex-wrap">
              <span className="text-xs text-slate-500">Legend:</span>
              <div className="flex items-center gap-1.5">
                <div
                  className="w-4 h-4 rounded"
                  style={{ backgroundColor: 'rgb(30, 64, 175)' }}
                />
                <span className="text-xs text-slate-500">Self (1.0)</span>
              </div>
              <div className="flex items-center gap-1.5">
                <div
                  className="w-4 h-4 rounded"
                  style={{ backgroundColor: 'rgba(59, 130, 246, 0.4)' }}
                />
                <span className="text-xs text-slate-500">Moderate (0.3-0.5)</span>
              </div>
              <div className="flex items-center gap-1.5">
                <div
                  className="w-4 h-4 rounded"
                  style={{ backgroundColor: 'rgba(96, 165, 250, 0.2)' }}
                />
                <span className="text-xs text-slate-500">Low (0-0.3)</span>
              </div>
              <div className="flex items-center gap-1.5">
                <div
                  className="w-4 h-4 rounded"
                  style={{ backgroundColor: 'rgba(244, 63, 94, 0.3)' }}
                />
                <span className="text-xs text-slate-500">Negative</span>
              </div>
            </div>
          </div>
        </div>

        {/* Analysis Card */}
        <div className="flex-1 bg-slate-900 border border-slate-700 rounded-xl p-6">
          <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-widest mb-5">
            Diversification Analysis
          </h3>

          {/* Big correlation number */}
          <div className="mb-4">
            <div className="flex items-baseline gap-3">
              <span className="font-mono text-5xl font-bold text-blue-400">
                {offDiagonalValue.toFixed(2)}
              </span>
              <div>
                <div className="text-sm font-semibold text-slate-200">
                  {getInterpretation(offDiagonalValue)}
                </div>
                <div className="text-xs text-slate-500 mt-0.5">
                  Pearson correlation coefficient
                </div>
              </div>
            </div>
          </div>

          {/* Divider */}
          <div className="border-t border-slate-700 mb-4" />

          {/* Explanation */}
          <p className="text-sm text-slate-300 leading-relaxed mb-4">
            The two profitable hybrids show moderate correlation ({offDiagonalValue.toFixed(2)}) in
            daily PnL. This indicates meaningful diversification - they capture
            partially different market dynamics despite sharing the same 4
            champion strategies.
          </p>

          {/* Detail note */}
          <div className="bg-slate-800 border border-slate-700 rounded-lg p-3">
            <p className="text-xs text-slate-400 leading-relaxed">
              Consensus Gate fires only on 3/4 agreement (selective, {profitableHybrids[0]?.trades.toLocaleString()} trades),
              while Regime Router delegates to the best champion per regime bucket
              (broader, {profitableHybrids[1]?.trades.toLocaleString()} trades). The difference in trade selection produces
              the observed decorrelation.
            </p>
          </div>

          {/* Variance explained */}
          <div className="mt-4 grid grid-cols-2 gap-3">
            <div className="bg-slate-800/60 rounded-lg p-3 border border-slate-700/50">
              <div className="text-xs text-slate-500 mb-1">Shared variance</div>
              <div className="font-mono text-lg font-semibold text-slate-200">
                {(offDiagonalValue * offDiagonalValue * 100).toFixed(1)}%
              </div>
              <div className="text-xs text-slate-500 mt-0.5">r^2 = {(offDiagonalValue * offDiagonalValue).toFixed(4)}</div>
            </div>
            <div className="bg-slate-800/60 rounded-lg p-3 border border-slate-700/50">
              <div className="text-xs text-slate-500 mb-1">Independent variance</div>
              <div className="font-mono text-lg font-semibold text-slate-200">
                {((1 - offDiagonalValue * offDiagonalValue) * 100).toFixed(1)}%
              </div>
              <div className="text-xs text-slate-500 mt-0.5">unique signal per strategy</div>
            </div>
          </div>
        </div>
      </div>

      {/* Strategy Summary Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-6">
        {profitableHybrids.map((h) => (
          <div
            key={h.name}
            className="bg-slate-900 border border-slate-700 rounded-xl p-5"
          >
            <div className="flex items-start justify-between mb-3">
              <div>
                <div className="flex items-center gap-2 mb-0.5">
                  <span className="text-xs font-mono text-emerald-400 bg-emerald-400/10 px-2 py-0.5 rounded-full border border-emerald-400/20">
                    Rank #{h.rank}
                  </span>
                </div>
                <div className="text-base font-semibold text-slate-100 mt-1.5">
                  {h.displayName}
                </div>
                <div className="text-xs text-slate-500 mt-0.5">{h.description}</div>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-x-4 gap-y-2 mt-3">
              <div>
                <div className="text-xs text-slate-500">Ann. Sharpe</div>
                <div className="font-mono text-sm font-semibold text-emerald-400">
                  {h.annSharpe.toFixed(2)}
                </div>
              </div>
              <div>
                <div className="text-xs text-slate-500">Total Return</div>
                <div className="font-mono text-sm font-semibold text-emerald-400">
                  +{h.totalReturn.toFixed(1)}%
                </div>
              </div>
              <div>
                <div className="text-xs text-slate-500">Trades</div>
                <div className="font-mono text-sm font-semibold text-slate-200">
                  {h.trades.toLocaleString()}
                </div>
              </div>
              <div>
                <div className="text-xs text-slate-500">Win Rate</div>
                <div className="font-mono text-sm font-semibold text-slate-200">
                  {h.winRate.toFixed(1)}%
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Excluded Strategy Note */}
      {hybrids.filter(h => h.sharpe <= 0).map(excluded => (
        <div key={excluded.name} className="bg-slate-900/50 border border-slate-800 rounded-lg px-5 py-3 flex flex-wrap items-center gap-x-6 gap-y-1">
          <span className="text-xs text-slate-500">
            Excluded from correlation:
          </span>
          <span className="text-xs font-semibold text-slate-400">
            {excluded.displayName}
          </span>
          <span className="text-xs text-slate-600">
            Sharpe = {excluded.sharpe.toFixed(3)} (negative)
          </span>
          <span className="font-mono text-xs text-rose-500/70">{excluded.totalReturn.toFixed(1)}% return</span>
          <span className="font-mono text-xs text-slate-500">{excluded.trades.toLocaleString()} trades</span>
        </div>
      ))}
    </div>
  );
}
