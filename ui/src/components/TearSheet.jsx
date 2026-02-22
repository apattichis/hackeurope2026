import { useState } from 'react';
import { hybrids } from '../data.js';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  ReferenceLine,
} from 'recharts';

// --- Formatters ---

function fmtDollarsK(value) {
  return '$' + (value / 1000).toFixed(0) + 'k';
}

function fmtPct(value) {
  return value.toFixed(1) + '%';
}

function fmtReturn(value) {
  return (value >= 0 ? '+' : '') + value.toFixed(1) + '%';
}

function fmtComma(value) {
  return value.toLocaleString();
}

// Pick ~6-8 evenly spaced tick indices from the equity curve dates
function pickTickIndices(data, count = 7) {
  if (!data || data.length === 0) return [];
  const step = Math.max(1, Math.floor(data.length / (count - 1)));
  const indices = new Set();
  for (let i = 0; i < data.length; i += step) {
    indices.add(i);
  }
  indices.add(data.length - 1);
  return Array.from(indices);
}

function buildTickSet(data, count = 7) {
  const indices = pickTickIndices(data, count);
  return new Set(indices.map((i) => data[i]?.date).filter(Boolean));
}

function formatDateTick(dateStr) {
  if (!dateStr) return '';
  const d = new Date(dateStr);
  const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  return months[d.getUTCMonth()] + " '" + String(d.getUTCFullYear()).slice(2);
}

// --- Custom Tooltip ---

function DarkTooltip({ active, payload, label, valueFormatter }) {
  if (!active || !payload || !payload.length) return null;
  return (
    <div
      style={{
        background: '#1e293b',
        border: '1px solid #334155',
        borderRadius: 6,
        padding: '8px 12px',
        fontSize: 12,
        fontFamily: 'monospace',
        color: '#cbd5e1',
      }}
    >
      <div style={{ color: '#94a3b8', marginBottom: 2 }}>{label}</div>
      {payload.map((p, i) => (
        <div key={i} style={{ color: p.color || '#cbd5e1' }}>
          {valueFormatter ? valueFormatter(p.value) : p.value}
        </div>
      ))}
    </div>
  );
}

// --- KPI Card ---

function KpiCard({ label, value, color }) {
  const colorClass =
    color === 'emerald'
      ? 'text-emerald-500'
      : color === 'rose'
      ? 'text-rose-500'
      : 'text-white';

  return (
    <div className="flex flex-col gap-1 bg-slate-800 border border-slate-700 rounded-lg px-4 py-3 min-w-0">
      <span className="text-xs uppercase tracking-widest text-slate-500 font-semibold truncate">
        {label}
      </span>
      <span className={`text-2xl font-mono font-semibold ${colorClass} truncate`}>
        {value}
      </span>
    </div>
  );
}

// --- Diagnostics Table ---

const GRANULARITY_ORDER = { GLOBAL: 0, '1D': 1, '2D': 2, '3D': 3 };

function DiagnosticsTable({ diagnostics }) {
  const sorted = [...diagnostics].sort((a, b) => {
    const ga = GRANULARITY_ORDER[a.granularity] ?? 99;
    const gb = GRANULARITY_ORDER[b.granularity] ?? 99;
    return ga - gb;
  });

  return (
    <div className="overflow-x-auto rounded-lg border border-slate-700">
      <table className="w-full text-xs font-mono" style={{ borderCollapse: 'collapse' }}>
        <thead>
          <tr className="bg-slate-800 text-slate-400 uppercase tracking-wider">
            {['Granularity', 'Session', 'Trend', 'Vol', 'Trades', 'Win Rate', 'Sharpe', 'Sufficient'].map(
              (col) => (
                <th
                  key={col}
                  className="text-left px-3 py-2 font-semibold border-b border-slate-700 whitespace-nowrap"
                >
                  {col}
                </th>
              )
            )}
          </tr>
        </thead>
        <tbody>
          {sorted.map((row, i) => {
            const rowBg = i % 2 === 0 ? '#0f172a' : '#1e293b';
            const sharpeColor = row.sharpe > 0 ? '#10b981' : '#f43f5e';
            const suffColor = row.sufficientEvidence ? '#10b981' : '#f43f5e';
            return (
              <tr key={i} style={{ background: rowBg }}>
                <td className="px-3 py-2 text-slate-300 whitespace-nowrap">{row.granularity}</td>
                <td className="px-3 py-2 text-slate-400 whitespace-nowrap">{row.session}</td>
                <td className="px-3 py-2 text-slate-400 whitespace-nowrap">{row.trendRegime}</td>
                <td className="px-3 py-2 text-slate-400 whitespace-nowrap">{row.volRegime}</td>
                <td className="px-3 py-2 text-slate-300">{fmtComma(row.tradeCount)}</td>
                <td className="px-3 py-2 text-slate-300">{row.winRate.toFixed(1)}%</td>
                <td className="px-3 py-2 font-semibold" style={{ color: sharpeColor }}>
                  {row.sharpe.toFixed(4)}
                </td>
                <td className="px-3 py-2 font-semibold" style={{ color: suffColor }}>
                  {row.sufficientEvidence ? 'Yes' : 'No'}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

// --- Main Component ---

export default function TearSheet() {
  const [selectedIdx, setSelectedIdx] = useState(0);
  const hybrid = hybrids[selectedIdx];

  const lineColor = hybrid.totalReturn > 0 ? '#10b981' : '#f43f5e';
  const equityTickSet = buildTickSet(hybrid.equityCurve, 7);
  const drawdownTickSet = buildTickSet(hybrid.drawdown, 7);

  // KPI row 1
  const fitnessColor = hybrid.fitness > 0 ? 'emerald' : 'rose';
  const sharpeColor = hybrid.annSharpe > 0 ? 'emerald' : 'rose';
  const returnColor = hybrid.totalReturn > 0 ? 'emerald' : 'rose';

  return (
    <div className="flex flex-col gap-6 p-6 min-h-screen bg-slate-950 text-white">
      {/* Dropdown */}
      <div className="flex items-center gap-3">
        <label className="text-xs uppercase tracking-widest text-slate-500 font-semibold">
          Strategy
        </label>
        <select
          value={selectedIdx}
          onChange={(e) => setSelectedIdx(Number(e.target.value))}
          className="bg-slate-800 border border-slate-700 text-white text-sm font-mono rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-emerald-500/50 cursor-pointer"
        >
          {hybrids.map((h, i) => (
            <option key={i} value={i}>
              #{h.rank} - {h.displayName}
            </option>
          ))}
        </select>
      </div>

      {/* KPI Cards - Row 1 */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <KpiCard label="Fitness Score" value={hybrid.fitness.toFixed(3)} color={fitnessColor} />
        <KpiCard label="Ann. Sharpe" value={hybrid.annSharpe.toFixed(2)} color={sharpeColor} />
        <KpiCard label="Win Rate" value={hybrid.winRate.toFixed(1) + '%'} color="white" />
        <KpiCard label="Total Trades" value={fmtComma(hybrid.trades)} color="white" />
      </div>

      {/* KPI Cards - Row 2 */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <KpiCard label="Total Return" value={fmtReturn(hybrid.totalReturn)} color={returnColor} />
        <KpiCard label="Max Drawdown" value={hybrid.maxDrawdown.toFixed(1) + '%'} color="rose" />
        <KpiCard label="Avg Hold Time" value={hybrid.avgHoldHours.toFixed(1) + 'h'} color="white" />
        <KpiCard label="Max Consec Losses" value={hybrid.maxConsecLosses} color="white" />
      </div>

      {/* Equity Curve */}
      <div className="flex flex-col gap-2">
        <span className="text-sm text-slate-400 font-semibold">Cumulative Equity</span>
        <div className="bg-slate-900 border border-slate-700 rounded-xl p-4">
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={hybrid.equityCurve} margin={{ top: 8, right: 16, left: 0, bottom: 32 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" strokeOpacity={0.5} />
              <XAxis
                dataKey="date"
                tick={{ fill: '#64748b', fontSize: 11, fontFamily: 'monospace' }}
                angle={-45}
                textAnchor="end"
                interval={0}
                tickFormatter={(v) => (equityTickSet.has(v) ? formatDateTick(v) : '')}
                height={50}
                axisLine={{ stroke: '#334155' }}
                tickLine={false}
              />
              <YAxis
                tickFormatter={fmtDollarsK}
                tick={{ fill: '#64748b', fontSize: 11, fontFamily: 'monospace' }}
                axisLine={false}
                tickLine={false}
                width={56}
              />
              <Tooltip
                content={
                  <DarkTooltip valueFormatter={(v) => fmtDollarsK(v)} />
                }
              />
              <ReferenceLine
                y={100000}
                stroke="#475569"
                strokeDasharray="4 3"
                strokeWidth={1.5}
              />
              <Line
                type="monotone"
                dataKey="balance"
                stroke={lineColor}
                strokeWidth={1.5}
                dot={false}
                activeDot={{ r: 3, fill: lineColor }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Drawdown Chart */}
      <div className="flex flex-col gap-2">
        <span className="text-sm text-slate-400 font-semibold">Drawdown</span>
        <div className="bg-slate-900 border border-slate-700 rounded-xl p-4">
          <ResponsiveContainer width="100%" height={150}>
            <AreaChart data={hybrid.drawdown} margin={{ top: 8, right: 16, left: 0, bottom: 32 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" strokeOpacity={0.5} />
              <XAxis
                dataKey="date"
                tick={{ fill: '#64748b', fontSize: 11, fontFamily: 'monospace' }}
                angle={-45}
                textAnchor="end"
                interval={0}
                tickFormatter={(v) => (drawdownTickSet.has(v) ? formatDateTick(v) : '')}
                height={50}
                axisLine={{ stroke: '#334155' }}
                tickLine={false}
              />
              <YAxis
                tickFormatter={fmtPct}
                tick={{ fill: '#64748b', fontSize: 11, fontFamily: 'monospace' }}
                axisLine={false}
                tickLine={false}
                width={48}
              />
              <Tooltip
                content={<DarkTooltip valueFormatter={(v) => fmtPct(v)} />}
              />
              <Area
                type="monotone"
                dataKey="dd"
                stroke="#f43f5e"
                strokeWidth={1.5}
                fill="#f43f5e"
                fillOpacity={0.2}
                dot={false}
                activeDot={{ r: 3, fill: '#f43f5e' }}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Diagnostics Table */}
      <div className="flex flex-col gap-2">
        <span className="text-sm text-slate-400 font-semibold">Regime Diagnostics</span>
        <DiagnosticsTable diagnostics={hybrid.diagnostics} />
      </div>

    </div>
  );
}
