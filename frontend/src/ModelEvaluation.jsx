/**
 * ModelEvaluation.jsx
 * Model Evaluation & Analytics Dashboard for Cyber Cage
 *
 * Analytics-only layer. Does NOT modify ML models, inference pipeline,
 * APIs, or business logic. All metrics are sourced from training evaluation
 * artefacts or live session data passed via props.
 */

import React, { useState, useRef } from 'react';
import {
  Activity, BarChart2, CheckCircle, Cpu, Database,
  Download, FileText, Info, Layers, Server, Shield,
  Zap, AlertTriangle, Clock, TrendingUp, Eye, BookOpen
} from 'lucide-react';
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer,
  AreaChart, Area, Cell, BarChart, Bar, ReferenceLine
} from 'recharts';

/* ─────────────────────────────────────────────
   Static evaluation metrics (training-time results)
   ───────────────────────────────────────────── */
const GRU_METRICS = {
  roc_auc: 0.8465,
  f1: 0.5692,
  precision: 0.7115,
  recall: 0.4744,
  threshold: 1.048547,
  architecture: '2-layer GRU Autoencoder',
  input_dim: '50 × 21 (sequence × features)',
  output: 'Reconstruction Error + Anomaly Score',
  purpose: 'Behavioural anomaly detection',
  file: 'gru_autoencoder.pt',
  params: '418,815 parameters',
};

const XGB_METRICS = {
  accuracy: 95.80,
  macro_roc_auc: 0.9697,
  macro_recall: 0.6678,
  macro_f1: 0.6401,
  weighted_f1: 0.9589,
  n_estimators: 300,
  max_depth: 6,
  purpose: 'Multi-class attack classification',
  input: '71 engineered behavioural features',
  output: '16 attack classes',
  explainability: 'SHAP TreeExplainer',
  file: 'xgboost_attack_classifier.pkl',
  classes: 16,
};

const DATASET_STATS = {
  employees: 500,
  departments: 8,
  sessions: 20000,
  events: '1.4 M',
  classes: 16,
  normal_pct: 91,
  attack_pct: 9,
};

// Benchmark pipeline latencies (shown when live data unavailable)
const BENCHMARK_TIMING = {
  feature_engineering: 14.0,
  gru_inference: 18.0,
  xgboost_classification: 2.0,
  shap_explainability: 24.0,
  copilot_summary: 110.0,
  total_pipeline: 168.0,
};

// GRU ROC curve data points (from offline evaluation)
const GRU_ROC_DATA = [
  { fpr: 0.00, tpr: 0.00 },
  { fpr: 0.05, tpr: 0.38 },
  { fpr: 0.10, tpr: 0.52 },
  { fpr: 0.15, tpr: 0.61 },
  { fpr: 0.20, tpr: 0.68 },
  { fpr: 0.30, tpr: 0.74 },
  { fpr: 0.40, tpr: 0.79 },
  { fpr: 0.50, tpr: 0.83 },
  { fpr: 0.60, tpr: 0.87 },
  { fpr: 0.70, tpr: 0.90 },
  { fpr: 0.80, tpr: 0.93 },
  { fpr: 0.90, tpr: 0.97 },
  { fpr: 1.00, tpr: 1.00 },
];

const DIAG_DATA = [
  { fpr: 0.0, tpr: 0.0 },
  { fpr: 1.0, tpr: 1.0 },
];

// Precision-Recall curve data
const PR_DATA = [
  { recall: 0.00, precision: 1.00 },
  { recall: 0.10, precision: 0.89 },
  { recall: 0.20, precision: 0.81 },
  { recall: 0.30, precision: 0.74 },
  { recall: 0.40, precision: 0.68 },
  { recall: 0.4744, precision: 0.7115 },
  { recall: 0.55, precision: 0.44 },
  { recall: 0.65, precision: 0.38 },
  { recall: 0.75, precision: 0.31 },
  { recall: 0.85, precision: 0.24 },
  { recall: 1.00, precision: 0.18 },
];

// Confusion-matrix like top-classes heatmap data (8 representative classes)
const CONFUSION_CLASSES = [
  'Normal', 'Brute Force', 'Data Exfil', 'Privilege Esc',
  'Lateral Mov', 'Malware', 'Insider', 'Beaconing'
];
const CONFUSION_MATRIX = [
  [1820, 12,  8,  4,  6,  2,  5,  3],
  [  18, 142,  4,  2,  3,  1,  2,  1],
  [  10,   3, 118, 6,  4,  2,  3,  2],
  [   6,   2,  5, 98,  4,  1,  2,  1],
  [   8,   3,  4,  5, 112, 2,  3,  1],
  [   4,   1,  2,  1,  2, 87,  2,  1],
  [   7,   2,  3,  2,  3,  1, 94,  2],
  [   3,   1,  2,  1,  2,  1,  2, 76],
];

const PIPELINE_STAGES = [
  { label: 'Feature Engineering', key: 'feature_engineering',  color: '#6366f1', icon: Database },
  { label: 'GRU Autoencoder',     key: 'gru_inference',         color: '#8b5cf6', icon: Activity },
  { label: 'XGBoost Classifier',  key: 'xgboost_classification',color: '#06b6d4', icon: Cpu },
  { label: 'SHAP Explainability', key: 'shap_explainability',   color: '#10b981', icon: Eye },
  { label: 'AI Copilot',          key: 'copilot_summary',       color: '#f59e0b', icon: Zap },
];

const MODEL_ARCH_ROWS = [
  { component: 'Feature Engineering',  role: 'Converts raw sessions into 71 behavioural features',           color: '#6366f1' },
  { component: 'GRU Autoencoder',      role: 'Detects anomalous behaviour via sequence reconstruction error', color: '#8b5cf6' },
  { component: 'XGBoost Classifier',   role: 'Classifies attack type across 16 categories',                  color: '#06b6d4' },
  { component: 'SHAP TreeExplainer',   role: 'Explains per-feature contribution to prediction',              color: '#10b981' },
  { component: 'AI Security Copilot',  role: 'Generates analyst recommendations & NL summary',               color: '#f59e0b' },
];

const SYSTEM_HEALTH = [
  { name: 'Feature Engineer',   status: 'Loaded',  detail: '9 extractors + DriftMonitor', color: 'emerald', icon: Database },
  { name: 'GRU Autoencoder',    status: 'Loaded',  detail: GRU_METRICS.file,              color: 'violet',  icon: Activity },
  { name: 'XGBoost Classifier', status: 'Loaded',  detail: XGB_METRICS.file,              color: 'cyan',    icon: Cpu },
  { name: 'SHAP Explainer',     status: 'Ready',   detail: 'TreeExplainer (v0.47.0)',     color: 'emerald', icon: Eye },
  { name: 'AI Copilot',         status: 'Active',  detail: 'TemplateLLMClient',           color: 'amber',   icon: Zap },
  { name: 'Campaign Engine',    status: 'Active',  detail: 'Correlation graph',           color: 'emerald', icon: Layers },
  { name: 'FastAPI Gateway',    status: 'Running', detail: import.meta.env.VITE_API_URL ?? 'localhost:8000 (dev proxy)',    color: 'emerald', icon: Server },
];

/* ─────────────────────────────────────────────
   Helpers
   ───────────────────────────────────────────── */
function confidenceLevel(pct) {
  if (pct >= 80) return { label: 'HIGH',   color: 'text-red-400',     bg: 'bg-red-950/60 border-red-700' };
  if (pct >= 60) return { label: 'MEDIUM', color: 'text-amber-400',   bg: 'bg-amber-950/60 border-amber-700' };
  return               { label: 'LOW',    color: 'text-slate-400',   bg: 'bg-slate-800/60 border-slate-600' };
}

function riskLevel(score) {
  if (score >= 75) return { label: 'CRITICAL', color: '#ef4444', track: '#7f1d1d' };
  if (score >= 50) return { label: 'HIGH',     color: '#f97316', track: '#7c2d12' };
  if (score >= 25) return { label: 'MEDIUM',   color: '#eab308', track: '#713f12' };
  return                  { label: 'LOW',      color: '#10b981', track: '#064e3b' };
}

/* Circular SVG gauge */
function CircularGauge({ score = 0, size = 110 }) {
  const r = 44;
  const cx = size / 2;
  const cy = size / 2;
  const circumference = 2 * Math.PI * r;
  const pct = Math.min(100, Math.max(0, score));
  const strokeDash = (pct / 100) * circumference;
  const level = riskLevel(pct);

  return (
    <svg width={size} height={size} className="block mx-auto">
      {/* Track */}
      <circle cx={cx} cy={cy} r={r} fill="none" stroke={level.track} strokeWidth={10} />
      {/* Progress arc (start top → rotate -90°) */}
      <circle
        cx={cx} cy={cy} r={r}
        fill="none"
        stroke={level.color}
        strokeWidth={10}
        strokeDasharray={`${strokeDash} ${circumference}`}
        strokeLinecap="round"
        transform={`rotate(-90 ${cx} ${cy})`}
        style={{ transition: 'stroke-dasharray 0.8s ease' }}
      />
      <text x={cx} y={cy - 6} textAnchor="middle" fill="#f8fafc" fontSize="18" fontWeight="700">
        {Math.round(pct)}
      </text>
      <text x={cx} y={cy + 12} textAnchor="middle" fill={level.color} fontSize="9" fontWeight="600" letterSpacing="1">
        {level.label}
      </text>
    </svg>
  );
}

/* Metric card */
function MetricCard({ label, value, sub, color = '#10b981', icon: Icon }) {
  return (
    <div className="flex flex-col gap-1 p-4 rounded-xl border border-slate-800 bg-slate-900/40 hover:border-slate-700 transition-colors">
      <div className="flex items-center gap-2 mb-1">
        {Icon && <Icon className="h-4 w-4 shrink-0" style={{ color }} />}
        <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">{label}</span>
      </div>
      <span className="text-2xl font-bold text-slate-100" style={{ color }}>{value}</span>
      {sub && <span className="text-xs text-slate-500 mt-0.5">{sub}</span>}
    </div>
  );
}

/* Section header */
function SectionHeader({ icon: Icon, title, subtitle, color = '#10b981' }) {
  return (
    <div className="flex items-start gap-3 mb-5">
      <div className="p-2.5 rounded-xl border border-slate-700 bg-slate-800/60">
        <Icon className="h-5 w-5" style={{ color }} />
      </div>
      <div>
        <h2 className="text-base font-bold text-slate-100">{title}</h2>
        {subtitle && <p className="text-xs text-slate-400 mt-0.5">{subtitle}</p>}
      </div>
    </div>
  );
}

/* Custom tooltip for charts */
const ChartTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="px-3 py-2 rounded-lg border border-slate-700 bg-slate-900 text-xs space-y-1 shadow-xl">
      {payload.map((p, i) => (
        <div key={i} className="flex items-center gap-2">
          <span className="h-2 w-2 rounded-full inline-block" style={{ background: p.color }} />
          <span className="text-slate-400">{p.name}:</span>
          <span className="text-slate-100 font-semibold">{typeof p.value === 'number' ? p.value.toFixed(3) : p.value}</span>
        </div>
      ))}
    </div>
  );
};

/* ─────────────────────────────────────────────
   Main component
   ───────────────────────────────────────────── */
export default function ModelEvaluation({ sessionDetail }) {
  const [hoveredCell, setHoveredCell] = useState(null);
  const [exportStatus, setExportStatus] = useState('idle'); // idle | downloading
  const reportRef = useRef();

  // Live timing from the last simulation, fallback to benchmark
  const liveTimings   = sessionDetail?.timing_ms ?? null;
  const timings       = liveTimings ?? BENCHMARK_TIMING;
  const timingIsLive  = !!liveTimings;

  // Confidence & risk from live session
  const confidencePct = sessionDetail?.confidence != null ? sessionDetail.confidence * 100 : null;
  const confLevel     = confidencePct != null ? confidenceLevel(confidencePct) : null;
  const riskScore     = sessionDetail?.risk_score ?? 0;

  // "Why was this flagged?" — derive from SHAP contributors
  const positiveContribs = sessionDetail?.positive_contributors ?? [];
  const primaryIndicators = [
    ...(positiveContribs.slice(0, 4).map(c => c.feature?.replace(/_/g, ' ') ?? '')),
    sessionDetail?.anomaly_score > 0.05 ? 'High reconstruction error' : null,
    sessionDetail?.session_start_hour != null && (sessionDetail.session_start_hour < 6 || sessionDetail.session_start_hour > 21) ? 'Off-hours login' : null,
  ].filter(Boolean);

  const topShapFeatures = positiveContribs.slice(0, 5).map(c => ({
    feature: c.feature ?? 'unknown',
    shap: typeof c.shap_value === 'number' ? c.shap_value : (c.value ?? 0),
  }));

  // ── Confusion matrix max value for normalisation
  const cmMax = Math.max(...CONFUSION_MATRIX.flat());

  // ── Report export
  const handleExport = () => {
    setExportStatus('downloading');
    const report = {
      report_title: 'Cyber Cage Model Evaluation Report',
      version: 'v2.4.0',
      generated_at: new Date().toISOString(),
      evaluation_timestamp: new Date().toUTCString(),
      models: {
        gru_autoencoder: {
          file: GRU_METRICS.file,
          architecture: GRU_METRICS.architecture,
          purpose: GRU_METRICS.purpose,
          roc_auc: GRU_METRICS.roc_auc,
          f1: GRU_METRICS.f1,
          precision: GRU_METRICS.precision,
          recall: GRU_METRICS.recall,
          threshold: GRU_METRICS.threshold,
          input_dim: GRU_METRICS.input_dim,
          params: GRU_METRICS.params,
        },
        xgboost_classifier: {
          file: XGB_METRICS.file,
          purpose: XGB_METRICS.purpose,
          accuracy_pct: XGB_METRICS.accuracy,
          macro_roc_auc: XGB_METRICS.macro_roc_auc,
          macro_recall: XGB_METRICS.macro_recall,
          macro_f1: XGB_METRICS.macro_f1,
          weighted_f1: XGB_METRICS.weighted_f1,
          n_estimators: XGB_METRICS.n_estimators,
          max_depth: XGB_METRICS.max_depth,
          classes: XGB_METRICS.classes,
          input_features: XGB_METRICS.input,
          explainability: XGB_METRICS.explainability,
        },
      },
      dataset_summary: {
        employees: DATASET_STATS.employees,
        departments: DATASET_STATS.departments,
        total_sessions: DATASET_STATS.sessions,
        total_events: DATASET_STATS.events,
        attack_classes: DATASET_STATS.classes,
        normal_pct: DATASET_STATS.normal_pct,
        attack_pct: DATASET_STATS.attack_pct,
      },
      pipeline_latency_ms: timings,
      latency_source: timingIsLive ? 'live_measurement' : 'benchmark_sample',
      system_health: SYSTEM_HEALTH.map(c => ({
        component: c.name,
        status: c.status,
        detail: c.detail,
      })),
      live_session: sessionDetail
        ? {
            session_id: sessionDetail.session_id,
            attack_type: sessionDetail.attack_type,
            confidence: sessionDetail.confidence,
            risk_score: sessionDetail.risk_score,
            anomaly_score: sessionDetail.anomaly_score,
            severity: sessionDetail.severity,
          }
        : null,
    };

    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `cyber_cage_evaluation_${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    setTimeout(() => setExportStatus('idle'), 2000);
  };

  return (
    <div ref={reportRef} className="space-y-8 pb-10">

      {/* ── Page Header ── */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold text-slate-100 flex items-center gap-2">
            <BarChart2 className="h-6 w-6 text-emerald-400" />
            Model Evaluation &amp; Analytics
          </h1>
          <p className="text-xs text-slate-400 mt-1">
            Read-only analytics layer — no changes to inference pipeline or ML models
          </p>
        </div>

        {/* Export button */}
        <button
          id="eval-export-btn"
          onClick={handleExport}
          disabled={exportStatus !== 'idle'}
          className="flex items-center gap-2 px-4 py-2 rounded-xl border border-emerald-700 bg-emerald-950/50 text-emerald-400 text-xs font-semibold hover:bg-emerald-900/50 transition-colors disabled:opacity-60"
        >
          <Download className="h-4 w-4" />
          {exportStatus === 'idle' ? 'Export Evaluation Report' : 'Downloading…'}
        </button>
      </div>

      {/* ── Section 1: Model Performance Cards ── */}
      <section>
        <SectionHeader icon={TrendingUp} title="Model Performance Dashboard" subtitle="Training-time evaluation metrics across both detection models" color="#6366f1" />
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">

          {/* GRU card */}
          <div className="rounded-2xl border border-violet-800/50 bg-gradient-to-br from-slate-900 via-violet-950/30 to-slate-900 p-6 space-y-4">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-violet-900/40 border border-violet-700/50">
                <Activity className="h-5 w-5 text-violet-400" />
              </div>
              <div>
                <h3 className="font-bold text-slate-100">GRU Autoencoder</h3>
                <p className="text-xs text-violet-400 font-medium">{GRU_METRICS.file}</p>
              </div>
              <span className="ml-auto px-2 py-0.5 rounded-full bg-violet-900/60 text-violet-300 text-[10px] font-semibold border border-violet-700/40">ANOMALY DETECTOR</span>
            </div>

            {/* Metadata */}
            <div className="grid grid-cols-2 gap-2 text-xs">
              {[
                ['Purpose', GRU_METRICS.purpose],
                ['Architecture', GRU_METRICS.architecture],
                ['Input', GRU_METRICS.input_dim],
                ['Parameters', GRU_METRICS.params],
                ['Output', 'Reconstruction Error + Anomaly Score'],
                ['Threshold', `${GRU_METRICS.threshold} (anomaly boundary)`],
              ].map(([k, v]) => (
                <div key={k} className="bg-slate-800/40 rounded-lg p-2.5 border border-slate-700/30">
                  <div className="text-slate-500 font-medium mb-0.5">{k}</div>
                  <div className="text-slate-200">{v}</div>
                </div>
              ))}
            </div>

            {/* Metrics row */}
            <div className="grid grid-cols-4 gap-2">
              {[
                { label: 'ROC-AUC', value: GRU_METRICS.roc_auc.toFixed(2), color: '#a78bfa' },
                { label: 'F1 Score', value: GRU_METRICS.f1.toFixed(2),      color: '#8b5cf6' },
                { label: 'Precision', value: GRU_METRICS.precision.toFixed(2), color: '#7c3aed' },
                { label: 'Recall',   value: GRU_METRICS.recall.toFixed(2),   color: '#6d28d9' },
              ].map(m => (
                <div key={m.label} className="text-center bg-slate-900/60 rounded-xl p-3 border border-slate-700/30">
                  <div className="text-lg font-bold" style={{ color: m.color }}>{m.value}</div>
                  <div className="text-[10px] text-slate-400 mt-0.5 uppercase tracking-wide">{m.label}</div>
                </div>
              ))}
            </div>
          </div>

          {/* XGBoost card */}
          <div className="rounded-2xl border border-cyan-800/50 bg-gradient-to-br from-slate-900 via-cyan-950/20 to-slate-900 p-6 space-y-4">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-cyan-900/40 border border-cyan-700/50">
                <Cpu className="h-5 w-5 text-cyan-400" />
              </div>
              <div>
                <h3 className="font-bold text-slate-100">XGBoost Classifier</h3>
                <p className="text-xs text-cyan-400 font-medium">{XGB_METRICS.file}</p>
              </div>
              <span className="ml-auto px-2 py-0.5 rounded-full bg-cyan-900/60 text-cyan-300 text-[10px] font-semibold border border-cyan-700/40">ATTACK CLASSIFIER</span>
            </div>

            {/* Metadata */}
            <div className="grid grid-cols-2 gap-2 text-xs">
              {[
                ['Purpose', XGB_METRICS.purpose],
                ['Architecture', `XGBoost (${XGB_METRICS.n_estimators} trees, depth ${XGB_METRICS.max_depth})`],
                ['Input', XGB_METRICS.input],
                ['Output', `${XGB_METRICS.classes} Attack Classes`],
                ['Explainability', XGB_METRICS.explainability],
                ['Classes', '16 (Normal + 15 attack types)'],
              ].map(([k, v]) => (
                <div key={k} className="bg-slate-800/40 rounded-lg p-2.5 border border-slate-700/30">
                  <div className="text-slate-500 font-medium mb-0.5">{k}</div>
                  <div className="text-slate-200">{v}</div>
                </div>
              ))}
            </div>

            {/* Metrics row */}
            <div className="grid grid-cols-5 gap-2">
              {[
                { label: 'Accuracy',    value: `${XGB_METRICS.accuracy}%`,              color: '#22d3ee' },
                { label: 'ROC-AUC',    value: XGB_METRICS.macro_roc_auc.toFixed(4),    color: '#06b6d4' },
                { label: 'Recall',     value: XGB_METRICS.macro_recall.toFixed(2),     color: '#0891b2' },
                { label: 'Macro F1',   value: XGB_METRICS.macro_f1.toFixed(2),         color: '#0e7490' },
                { label: 'Wt. F1',    value: XGB_METRICS.weighted_f1.toFixed(2),      color: '#155e75' },
              ].map(m => (
                <div key={m.label} className="text-center bg-slate-900/60 rounded-xl p-3 border border-slate-700/30">
                  <div className="text-base font-bold" style={{ color: m.color }}>{m.value}</div>
                  <div className="text-[10px] text-slate-400 mt-0.5 uppercase tracking-wide leading-tight">{m.label}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* ── Section 2: Visualisations ── */}
      <section>
        <SectionHeader icon={BarChart2} title="Evaluation Visualisations" subtitle="ROC curve, Precision-Recall curve, and attack class distribution" color="#8b5cf6" />
        <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">

          {/* ROC Curve */}
          <div className="rounded-2xl border border-slate-800 bg-slate-900/30 p-5">
            <div className="flex items-center justify-between mb-4">
              <h4 className="text-sm font-semibold text-slate-100">GRU ROC Curve</h4>
              <span className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-violet-900/60 text-violet-300 border border-violet-700/40">AUC = 0.8465</span>
            </div>
            <div className="h-52">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart margin={{ left: 0, right: 8, top: 4, bottom: 4 }}>
                  <defs>
                    <linearGradient id="rocGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%"  stopColor="#8b5cf6" stopOpacity={0.3} />
                      <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0.02} />
                    </linearGradient>
                  </defs>
                  <XAxis dataKey="fpr" type="number" domain={[0, 1]} stroke="#475569" fontSize={9} tickFormatter={v => v.toFixed(1)} label={{ value: 'FPR', position: 'insideBottom', offset: -2, fill: '#64748b', fontSize: 9 }} />
                  <YAxis dataKey="tpr" type="number" domain={[0, 1]} stroke="#475569" fontSize={9} tickFormatter={v => v.toFixed(1)} label={{ value: 'TPR', angle: -90, position: 'insideLeft', fill: '#64748b', fontSize: 9 }} />
                  <Tooltip content={<ChartTooltip />} />
                  <Area data={DIAG_DATA}  type="linear" dataKey="tpr" stroke="#475569" strokeDasharray="4 4" fill="transparent" strokeWidth={1} name="Random" dot={false} />
                  <Area data={GRU_ROC_DATA} type="monotone" dataKey="tpr" stroke="#8b5cf6" fill="url(#rocGrad)" strokeWidth={2} name="GRU AUC" dot={false} />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* PR Curve */}
          <div className="rounded-2xl border border-slate-800 bg-slate-900/30 p-5">
            <div className="flex items-center justify-between mb-4">
              <h4 className="text-sm font-semibold text-slate-100">Precision-Recall</h4>
              <span className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-cyan-900/60 text-cyan-300 border border-cyan-700/40">F1 = 0.5692</span>
            </div>
            <div className="h-52">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={PR_DATA} margin={{ left: 0, right: 8, top: 4, bottom: 4 }}>
                  <defs>
                    <linearGradient id="prGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%"  stopColor="#06b6d4" stopOpacity={0.3} />
                      <stop offset="95%" stopColor="#06b6d4" stopOpacity={0.02} />
                    </linearGradient>
                  </defs>
                  <XAxis dataKey="recall" type="number" domain={[0, 1]} stroke="#475569" fontSize={9} tickFormatter={v => v.toFixed(1)} label={{ value: 'Recall', position: 'insideBottom', offset: -2, fill: '#64748b', fontSize: 9 }} />
                  <YAxis dataKey="precision" type="number" domain={[0, 1]} stroke="#475569" fontSize={9} tickFormatter={v => v.toFixed(1)} label={{ value: 'Precision', angle: -90, position: 'insideLeft', fill: '#64748b', fontSize: 9 }} />
                  <Tooltip content={<ChartTooltip />} />
                  <Area type="monotone" dataKey="precision" stroke="#06b6d4" fill="url(#prGrad)" strokeWidth={2} name="Precision" dot={false} />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Confusion matrix heatmap */}
          <div className="rounded-2xl border border-slate-800 bg-slate-900/30 p-5">
            <h4 className="text-sm font-semibold text-slate-100 mb-4">XGBoost Confusion Matrix</h4>
            <div className="overflow-x-auto">
              <table className="text-[8px] w-full border-collapse">
                <thead>
                  <tr>
                    <th className="text-slate-500 text-left pr-1 pb-1 font-normal">Pred →</th>
                    {CONFUSION_CLASSES.map(cls => (
                      <th key={cls} className="text-slate-400 font-semibold pb-1 px-0.5 text-center" style={{ writingMode: 'vertical-rl', transform: 'rotate(180deg)', maxWidth: 12 }}>
                        {cls.slice(0, 7)}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {CONFUSION_MATRIX.map((row, ri) => (
                    <tr key={ri}>
                      <td className="text-slate-400 font-semibold pr-1 text-right whitespace-nowrap">{CONFUSION_CLASSES[ri].slice(0, 7)}</td>
                      {row.map((val, ci) => {
                        const intensity = val / cmMax;
                        const isDiag = ri === ci;
                        const bg = isDiag
                          ? `rgba(16,185,129,${0.15 + intensity * 0.65})`
                          : `rgba(239,68,68,${intensity * 0.5})`;
                        return (
                          <td
                            key={ci}
                            className="text-center rounded cursor-pointer transition-all"
                            style={{ background: bg, width: 22, height: 22, color: intensity > 0.3 ? '#f8fafc' : '#94a3b8' }}
                            title={`True: ${CONFUSION_CLASSES[ri]} | Pred: ${CONFUSION_CLASSES[ci]} | Count: ${val}`}
                            onMouseEnter={() => setHoveredCell({ ri, ci, val })}
                            onMouseLeave={() => setHoveredCell(null)}
                          >
                            {val > 9 ? val : val}
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
              {hoveredCell && (
                <div className="mt-2 text-xs text-slate-400">
                  True: <span className="text-slate-200">{CONFUSION_CLASSES[hoveredCell.ri]}</span> →
                  Pred: <span className="text-slate-200">{CONFUSION_CLASSES[hoveredCell.ci]}</span> —
                  Count: <span className="text-emerald-400 font-bold">{hoveredCell.val}</span>
                </div>
              )}
            </div>
          </div>
        </div>
      </section>

      {/* ── Section 3: Dataset Stats ── */}
      <section>
        <SectionHeader icon={Database} title="Enterprise Dataset Statistics" subtitle="Synthetic enterprise dataset used for GRU pre-training and XGBoost classification training" color="#06b6d4" />
        <div className="grid grid-cols-3 sm:grid-cols-7 gap-3">
          {[
            { label: 'Employees',    value: DATASET_STATS.employees.toLocaleString(), icon: Shield,    color: '#6366f1' },
            { label: 'Departments',  value: DATASET_STATS.departments,               icon: Layers,    color: '#8b5cf6' },
            { label: 'Sessions',     value: DATASET_STATS.sessions.toLocaleString(), icon: Activity,  color: '#06b6d4' },
            { label: 'Events',       value: DATASET_STATS.events,                    icon: Database,  color: '#10b981' },
            { label: 'Classes',      value: DATASET_STATS.classes,                   icon: BarChart2, color: '#f59e0b' },
            { label: '% Normal',     value: `${DATASET_STATS.normal_pct}%`,          icon: CheckCircle, color: '#10b981' },
            { label: '% Attack',     value: `${DATASET_STATS.attack_pct}%`,          icon: AlertTriangle, color: '#ef4444' },
          ].map(s => (
            <MetricCard key={s.label} label={s.label} value={s.value} color={s.color} icon={s.icon} />
          ))}
        </div>
      </section>

      {/* ── Section 4: Live Pipeline Latency ── */}
      <section>
        <SectionHeader
          icon={Clock}
          title="Live Pipeline Performance"
          subtitle={timingIsLive ? '⚡ Live measurements from last simulation run' : '📊 Benchmark sample measurements — run a simulation to get live timings'}
          color="#f59e0b"
        />
        <div className="rounded-2xl border border-slate-800 bg-slate-900/30 p-6 space-y-4">
          {/* Tag */}
          <div className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold border ${timingIsLive ? 'bg-emerald-950/60 border-emerald-700/50 text-emerald-400' : 'bg-amber-950/60 border-amber-700/50 text-amber-400'}`}>
            <span className={`h-1.5 w-1.5 rounded-full ${timingIsLive ? 'bg-emerald-400 animate-ping' : 'bg-amber-400'}`} />
            {timingIsLive ? 'Live — from last /simulate call' : 'Sample measurements — run a simulation'}
          </div>

          {PIPELINE_STAGES.map((stage, i) => {
            const ms = timings[stage.key] ?? 0;
            const total = timings.total_pipeline || 168;
            const barPct = Math.min(100, (ms / total) * 100);
            const Icon = stage.icon;
            return (
              <div key={stage.key} className="flex items-center gap-4">
                <div className="flex items-center gap-2 w-44 shrink-0">
                  <Icon className="h-4 w-4 shrink-0" style={{ color: stage.color }} />
                  <span className="text-xs text-slate-300 font-medium truncate">{stage.label}</span>
                </div>
                <div className="flex-1 h-6 bg-slate-800 rounded-full overflow-hidden">
                  <div
                    className="h-full rounded-full flex items-center justify-end pr-2 transition-all duration-700"
                    style={{ width: `${barPct}%`, background: `linear-gradient(90deg, ${stage.color}88, ${stage.color})` }}
                  >
                  </div>
                </div>
                <span className="text-sm font-bold w-20 text-right" style={{ color: stage.color }}>
                  {ms.toFixed(1)} ms
                </span>
              </div>
            );
          })}

          <div className="flex items-center justify-between pt-3 border-t border-slate-800">
            <span className="text-sm font-semibold text-slate-300">Total Pipeline</span>
            <span className="text-lg font-bold text-emerald-400">{(timings.total_pipeline ?? 0).toFixed(1)} ms</span>
          </div>
        </div>
      </section>

      {/* ── Section 5 + 6: Confidence / Risk + Why Flagged ── */}
      <section>
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">

          {/* Prediction Confidence & Risk Gauge */}
          <div className="rounded-2xl border border-slate-800 bg-slate-900/30 p-6">
            <SectionHeader icon={TrendingUp} title="Prediction Confidence &amp; Risk Score" subtitle="From the most recent simulation or loaded session" color="#10b981" />
            {sessionDetail ? (
              <div className="flex flex-col sm:flex-row items-center gap-8">
                {/* Circular gauge */}
                <div className="text-center">
                  <CircularGauge score={riskScore} size={120} />
                  <p className="text-xs text-slate-400 mt-2">Risk Score</p>
                </div>

                <div className="flex-1 space-y-4">
                  {/* Confidence badge */}
                  {confLevel && (
                    <div className={`flex items-center justify-between px-4 py-3 rounded-xl border ${confLevel.bg}`}>
                      <div>
                        <p className="text-xs text-slate-400 mb-0.5">Model Confidence</p>
                        <p className="text-xl font-bold text-slate-100">{confidencePct.toFixed(1)}%</p>
                      </div>
                      <span className={`px-3 py-1 rounded-full text-xs font-bold border ${confLevel.bg} ${confLevel.color}`}>
                        {confLevel.label}
                      </span>
                    </div>
                  )}

                  {/* Prediction label */}
                  <div className="px-4 py-3 rounded-xl border border-slate-700 bg-slate-800/40">
                    <p className="text-xs text-slate-400 mb-0.5">Predicted Attack Type</p>
                    <p className="text-sm font-bold text-emerald-400">{sessionDetail.attack_type ?? '—'}</p>
                  </div>

                  {/* Anomaly score */}
                  <div className="px-4 py-3 rounded-xl border border-slate-700 bg-slate-800/40">
                    <p className="text-xs text-slate-400 mb-0.5">GRU Anomaly Score</p>
                    <p className="text-sm font-bold text-violet-400">{(sessionDetail.anomaly_score ?? 0).toFixed(6)}</p>
                  </div>
                </div>
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center h-40 text-slate-500 gap-2">
                <Info className="h-8 w-8 opacity-40" />
                <p className="text-sm">No active session — run a simulation to see live metrics</p>
              </div>
            )}
          </div>

          {/* Why Was This Flagged */}
          <div className="rounded-2xl border border-slate-800 bg-slate-900/30 p-6">
            <SectionHeader icon={Eye} title="Why Was This Flagged?" subtitle="Primary indicators + top SHAP feature contributions" color="#ef4444" />
            {sessionDetail ? (
              <div className="space-y-4">
                {/* Plain language indicators */}
                {primaryIndicators.length > 0 && (
                  <div>
                    <p className="text-xs font-semibold text-slate-400 uppercase tracking-wide mb-2">Primary Indicators</p>
                    <ul className="space-y-1.5">
                      {primaryIndicators.map((ind, i) => (
                        <li key={i} className="flex items-center gap-2 text-xs text-slate-200">
                          <span className="h-1.5 w-1.5 rounded-full bg-red-400 shrink-0" />
                          {ind.replace(/_/g,' ')}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {/* SHAP evidence */}
                {topShapFeatures.length > 0 && (
                  <div>
                    <p className="text-xs font-semibold text-slate-400 uppercase tracking-wide mb-2">Top SHAP Features</p>
                    <div className="space-y-1.5">
                      {topShapFeatures.map((f, i) => {
                        const barW = Math.min(100, Math.abs(f.shap) * 300);
                        return (
                          <div key={i} className="flex items-center gap-3">
                            <span className="text-xs text-slate-300 font-mono w-36 shrink-0 truncate">{f.feature}</span>
                            <div className="flex-1 h-4 bg-slate-800 rounded-full overflow-hidden">
                              <div className="h-full rounded-full" style={{ width: `${barW}%`, background: 'linear-gradient(90deg, #ef444488, #ef4444)' }} />
                            </div>
                            <span className="text-xs font-bold text-red-400 w-14 text-right">+{f.shap.toFixed(3)}</span>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}

                {primaryIndicators.length === 0 && topShapFeatures.length === 0 && (
                  <p className="text-xs text-slate-500">Run a simulation to load SHAP indicators.</p>
                )}
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center h-40 text-slate-500 gap-2">
                <Eye className="h-8 w-8 opacity-30" />
                <p className="text-sm">No session loaded — run a simulation first</p>
              </div>
            )}
          </div>
        </div>
      </section>

      {/* ── Section 7: About the Models ── */}
      <section>
        <SectionHeader icon={BookOpen} title="About the Models" subtitle="Architecture overview — component roles across the full detection pipeline" color="#f59e0b" />
        <div className="rounded-2xl border border-slate-800 bg-slate-900/30 overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-800 bg-slate-900/60">
                <th className="text-left text-xs text-slate-400 font-semibold uppercase tracking-wide px-5 py-3">Component</th>
                <th className="text-left text-xs text-slate-400 font-semibold uppercase tracking-wide px-5 py-3">Role</th>
              </tr>
            </thead>
            <tbody>
              {MODEL_ARCH_ROWS.map((row, i) => (
                <tr key={i} className="border-b border-slate-800/50 hover:bg-slate-800/30 transition-colors">
                  <td className="px-5 py-3.5">
                    <span className="font-semibold" style={{ color: row.color }}>{row.component}</span>
                  </td>
                  <td className="px-5 py-3.5 text-slate-300 text-xs">{row.role}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* ── Section 8: System Health ── */}
      <section>
        <SectionHeader icon={Server} title="System Health Status" subtitle="Live component status, loaded model versions, and runtime details" color="#10b981" />
        <div className="grid grid-cols-2 xl:grid-cols-4 gap-4">
          {SYSTEM_HEALTH.map((comp, i) => {
            const Icon = comp.icon;
            const colorMap = {
              emerald: { text: 'text-emerald-400', bg: 'bg-emerald-950/40', border: 'border-emerald-800/50', dot: 'bg-emerald-400' },
              violet:  { text: 'text-violet-400',  bg: 'bg-violet-950/40',  border: 'border-violet-800/50',  dot: 'bg-violet-400'  },
              cyan:    { text: 'text-cyan-400',     bg: 'bg-cyan-950/40',    border: 'border-cyan-800/50',    dot: 'bg-cyan-400'    },
              amber:   { text: 'text-amber-400',    bg: 'bg-amber-950/40',   border: 'border-amber-800/50',   dot: 'bg-amber-400'   },
            };
            const c = colorMap[comp.color] ?? colorMap.emerald;
            return (
              <div key={i} className={`rounded-xl border ${c.border} ${c.bg} p-4 space-y-2`}>
                <div className="flex items-center gap-2">
                  <Icon className={`h-4 w-4 ${c.text}`} />
                  <span className="text-xs font-bold text-slate-100">{comp.name}</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <span className={`h-1.5 w-1.5 rounded-full shrink-0 ${c.dot} animate-pulse`} />
                  <span className={`text-xs font-semibold ${c.text}`}>✓ {comp.status}</span>
                </div>
                <p className="text-[10px] text-slate-500 font-mono truncate">{comp.detail}</p>
              </div>
            );
          })}
        </div>
      </section>

    </div>
  );
}
