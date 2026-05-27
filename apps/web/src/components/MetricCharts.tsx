"use client";

import { useMemo } from "react";

/**
 * Parses log lines emitted by the studio's training callbacks.
 *
 * Whisper / LLM finetune handlers log lines like:
 *   `[<iso>] step loss=0.4231 learning_rate=0.0001 epoch=1.5`
 *   `[<iso>] eval: {'eval_loss': 0.31, 'eval_wer': 0.18}`
 *
 * We pull `key=number` tokens from "step" lines and `'key': number` pairs from
 * `eval:` lines. Everything else is ignored so noisy log noise can't crash the
 * parser.
 */
export type MetricPoint = { idx: number; value: number };
export type MetricSeries = Record<string, MetricPoint[]>;

const STEP_LINE = /\bstep\b\s+(.+)$/;
const KV_NUM = /\b([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)/g;
const EVAL_LINE = /\beval:\s*\{([^}]+)\}/;
const EVAL_KV = /'([^']+)'\s*:\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)/g;

export function parseMetrics(text: string): MetricSeries {
  const series: MetricSeries = {};
  const lines = text.split("\n");
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const stepMatch = line.match(STEP_LINE);
    if (stepMatch) {
      KV_NUM.lastIndex = 0;
      let m: RegExpExecArray | null;
      while ((m = KV_NUM.exec(stepMatch[1])) !== null) {
        push(series, m[1], parseFloat(m[2]), i);
      }
      continue;
    }
    const evalMatch = line.match(EVAL_LINE);
    if (evalMatch) {
      EVAL_KV.lastIndex = 0;
      let m: RegExpExecArray | null;
      while ((m = EVAL_KV.exec(evalMatch[1])) !== null) {
        push(series, m[1], parseFloat(m[2]), i);
      }
    }
  }
  return series;
}

function push(series: MetricSeries, key: string, value: number, idx: number): void {
  if (!Number.isFinite(value)) return;
  (series[key] ??= []).push({ idx, value });
}

const PREFERRED_ORDER = [
  "loss",
  "eval_loss",
  "wer",
  "eval_wer",
  "learning_rate",
  "epoch",
  "grad_norm",
];

export function MetricCharts({ logs }: { logs: string }) {
  const series = useMemo(() => parseMetrics(logs), [logs]);
  const keys = useMemo(() => {
    const all = Object.keys(series);
    const ordered = PREFERRED_ORDER.filter((k) => all.includes(k));
    const rest = all.filter((k) => !PREFERRED_ORDER.includes(k));
    return [...ordered, ...rest];
  }, [series]);

  if (keys.length === 0) {
    return (
      <p className="text-xs text-muted">
        No numeric metrics parsed yet. Training callbacks emit `step k=v` lines that appear here.
      </p>
    );
  }

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
      {keys.map((k) => (
        <Sparkline key={k} label={k} points={series[k]} />
      ))}
    </div>
  );
}

function Sparkline({ label, points }: { label: string; points: MetricPoint[] }) {
  if (points.length < 1) return null;

  const w = 280;
  const h = 80;
  const pad = 4;

  const xs = points.map((p) => p.idx);
  const ys = points.map((p) => p.value);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const rangeX = Math.max(1, maxX - minX);
  const rangeY = Math.max(Number.EPSILON, maxY - minY);

  const path = points
    .map((p, i) => {
      const x = pad + ((p.idx - minX) / rangeX) * (w - 2 * pad);
      const y = h - pad - ((p.value - minY) / rangeY) * (h - 2 * pad);
      return `${i === 0 ? "M" : "L"}${x.toFixed(2)},${y.toFixed(2)}`;
    })
    .join(" ");

  const last = points[points.length - 1].value;

  return (
    <div className="border border-border rounded-md p-2">
      <div className="flex items-center justify-between text-xs mb-1">
        <span className="text-muted">{label}</span>
        <span className="font-mono">{format(last)}</span>
      </div>
      <svg width="100%" viewBox={`0 0 ${w} ${h}`} className="block">
        <path d={path} fill="none" stroke="currentColor" className="text-accent" strokeWidth={1.5} />
      </svg>
      <div className="flex items-center justify-between text-[10px] text-muted mt-1">
        <span>{format(minY)}</span>
        <span>n={points.length}</span>
        <span>{format(maxY)}</span>
      </div>
    </div>
  );
}

function format(v: number): string {
  if (!Number.isFinite(v)) return "—";
  if (Math.abs(v) >= 1000 || (v !== 0 && Math.abs(v) < 0.001)) return v.toExponential(2);
  return v.toFixed(4).replace(/\.?0+$/, "");
}
