"use client";

import { useState, useMemo, useRef } from "react";
import useSWR from "swr";
import Link from "next/link";
import { Card, CardTitle } from "@/components/Card";
import { StatusPill } from "@/components/StatusPill";
import { relativeTime } from "@/lib/utils";

type EvalResult = {
  job_id: string;
  kind: "asr_eval" | "llm_eval" | "tts_eval";
  status: string;
  created_at: string;
  finished_at: string | null;
  metrics: Record<string, unknown>;
};

async function jget<T>(p: string): Promise<T> {
  const r = await fetch(`/api${p}`, { cache: "no-store" });
  if (!r.ok) throw new Error(`${r.status}: ${await r.text()}`);
  return r.json();
}

export function EvalResults({ versionId }: { versionId: string }) {
  const { data, isLoading } = useSWR<EvalResult[]>(
    ["evals", versionId],
    () => jget<EvalResult[]>(`/models/versions/${versionId}/evals`),
    { refreshInterval: 5000 },
  );

  if (isLoading) return <p className="text-xs text-muted">loading evals…</p>;
  if (!data?.length)
    return (
      <p className="text-xs text-muted">
        No evals run yet. Use the panel on the right to queue one.
      </p>
    );

  return (
    <div className="space-y-4">
      {data.map((e) => (
        <EvalCard key={e.job_id} eval={e} />
      ))}
    </div>
  );
}

function EvalCard({ eval: e }: { eval: EvalResult }) {
  return (
    <Card>
      <div className="flex items-center justify-between mb-2">
        <CardTitle>
          <span className="font-mono text-xs">{e.kind}</span>
        </CardTitle>
        <div className="flex items-center gap-2 text-xs text-muted">
          <StatusPill status={e.status} />
          <Link href={`/jobs/${e.job_id}`} className="text-accent">
            job →
          </Link>
          <span>{relativeTime(e.created_at)}</span>
        </div>
      </div>

      {e.status !== "succeeded" ? (
        <p className="text-xs text-muted">Run not finished — open the job for live logs.</p>
      ) : e.kind === "asr_eval" ? (
        <ASREvalView metrics={e.metrics} />
      ) : e.kind === "llm_eval" ? (
        <LLMEvalView metrics={e.metrics} />
      ) : (
        <TTSEvalView metrics={e.metrics} />
      )}
    </Card>
  );
}

function HeadlineMetric({ label, value, unit = "" }: { label: string; value: string | number | undefined; unit?: string }) {
  return (
    <div className="border border-border rounded-md p-3">
      <div className="text-[10px] uppercase tracking-wide text-muted">{label}</div>
      <div className="text-xl font-mono mt-1">
        {value === undefined || value === null ? "—" : typeof value === "number" ? value.toFixed(4) : value}
        {unit && <span className="text-xs text-muted ml-1">{unit}</span>}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------

function ASREvalView({ metrics }: { metrics: Record<string, unknown> }) {
  const wer = metrics.wer as number | undefined;
  const cer = metrics.cer as number | undefined;
  const n = metrics.n_samples as number | undefined;
  const slices = (metrics.slices ?? {}) as Record<string, Record<string, { wer: number; n: number }>>;
  const worst = useMemo(() => (metrics.worst ?? []) as Array<{
    id?: string;
    audio_uri?: string;
    ref: string;
    hyp: string;
    wer: number;
    subs: number;
    ins: number;
    del: number;
  }>, [metrics.worst]);

  const [search, setSearch] = useState("");

  const filteredWorst = useMemo(() => {
    if (!search) return worst;
    const term = search.toLowerCase();
    return worst.filter(
      (w) => w.ref.toLowerCase().includes(term) || w.hyp.toLowerCase().includes(term)
    );
  }, [worst, search]);

  return (
    <div className="space-y-3">
      <div className="grid grid-cols-3 gap-2">
        <HeadlineMetric label="WER" value={wer !== undefined ? wer * 100 : undefined} unit="%" />
        <HeadlineMetric label="CER" value={cer !== undefined ? cer * 100 : undefined} unit="%" />
        <HeadlineMetric label="Samples" value={n} />
      </div>

      {Object.entries(slices).map(([sliceKey, buckets]) => (
        <SliceTable key={sliceKey} sliceKey={sliceKey} buckets={buckets} baseline={wer ?? 0} />
      ))}

      {worst.length > 0 && (
        <Card className="p-3 border border-border/80 bg-black/10">
          <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2 mb-3 border-b border-border/60 pb-2">
            <span className="text-xs font-semibold uppercase tracking-wider text-muted">Transcript Error Explorer</span>
            <input
              type="text"
              placeholder="Search transcript errors..."
              className="text-xs px-2 py-1 bg-bg border border-border rounded max-w-xs w-full focus:border-accent outline-none font-mono"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
            />
          </div>
          <div className="space-y-3 divide-y divide-border/60">
            {filteredWorst.length === 0 ? (
              <p className="text-xs text-muted py-2 italic">No search results matching &quot;{search}&quot;</p>
            ) : (
              filteredWorst.map((w, i) => (
                <div key={w.id || i} className={`pt-3 ${i === 0 ? "pt-0" : ""} flex gap-3 items-start`}>
                  <div className="pt-0.5">
                    <ErrorAudioPlayer audioUri={w.audio_uri} />
                  </div>
                  <div className="flex-1 space-y-1 text-xs min-w-0">
                    <div className="flex items-center justify-between text-muted font-mono text-[10px]">
                      <span className="font-semibold text-accent/90">WER {(w.wer * 100).toFixed(1)}%</span>
                      <span className="flex items-center gap-1.5">
                        <span className="text-red-400 bg-red-400/5 border border-red-500/10 px-1 rounded-sm">Del {w.del}</span>
                        <span className="text-emerald-400 bg-emerald-400/5 border border-emerald-500/10 px-1 rounded-sm">Ins {w.ins}</span>
                        <span className="text-amber-400 bg-amber-400/5 border border-amber-500/10 px-1 rounded-sm">Sub {w.subs}</span>
                      </span>
                    </div>
                    <div className="grid grid-cols-1 gap-1.5 bg-black/20 p-2 rounded border border-border/40">
                      <div className="flex items-start gap-2">
                        <span className="text-[10px] font-semibold text-muted w-8 select-none pt-0.5 font-mono">REF</span>
                        <div className="flex-1">
                          <DiffText text={w.ref} otherText={w.hyp} type="ref" />
                        </div>
                      </div>
                      <div className="flex items-start gap-2">
                        <span className="text-[10px] font-semibold text-muted w-8 select-none pt-0.5 font-mono">HYP</span>
                        <div className="flex-1">
                          <DiffText text={w.hyp} otherText={w.ref} type="hyp" />
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              ))
            )}
          </div>
        </Card>
      )}
    </div>
  );
}

function SliceTable({
  sliceKey,
  buckets,
  baseline,
}: {
  sliceKey: string;
  buckets: Record<string, { wer: number; n: number }>;
  baseline: number;
}) {
  const rows = Object.entries(buckets).sort((a, b) => b[1].wer - a[1].wer);
  const maxWer = Math.max(0.01, ...rows.map(([, v]) => v.wer));
  return (
    <div>
      <div className="text-xs uppercase tracking-wide text-muted mb-1">By {sliceKey}</div>
      <div className="space-y-1">
        {rows.map(([bucket, v]) => {
          const width = (v.wer / maxWer) * 100;
          const worseThanOverall = v.wer > baseline * 1.1;
          return (
            <div key={bucket} className="text-xs flex items-center gap-2">
              <span className="w-24 truncate font-mono">{bucket}</span>
              <div className="flex-1 bg-border/30 rounded h-3 relative overflow-hidden">
                <div
                  className={`h-full ${worseThanOverall ? "bg-red-500/60" : "bg-accent/60"}`}
                  style={{ width: `${width}%` }}
                />
              </div>
              <span className="w-16 text-right font-mono">{(v.wer * 100).toFixed(1)}%</span>
              <span className="w-12 text-right text-muted">n={v.n}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------

function LLMEvalView({ metrics }: { metrics: Record<string, unknown> }) {
  const loss = metrics.eval_loss as number | undefined;
  const ppl = metrics.perplexity as number | undefined;
  const tokens = metrics.n_tokens as number | undefined;
  const n = metrics.n_samples as number | undefined;
  const lens = (metrics.assistant_token_lengths ?? {}) as Record<string, number>;
  return (
    <div className="space-y-3">
      <div className="grid grid-cols-4 gap-2">
        <HeadlineMetric label="Eval loss" value={loss} />
        <HeadlineMetric
          label="Perplexity"
          value={ppl !== undefined && Number.isFinite(ppl) ? ppl : "∞"}
        />
        <HeadlineMetric label="Samples" value={n} />
        <HeadlineMetric label="Tokens" value={tokens} />
      </div>
      <div className="border border-border rounded-md p-3">
        <div className="text-xs uppercase tracking-wide text-muted mb-2">
          Assistant token-length distribution
        </div>
        <div className="grid grid-cols-5 text-center text-xs">
          {["min", "median", "p95", "max", "n"].map((k) => (
            <div key={k}>
              <div className="text-muted">{k}</div>
              <div className="font-mono">{lens[k] ?? "—"}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------

function TTSEvalView({ metrics }: { metrics: Record<string, unknown> }) {
  const wer = metrics.round_trip_wer as number | undefined;
  const dur = metrics.mean_duration_s as number | undefined;
  const total = metrics.total_audio_s as number | undefined;
  const n = metrics.n_samples as number | undefined;
  return (
    <div>
      <div className="grid grid-cols-4 gap-2">
        <HeadlineMetric label="Round-trip WER" value={wer !== undefined ? wer * 100 : undefined} unit="%" />
        <HeadlineMetric label="Mean dur" value={dur} unit="s" />
        <HeadlineMetric label="Total audio" value={total} unit="s" />
        <HeadlineMetric label="Samples" value={n} />
      </div>
      <p className="text-xs text-muted mt-2">
        Round-trip WER is intelligibility via ASR; lower is better. MOS estimators (UTMOS / NISQA) and
        speaker similarity will appear here once those packages land in the <span className="font-mono">[tts_eval]</span> extras.
      </p>
    </div>
  );
}

// ---------------------------------------------------------------------------

function DiffText({ text, otherText, type }: { text: string; otherText: string; type: "ref" | "hyp" }) {
  if (!text) return <span className="text-muted italic">empty</span>;
  const clean = (t: string) => t.toLowerCase().replace(/[.,\/#!$%\^&\*;:{}=\-_`~()?]/g, "").trim();

  const otherWords = clean(otherText).split(/\s+/);
  const words = text.split(/\s+/);

  return (
    <span className="font-mono break-all text-xs">
      {words.map((w, idx) => {
        const cleanW = clean(w);
        const isMatch = otherWords.includes(cleanW);

        if (type === "ref") {
          // reference words not in hypothesis are DELETIONS (red strike-through)
          return (
            <span
              key={idx}
              className={`mr-1 inline-block ${!isMatch ? "bg-red-500/10 text-red-400 line-through px-0.5 rounded border border-red-500/20" : ""}`}
            >
              {w}
            </span>
          );
        } else {
          // hypothesis words not in reference are INSERTIONS (green underline)
          return (
            <span
              key={idx}
              className={`mr-1 inline-block ${!isMatch ? "bg-emerald-500/10 text-emerald-400 underline decoration-emerald-500/40 decoration-2 px-0.5 rounded border border-emerald-500/20" : ""}`}
            >
              {w}
            </span>
          );
        }
      })}
    </span>
  );
}

function ErrorAudioPlayer({ audioUri }: { audioUri?: string }) {
  const [playing, setPlaying] = useState(false);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  if (!audioUri) return null;

  const togglePlay = () => {
    if (!audioRef.current) {
      audioRef.current = new Audio(`/api/uploads/file?uri=${encodeURIComponent(audioUri)}`);
      audioRef.current.onended = () => setPlaying(false);
    }
    if (playing) {
      audioRef.current.pause();
      setPlaying(false);
    } else {
      audioRef.current.play().catch(() => {});
      setPlaying(true);
    }
  };

  return (
    <button
      onClick={togglePlay}
      className={`p-1.5 rounded-full border transition-all flex items-center justify-center ${
        playing
          ? "bg-accent/15 border-accent text-accent animate-pulse"
          : "bg-border/20 border-border text-muted hover:text-fg hover:bg-border/40"
      }`}
      title={playing ? "Pause Audio" : "Play Audio"}
    >
      {playing ? (
        <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
          <rect x="14" y="4" width="4" height="16" rx="1" />
          <rect x="6" y="4" width="4" height="16" rx="1" />
        </svg>
      ) : (
        <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="currentColor" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
          <polygon points="6 3 20 12 6 21 6 3" />
        </svg>
      )}
    </button>
  );
}
