"use client";

import { use, useState } from "react";
import useSWR from "swr";
import Link from "next/link";
import { PageHeader } from "@/components/PageHeader";
import { Card } from "@/components/Card";

type Sample = {
  id: string;
  modality: string;
  transcript?: string;
  language?: string | null;
  split?: string;
  audio?: { uri: string; duration_s: number; sample_rate: number } | null;
  metadata?: Record<string, unknown> | null;
};

async function jget<T>(p: string): Promise<T> {
  const r = await fetch(`/api${p}`, { cache: "no-store" });
  if (!r.ok) throw new Error(`${r.status}: ${await r.text()}`);
  return r.json();
}

export default function SampleEditorPage({
  params,
}: {
  params: any;
}) {
  const resolvedParams = params && typeof params.then === "function" ? use(params) : params;
  const { id, vid } = resolvedParams;
  const [offset, setOffset] = useState(0);
  const limit = 1;
  const { data, isLoading, mutate } = useSWR<{ total: number; items: Sample[] }>(
    ["sample", vid, offset],
    () => jget(`/datasets/versions/${vid}/samples?offset=${offset}&limit=${limit}`)
  );

  const sample = data?.items[0];

  return (
    <>
      <PageHeader
        title="Review & correct"
        subtitle="Listen, edit the transcript, advance. Marks samples as reviewed."
        actions={
          <Link href={`/datasets/${id}`} className="px-3 py-1.5 rounded-md border border-border text-sm">
            ← Dataset
          </Link>
        }
      />

      <Card>
        {isLoading || !data ? (
          <p className="text-muted text-sm">loading…</p>
        ) : !sample ? (
          <p className="text-muted text-sm">No more samples.</p>
        ) : (
          <SampleEditor
            key={sample.id}
            sample={sample}
            position={offset + 1}
            total={data.total}
            onSaved={async () => {
              await mutate();
              setOffset(Math.min(offset + 1, data.total - 1));
            }}
            onPrev={() => setOffset(Math.max(0, offset - 1))}
            onNext={() => setOffset(Math.min(data.total - 1, offset + 1))}
            versionId={vid}
          />
        )}
      </Card>
    </>
  );
}

function SampleEditor({
  sample,
  position,
  total,
  onSaved,
  onPrev,
  onNext,
  versionId,
}: {
  sample: Sample;
  position: number;
  total: number;
  onSaved: () => void;
  onPrev: () => void;
  onNext: () => void;
  versionId: string;
}) {
  const [transcript, setTranscript] = useState(sample.transcript ?? "");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  async function save() {
    setBusy(true);
    setErr(null);
    try {
      const r = await fetch(`/api/datasets/versions/${versionId}/samples/${sample.id}`, {
        method: "PATCH",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          transcript,
          metadata: { ...(sample.metadata ?? {}), needs_review: false },
        }),
      });
      if (!r.ok) throw new Error(`${r.status}: ${await r.text()}`);
      await onSaved();
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-3 text-xs text-muted">
        <span>
          Sample {position} / {total} · id <span className="font-mono">{sample.id.slice(0, 8)}</span>
        </span>
        <div className="flex gap-2">
          <button onClick={onPrev} className="px-2 py-1 border border-border rounded">
            ← Prev
          </button>
          <button onClick={onNext} className="px-2 py-1 border border-border rounded">
            Next →
          </button>
        </div>
      </div>

      {sample.audio?.uri && (
        <audio
          controls
          className="w-full mb-3"
          src={`/api/uploads/file?uri=${encodeURIComponent(sample.audio.uri)}`}
        />
      )}

      <label className="block text-xs text-muted mb-1">Transcript</label>
      <textarea
        value={transcript}
        onChange={(e) => setTranscript(e.target.value)}
        rows={4}
        className="w-full bg-bg border border-border rounded p-2 text-sm font-mono"
      />

      <div className="grid grid-cols-3 gap-3 mt-3 text-xs">
        <Meta label="Language" value={sample.language ?? "—"} />
        <Meta label="Split" value={sample.split ?? "train"} />
        <Meta
          label="Duration"
          value={sample.audio ? `${sample.audio.duration_s.toFixed(2)}s` : "—"}
        />
      </div>

      {err && <p className="text-red-400 text-sm mt-2">{err}</p>}

      <div className="flex justify-end gap-2 mt-3">
        <button
          onClick={save}
          disabled={busy}
          className="px-3 py-1.5 rounded-md bg-accent text-white text-sm font-medium disabled:opacity-50"
        >
          {busy ? "Saving…" : "Save & next"}
        </button>
      </div>
    </div>
  );
}

function Meta({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="text-muted">{label}</div>
      <div className="font-mono">{value}</div>
    </div>
  );
}
