"use client";

import { use, useState } from "react";
import useSWR from "swr";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { api } from "@/lib/api";
import { PageHeader } from "@/components/PageHeader";
import { Card, CardTitle } from "@/components/Card";
import { AudioUploader, type UploadResult } from "@/components/AudioUploader";
import { relativeTime } from "@/lib/utils";
import { LANGUAGES } from "@/lib/languages";

type DatasetVersion = {
  id: string;
  version: string;
  manifest_uri: string;
  num_samples: number;
  total_audio_s: number;
  notes: string | null;
  created_at: string;
  parent_version_id: string | null;
};

type Dataset = {
  id: string;
  name: string;
  slug: string;
  modality: string;
  description: string | null;
  source: string | null;
  created_at: string;
};

async function jget<T>(path: string): Promise<T> {
  const r = await fetch(`/api${path}`, { cache: "no-store" });
  if (!r.ok) throw new Error(`${r.status}: ${await r.text()}`);
  return r.json();
}

export default function DatasetDetailPage({ params }: { params: any }) {
  const resolvedParams = params && typeof params.then === "function" ? use(params) : params;
  const { id } = resolvedParams;
  const ds = useSWR<Dataset>(["dataset", id], () => jget<Dataset>(`/datasets/${id}`));
  const versions = useSWR<DatasetVersion[]>(["versions", id], () =>
    jget<DatasetVersion[]>(`/datasets/${id}/versions`)
  );
  const [activeVersion, setActiveVersion] = useState<string | null>(null);
  const versionId = activeVersion ?? versions.data?.[0]?.id ?? null;

  return (
    <>
      <PageHeader
        title={ds.data?.name ?? "Dataset"}
        subtitle={ds.data ? `${ds.data.slug} · ${ds.data.modality.toUpperCase()}${ds.data.source ? ` · ${ds.data.source}` : ""}` : ""}
        actions={
          <Link href="/datasets" className="px-3 py-1.5 rounded-md border border-border text-sm">
            ← All datasets
          </Link>
        }
      />

      {ds.data?.modality === "asr" && (
        <div className="mb-4">
          <BuildSection
            datasetId={id}
            projectId={(ds.data as { project_id?: string } | undefined)?.project_id ?? ""}
            onJobQueued={() => versions.mutate()}
          />
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="md:col-span-1 space-y-3">
          <Card>
            <h2 className="text-sm font-medium mb-2">Versions</h2>
            {versions.isLoading ? (
              <p className="text-muted text-sm">loading…</p>
            ) : !versions.data?.length ? (
              <p className="text-muted text-sm">No versions yet — import or build one.</p>
            ) : (
              <>
                <ul className="space-y-1">
                  {versions.data.map((v) => {
                    const parent = v.parent_version_id
                      ? versions.data?.find((x) => x.id === v.parent_version_id)
                      : null;
                    return (
                      <li key={v.id}>
                        <button
                          onClick={() => setActiveVersion(v.id)}
                          className={`w-full text-left rounded-md px-2 py-1.5 text-sm ${
                            versionId === v.id ? "bg-accent/10 text-accent" : "hover:bg-border/40"
                          }`}
                        >
                          <div className="flex items-center justify-between">
                            <span className="font-mono">{v.version}</span>
                            <span className="text-xs text-muted">{v.num_samples} samples</span>
                          </div>
                          <div className="text-xs text-muted flex items-center gap-2">
                            <span>{relativeTime(v.created_at)}</span>
                            {parent && (
                              <span
                                className="rounded-full bg-border/40 px-1.5 py-0.5 font-mono text-[10px]"
                                title={`forked from ${parent.version}`}
                              >
                                ← {parent.version}
                              </span>
                            )}
                          </div>
                        </button>
                      </li>
                    );
                  })}
                </ul>
                {versionId && (
                  <>
                    <ForkButton
                      datasetId={id}
                      parentId={versionId}
                      parentVersion={versions.data.find((v) => v.id === versionId)?.version ?? ""}
                      onForked={() => versions.mutate()}
                    />
                    <AssignSplitsButton
                      datasetId={id}
                      parentId={versionId}
                      parentVersion={versions.data.find((v) => v.id === versionId)?.version ?? ""}
                      onSplit={() => versions.mutate()}
                    />
                    <ApplyFiltersButton
                      datasetId={id}
                      parentId={versionId}
                      parentVersion={versions.data.find((v) => v.id === versionId)?.version ?? ""}
                      onFilter={() => versions.mutate()}
                    />
                    <ApplyDedupButton
                      datasetId={id}
                      parentId={versionId}
                      parentVersion={versions.data.find((v) => v.id === versionId)?.version ?? ""}
                      onDedup={() => versions.mutate()}
                    />
                    {ds.data?.modality && ["tts", "llm", "s2s"].includes(ds.data.modality) && (
                      <Link
                        href={`/datasets/${id}/build-${ds.data.modality}`}
                        className="mt-3 block w-full text-center rounded-md bg-accent px-2 py-2 text-xs font-semibold text-white hover:bg-accent/90 transition shadow-sm"
                      >
                        🛠️ Open {ds.data.modality.toUpperCase()} Builder
                      </Link>
                    )}
                  </>
                )}
              </>
            )}
          </Card>
        </div>

        <div className="md:col-span-2">
          {versionId ? (
            <SamplePreview datasetId={id} versionId={versionId} />
          ) : (
            <Card>
              {ds.data && ds.data.modality !== "asr" && !versions.data?.length ? (
                <InitializeVersionSection
                  datasetId={id}
                  onCreated={(newVer) => {
                    versions.mutate();
                    setActiveVersion(newVer.id);
                  }}
                />
              ) : (
                <p className="text-sm text-muted">Select a version to preview samples.</p>
              )}
            </Card>
          )}
        </div>
      </div>
    </>
  );
}

function InitializeVersionSection({
  datasetId,
  onCreated,
}: {
  datasetId: string;
  onCreated: (newVersion: DatasetVersion) => void;
}) {
  const [version, setVersion] = useState("0.1.0");
  const [licenseSpdx, setLicenseSpdx] = useState("CC-BY-4.0");
  const [notes, setNotes] = useState("Initialized via Studio Dashboard");
  const [submitting, setSubmitting] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setErr(null);
    if (!/^\d+\.\d+\.\d+$/.test(version)) {
      setErr("Version must follow semantic versioning format (e.g., 0.1.0)");
      return;
    }
    setSubmitting(true);
    try {
      const token = typeof window !== "undefined" ? window.localStorage.getItem("oas_token") : null;
      const headers: Record<string, string> = { "Content-Type": "application/json" };
      if (token) {
        headers["Authorization"] = `Bearer ${token}`;
      }

      const r = await fetch(`/api/datasets/${datasetId}/versions/init`, {
        method: "POST",
        headers,
        body: JSON.stringify({
          version,
          license: { spdx: licenseSpdx },
          notes: notes || null,
        }),
      });

      if (!r.ok) {
        const text = await r.text();
        throw new Error(`${r.status} ${r.statusText}: ${text}`);
      }

      const newVersion = await r.json();
      onCreated(newVersion);
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="space-y-4">
      <div>
        <h3 className="text-lg font-semibold text-accent mb-1 flex items-center gap-2">
          <span>🚀</span> Initialize Dataset Version
        </h3>
        <p className="text-xs text-muted">
          Every custom dataset needs an active version manifest to begin collecting and preparing samples. Initialize your first version (e.g., <code className="font-mono text-accent">0.1.0</code>) below.
        </p>
      </div>

      <form onSubmit={handleSubmit} className="space-y-4 pt-2">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="space-y-1">
            <label className="block text-xs font-medium text-muted">Version</label>
            <input
              type="text"
              className="w-full bg-bg border border-border rounded-md px-3 py-2 text-sm font-mono text-accent focus:outline-none focus:border-accent"
              value={version}
              onChange={(e) => setVersion(e.target.value)}
              pattern="\d+\.\d+\.\d+"
              required
              placeholder="e.g. 0.1.0"
            />
          </div>

          <div className="space-y-1">
            <label className="block text-xs font-medium text-muted">Default License (SPDX)</label>
            <input
              type="text"
              className="w-full bg-bg border border-border rounded-md px-3 py-2 text-sm focus:outline-none focus:border-accent"
              value={licenseSpdx}
              onChange={(e) => setLicenseSpdx(e.target.value)}
              required
              placeholder="e.g. CC-BY-4.0"
            />
          </div>
        </div>

        <div className="space-y-1">
          <label className="block text-xs font-medium text-muted">Notes / Version Description</label>
          <textarea
            rows={2}
            className="w-full bg-bg border border-border rounded-md px-3 py-2 text-sm focus:outline-none focus:border-accent"
            value={notes}
            onChange={(e) => setNotes(e.target.value)}
            placeholder="Describe what is in this version..."
          />
        </div>

        {err && (
          <div className="p-3 bg-red-500/10 border border-red-500/20 rounded-md">
            <p className="text-red-400 text-xs leading-relaxed font-mono">{err}</p>
          </div>
        )}

        <div className="flex justify-end gap-3 pt-2">
          <button
            type="submit"
            disabled={submitting}
            className="px-4 py-2 bg-accent text-white rounded-md text-sm font-medium transition duration-150 disabled:opacity-50 flex items-center gap-2"
          >
            {submitting ? (
              <>
                <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                </svg>
                Initializing...
              </>
            ) : (
              "Initialize version 0.1.0"
            )}
          </button>
        </div>
      </form>
    </div>
  );
}


function BuildSection({
  datasetId,
  projectId,
  onJobQueued,
}: {
  datasetId: string;
  projectId: string;
  onJobQueued: () => void;
}) {
  const router = useRouter();
  const [uploads, setUploads] = useState<UploadResult[]>([]);
  const [version, setVersion] = useState("0.1.0");
  const [language, setLanguage] = useState("en");
  const [whisperModel, setWhisperModel] = useState("openai/whisper-tiny");
  const [licenseSpdx, setLicenseSpdx] = useState("CC-BY-4.0");
  const [submitting, setSubmitting] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  async function bootstrap() {
    setErr(null);
    setSubmitting(true);
    try {
      if (uploads.length === 0) throw new Error("upload at least one audio file");
      const job = await api.jobs.submit({
        project_id: projectId,
        kind: "asr_bootstrap",
        name: `bootstrap ${uploads.length} file(s)`,
        config: {
          audio_uris: uploads.map((u) => u.uri),
          dataset_id: datasetId,
          version,
          language,
          whisper_model: whisperModel,
          license: { spdx: licenseSpdx },
        },
      });
      onJobQueued();
      router.push(`/jobs/${job.id}`);
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <Card>
      <div className="flex items-center justify-between mb-3">
        <CardTitle>Build a new version (audio → VAD → Whisper auto-transcribe)</CardTitle>
        <span className="text-[11px] text-muted">{uploads.length} uploaded</span>
      </div>

      <AudioUploader onUploaded={(rs) => setUploads((prev) => [...prev, ...rs])} />

      <div className="grid grid-cols-5 gap-3 mt-4 text-sm">
        <Field label="Version">
          <input className="input" pattern="\d+\.\d+\.\d+" value={version} onChange={(e) => setVersion(e.target.value)} />
        </Field>
        <Field label="Preset Language">
          <select
            className="input"
            value={language}
            onChange={(e) => setLanguage(e.target.value)}
          >
            {LANGUAGES.map((l) => (
              <option key={l.value} value={l.value}>
                {l.label} ({l.value})
              </option>
            ))}
          </select>
        </Field>
        <Field label="Or Custom ISO Code">
          <input className="input" placeholder="e.g. hi-IN, mr" value={language} onChange={(e) => setLanguage(e.target.value)} />
        </Field>
        <Field label="Whisper bootstrap model">
          <select className="input" value={whisperModel} onChange={(e) => setWhisperModel(e.target.value)}>
            <option>openai/whisper-tiny</option>
            <option>openai/whisper-base</option>
            <option>openai/whisper-small</option>
            <option>openai/whisper-medium</option>
            <option>openai/whisper-large-v3</option>
          </select>
        </Field>
        <Field label="License (SPDX)">
          <input className="input" value={licenseSpdx} onChange={(e) => setLicenseSpdx(e.target.value)} />
        </Field>
      </div>

      {err && <p className="text-red-400 text-sm mt-2">{err}</p>}
      <div className="flex justify-end mt-2">
        <button
          onClick={bootstrap}
          disabled={submitting || uploads.length === 0}
          className="px-3 py-1.5 rounded-md bg-accent text-white text-sm font-medium disabled:opacity-50"
        >
          {submitting ? "Submitting…" : "Bootstrap dataset version"}
        </button>
      </div>

      <style jsx global>{`
        .input {
          width: 100%;
          background: rgb(var(--bg));
          border: 1px solid rgb(var(--border));
          border-radius: 6px;
          padding: 6px 10px;
          font-size: 14px;
        }
      `}</style>
    </Card>
  );
}

function ForkButton({
  datasetId,
  parentId,
  parentVersion,
  onForked,
}: {
  datasetId: string;
  parentId: string;
  parentVersion: string;
  onForked: () => void;
}) {
  const [open, setOpen] = useState(false);
  const [version, setVersion] = useState("");
  const [notes, setNotes] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  async function submit() {
    setErr(null);
    if (!/^\d+\.\d+\.\d+$/.test(version)) {
      setErr("version must look like 0.2.0");
      return;
    }
    setBusy(true);
    try {
      const r = await fetch(
        `/api/datasets/${datasetId}/versions/${parentId}/fork`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ version, notes: notes || null }),
        },
      );
      if (!r.ok) throw new Error(`${r.status}: ${await r.text()}`);
      setOpen(false);
      setVersion("");
      setNotes("");
      onForked();
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  if (!open) {
    return (
      <button
        onClick={() => setOpen(true)}
        className="mt-3 w-full rounded-md border border-border px-2 py-1.5 text-xs text-muted hover:bg-border/40"
      >
        Fork from {parentVersion} →
      </button>
    );
  }
  return (
    <div className="mt-3 space-y-2 rounded-md border border-border p-2">
      <div className="text-xs text-muted">Forking from <span className="font-mono">{parentVersion}</span></div>
      <input
        className="input"
        placeholder="new version (e.g. 0.2.0)"
        value={version}
        onChange={(e) => setVersion(e.target.value)}
      />
      <input
        className="input"
        placeholder="notes (optional)"
        value={notes}
        onChange={(e) => setNotes(e.target.value)}
      />
      {err && <p className="text-red-400 text-xs">{err}</p>}
      <div className="flex justify-end gap-2">
        <button
          onClick={() => setOpen(false)}
          className="rounded-md px-2 py-1 text-xs text-muted hover:bg-border/40"
        >
          Cancel
        </button>
        <button
          onClick={submit}
          disabled={busy}
          className="rounded-md bg-accent px-2 py-1 text-xs font-medium text-white disabled:opacity-50"
        >
          {busy ? "Forking…" : "Fork"}
        </button>
      </div>
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <label className="block text-xs text-muted mb-1">{label}</label>
      {children}
    </div>
  );
}

function SamplePreview({ datasetId, versionId }: { datasetId: string; versionId: string }) {
  const [offset, setOffset] = useState(0);
  const limit = 25;
  const page = useSWR<{ total: number; items: Record<string, unknown>[] }>(
    ["samples", versionId, offset],
    () => jget(`/datasets/versions/${versionId}/samples?offset=${offset}&limit=${limit}`)
  );

  return (
    <Card>
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-sm font-medium">
          Samples{" "}
          <Link
            href={`/datasets/${datasetId}/versions/${versionId}/edit`}
            className="ml-2 text-xs text-accent"
          >
            review →
          </Link>
        </h2>
        <div className="flex items-center gap-2 text-xs text-muted">
          {page.data && (
            <>
              <span>
                {offset + 1}–{Math.min(offset + limit, page.data.total)} of {page.data.total}
              </span>
              <button
                disabled={offset === 0}
                onClick={() => setOffset(Math.max(0, offset - limit))}
                className="px-2 py-1 border border-border rounded disabled:opacity-30"
              >
                Prev
              </button>
              <button
                disabled={offset + limit >= page.data.total}
                onClick={() => setOffset(offset + limit)}
                className="px-2 py-1 border border-border rounded disabled:opacity-30"
              >
                Next
              </button>
            </>
          )}
        </div>
      </div>

      {page.isLoading ? (
        <p className="text-muted text-sm">loading…</p>
      ) : !page.data?.items.length ? (
        <p className="text-muted text-sm">No samples.</p>
      ) : (
        <div className="space-y-2 max-h-[60vh] overflow-y-auto">
          {page.data.items.map((s, i) => (
            <details key={i} className="rounded-md border border-border p-2">
              <summary className="cursor-pointer text-sm">
                <span className="font-mono text-xs text-muted">#{offset + i}</span>{" "}
                <span className="text-accent text-xs">{String(s.modality ?? "")}</span>{" "}
                {String((s as { transcript?: string; text?: string }).transcript ?? (s as { text?: string }).text ?? "—").slice(0, 100)}
              </summary>
              <pre className="mt-2 text-xs font-mono overflow-x-auto bg-bg p-2 rounded">
                {JSON.stringify(s, null, 2)}
              </pre>
            </details>
          ))}
        </div>
      )}
    </Card>
  );
}


function AssignSplitsButton({
  datasetId,
  parentId,
  parentVersion,
  onSplit,
}: {
  datasetId: string;
  parentId: string;
  parentVersion: string;
  onSplit: () => void;
}) {
  const [open, setOpen] = useState(false);
  const [version, setVersion] = useState("");
  const [train, setTrain] = useState(80);
  const [val, setVal] = useState(10);
  const [test, setTest] = useState(10);
  const [holdout, setHoldout] = useState(0);
  const [strategy, setStrategy] = useState<"random" | "speaker_disjoint">("random");
  const [notes, setNotes] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  async function submit() {
    setErr(null);
    if (!/^\d+\.\d+\.\d+$/.test(version)) {
      setErr("version must look like 0.2.0");
      return;
    }
    const sum = Number(train) + Number(val) + Number(test) + Number(holdout);
    if (Math.abs(sum - 100) > 0.01) {
      setErr(`Split percentages must sum to 100 (current: ${sum})`);
      return;
    }

    setBusy(true);
    try {
      const r = await fetch(
        `/api/datasets/${datasetId}/versions/${parentId}/split`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            version,
            train_pct: Number(train) / 100,
            val_pct: Number(val) / 100,
            test_pct: Number(test) / 100,
            holdout_pct: Number(holdout) / 100,
            strategy,
            notes: notes || null,
          }),
        },
      );
      if (!r.ok) throw new Error(`${r.status}: ${await r.text()}`);
      setOpen(false);
      setVersion("");
      setNotes("");
      onSplit();
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  if (!open) {
    return (
      <button
        onClick={() => setOpen(true)}
        className="mt-2 w-full rounded-md border border-border px-2 py-1.5 text-xs text-muted hover:bg-border/40"
      >
        Assign splits to {parentVersion} →
      </button>
    );
  }

  return (
    <div className="mt-3 space-y-2 rounded-md border border-border p-3 bg-bg/50">
      <div className="text-xs text-muted font-medium mb-1">
        Assign splits to <span className="font-mono">{parentVersion}</span>
      </div>

      <div className="space-y-1.5">
        <label className="block text-[11px] text-muted font-medium">New Version Name</label>
        <input
          className="input"
          placeholder="e.g. 0.2.0"
          value={version}
          onChange={(e) => setVersion(e.target.value)}
        />
      </div>

      <div className="grid grid-cols-2 gap-2 text-xs">
        <div>
          <label className="block text-[11px] text-muted font-medium">Train %</label>
          <input
            type="number"
            className="input font-mono"
            value={train}
            min={0}
            max={100}
            onChange={(e) => setTrain(Number(e.target.value))}
          />
        </div>
        <div>
          <label className="block text-[11px] text-muted font-medium">Val %</label>
          <input
            type="number"
            className="input font-mono"
            value={val}
            min={0}
            max={100}
            onChange={(e) => setVal(Number(e.target.value))}
          />
        </div>
        <div>
          <label className="block text-[11px] text-muted font-medium">Test %</label>
          <input
            type="number"
            className="input font-mono"
            value={test}
            min={0}
            max={100}
            onChange={(e) => setTest(Number(e.target.value))}
          />
        </div>
        <div>
          <label className="block text-[11px] text-muted font-medium">Holdout %</label>
          <input
            type="number"
            className="input font-mono"
            value={holdout}
            min={0}
            max={100}
            onChange={(e) => setHoldout(Number(e.target.value))}
          />
        </div>
      </div>

      <div className="space-y-1.5">
        <label className="block text-[11px] text-muted font-medium">Splitting Strategy</label>
        <select
          className="input bg-transparent"
          value={strategy}
          onChange={(e) => setStrategy(e.target.value as "random" | "speaker_disjoint")}
        >
          <option value="random">Random Partitioning</option>
          <option value="speaker_disjoint">Speaker Disjoint (disallow overlap)</option>
        </select>
        <p className="text-[10px] text-muted leading-tight">
          {strategy === "speaker_disjoint"
            ? "Disjoint ensures samples of a given speaker do not cross splits (train vs test)."
            : "Random shuffles samples individually across the split bins."}
        </p>
      </div>

      <div className="space-y-1.5">
        <label className="block text-[11px] text-muted font-medium">Notes</label>
        <input
          className="input"
          placeholder="optional notes"
          value={notes}
          onChange={(e) => setNotes(e.target.value)}
        />
      </div>

      {err && <p className="text-red-400 text-xs mt-1 leading-tight">{err}</p>}

      <div className="flex justify-end gap-2 pt-1">
        <button
          onClick={() => setOpen(false)}
          className="rounded-md px-2 py-1 text-xs text-muted hover:bg-border/40"
        >
          Cancel
        </button>
        <button
          onClick={submit}
          disabled={busy}
          className="rounded-md bg-accent px-2 py-1 text-xs font-medium text-white disabled:opacity-50"
        >
          {busy ? "Running…" : "Assign Splits"}
        </button>
      </div>
    </div>
  );
}


function ApplyFiltersButton({
  datasetId,
  parentId,
  parentVersion,
  onFilter,
}: {
  datasetId: string;
  parentId: string;
  parentVersion: string;
  onFilter: () => void;
}) {
  const [open, setOpen] = useState(false);
  const [version, setVersion] = useState("");
  const [minDuration, setMinDuration] = useState("");
  const [maxDuration, setMaxDuration] = useState("");
  const [minSnr, setMinSnr] = useState("");
  const [minQuality, setMinQuality] = useState("");
  const [minText, setMinText] = useState("");
  const [maxText, setMaxText] = useState("");
  const [notes, setNotes] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  async function submit() {
    setErr(null);
    if (!/^\d+\.\d+\.\d+$/.test(version)) {
      setErr("version must look like 0.2.0");
      return;
    }
    if (minDuration && maxDuration && Number(minDuration) > Number(maxDuration)) {
      setErr("Min duration must be <= Max duration");
      return;
    }
    if (minText && maxText && Number(minText) > Number(maxText)) {
      setErr("Min text length must be <= Max text length");
      return;
    }

    setBusy(true);
    try {
      const r = await fetch(
        `/api/datasets/${datasetId}/versions/${parentId}/filter`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            version,
            min_duration_s: minDuration ? Number(minDuration) : null,
            max_duration_s: maxDuration ? Number(maxDuration) : null,
            min_snr_db: minSnr ? Number(minSnr) : null,
            min_quality_score: minQuality ? Number(minQuality) : null,
            min_text_len: minText ? Number(minText) : null,
            max_text_len: maxText ? Number(maxText) : null,
            notes: notes || null,
          }),
        },
      );
      if (!r.ok) throw new Error(`${r.status}: ${await r.text()}`);
      setOpen(false);
      setVersion("");
      setMinDuration("");
      setMaxDuration("");
      setMinSnr("");
      setMinQuality("");
      setMinText("");
      setMaxText("");
      setNotes("");
      onFilter();
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  if (!open) {
    return (
      <button
        onClick={() => setOpen(true)}
        className="mt-2 w-full rounded-md border border-border px-2 py-1.5 text-xs text-muted hover:bg-border/40"
      >
        Filter quality of {parentVersion} →
      </button>
    );
  }

  return (
    <div className="mt-3 space-y-2 rounded-md border border-border p-3 bg-bg/50">
      <div className="text-xs text-muted font-medium mb-1">
        Apply quality filters to <span className="font-mono">{parentVersion}</span>
      </div>

      <div className="space-y-1.5">
        <label className="block text-[11px] text-muted font-medium">New Version Name</label>
        <input
          className="input"
          placeholder="e.g. 0.2.0"
          value={version}
          onChange={(e) => setVersion(e.target.value)}
        />
      </div>

      <div className="grid grid-cols-2 gap-2 text-xs">
        <div>
          <label className="block text-[11px] text-muted font-medium">Min Duration (s)</label>
          <input
            type="number"
            step="any"
            className="input font-mono"
            placeholder="none"
            value={minDuration}
            onChange={(e) => setMinDuration(e.target.value)}
          />
        </div>
        <div>
          <label className="block text-[11px] text-muted font-medium">Max Duration (s)</label>
          <input
            type="number"
            step="any"
            className="input font-mono"
            placeholder="none"
            value={maxDuration}
            onChange={(e) => setMaxDuration(e.target.value)}
          />
        </div>
        <div>
          <label className="block text-[11px] text-muted font-medium">Min SNR (dB)</label>
          <input
            type="number"
            step="any"
            className="input font-mono"
            placeholder="none"
            value={minSnr}
            onChange={(e) => setMinSnr(e.target.value)}
          />
        </div>
        <div>
          <label className="block text-[11px] text-muted font-medium">Min Quality (0-1)</label>
          <input
            type="number"
            step="any"
            className="input font-mono"
            placeholder="none"
            value={minQuality}
            onChange={(e) => setMinQuality(e.target.value)}
          />
        </div>
        <div>
          <label className="block text-[11px] text-muted font-medium">Min Text Length</label>
          <input
            type="number"
            className="input font-mono"
            placeholder="none"
            value={minText}
            onChange={(e) => setMinText(e.target.value)}
          />
        </div>
        <div>
          <label className="block text-[11px] text-muted font-medium">Max Text Length</label>
          <input
            type="number"
            className="input font-mono"
            placeholder="none"
            value={maxText}
            onChange={(e) => setMaxText(e.target.value)}
          />
        </div>
      </div>

      <div className="space-y-1.5">
        <label className="block text-[11px] text-muted font-medium">Notes</label>
        <input
          className="input"
          placeholder="optional filter details"
          value={notes}
          onChange={(e) => setNotes(e.target.value)}
        />
      </div>

      {err && <p className="text-red-400 text-xs mt-1 leading-tight">{err}</p>}

      <div className="flex justify-end gap-2 pt-1">
        <button
          onClick={() => setOpen(false)}
          className="rounded-md px-2 py-1 text-xs text-muted hover:bg-border/40"
        >
          Cancel
        </button>
        <button
          onClick={submit}
          disabled={busy}
          className="rounded-md bg-accent px-2 py-1 text-xs font-medium text-white disabled:opacity-50"
        >
          {busy ? "Filtering…" : "Apply Filters"}
        </button>
      </div>
    </div>
  );
}


function ApplyDedupButton({
  datasetId,
  parentId,
  parentVersion,
  onDedup,
}: {
  datasetId: string;
  parentId: string;
  parentVersion: string;
  onDedup: () => void;
}) {
  const [open, setOpen] = useState(false);
  const [version, setVersion] = useState("");
  const [strategy, setStrategy] = useState<"exact_text" | "audio_hash" | "similar_text">("exact_text");
  const [threshold, setThreshold] = useState(85);
  const [notes, setNotes] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  async function submit() {
    setErr(null);
    if (!/^\d+\.\d+\.\d+$/.test(version)) {
      setErr("version must look like 0.2.0");
      return;
    }

    setBusy(true);
    try {
      const r = await fetch(
        `/api/datasets/${datasetId}/versions/${parentId}/dedup`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            version,
            strategy,
            threshold: Number(threshold) / 100,
            notes: notes || null,
          }),
        },
      );
      if (!r.ok) throw new Error(`${r.status}: ${await r.text()}`);
      setOpen(false);
      setVersion("");
      setNotes("");
      onDedup();
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  if (!open) {
    return (
      <button
        onClick={() => setOpen(true)}
        className="mt-2 w-full rounded-md border border-border px-2 py-1.5 text-xs text-muted hover:bg-border/40"
      >
        Deduplicate {parentVersion} →
      </button>
    );
  }

  return (
    <div className="mt-3 space-y-2 rounded-md border border-border p-3 bg-bg/50">
      <div className="text-xs text-muted font-medium mb-1">
        Deduplicate dataset <span className="font-mono">{parentVersion}</span>
      </div>

      <div className="space-y-1.5">
        <label className="block text-[11px] text-muted font-medium">New Version Name</label>
        <input
          className="input"
          placeholder="e.g. 0.2.0"
          value={version}
          onChange={(e) => setVersion(e.target.value)}
        />
      </div>

      <div className="space-y-1.5">
        <label className="block text-[11px] text-muted font-medium">Deduplication Strategy</label>
        <select
          className="input bg-transparent"
          value={strategy}
          onChange={(e) => setStrategy(e.target.value as "exact_text" | "audio_hash" | "similar_text")}
        >
          <option value="exact_text">Exact Text Match</option>
          <option value="audio_hash">Audio Hash (SHA256) Match</option>
          <option value="similar_text">Similar Text (Jaccard) Match</option>
        </select>
      </div>

      {strategy === "similar_text" && (
        <div className="space-y-1">
          <div className="flex justify-between text-[11px]">
            <span className="text-muted font-medium">Similarity Threshold</span>
            <span className="font-mono font-medium">{threshold}%</span>
          </div>
          <input
            type="range"
            min="50"
            max="100"
            className="w-full h-1 bg-border rounded-lg appearance-none cursor-pointer accent-accent"
            value={threshold}
            onChange={(e) => setThreshold(Number(e.target.value))}
          />
          <p className="text-[10px] text-muted leading-tight">
            Discard sample if it shares at least {threshold}% of its vocabulary words with a retained sample.
          </p>
        </div>
      )}

      <div className="space-y-1.5">
        <label className="block text-[11px] text-muted font-medium">Notes</label>
        <input
          className="input"
          placeholder="optional deduplication details"
          value={notes}
          onChange={(e) => setNotes(e.target.value)}
        />
      </div>

      {err && <p className="text-red-400 text-xs mt-1 leading-tight">{err}</p>}

      <div className="flex justify-end gap-2 pt-1">
        <button
          onClick={() => setOpen(false)}
          className="rounded-md px-2 py-1 text-xs text-muted hover:bg-border/40"
        >
          Cancel
        </button>
        <button
          onClick={submit}
          disabled={busy}
          className="rounded-md bg-accent px-2 py-1 text-xs font-medium text-white disabled:opacity-50"
        >
          {busy ? "Running…" : "Deduplicate"}
        </button>
      </div>
    </div>
  );
}



