"use client";

import { useMemo, useState } from "react";
import useSWR from "swr";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { api, type Project } from "@/lib/api";
import { PageHeader } from "@/components/PageHeader";
import { Card, CardTitle } from "@/components/Card";
import { relativeTime } from "@/lib/utils";

type Pipeline = {
  id: string;
  project_id: string;
  slug: string;
  name: string;
  description: string | null;
  asr_version_id: string | null;
  llm_version_id: string | null;
  tts_version_id: string | null;
  asr_fallback: string | null;
  llm_fallback: string | null;
  tts_fallback: string | null;
  system_prompt: string | null;
  created_at: string;
};

async function jget<T>(p: string): Promise<T> {
  const r = await fetch(`/api${p}`, { cache: "no-store" });
  if (!r.ok) throw new Error(`${r.status}: ${await r.text()}`);
  return r.json();
}

export default function S2SPage() {
  const projects = useSWR("projects", () => api.projects.list());
  const pipelines = useSWR<Pipeline[]>("pipelines", () => jget<Pipeline[]>("/s2s/pipelines"));
  const [showCreate, setShowCreate] = useState(false);

  return (
    <>
      <PageHeader
        title="Speech-to-Speech Studio"
        subtitle="Wire an ASR + LLM + TTS into a realtime voice assistant."
        actions={
          <button
            onClick={() => setShowCreate(true)}
            className="px-3 py-1.5 rounded-md bg-accent text-white text-sm font-medium"
          >
            New pipeline
          </button>
        }
      />

      {pipelines.isLoading ? (
        <p className="text-muted text-sm">loading…</p>
      ) : !pipelines.data?.length ? (
        <Card>
          <p className="text-sm text-muted">
            No pipelines yet. Create one — you can use registered models or fall back to HF model
            IDs (e.g. <span className="font-mono">openai/whisper-tiny</span>).
          </p>
        </Card>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {pipelines.data.map((p) => (
            <Link
              key={p.id}
              href={`/s2s/${p.id}/play`}
              className="block rounded-lg border border-border bg-card p-4 hover:border-accent transition"
            >
              <div className="flex items-center justify-between">
                <div className="font-medium">{p.name}</div>
                <span className="text-[11px] text-muted">{relativeTime(p.created_at)}</span>
              </div>
              <div className="text-xs text-muted mt-1">{p.slug}</div>
              <div className="text-xs mt-2 space-y-0.5 font-mono text-fg/70">
                <div>ASR: {p.asr_version_id ?? p.asr_fallback ?? "—"}</div>
                <div>LLM: {p.llm_version_id ?? p.llm_fallback ?? "—"}</div>
                <div>TTS: {p.tts_version_id ?? p.tts_fallback ?? "—"}</div>
              </div>
            </Link>
          ))}
        </div>
      )}

      {showCreate && (
        <CreatePipelineModal
          projects={projects.data ?? []}
          onClose={() => setShowCreate(false)}
          onCreated={() => {
            setShowCreate(false);
            pipelines.mutate();
          }}
        />
      )}
    </>
  );
}

function CreatePipelineModal({
  projects,
  onClose,
  onCreated,
}: {
  projects: Project[];
  onClose: () => void;
  onCreated: () => void;
}) {
  const router = useRouter();
  const [projectId, setProjectId] = useState(projects[0]?.id ?? "");
  const asrModels = useSWR(projectId ? ["m-asr", projectId] : null, () => api.models.list(projectId, "asr"));
  const llmModels = useSWR(projectId ? ["m-llm", projectId] : null, () => api.models.list(projectId, "llm"));
  const ttsModels = useSWR(projectId ? ["m-tts", projectId] : null, () => api.models.list(projectId, "tts"));

  const [mode, setMode] = useState<"pipeline" | "native">("pipeline");
  const [form, setForm] = useState({
    slug: "",
    name: "",
    asr_version_id: "",
    llm_version_id: "",
    tts_version_id: "",
    asr_fallback: "openai/whisper-tiny",
    llm_fallback: "Qwen/Qwen2.5-0.5B-Instruct",
    tts_fallback: "facebook/mms-tts-eng",
    system_prompt: "You are a helpful spoken assistant. Keep replies brief and natural.",
  });
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const versionOptions = useMemo(
    () => async (modelId: string) =>
      (await jget<Array<{ id: string; version: string; stage: string }>>(`/models/${modelId}/versions`)),
    [],
  );

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setErr(null);
    setBusy(true);
    try {
      const body = {
        project_id: projectId,
        slug: form.slug,
        name: form.name,
        asr_version_id: form.asr_version_id || null,
        llm_version_id: form.llm_version_id || null,
        tts_version_id: form.tts_version_id || null,
        asr_fallback: mode === "native" ? null : form.asr_fallback || null,
        llm_fallback: form.llm_fallback || null,
        tts_fallback: form.tts_fallback || null,
        system_prompt: form.system_prompt,
        vad_config: {},
        runtime_config: { mode },
      };
      const r = await fetch("/api/s2s/pipelines", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!r.ok) throw new Error(`${r.status}: ${await r.text()}`);
      const pipeline = await r.json();
      onCreated();
      router.push(`/s2s/${pipeline.id}/play`);
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50" onClick={onClose}>
      <form
        onClick={(e) => e.stopPropagation()}
        onSubmit={submit}
        className="bg-card border border-border rounded-lg p-6 w-full max-w-xl space-y-3 max-h-[90vh] overflow-y-auto"
      >
        <h2 className="text-lg font-semibold">New S2S pipeline</h2>

        <div>
          <label className="block text-xs text-muted mb-1">Mode</label>
          <div className="flex gap-2">
            <ModeChip active={mode === "pipeline"} onClick={() => setMode("pipeline")}>
              Pipeline (ASR + LLM + TTS)
            </ModeChip>
            <ModeChip active={mode === "native"} onClick={() => setMode("native")}>
              Native audio (Qwen-Omni-style)
            </ModeChip>
          </div>
          <p className="text-[11px] text-muted mt-1">
            {mode === "pipeline"
              ? "Streams partial transcripts, sentence-chunked TTS, and supports barge-in."
              : "Audio goes directly into a multimodal LM. The TTS slot becomes a fallback vocoder for text-out models."}
          </p>
        </div>

        <Row>
          <Field label="Project">
            <select className="input" value={projectId} onChange={(e) => setProjectId(e.target.value)} required>
              {projects.map((p) => (
                <option key={p.id} value={p.id}>{p.name}</option>
              ))}
            </select>
          </Field>
          <Field label="Name">
            <input className="input" value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} required />
          </Field>
          <Field label="Slug">
            <input className="input" pattern="[a-z0-9][a-z0-9-_]*" value={form.slug} onChange={(e) => setForm({ ...form, slug: e.target.value })} required />
          </Field>
        </Row>

        {mode === "pipeline" && (
          <StageStack
            label="ASR"
            models={asrModels.data ?? []}
            versionId={form.asr_version_id}
            onVersion={(v) => setForm({ ...form, asr_version_id: v })}
            fallback={form.asr_fallback}
            onFallback={(v) => setForm({ ...form, asr_fallback: v })}
            loadVersions={versionOptions}
            fallbackHint="openai/whisper-tiny"
          />
        )}
        <StageStack
          label={mode === "native" ? "Audio-LM" : "LLM"}
          models={llmModels.data ?? []}
          versionId={form.llm_version_id}
          onVersion={(v) => setForm({ ...form, llm_version_id: v })}
          fallback={form.llm_fallback}
          onFallback={(v) => setForm({ ...form, llm_fallback: v })}
          loadVersions={versionOptions}
          fallbackHint={mode === "native" ? "Qwen/Qwen2.5-Omni-7B" : "Qwen/Qwen2.5-0.5B-Instruct"}
        />
        <StageStack
          label={mode === "native" ? "TTS fallback (optional)" : "TTS"}
          models={ttsModels.data ?? []}
          versionId={form.tts_version_id}
          onVersion={(v) => setForm({ ...form, tts_version_id: v })}
          fallback={form.tts_fallback}
          onFallback={(v) => setForm({ ...form, tts_fallback: v })}
          loadVersions={versionOptions}
          fallbackHint="facebook/mms-tts-eng"
        />

        <Field label="System prompt">
          <textarea
            className="input"
            rows={3}
            value={form.system_prompt}
            onChange={(e) => setForm({ ...form, system_prompt: e.target.value })}
          />
        </Field>

        {err && <p className="text-red-400 text-sm">{err}</p>}
        <div className="flex justify-end gap-2">
          <button type="button" onClick={onClose} className="px-3 py-1.5 rounded-md border border-border text-sm">
            Cancel
          </button>
          <button type="submit" disabled={busy} className="px-3 py-1.5 rounded-md bg-accent text-white text-sm font-medium disabled:opacity-50">
            {busy ? "Creating…" : "Create & open playground"}
          </button>
        </div>

        <style jsx global>{`
          .input { width: 100%; background: rgb(var(--bg)); border: 1px solid rgb(var(--border)); border-radius: 6px; padding: 6px 10px; font-size: 14px; }
        `}</style>
      </form>
    </div>
  );
}

function StageStack({
  label,
  models,
  versionId,
  onVersion,
  fallback,
  onFallback,
  loadVersions,
  fallbackHint,
}: {
  label: string;
  models: { id: string; name: string }[];
  versionId: string;
  onVersion: (v: string) => void;
  fallback: string;
  onFallback: (v: string) => void;
  loadVersions: (id: string) => Promise<Array<{ id: string; version: string; stage: string }>>;
  fallbackHint: string;
}) {
  const [modelId, setModelId] = useState("");
  const versions = useSWR(modelId ? ["mv", modelId] : null, () => loadVersions(modelId));

  return (
    <div className="border border-border rounded-md p-3">
      <div className="text-xs uppercase tracking-wide text-muted mb-2">{label}</div>
      <Row>
        <Field label="Registry model (optional)">
          <select
            className="input"
            value={modelId}
            onChange={(e) => {
              setModelId(e.target.value);
              onVersion("");
            }}
          >
            <option value="">— use fallback —</option>
            {models.map((m) => (
              <option key={m.id} value={m.id}>{m.name}</option>
            ))}
          </select>
        </Field>
        <Field label="Version">
          <select
            className="input"
            value={versionId}
            onChange={(e) => onVersion(e.target.value)}
            disabled={!modelId}
          >
            <option value="">{modelId ? "— pick one —" : "—"}</option>
            {versions.data?.map((v) => (
              <option key={v.id} value={v.id}>{v.version} ({v.stage})</option>
            ))}
          </select>
        </Field>
        <Field label="HF fallback id">
          <input className="input" value={fallback} onChange={(e) => onFallback(e.target.value)} placeholder={fallbackHint} />
        </Field>
      </Row>
    </div>
  );
}

function Row({ children }: { children: React.ReactNode }) {
  return <div className="grid grid-cols-3 gap-3">{children}</div>;
}

function ModeChip({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`px-3 py-1.5 rounded-md text-xs font-medium border transition ${
        active ? "bg-accent text-white border-accent" : "border-border hover:border-accent/60"
      }`}
    >
      {children}
    </button>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="mb-2">
      <label className="block text-xs text-muted mb-1">{label}</label>
      {children}
    </div>
  );
}
