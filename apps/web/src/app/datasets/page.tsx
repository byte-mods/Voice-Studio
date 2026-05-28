"use client";

import { useState } from "react";
import useSWR from "swr";
import Link from "next/link";
import { api, type Dataset, type Project } from "@/lib/api";
import { PageHeader } from "@/components/PageHeader";
import { Card } from "@/components/Card";
import { relativeTime } from "@/lib/utils";

export default function DatasetsPage() {
  const projects = useSWR("projects", () => api.projects.list());
  const datasets = useSWR("datasets", () => api.datasets.list());
  const [showCreate, setShowCreate] = useState(false);
  const [showImport, setShowImport] = useState(false);

  return (
    <>
      <PageHeader
        title="Datasets"
        subtitle="Pull from Hugging Face or build custom ASR / TTS / LLM / S2S datasets."
        actions={
          <div className="flex items-center gap-2">
            <button
              onClick={() => setShowImport(true)}
              className="px-3 py-1.5 rounded-md border border-border text-sm font-medium"
            >
              Import from HF
            </button>
            <button
              onClick={() => setShowCreate(true)}
              className="px-3 py-1.5 rounded-md bg-accent text-white text-sm font-medium"
            >
              New dataset
            </button>
          </div>
        }
      />

      {datasets.isLoading ? (
        <p className="text-muted text-sm">Loading…</p>
      ) : datasets.error ? (
        <p className="text-red-400 text-sm">Failed to load datasets</p>
      ) : !datasets.data?.length ? (
        <EmptyState onCreate={() => setShowCreate(true)} hasProjects={!!projects.data?.length} />
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {datasets.data.map((d) => (
            <DatasetCard key={d.id} d={d} />
          ))}
        </div>
      )}

      {showCreate && (
        <CreateDatasetModal
          projects={projects.data ?? []}
          onClose={() => setShowCreate(false)}
          onCreated={() => {
            setShowCreate(false);
            datasets.mutate();
          }}
        />
      )}
      {showImport && (
        <HFImportModal
          datasets={datasets.data ?? []}
          onClose={() => setShowImport(false)}
          onSubmitted={() => {
            setShowImport(false);
            datasets.mutate();
          }}
        />
      )}
    </>
  );
}

function HFImportModal({
  datasets,
  onClose,
  onSubmitted,
}: {
  datasets: Dataset[];
  onClose: () => void;
  onSubmitted: () => void;
}) {
  const [form, setForm] = useState({
    dataset_id: datasets[0]?.id ?? "",
    hf_id: "",
    hf_config: "",
    hf_split: "train",
    version: "0.1.0",
    language: "en",
    license_spdx: "CC-BY-4.0",
    audio_field: "audio",
    transcript_field: "sentence",
    speaker_field: "client_id",
    max_samples: 100,
  });
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setErr(null);
    try {
      const ds = datasets.find((d) => d.id === form.dataset_id);
      if (!ds) throw new Error("select a dataset");
      const config: Record<string, unknown> = {
        hf_id: form.hf_id,
        hf_split: form.hf_split,
        modality: ds.modality,
        dataset_id: ds.id,
        version: form.version,
        language: form.language,
        license: { spdx: form.license_spdx },
        field_map: {
          audio: form.audio_field,
          transcript: form.transcript_field,
          text: form.transcript_field,
          speaker_id: form.speaker_field,
        },
        max_samples: form.max_samples,
      };
      if (form.hf_config) config.hf_config = form.hf_config;
      await api.jobs.submit({
        project_id: ds.project_id,
        kind: "hf_import",
        name: `import ${form.hf_id}`,
        config,
      });
      onSubmitted();
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
        className="bg-card border border-border rounded-lg p-6 w-full max-w-lg space-y-3 max-h-[90vh] overflow-y-auto"
      >
        <h2 className="text-lg font-semibold">Import from Hugging Face</h2>
        <p className="text-xs text-muted">
          Streams a dataset, materializes audio to storage, writes a manifest v1 dataset version.
        </p>

        <Row>
          <Field label="Target dataset">
            <select
              className="input"
              value={form.dataset_id}
              onChange={(e) => setForm({ ...form, dataset_id: e.target.value })}
              required
            >
              {datasets.map((d) => (
                <option key={d.id} value={d.id}>
                  {d.name} ({d.modality})
                </option>
              ))}
            </select>
          </Field>
          <Field label="Version">
            <input
              className="input"
              pattern="\d+\.\d+\.\d+"
              value={form.version}
              onChange={(e) => setForm({ ...form, version: e.target.value })}
            />
          </Field>
        </Row>

        <Field label="HF dataset id">
          <input
            className="input"
            placeholder="mozilla-foundation/common_voice_16_1"
            value={form.hf_id}
            onChange={(e) => setForm({ ...form, hf_id: e.target.value })}
            required
          />
        </Field>

        <Row>
          <Field label="HF config (optional)">
            <input className="input" value={form.hf_config} onChange={(e) => setForm({ ...form, hf_config: e.target.value })} placeholder="en" />
          </Field>
          <Field label="Split">
            <input className="input" value={form.hf_split} onChange={(e) => setForm({ ...form, hf_split: e.target.value })} />
          </Field>
        </Row>

        <Row>
          <Field label="Preset Language">
            <select
              className="input"
              value={form.language}
              onChange={(e) => setForm({ ...form, language: e.target.value })}
            >
              <option value="en">🇺🇸 English (en)</option>
              <option value="hi">🇮🇳 Hindi (hi / हिंदी)</option>
              <option value="es">🇪🇸 Spanish (es)</option>
              <option value="fr">🇫🇷 French (fr)</option>
              <option value="de">🇩🇪 German (de)</option>
              <option value="zh">🇨🇳 Chinese (zh)</option>
              <option value="bn">🇮🇳 Bengali (bn / বাংলা)</option>
              <option value="ta">🇮🇳 Tamil (ta / தமிழ்)</option>
            </select>
          </Field>
          <Field label="Or Custom ISO Code">
            <input className="input" placeholder="e.g. hi-IN, mr" value={form.language} onChange={(e) => setForm({ ...form, language: e.target.value })} />
          </Field>
        </Row>
        <Row>
          <Field label="License (SPDX)">
            <input className="input" value={form.license_spdx} onChange={(e) => setForm({ ...form, license_spdx: e.target.value })} />
          </Field>
          <div />
        </Row>

        <div className="text-xs uppercase tracking-wide text-muted pt-2">Field mapping</div>
        <Row>
          <Field label="Audio field">
            <input className="input" value={form.audio_field} onChange={(e) => setForm({ ...form, audio_field: e.target.value })} />
          </Field>
          <Field label="Transcript / text field">
            <input className="input" value={form.transcript_field} onChange={(e) => setForm({ ...form, transcript_field: e.target.value })} />
          </Field>
        </Row>
        <Row>
          <Field label="Speaker field (optional)">
            <input className="input" value={form.speaker_field} onChange={(e) => setForm({ ...form, speaker_field: e.target.value })} />
          </Field>
          <Field label="Max samples">
            <input className="input" type="number" min={1} value={form.max_samples} onChange={(e) => setForm({ ...form, max_samples: Number(e.target.value) })} />
          </Field>
        </Row>

        {err && <p className="text-red-400 text-sm">{err}</p>}
        <div className="flex justify-end gap-2 pt-2">
          <button type="button" onClick={onClose} className="px-3 py-1.5 rounded-md border border-border text-sm">
            Cancel
          </button>
          <button type="submit" disabled={busy} className="px-3 py-1.5 rounded-md bg-accent text-white text-sm font-medium disabled:opacity-50">
            {busy ? "Queueing…" : "Start import"}
          </button>
        </div>
      </form>
    </div>
  );
}

function Row({ children }: { children: React.ReactNode }) {
  return <div className="grid grid-cols-2 gap-3">{children}</div>;
}

function DatasetCard({ d }: { d: Dataset }) {
  return (
    <Link
      href={`/datasets/${d.id}`}
      className="block rounded-lg border border-border bg-card p-4 hover:border-accent transition"
    >
      <div className="flex items-center justify-between">
        <div className="font-medium">{d.name}</div>
        <span className="text-[11px] uppercase tracking-wide text-accent">{d.modality}</span>
      </div>
      <div className="text-xs text-muted mt-1">
        {d.slug} · created {relativeTime(d.created_at)}
      </div>
      {d.description && <p className="text-sm text-fg/80 mt-2 line-clamp-2">{d.description}</p>}
    </Link>
  );
}

function EmptyState({ onCreate, hasProjects }: { onCreate: () => void; hasProjects: boolean }) {
  return (
    <Card>
      <div className="text-center py-8">
        <div className="text-lg font-medium mb-1">No datasets yet</div>
        <p className="text-sm text-muted mb-4">
          {hasProjects
            ? "Create your first dataset or import one from Hugging Face."
            : "Create a project first, then start adding datasets."}
        </p>
        {hasProjects ? (
          <button
            onClick={onCreate}
            className="px-3 py-1.5 rounded-md bg-accent text-white text-sm font-medium"
          >
            New dataset
          </button>
        ) : (
          <Link
            href="/projects"
            className="inline-block px-3 py-1.5 rounded-md bg-accent text-white text-sm font-medium"
          >
            Create a project
          </Link>
        )}
      </div>
    </Card>
  );
}

function CreateDatasetModal({
  projects,
  onClose,
  onCreated,
}: {
  projects: Project[];
  onClose: () => void;
  onCreated: () => void;
}) {
  const [form, setForm] = useState({
    project_id: projects[0]?.id ?? "",
    slug: "",
    name: "",
    modality: "asr",
  });
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setErr(null);
    try {
      await api.datasets.create(form);
      onCreated();
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
        className="bg-card border border-border rounded-lg p-6 w-full max-w-md space-y-4"
      >
        <h2 className="text-lg font-semibold">New dataset</h2>
        <Field label="Project">
          <select
            className="input"
            value={form.project_id}
            onChange={(e) => setForm({ ...form, project_id: e.target.value })}
            required
          >
            {projects.map((p) => (
              <option key={p.id} value={p.id}>
                {p.name}
              </option>
            ))}
          </select>
        </Field>
        <Field label="Modality">
          <select
            className="input"
            value={form.modality}
            onChange={(e) => setForm({ ...form, modality: e.target.value })}
          >
            <option value="asr">ASR</option>
            <option value="tts">TTS</option>
            <option value="llm">LLM</option>
            <option value="s2s">Speech-to-Speech</option>
          </select>
        </Field>
        <Field label="Name">
          <input
            className="input"
            value={form.name}
            onChange={(e) => setForm({ ...form, name: e.target.value })}
            required
          />
        </Field>
        <Field label="Slug">
          <input
            className="input"
            pattern="[a-z0-9][a-z0-9-_]*"
            value={form.slug}
            onChange={(e) => setForm({ ...form, slug: e.target.value })}
            placeholder="my-dataset"
            required
          />
        </Field>
        {err && <p className="text-red-400 text-sm">{err}</p>}
        <div className="flex justify-end gap-2">
          <button type="button" onClick={onClose} className="px-3 py-1.5 rounded-md border border-border text-sm">
            Cancel
          </button>
          <button
            type="submit"
            disabled={busy}
            className="px-3 py-1.5 rounded-md bg-accent text-white text-sm font-medium disabled:opacity-50"
          >
            {busy ? "Creating…" : "Create"}
          </button>
        </div>
      </form>
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
