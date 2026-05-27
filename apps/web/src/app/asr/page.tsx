"use client";

import { useMemo, useState } from "react";
import useSWR from "swr";
import { useRouter } from "next/navigation";
import { api } from "@/lib/api";
import { PageHeader } from "@/components/PageHeader";
import { Card, CardTitle } from "@/components/Card";

type DatasetVersion = {
  id: string;
  dataset_id: string;
  version: string;
  num_samples: number;
};

async function jget<T>(p: string): Promise<T> {
  const r = await fetch(`/api${p}`, { cache: "no-store" });
  if (!r.ok) throw new Error(`${r.status}: ${await r.text()}`);
  return r.json();
}

const BASE_MODELS = [
  "openai/whisper-tiny",
  "openai/whisper-base",
  "openai/whisper-small",
  "openai/whisper-medium",
  "openai/whisper-large-v3",
  "distil-whisper/distil-small.en",
  "distil-whisper/distil-large-v3",
];

export default function ASRStudio() {
  const router = useRouter();
  const projects = useSWR("projects", () => api.projects.list());
  const datasets = useSWR(
    "datasets-asr",
    () => api.datasets.list(undefined, "asr"),
  );

  const [datasetId, setDatasetId] = useState<string>("");
  const versions = useSWR<DatasetVersion[]>(
    datasetId ? ["versions", datasetId] : null,
    () => jget<DatasetVersion[]>(`/datasets/${datasetId}/versions`),
  );

  const [form, setForm] = useState({
    version_id: "",
    base_model: "openai/whisper-small",
    mode: "lora" as "lora" | "full",
    epochs: 3,
    batch_size: 8,
    learning_rate: 1e-4,
    max_audio_s: 30,
    language: "en",
    publish_model_slug: "",
    publish_version: "0.1.0",
  });
  const [augForm, setAugForm] = useState({
    speedEnabled: false,
    speedMin: 0.9,
    speedMax: 1.1,
    noiseEnabled: false,
    noiseMinSnr: 10,
    noiseMaxSnr: 30,
    reverbEnabled: false,
    reverbDecay: 0.5,
  });
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const dataset = useMemo(
    () => datasets.data?.find((d) => d.id === datasetId),
    [datasets.data, datasetId],
  );

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setErr(null);
    setBusy(true);
    try {
      if (!dataset) throw new Error("pick a dataset");
      const versionId = form.version_id || versions.data?.[0]?.id;
      if (!versionId) throw new Error("pick a dataset version");

      let modelId: string | undefined;
      if (form.publish_model_slug) {
        const existing = await api.models.list(dataset.project_id, "asr");
        const match = existing.find((m) => m.slug === form.publish_model_slug);
        modelId = match
          ? match.id
          : (
              await api.models.create({
                project_id: dataset.project_id,
                slug: form.publish_model_slug,
                name: form.publish_model_slug,
                modality: "asr",
              })
            ).id;
      }

      const config: Record<string, unknown> = {
        dataset_version_id: versionId,
        base_model: form.base_model,
        training: {
          mode: form.mode,
          epochs: form.epochs,
          batch_size: form.batch_size,
          learning_rate: form.learning_rate,
          max_audio_s: form.max_audio_s,
          language: form.language,
          task: "transcribe",
          fp16: true,
        },
        lora: { r: 16, alpha: 32, dropout: 0.05, target_modules: ["q_proj", "v_proj"] },
        augmentations: {
          speed: {
            enabled: augForm.speedEnabled,
            min_factor: augForm.speedMin,
            max_factor: augForm.speedMax,
          },
          noise: {
            enabled: augForm.noiseEnabled,
            min_snr_db: augForm.noiseMinSnr,
            max_snr_db: augForm.noiseMaxSnr,
          },
          reverb: {
            enabled: augForm.reverbEnabled,
            decay: augForm.reverbDecay,
          },
        },
        registry: modelId ? { model_id: modelId, version: form.publish_version } : undefined,
      };

      const job = await api.jobs.submit({
        project_id: dataset.project_id,
        kind: "whisper_finetune",
        name: `whisper-ft ${form.base_model.split("/").pop()}`,
        config,
      });
      router.push(`/jobs/${job.id}`);
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      <PageHeader
        title="ASR Studio"
        subtitle="Fine-tune Whisper-family models with LoRA or full fine-tune, then publish to the registry."
      />

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <form onSubmit={submit} className="md:col-span-2 space-y-3">
          <Card>
            <CardTitle>Data</CardTitle>
            <Field label="Dataset (ASR only)">
              <select
                className="input"
                value={datasetId}
                onChange={(e) => {
                  setDatasetId(e.target.value);
                  setForm({ ...form, version_id: "" });
                }}
                required
              >
                <option value="">— pick one —</option>
                {datasets.data?.map((d) => (
                  <option key={d.id} value={d.id}>{d.name}</option>
                ))}
              </select>
            </Field>
            <Field label="Dataset version">
              <select
                className="input"
                value={form.version_id}
                onChange={(e) => setForm({ ...form, version_id: e.target.value })}
                disabled={!datasetId}
                required
              >
                <option value="">{versions.isLoading ? "loading…" : "— pick one —"}</option>
                {versions.data?.map((v) => (
                  <option key={v.id} value={v.id}>
                    {v.version} ({v.num_samples} samples)
                  </option>
                ))}
              </select>
            </Field>
          </Card>

          <Card>
            <CardTitle>Model</CardTitle>
            <Field label="Base model">
              <select
                className="input"
                value={form.base_model}
                onChange={(e) => setForm({ ...form, base_model: e.target.value })}
              >
                {BASE_MODELS.map((m) => (
                  <option key={m} value={m}>{m}</option>
                ))}
              </select>
            </Field>
            <Field label="Training mode">
              <select
                className="input"
                value={form.mode}
                onChange={(e) => setForm({ ...form, mode: e.target.value as "lora" | "full" })}
              >
                <option value="lora">LoRA (recommended)</option>
                <option value="full">Full fine-tune</option>
              </select>
            </Field>
          </Card>

          <Card>
            <CardTitle>Hyperparameters</CardTitle>
            <Row>
              <Field label="Epochs">
                <input type="number" min={1} className="input" value={form.epochs} onChange={(e) => setForm({ ...form, epochs: Number(e.target.value) })} />
              </Field>
              <Field label="Batch size">
                <input type="number" min={1} className="input" value={form.batch_size} onChange={(e) => setForm({ ...form, batch_size: Number(e.target.value) })} />
              </Field>
              <Field label="Learning rate">
                <input type="number" step="any" className="input" value={form.learning_rate} onChange={(e) => setForm({ ...form, learning_rate: Number(e.target.value) })} />
              </Field>
            </Row>
            <Row>
              <Field label="Max audio (s)">
                <input type="number" min={1} className="input" value={form.max_audio_s} onChange={(e) => setForm({ ...form, max_audio_s: Number(e.target.value) })} />
              </Field>
              <Field label="Language (BCP-47)">
                <input className="input" value={form.language} onChange={(e) => setForm({ ...form, language: e.target.value })} />
              </Field>
            </Row>
          </Card>

          <Card>
            <CardTitle>Audio Data Augmentations (On-the-fly)</CardTitle>
            <div className="space-y-4">
              <div className="border border-border/60 rounded-md p-3 bg-bg/20">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium">Speed Perturbation</span>
                  <input
                    type="checkbox"
                    checked={augForm.speedEnabled}
                    onChange={(e) => setAugForm({ ...augForm, speedEnabled: e.target.checked })}
                    className="w-4 h-4 cursor-pointer accent-accent"
                  />
                </div>
                {augForm.speedEnabled && (
                  <Row>
                    <Field label="Min speed factor">
                      <input
                        type="number"
                        step="0.05"
                        min="0.5"
                        max="2.0"
                        className="input"
                        value={augForm.speedMin}
                        onChange={(e) => setAugForm({ ...augForm, speedMin: Number(e.target.value) })}
                      />
                    </Field>
                    <Field label="Max speed factor">
                      <input
                        type="number"
                        step="0.05"
                        min="0.5"
                        max="2.0"
                        className="input"
                        value={augForm.speedMax}
                        onChange={(e) => setAugForm({ ...augForm, speedMax: Number(e.target.value) })}
                      />
                    </Field>
                  </Row>
                )}
              </div>

              <div className="border border-border/60 rounded-md p-3 bg-bg/20">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium">White Noise Injection</span>
                  <input
                    type="checkbox"
                    checked={augForm.noiseEnabled}
                    onChange={(e) => setAugForm({ ...augForm, noiseEnabled: e.target.checked })}
                    className="w-4 h-4 cursor-pointer accent-accent"
                  />
                </div>
                {augForm.noiseEnabled && (
                  <Row>
                    <Field label="Min SNR (dB)">
                      <input
                        type="number"
                        min="0"
                        max="100"
                        className="input"
                        value={augForm.noiseMinSnr}
                        onChange={(e) => setAugForm({ ...augForm, noiseMinSnr: Number(e.target.value) })}
                      />
                    </Field>
                    <Field label="Max SNR (dB)">
                      <input
                        type="number"
                        min="0"
                        max="100"
                        className="input"
                        value={augForm.noiseMaxSnr}
                        onChange={(e) => setAugForm({ ...augForm, noiseMaxSnr: Number(e.target.value) })}
                      />
                    </Field>
                  </Row>
                )}
              </div>

              <div className="border border-border/60 rounded-md p-3 bg-bg/20">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium">Reverb / Room Simulation</span>
                  <input
                    type="checkbox"
                    checked={augForm.reverbEnabled}
                    onChange={(e) => setAugForm({ ...augForm, reverbEnabled: e.target.checked })}
                    className="w-4 h-4 cursor-pointer accent-accent"
                  />
                </div>
                {augForm.reverbEnabled && (
                  <div className="space-y-1">
                    <div className="flex justify-between text-xs text-muted">
                      <span>Damping / Decay</span>
                      <span>{augForm.reverbDecay.toFixed(2)}</span>
                    </div>
                    <input
                      type="range"
                      min="0.1"
                      max="0.9"
                      step="0.05"
                      className="w-full h-1 bg-border rounded-lg appearance-none cursor-pointer accent-accent animate-pulse"
                      value={augForm.reverbDecay}
                      onChange={(e) => setAugForm({ ...augForm, reverbDecay: Number(e.target.value) })}
                    />
                  </div>
                )}
              </div>
            </div>
          </Card>

          <Card>
            <CardTitle>Publish to registry (optional)</CardTitle>
            <Row>
              <Field label="Model slug">
                <input className="input" placeholder="whisper-en" value={form.publish_model_slug} onChange={(e) => setForm({ ...form, publish_model_slug: e.target.value })} />
              </Field>
              <Field label="Version (semver)">
                <input className="input" pattern="\d+\.\d+\.\d+" value={form.publish_version} onChange={(e) => setForm({ ...form, publish_version: e.target.value })} />
              </Field>
            </Row>
            <p className="text-xs text-muted mt-1">
              If a model with this slug doesn&apos;t exist, it&apos;ll be created in dev stage on success.
            </p>
          </Card>

          {err && <p className="text-red-400 text-sm">{err}</p>}
          <div className="flex justify-end">
            <button
              type="submit"
              disabled={busy}
              className="px-4 py-2 rounded-md bg-accent text-white text-sm font-medium disabled:opacity-50"
            >
              {busy ? "Submitting…" : "Start fine-tune"}
            </button>
          </div>
        </form>

        <div>
          <Card>
            <CardTitle>About this flow</CardTitle>
            <p className="text-sm text-muted">
              Submits a <span className="font-mono text-fg">whisper_finetune</span> job that:
            </p>
            <ul className="text-sm text-muted list-disc list-inside mt-2 space-y-1">
              <li>Reads ASR samples from the chosen manifest split</li>
              <li>Loads the base Whisper checkpoint + processor</li>
              <li>Applies LoRA (or trains full)</li>
              <li>Streams loss / WER to job logs</li>
              <li>Saves to the run artifacts dir</li>
              <li>Publishes a Model Version on success (if registry filled)</li>
            </ul>
            <p className="text-xs text-muted mt-3">
              Requires <span className="font-mono">pip install -e &apos;apps/server[asr]&apos;</span> on the server.
            </p>
          </Card>
        </div>
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
    </>
  );
}

function Row({ children }: { children: React.ReactNode }) {
  return <div className="grid grid-cols-3 gap-3">{children}</div>;
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="mb-2">
      <label className="block text-xs text-muted mb-1">{label}</label>
      {children}
    </div>
  );
}
