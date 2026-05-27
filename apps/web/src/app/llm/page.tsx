"use client";

import { useMemo, useState } from "react";
import useSWR from "swr";
import { useRouter } from "next/navigation";
import { api } from "@/lib/api";
import { PageHeader } from "@/components/PageHeader";
import { Card, CardTitle } from "@/components/Card";

type DatasetVersion = { id: string; version: string; num_samples: number };

async function jget<T>(p: string): Promise<T> {
  const r = await fetch(`/api${p}`, { cache: "no-store" });
  if (!r.ok) throw new Error(`${r.status}: ${await r.text()}`);
  return r.json();
}

const BASE_MODELS = [
  "Qwen/Qwen2.5-0.5B-Instruct",
  "Qwen/Qwen2.5-1.5B-Instruct",
  "Qwen/Qwen2.5-3B-Instruct",
  "Qwen/Qwen2.5-7B-Instruct",
  "meta-llama/Llama-3.2-1B-Instruct",
  "meta-llama/Llama-3.2-3B-Instruct",
  "meta-llama/Llama-3.1-8B-Instruct",
  "google/gemma-2-2b-it",
  "google/gemma-2-9b-it",
  "mistralai/Mistral-7B-Instruct-v0.3",
  "microsoft/Phi-3.5-mini-instruct",
];

export default function LLMStudio() {
  const router = useRouter();
  const datasets = useSWR("datasets-llm", () => api.datasets.list(undefined, "llm"));
  const [datasetId, setDatasetId] = useState("");
  const versions = useSWR<DatasetVersion[]>(
    datasetId ? ["versions-llm", datasetId] : null,
    () => jget<DatasetVersion[]>(`/datasets/${datasetId}/versions`),
  );

  const [form, setForm] = useState({
    version_id: "",
    base_model: BASE_MODELS[0],
    epochs: 3,
    batch_size: 4,
    grad_accum: 4,
    learning_rate: 2e-4,
    max_seq_len: 2048,
    lora_r: 16,
    lora_alpha: 32,
    quantization: "none" as "none" | "4bit" | "8bit",
    publish_model_slug: "",
    publish_version: "0.1.0",
  });
  const [robustForm, setRobustForm] = useState({
    asrNoiseEnabled: false,
    asrErrorRate: 0.1,
    spokenStyleEnabled: false,
    stripMarkdown: true,
    stripEmoji: true,
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
        const existing = await api.models.list(dataset.project_id, "llm");
        const match = existing.find((m) => m.slug === form.publish_model_slug);
        modelId = match
          ? match.id
          : (
              await api.models.create({
                project_id: dataset.project_id,
                slug: form.publish_model_slug,
                name: form.publish_model_slug,
                modality: "llm",
              })
            ).id;
      }

      const config: Record<string, unknown> = {
        dataset_version_id: versionId,
        base_model: form.base_model,
        training: {
          epochs: form.epochs,
          batch_size: form.batch_size,
          grad_accum_steps: form.grad_accum,
          learning_rate: form.learning_rate,
          max_seq_len: form.max_seq_len,
          bf16: true,
          gradient_checkpointing: true,
        },
        lora: { r: form.lora_r, alpha: form.lora_alpha, dropout: 0.05, target_modules: "auto" },
        quantization: form.quantization,
        robustness: {
          asr_noise_enabled: robustForm.asrNoiseEnabled,
          asr_error_rate: robustForm.asrErrorRate,
          spoken_style_enabled: robustForm.spokenStyleEnabled,
          strip_markdown: robustForm.stripMarkdown,
          strip_emoji: robustForm.stripEmoji,
        },
        registry: modelId ? { model_id: modelId, version: form.publish_version } : undefined,
      };

      const job = await api.jobs.submit({
        project_id: dataset.project_id,
        kind: "llm_finetune_sft",
        name: `sft ${form.base_model.split("/").pop()}`,
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
        title="LLM Studio"
        subtitle="LoRA / QLoRA SFT on chat LLMs. Llama, Qwen, Gemma, Mistral, Phi. Tool-use and spoken-style trainers to follow."
      />

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <form onSubmit={submit} className="md:col-span-2 space-y-3">
          <Card>
            <CardTitle>Data</CardTitle>
            <Field label="Dataset (LLM only)">
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
                  <option key={v.id} value={v.id}>{v.version} ({v.num_samples} samples)</option>
                ))}
              </select>
            </Field>
          </Card>

          <Card>
            <CardTitle>Model</CardTitle>
            <Field label="Base model">
              <select className="input" value={form.base_model} onChange={(e) => setForm({ ...form, base_model: e.target.value })}>
                {BASE_MODELS.map((m) => <option key={m} value={m}>{m}</option>)}
              </select>
            </Field>
            <Row>
              <Field label="Quantization">
                <select className="input" value={form.quantization} onChange={(e) => setForm({ ...form, quantization: e.target.value as "none" | "4bit" | "8bit" })}>
                  <option value="none">None (full precision)</option>
                  <option value="4bit">4-bit (QLoRA)</option>
                  <option value="8bit">8-bit</option>
                </select>
              </Field>
              <Field label="LoRA rank">
                <input type="number" min={1} className="input" value={form.lora_r} onChange={(e) => setForm({ ...form, lora_r: Number(e.target.value) })} />
              </Field>
              <Field label="LoRA alpha">
                <input type="number" min={1} className="input" value={form.lora_alpha} onChange={(e) => setForm({ ...form, lora_alpha: Number(e.target.value) })} />
              </Field>
            </Row>
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
              <Field label="Grad accum steps">
                <input type="number" min={1} className="input" value={form.grad_accum} onChange={(e) => setForm({ ...form, grad_accum: Number(e.target.value) })} />
              </Field>
            </Row>
            <Row>
              <Field label="Learning rate">
                <input type="number" step="any" className="input" value={form.learning_rate} onChange={(e) => setForm({ ...form, learning_rate: Number(e.target.value) })} />
              </Field>
              <Field label="Max seq len">
                <input type="number" min={128} className="input" value={form.max_seq_len} onChange={(e) => setForm({ ...form, max_seq_len: Number(e.target.value) })} />
              </Field>
              <div />
            </Row>
          </Card>

          <Card>
            <CardTitle>Robustness & Spoken Sanitization (On-the-fly)</CardTitle>
            <div className="space-y-4">
              <div className="border border-border/60 rounded-md p-3 bg-bg/20">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex flex-col">
                    <span className="text-sm font-medium">ASR Noise Simulation</span>
                    <span className="text-[10px] text-muted">Inject realistic typos and homophone swaps on User turns</span>
                  </div>
                  <input
                    type="checkbox"
                    checked={robustForm.asrNoiseEnabled}
                    onChange={(e) => setRobustForm({ ...robustForm, asrNoiseEnabled: e.target.checked })}
                    className="w-4 h-4 cursor-pointer accent-accent"
                  />
                </div>
                {robustForm.asrNoiseEnabled && (
                  <div className="space-y-1 mt-2">
                    <div className="flex justify-between text-xs text-muted">
                      <span>Typo / Error Rate</span>
                      <span>{(robustForm.asrErrorRate * 100).toFixed(0)}%</span>
                    </div>
                    <input
                      type="range"
                      min="0.05"
                      max="0.3"
                      step="0.05"
                      className="w-full h-1 bg-border rounded-lg appearance-none cursor-pointer accent-accent"
                      value={robustForm.asrErrorRate}
                      onChange={(e) => setRobustForm({ ...robustForm, asrErrorRate: Number(e.target.value) })}
                    />
                  </div>
                )}
              </div>

              <div className="border border-border/60 rounded-md p-3 bg-bg/20">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex flex-col">
                    <span className="text-sm font-medium">Spoken-Style Constraints</span>
                    <span className="text-[10px] text-muted">Strip markdown and emojis from Assistant responses</span>
                  </div>
                  <input
                    type="checkbox"
                    checked={robustForm.spokenStyleEnabled}
                    onChange={(e) => setRobustForm({ ...robustForm, spokenStyleEnabled: e.target.checked })}
                    className="w-4 h-4 cursor-pointer accent-accent"
                  />
                </div>
                {robustForm.spokenStyleEnabled && (
                  <Row>
                    <div className="flex items-center gap-2 pt-1">
                      <input
                        type="checkbox"
                        checked={robustForm.stripMarkdown}
                        onChange={(e) => setRobustForm({ ...robustForm, stripMarkdown: e.target.checked })}
                        className="w-4 h-4 cursor-pointer accent-accent"
                        id="strip_md"
                      />
                      <label htmlFor="strip_md" className="text-xs cursor-pointer select-none">Strip Markdown</label>
                    </div>
                    <div className="flex items-center gap-2 pt-1">
                      <input
                        type="checkbox"
                        checked={robustForm.stripEmoji}
                        onChange={(e) => setRobustForm({ ...robustForm, stripEmoji: e.target.checked })}
                        className="w-4 h-4 cursor-pointer accent-accent"
                        id="strip_em"
                      />
                      <label htmlFor="strip_em" className="text-xs cursor-pointer select-none">Filter Emojis</label>
                    </div>
                  </Row>
                )}
              </div>
            </div>
          </Card>

          <Card>
            <CardTitle>Publish to registry (optional)</CardTitle>
            <Row>
              <Field label="Model slug">
                <input className="input" placeholder="my-llm" value={form.publish_model_slug} onChange={(e) => setForm({ ...form, publish_model_slug: e.target.value })} />
              </Field>
              <Field label="Version">
                <input className="input" pattern="\d+\.\d+\.\d+" value={form.publish_version} onChange={(e) => setForm({ ...form, publish_version: e.target.value })} />
              </Field>
              <div />
            </Row>
          </Card>

          {err && <p className="text-red-400 text-sm">{err}</p>}
          <div className="flex justify-end">
            <button type="submit" disabled={busy} className="px-4 py-2 rounded-md bg-accent text-white text-sm font-medium disabled:opacity-50">
              {busy ? "Submitting…" : "Start fine-tune"}
            </button>
          </div>
        </form>

        <Card>
          <CardTitle>About this flow</CardTitle>
          <p className="text-sm text-muted">
            Submits an <span className="font-mono text-fg">llm_finetune_sft</span> job that:
          </p>
          <ul className="text-sm text-muted list-disc list-inside mt-2 space-y-1">
            <li>Renders each manifest LLM sample via the base model&apos;s chat template</li>
            <li>Trains with <span className="font-mono">trl.SFTTrainer</span> + PEFT LoRA</li>
            <li>Optionally 4-bit / 8-bit quantized (QLoRA)</li>
            <li>Auto-discovers LoRA target modules (all linear layers minus lm_head)</li>
            <li>Publishes a Model Version (PEFT adapter format)</li>
          </ul>
          <p className="text-xs text-muted mt-3">
            Requires <span className="font-mono">pip install -e &apos;apps/server[llm]&apos;</span>.
          </p>
        </Card>
      </div>

      <style jsx global>{`
        .input { width: 100%; background: rgb(var(--bg)); border: 1px solid rgb(var(--border)); border-radius: 6px; padding: 6px 10px; font-size: 14px; }
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
