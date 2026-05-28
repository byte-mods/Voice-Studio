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

const BASE_VOICES = [
  "coqui/XTTS-v2",
  "bark/suno",
  "vits/en-ljspeech",
  "flow-matching/tts-synthesizer",
  "elevenlabs/multilingual-v2",
  "en_US-lessac-medium",
  "en_US-joe-medium",
];

export default function TTSStudio() {
  const router = useRouter();
  const datasets = useSWR("datasets-tts", () => api.datasets.list(undefined, "tts"));
  const [datasetId, setDatasetId] = useState("");
  const versions = useSWR<DatasetVersion[]>(
    datasetId ? ["versions-tts", datasetId] : null,
    () => jget<DatasetVersion[]>(`/datasets/${datasetId}/versions`),
  );

  const [form, setForm] = useState({
    version_id: "",
    voice_name: "my-voice",
    language: "en",
    quality: "medium" as "x_low" | "low" | "medium" | "high",
    sample_rate: 22050,
    max_epochs: 1000,
    batch_size: 32,
    checkpoint_epochs: 50,
    base_voice: "",
    select_base_voice: "coqui/XTTS-v2",
    publish_model_slug: "",
    publish_version: "0.1.0",
  });
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  // Phase 4 additions: Vocal Consent and Lexicon mapping
  const [speakerName, setSpeakerName] = useState("");
  const [consentVerified, setConsentVerified] = useState(false);
  const [consentStatement, setConsentStatement] = useState("");
  const [lexicon, setLexicon] = useState<Array<{ word: string; replacement: string }>>([]);

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

      // Verify ethical consent gate is checked
      if (!consentVerified) {
        throw new Error("vocal speaker consent must be verified and checked before starting voice fine-tuning");
      }

      let modelId: string | undefined;
      if (form.publish_model_slug) {
        const existing = await api.models.list(dataset.project_id, "tts");
        const match = existing.find((m) => m.slug === form.publish_model_slug);
        modelId = match
          ? match.id
          : (
              await api.models.create({
                project_id: dataset.project_id,
                slug: form.publish_model_slug,
                name: form.publish_model_slug,
                modality: "tts",
              })
            ).id;
      }

      // Convert phonetic lexicon array into key-value map
      const lexiconDict: Record<string, string> = {};
      for (const item of lexicon) {
        if (item.word.trim() && item.replacement.trim()) {
          lexiconDict[item.word.trim()] = item.replacement.trim();
        }
      }

      const config: Record<string, unknown> = {
        dataset_version_id: versionId,
        voice_name: form.voice_name,
        language: form.language,
        sample_rate: form.sample_rate,
        training: {
          max_epochs: form.max_epochs,
          batch_size: form.batch_size,
          checkpoint_epochs: form.checkpoint_epochs,
          quality: form.quality,
        },
        speaker_name: speakerName,
        consent_statement: consentStatement,
        lexicon: lexiconDict,
        registry: modelId ? { model_id: modelId, version: form.publish_version } : undefined,
      };
      config.base_voice = form.base_voice.trim() || form.select_base_voice;

      const job = await api.jobs.submit({
        project_id: dataset.project_id,
        kind: "tts_finetune_piper",
        name: `piper ${form.voice_name}`,
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
        title="TTS Studio"
        subtitle="Fine-tune Piper voices on TTS datasets. Streaming WebSocket serve and consent gating to follow."
      />

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <form onSubmit={submit} className="md:col-span-2 space-y-3">
          <Card>
            <CardTitle>Data</CardTitle>
            <Field label="Dataset (TTS only)">
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
            <CardTitle>Voice</CardTitle>
            <Row>
              <Field label="Voice name">
                <input className="input" value={form.voice_name} onChange={(e) => setForm({ ...form, voice_name: e.target.value })} required />
              </Field>
              <Field label="Language">
                <input className="input" value={form.language} onChange={(e) => setForm({ ...form, language: e.target.value })} />
              </Field>
              <Field label="Quality">
                <select className="input" value={form.quality} onChange={(e) => setForm({ ...form, quality: e.target.value as "x_low" | "low" | "medium" | "high" })}>
                  <option value="x_low">x_low (5k params, fastest)</option>
                  <option value="low">low</option>
                  <option value="medium">medium</option>
                  <option value="high">high (best)</option>
                </select>
              </Field>
            </Row>
            <Field label="Base model voice family">
              <select className="input" value={form.select_base_voice} onChange={(e) => setForm({ ...form, select_base_voice: e.target.value })}>
                {BASE_VOICES.map((v) => <option key={v} value={v}>{v}</option>)}
              </select>
            </Field>

            <Field label="Or manual Hugging Face ID / local path (overrides dropdown)">
              <input className="input" placeholder="e.g. coqui/XTTS-v2 or /path/to/local/voice" value={form.base_voice} onChange={(e) => setForm({ ...form, base_voice: e.target.value })} />
            </Field>

            <div className="mt-4 p-3 bg-cyan-500/5 border border-cyan-500/20 rounded-lg text-xs">
              <span className="font-bold text-cyan-400 block mb-1">🏗️ Build Architectures from Scratch</span>
              <p className="text-muted leading-relaxed mb-2">
                Want to construct a completely new custom Flow-Matching TTS or vocoder block from scratch?
              </p>
              <a
                href="/lab"
                className="inline-block px-2.5 py-1 bg-cyan-500/10 border border-cyan-500/35 hover:bg-cyan-500/20 text-cyan-300 font-semibold rounded text-[11px] transition-all"
              >
                Go to Architecture Lab →
              </a>
            </div>
          </Card>

          <Card>
            <CardTitle>Training</CardTitle>
            <Row>
              <Field label="Max epochs">
                <input type="number" min={1} className="input" value={form.max_epochs} onChange={(e) => setForm({ ...form, max_epochs: Number(e.target.value) })} />
              </Field>
              <Field label="Batch size">
                <input type="number" min={1} className="input" value={form.batch_size} onChange={(e) => setForm({ ...form, batch_size: Number(e.target.value) })} />
              </Field>
              <Field label="Checkpoint every">
                <input type="number" min={1} className="input" value={form.checkpoint_epochs} onChange={(e) => setForm({ ...form, checkpoint_epochs: Number(e.target.value) })} />
              </Field>
              <Field label="Sample rate (Hz)">
                <select className="input" value={form.sample_rate} onChange={(e) => setForm({ ...form, sample_rate: Number(e.target.value) })}>
                  <option value={16000}>16000</option>
                  <option value={22050}>22050</option>
                </select>
              </Field>
            </Row>
          </Card>

          <Card>
            <CardTitle>Vocal Consent & Gating</CardTitle>
            <p className="text-xs text-muted mb-3">
              Ethical voice cloning requires explicit, documented consent from the target speaker.
            </p>
            <div className="grid grid-cols-2 gap-3 mb-2">
              <Field label="Speaker Name">
                <input
                  className="input"
                  placeholder="e.g. Jane Doe"
                  value={speakerName}
                  onChange={(e) => setSpeakerName(e.target.value)}
                  required
                />
              </Field>
              <Field label="Vocal Authorization Text">
                <input
                  className="input"
                  placeholder="e.g. I hereby authorize cloning of my voice..."
                  value={consentStatement}
                  onChange={(e) => setConsentStatement(e.target.value)}
                />
              </Field>
            </div>
            <div className="flex items-center gap-2 mt-2 bg-accent/5 p-2.5 border border-accent/10 rounded-md">
              <input
                type="checkbox"
                id="consentCheckbox"
                checked={consentVerified}
                onChange={(e) => setConsentVerified(e.target.checked)}
                className="w-4 h-4 cursor-pointer accent-accent"
              />
              <label htmlFor="consentCheckbox" className="text-xs text-fg select-none cursor-pointer">
                I verify that I have obtained explicit, verifiable vocal or written consent from <strong>{speakerName || "the speaker"}</strong>.
              </label>
            </div>
          </Card>

          <Card>
            <CardTitle>Phonetic Lexicon Override Manager</CardTitle>
            <p className="text-xs text-muted mb-2">
              Define spelling-to-speech phonetic replacement overrides to expand acronyms, symbols, or custom terminology during SFT training.
            </p>
            <div className="space-y-2 max-h-[200px] overflow-y-auto mb-3 pr-1">
              {lexicon.length === 0 ? (
                <p className="text-xs text-muted/60 italic">No phonetic lexicon overrides defined yet. Add one below!</p>
              ) : (
                lexicon.map((item, index) => (
                  <div key={index} className="flex gap-2 items-center">
                    <input
                      type="text"
                      placeholder="e.g. OAS"
                      value={item.word}
                      onChange={(e) => {
                        const newLex = [...lexicon];
                        newLex[index].word = e.target.value;
                        setLexicon(newLex);
                      }}
                      className="input text-xs flex-1"
                      style={{ marginBottom: 0 }}
                      required
                    />
                    <span className="text-muted text-xs">→</span>
                    <input
                      type="text"
                      placeholder="e.g. Oh-Ay-Es"
                      value={item.replacement}
                      onChange={(e) => {
                        const newLex = [...lexicon];
                        newLex[index].replacement = e.target.value;
                        setLexicon(newLex);
                      }}
                      className="input text-xs flex-1"
                      style={{ marginBottom: 0 }}
                      required
                    />
                    <button
                      type="button"
                      onClick={() => {
                        setLexicon(lexicon.filter((_, i) => i !== index));
                      }}
                      className="text-xs text-red-400 hover:text-red-300 font-semibold px-2 py-1"
                    >
                      Remove
                    </button>
                  </div>
                ))
              )}
            </div>
            <button
              type="button"
              onClick={() => setLexicon([...lexicon, { word: "", replacement: "" }])}
              className="px-3 py-1 text-xs border border-border rounded hover:bg-border/20 transition font-semibold"
            >
              + Add Override Mapping
            </button>
          </Card>

          <Card>
            <CardTitle>Publish to registry (optional)</CardTitle>
            <Row>
              <Field label="Model slug">
                <input className="input" placeholder="my-voice" value={form.publish_model_slug} onChange={(e) => setForm({ ...form, publish_model_slug: e.target.value })} />
              </Field>
              <Field label="Version">
                <input className="input" pattern="\d+\.\d+\.\d+" value={form.publish_version} onChange={(e) => setForm({ ...form, publish_version: e.target.value })} />
              </Field>
            </Row>
          </Card>

          {err && <p className="text-red-400 text-sm">{err}</p>}
          <div className="flex justify-end">
            <button type="submit" disabled={busy || !consentVerified} className="px-4 py-2 rounded-md bg-accent text-white text-sm font-medium disabled:opacity-50">
              {busy ? "Submitting…" : "Start training"}
            </button>
          </div>
        </form>

        <Card>
          <CardTitle>About this flow</CardTitle>
          <p className="text-sm text-muted">
            Submits a <span className="font-mono text-fg">tts_finetune_piper</span> job that:
          </p>
          <ul className="text-sm text-muted list-disc list-inside mt-2 space-y-1">
            <li>Materializes manifest TTS samples into Piper&apos;s LJSpeech layout</li>
            <li>Runs <span className="font-mono">piper_train.preprocess</span></li>
            <li>Runs <span className="font-mono">piper_train</span> (VITS backbone)</li>
            <li>Exports ONNX + voice config</li>
            <li>Publishes a Model Version on success</li>
          </ul>
          <p className="text-xs text-muted mt-3">
            Requires <span className="font-mono">pip install -e &apos;apps/server[tts]&apos;</span>.
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
