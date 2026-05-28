"use client";

import { use, useEffect, useMemo, useRef, useState } from "react";
import useSWR from "swr";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { api } from "@/lib/api";
import { PageHeader } from "@/components/PageHeader";
import { Card, CardTitle } from "@/components/Card";
import { StatusPill } from "@/components/StatusPill";
import { EvalResults } from "@/components/EvalResults";
import { relativeTime } from "@/lib/utils";

type Version = {
  id: string;
  version: string;
  stage: string;
  artifact_uri: string;
  format: string;
  metrics: Record<string, unknown>;
  source_run_id: string | null;
  source_dataset_version_id: string | null;
  notes: string | null;
  created_at: string;
};

type Model = {
  id: string;
  project_id: string;
  slug: string;
  name: string;
  modality: string;
  family: string | null;
};

async function jget<T>(p: string): Promise<T> {
  const r = await fetch(`/api${p}`, { cache: "no-store" });
  if (!r.ok) throw new Error(`${r.status}: ${await r.text()}`);
  return r.json();
}

export default function ModelDetail({ params }: { params: any }) {
  const resolvedParams = params && typeof params.then === "function" ? use(params) : params;
  const { id } = resolvedParams;
  const model = useSWR<Model>(["model", id], () => jget<Model>(`/models/${id}`));
  const versions = useSWR<Version[]>(["versions", id], () => jget<Version[]>(`/models/${id}/versions`));
  const [activeVid, setActiveVid] = useState<string | null>(null);
  const [rightTab, setRightTab] = useState<"eval" | "serve">("eval");

  useEffect(() => {
    if (!activeVid && versions.data?.[0]) setActiveVid(versions.data[0].id);
  }, [versions.data, activeVid]);

  const versionId = activeVid ?? versions.data?.[0]?.id ?? null;

  return (
    <>
      <PageHeader
        title={model.data?.name ?? `Model ${id.slice(0, 8)}`}
        subtitle={model.data ? `${model.data.modality.toUpperCase()}${model.data.family ? ` · ${model.data.family}` : ""}` : ""}
        actions={<Link href="/models" className="px-3 py-1.5 rounded-md border border-border text-sm">← Registry</Link>}
      />

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="md:col-span-2 space-y-3">
          <Card>
            <CardTitle>Versions</CardTitle>
            {!versions.data?.length ? (
              <p className="text-sm text-muted">No versions yet.</p>
            ) : (
              <table className="w-full text-sm">
                <thead className="text-xs text-muted uppercase">
                  <tr>
                    <th className="text-left py-1"></th>
                    <th className="text-left py-1">Version</th>
                    <th className="text-left py-1">Stage</th>
                    <th className="text-left py-1">Format</th>
                    <th className="text-left py-1">Train metrics</th>
                    <th className="text-left py-1">Created</th>
                  </tr>
                </thead>
                <tbody>
                  {versions.data.map((v) => (
                    <tr
                      key={v.id}
                      className={`border-t border-border cursor-pointer ${
                        versionId === v.id ? "bg-accent/5" : "hover:bg-border/20"
                      }`}
                      onClick={() => setActiveVid(v.id)}
                    >
                      <td className="py-1">
                        <input type="radio" readOnly checked={versionId === v.id} />
                      </td>
                      <td className="py-1 font-mono">{v.version}</td>
                      <td className="py-1"><StatusPill status={v.stage} /></td>
                      <td className="py-1 text-muted text-xs font-mono">{v.format}</td>
                      <td className="py-1 text-xs">
                        {Object.entries(v.metrics)
                          .slice(0, 3)
                          .map(([k, val]) => (
                            <span key={k} className="mr-2">
                              <span className="text-muted">{k}:</span>{" "}
                              <span className="font-mono">{typeof val === "number" ? val.toFixed(4) : String(val)}</span>
                            </span>
                          ))}
                      </td>
                      <td className="py-1 text-xs text-muted">{relativeTime(v.created_at)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </Card>

          {versionId && (
            <div>
              <h2 className="text-sm font-medium text-muted mb-2">
                Evaluations for{" "}
                <span className="font-mono">
                  {versions.data?.find((v) => v.id === versionId)?.version}
                </span>
              </h2>
              <EvalResults versionId={versionId} />
            </div>
          )}
        </div>

        <div className="space-y-3">
          {versionId && model.data && (model.data.modality === "asr" || model.data.modality === "llm" || model.data.modality === "tts") && (
            <div className="flex bg-card border border-border p-1 rounded-lg">
              <button
                onClick={() => setRightTab("eval")}
                className={`flex-1 text-xs py-1.5 font-medium rounded-md transition-all ${
                  rightTab === "eval"
                    ? "bg-accent text-white shadow-sm"
                    : "text-muted hover:text-fg hover:bg-border/20"
                }`}
              >
                Run Evaluation
              </button>
              <button
                onClick={() => setRightTab("serve")}
                className={`flex-1 text-xs py-1.5 font-medium rounded-md transition-all ${
                  rightTab === "serve"
                    ? "bg-accent text-white shadow-sm"
                    : "text-muted hover:text-fg hover:bg-border/20"
                }`}
              >
                {model.data.modality === "asr" ? "Serving Sandbox" : model.data.modality === "llm" ? "Chat Sandbox" : "Voice Sandbox"}
              </button>
            </div>
          )}

          {versionId && model.data && (
            rightTab === "eval" || (model.data.modality !== "asr" && model.data.modality !== "llm" && model.data.modality !== "tts") ? (
              <EvalPanel
                modelId={id}
                projectId={model.data.project_id}
                modality={model.data.modality}
                activeVersionId={versionId}
                versions={versions.data ?? []}
              />
            ) : model.data.modality === "asr" ? (
              <ServingSandbox
                versionId={versionId}
                baseModel={versions.data?.find((v) => v.id === versionId)?.artifact_uri}
              />
            ) : model.data.modality === "llm" ? (
              <ChatSandbox
                versionId={versionId}
                baseModelUri={versions.data?.find((v) => v.id === versionId)?.artifact_uri}
              />
            ) : (
              <TTSSandbox
                versionId={versionId}
              />
            )
          )}
        </div>
      </div>
    </>
  );
}

function EvalPanel({
  projectId,
  modality,
  activeVersionId,
  versions,
}: {
  modelId: string;
  projectId: string;
  modality: string;
  activeVersionId: string;
  versions: Version[];
}) {
  const router = useRouter();
  const datasets = useSWR(["ds-eval", projectId, modality], () =>
    api.datasets.list(projectId, modality),
  );
  const [datasetId, setDatasetId] = useState("");
  const dsVersions = useSWR(datasetId ? ["dsv-eval", datasetId] : null, () =>
    jget<Array<{ id: string; version: string; num_samples: number }>>(`/datasets/${datasetId}/versions`),
  );
  const [form, setForm] = useState({
    dataset_version_id: "",
    split: "test",
    max_samples: 200,
    base_model: "Qwen/Qwen2.5-0.5B-Instruct",
  });
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const activeVersion = useMemo(
    () => versions.find((v) => v.id === activeVersionId),
    [versions, activeVersionId],
  );

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setErr(null);
    try {
      const kind = modality === "asr" ? "asr_eval" : modality === "tts" ? "tts_eval" : "llm_eval";
      const dsv = form.dataset_version_id || dsVersions.data?.[0]?.id;
      if (!dsv) throw new Error("pick a dataset version");
      const config: Record<string, unknown> = {
        model_version_id: activeVersionId,
        dataset_version_id: dsv,
        split: form.split,
        max_samples: form.max_samples,
      };
      if (kind === "llm_eval") config.base_model = form.base_model;

      const job = await api.jobs.submit({
        project_id: projectId,
        kind,
        name: `${kind} v${activeVersion?.version ?? ""}`,
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
    <form onSubmit={submit}>
      <Card>
        <CardTitle>Run eval on selected version</CardTitle>
        <p className="text-xs text-muted mb-2">
          Will use <span className="font-mono">{modality}_eval</span> handler.
        </p>

        <Field label="Dataset">
          <select className="input" value={datasetId} onChange={(e) => setDatasetId(e.target.value)}>
            <option value="">— pick one —</option>
            {datasets.data?.map((d) => <option key={d.id} value={d.id}>{d.name}</option>)}
          </select>
        </Field>

        <Field label="Dataset version">
          <select
            className="input"
            value={form.dataset_version_id}
            onChange={(e) => setForm({ ...form, dataset_version_id: e.target.value })}
            disabled={!datasetId}
          >
            {dsVersions.data?.map((v) => (
              <option key={v.id} value={v.id}>{v.version} ({v.num_samples})</option>
            ))}
          </select>
        </Field>

        <Field label="Split">
          <select className="input" value={form.split} onChange={(e) => setForm({ ...form, split: e.target.value })}>
            <option value="test">test</option>
            <option value="val">val</option>
            <option value="train">train</option>
          </select>
        </Field>

        <Field label="Max samples">
          <input type="number" className="input" min={1} value={form.max_samples} onChange={(e) => setForm({ ...form, max_samples: Number(e.target.value) })} />
        </Field>

        {modality === "llm" && (
          <Field label="Base model (PEFT adapters)">
            <input className="input" value={form.base_model} onChange={(e) => setForm({ ...form, base_model: e.target.value })} />
          </Field>
        )}

        {err && <p className="text-red-400 text-sm mb-2">{err}</p>}
        <button type="submit" disabled={busy} className="w-full px-3 py-1.5 rounded-md bg-accent text-white text-sm font-medium disabled:opacity-50">
          {busy ? "Queueing…" : "Run evaluation"}
        </button>
      </Card>

      <style jsx global>{`
        .input { width: 100%; background: rgb(var(--bg)); border: 1px solid rgb(var(--border)); border-radius: 6px; padding: 6px 10px; font-size: 14px; margin-bottom: 8px; }
      `}</style>
    </form>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="mb-1">
      <label className="block text-xs text-muted mb-1">{label}</label>
      {children}
    </div>
  );
}

function ServingSandbox({ versionId, baseModel }: { versionId: string; baseModel?: string }) {
  const [recording, setRecording] = useState(false);
  const [mediaRecorder, setMediaRecorder] = useState<MediaRecorder | null>(null);
  const [transcribing, setTranscribing] = useState(false);
  const [transcript, setTranscript] = useState("");
  const [details, setDetails] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);
  const [dragActive, setDragActive] = useState(false);

  async function startRecording() {
    setErr(null);
    setTranscript("");
    setDetails(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      const chunks: Blob[] = [];
      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunks.push(e.data);
      };
      recorder.onstop = async () => {
        const audioBlob = new Blob(chunks, { type: "audio/wav" });
        await sendAudio(audioBlob);
      };
      recorder.start();
      setMediaRecorder(recorder);
      setRecording(true);
    } catch (e) {
      setErr("Microphone access denied or not supported.");
    }
  }

  function stopRecording() {
    if (mediaRecorder && recording) {
      mediaRecorder.stop();
      mediaRecorder.stream.getTracks().forEach((track) => track.stop());
      setRecording(false);
    }
  }

  async function sendAudio(blob: Blob) {
    setTranscribing(true);
    setErr(null);
    try {
      const formData = new FormData();
      formData.append("file", blob, "recording.wav");
      const res = await fetch(`/api/serve/asr/${versionId}/transcribe`, {
        method: "POST",
        body: formData,
      });
      if (!res.ok) throw new Error(`${res.status}: ${await res.text()}`);
      const body = await res.json();
      setTranscript(body.text);
      setDetails(body);
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setTranscribing(false);
    }
  }

  async function handleFile(file: File) {
    if (!file.type.startsWith("audio/")) {
      setErr("Please upload an audio file.");
      return;
    }
    await sendAudio(file);
  }

  return (
    <Card className="relative overflow-hidden border border-border/80 bg-black/10">
      <CardTitle>Serving Sandbox</CardTitle>
      <p className="text-xs text-muted mb-3">
        Test transcriptions locally with microphone recording or audio uploads.
      </p>

      <div
        className={`border-2 border-dashed rounded-md p-4 flex flex-col items-center justify-center transition-all ${
          dragActive
            ? "border-accent bg-accent/5"
            : "border-border/60 hover:border-accent hover:bg-card/20"
        }`}
        onDragOver={(e) => { e.preventDefault(); setDragActive(true); }}
        onDragLeave={() => setDragActive(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragActive(false);
          if (e.dataTransfer.files?.[0]) handleFile(e.dataTransfer.files[0]);
        }}
      >
        <div className="mb-3 flex flex-col items-center">
          <button
            type="button"
            onClick={recording ? stopRecording : startRecording}
            className={`w-12 h-12 rounded-full flex items-center justify-center transition-all shadow-md ${
              recording
                ? "bg-red-500 text-white animate-pulse scale-105 shadow-red-500/25"
                : "bg-accent text-white hover:scale-105 shadow-accent/25"
            }`}
          >
            {recording ? (
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
                <rect x="4" y="4" width="16" height="16" rx="2" />
              </svg>
            ) : (
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
                <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z" />
                <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
                <line x1="12" x2="12" y1="19" y2="22" />
              </svg>
            )}
          </button>
          <span className="text-[10px] text-muted font-medium mt-1 uppercase tracking-wider">
            {recording ? "Recording..." : "Record Mic"}
          </span>
        </div>

        {recording && (
          <div className="flex gap-0.5 justify-center items-center h-5 mb-2 w-full">
            {[1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3, 4, 5, 4, 3, 2, 1].map((h, i) => (
              <div
                key={i}
                className="w-[2px] bg-red-400 rounded-full animate-bounce"
                style={{
                  height: `${h * 4}px`,
                  animationDelay: `${i * 0.05}s`,
                  animationDuration: "0.8s",
                }}
              />
            ))}
          </div>
        )}

        <div className="text-center text-xs text-muted">
          <label className="cursor-pointer text-accent hover:underline">
            Upload audio file
            <input
              type="file"
              accept="audio/*"
              className="hidden"
              onChange={(e) => {
                if (e.target.files?.[0]) handleFile(e.target.files[0]);
              }}
            />
          </label>
          <span className="mx-1">or drag & drop here</span>
        </div>
      </div>

      {err && <p className="text-red-400 text-xs mt-2 border border-red-500/10 p-2 bg-red-500/5 rounded font-mono">{err}</p>}

      {(transcribing || transcript) && (
        <div className="mt-3 bg-black/40 border border-border/40 rounded p-3 text-xs space-y-2">
          <div className="flex items-center justify-between text-[10px] text-muted font-mono uppercase tracking-wide border-b border-border/40 pb-1">
            <span>Console Output</span>
            {transcribing ? (
              <span className="text-accent animate-pulse font-bold">Transcribing...</span>
            ) : (
              <span className="text-emerald-400 font-bold">Success</span>
            )}
          </div>
          {transcribing ? (
            <div className="flex flex-col gap-1.5 py-2 items-center justify-center text-muted font-mono">
              <div className="w-4 h-4 border-2 border-accent border-t-transparent rounded-full animate-spin" />
              <span>Decoding audio features...</span>
            </div>
          ) : (
            <div className="space-y-2">
              <div className="font-mono text-emerald-300 break-words whitespace-pre-wrap leading-relaxed">
                {transcript || <span className="italic text-muted">Empty transcript received</span>}
              </div>
              {details && (
                <div className="flex flex-wrap gap-x-3 gap-y-1 text-[10px] text-muted border-t border-border/30 pt-1.5 font-mono">
                  {details.language && (
                    <div>
                      <span className="text-muted/65">Language:</span> {details.language}
                    </div>
                  )}
                  {details.words && details.words.length > 0 && (
                    <div>
                      <span className="text-muted/65">Words:</span> {details.words.length}
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </Card>
  );
}

function ChatSandbox({ versionId, baseModelUri }: { versionId: string; baseModelUri?: string }) {
  const [messages, setMessages] = useState<Array<{ role: "user" | "assistant"; content: string }>>([]);
  const [input, setInput] = useState("");
  const [baseModel, setBaseModel] = useState("Qwen/Qwen2.5-0.5B-Instruct");
  const [sending, setSending] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, [messages, sending]);

  async function send() {
    if (!input.trim() || sending) return;
    setErr(null);
    const userMsg = { role: "user" as const, content: input };
    setInput("");
    setSending(true);

    const fullHistory = [...messages, userMsg];
    setMessages(fullHistory);

    try {
      const response = await fetch(`/api/serve/llm/${versionId}/v1/chat/completions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          messages: fullHistory,
          stream: true,
          base_model: baseModel,
          max_tokens: 256,
        }),
      });

      if (!response.ok) throw new Error(`${response.status}: ${await response.text()}`);
      const reader = response.body?.getReader();
      const decoder = new TextDecoder();
      if (!reader) throw new Error("No readable stream in response");

      setMessages((prev) => [...prev, { role: "assistant" as const, content: "" }]);

      let assistantText = "";
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value);
        const lines = chunk.split("\n");
        for (const line of lines) {
          if (line.startsWith("data: ")) {
            const data = line.slice(6).trim();
            if (data === "[DONE]") break;
            try {
              const json = JSON.parse(data);
              const piece = json.choices?.[0]?.delta?.content || "";
              assistantText += piece;
              setMessages((prev) => {
                const copy = [...prev];
                copy[copy.length - 1] = { role: "assistant" as const, content: assistantText };
                return copy;
              });
            } catch {}
          }
        }
      }
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setSending(false);
    }
  }

  return (
    <Card className="relative overflow-hidden border border-border/80 bg-black/10 flex flex-col h-[55vh]">
      <CardTitle>Chat Sandbox</CardTitle>
      <p className="text-[10px] text-muted mb-2">
        Type prompts and witness real-time completions streaming from the fine-tuned adapter.
      </p>

      <div className="mb-2">
        <label className="block text-[10px] text-muted mb-0.5">Base Model (PEFT Adapters)</label>
        <input
          type="text"
          className="input text-xs"
          style={{ marginBottom: 0, padding: "4px 8px" }}
          value={baseModel}
          onChange={(e) => setBaseModel(e.target.value)}
        />
      </div>

      <div
        ref={scrollRef}
        className="flex-1 overflow-y-auto border border-border/40 bg-black/20 rounded p-2.5 mb-2 space-y-2.5 min-h-0"
      >
        {messages.length === 0 && (
          <div className="text-center text-xs text-muted/60 italic py-6">No messages yet. Send one below to start!</div>
        )}
        {messages.map((m, idx) => (
          <div
            key={idx}
            className={`flex flex-col max-w-[85%] rounded px-2.5 py-1.5 text-xs ${
              m.role === "user"
                ? "bg-accent/25 border border-accent/20 text-fg self-end ml-auto"
                : "bg-border/20 border border-border/30 text-fg self-start mr-auto"
            }`}
          >
            <span className="text-[8px] uppercase tracking-wide text-muted mb-0.5 font-semibold font-mono">
              {m.role}
            </span>
            <div className="whitespace-pre-wrap leading-relaxed">{m.content || <span className="animate-pulse">...</span>}</div>
          </div>
        ))}
        {sending && messages[messages.length - 1]?.role === "user" && (
          <div className="bg-border/20 border border-border/30 text-fg self-start mr-auto max-w-[85%] rounded px-2.5 py-1.5 text-xs">
            <span className="text-[8px] uppercase tracking-wide text-muted mb-0.5 font-semibold font-mono">assistant</span>
            <div className="flex gap-1 items-center py-1">
              <span className="w-1.5 h-1.5 bg-muted rounded-full animate-bounce" style={{ animationDelay: "0s" }} />
              <span className="w-1.5 h-1.5 bg-muted rounded-full animate-bounce" style={{ animationDelay: "0.15s" }} />
              <span className="w-1.5 h-1.5 bg-muted rounded-full animate-bounce" style={{ animationDelay: "0.3s" }} />
            </div>
          </div>
        )}
      </div>

      {err && <p className="text-red-400 text-[10px] mb-2 border border-red-500/10 p-1 bg-red-500/5 rounded font-mono">{err}</p>}

      <div className="flex gap-2">
        <input
          type="text"
          placeholder="Ask something..."
          className="input flex-1 text-xs"
          style={{ marginBottom: 0 }}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") send();
          }}
          disabled={sending}
        />
        <button
          type="button"
          onClick={send}
          disabled={sending || !input.trim()}
          className="px-3 py-1 bg-accent text-white rounded text-xs font-semibold hover:scale-105 disabled:opacity-50 transition"
        >
          Send
        </button>
      </div>
    </Card>
  );
}


function TTSSandbox({ versionId }: { versionId: string }) {
  const [text, setText] = useState("Hello, this is a custom voice streaming from Open Audio Studio! Enjoy the low latency speech synthesis.");
  const [stream, setStream] = useState(true);
  const [busy, setBusy] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [ttfa, setTtfa] = useState<number | null>(null);
  const [err, setErr] = useState<string | null>(null);

  // MOS Ratings
  const [averageMos, setAverageMos] = useState<number | null>(null);
  const [ratingsCount, setRatingsCount] = useState<number>(0);
  const [rating, setRating] = useState<number>(0);
  const [hoverRating, setHoverRating] = useState<number>(0);

  // Audio elements
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const nextPlayTimeRef = useRef<number>(0);

  // Stop any active playbacks
  function stopPlayback() {
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.src = "";
    }
    if (audioContextRef.current) {
      audioContextRef.current.close().catch(() => {});
      audioContextRef.current = null;
    }
    setPlaying(false);
  }

  // Web Audio playing base64 chunks
  async function playChunk(b64: string) {
    try {
      if (!audioContextRef.current) {
        audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)();
        nextPlayTimeRef.current = audioContextRef.current.currentTime;
      }
      const audioCtx = audioContextRef.current;

      const binaryStr = window.atob(b64);
      const bytes = new Uint8Array(binaryStr.length);
      for (let i = 0; i < binaryStr.length; i++) {
        bytes[i] = binaryStr.charCodeAt(i);
      }
      
      const audioBuffer = await audioCtx.decodeAudioData(bytes.buffer);
      const source = audioCtx.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(audioCtx.destination);

      const currentTime = audioCtx.currentTime;
      let playTime = nextPlayTimeRef.current;
      if (playTime < currentTime) {
        playTime = currentTime;
      }
      source.start(playTime);
      nextPlayTimeRef.current = playTime + audioBuffer.duration;
      setPlaying(true);
      
      // Stop playing state after buffer duration
      source.onended = () => {
        if (audioCtx.currentTime >= nextPlayTimeRef.current - 0.05) {
          setPlaying(false);
        }
      };
    } catch (e) {
      console.error("Audio chunk playback failed:", e);
    }
  }

  async function synthesize() {
    if (!text.trim() || busy) return;
    setErr(null);
    setTtfa(null);
    setBusy(true);
    stopPlayback();

    const startTime = performance.now();

    try {
      if (stream) {
        const response = await fetch(`/api/serve/tts/${versionId}/stream`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text }),
        });

        if (!response.ok) throw new Error(`${response.status}: ${await response.text()}`);
        const reader = response.body?.getReader();
        const decoder = new TextDecoder();
        if (!reader) throw new Error("No readable stream in response");

        let firstChunkCalculated = false;

        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          const chunk = decoder.decode(value);
          const lines = chunk.split("\n");
          for (const line of lines) {
            if (line.startsWith("data: ")) {
              const dataStr = line.slice(6).trim();
              if (dataStr === "[DONE]") break;
              try {
                const json = JSON.parse(dataStr);
                if (json.error) {
                  setErr(json.error);
                } else if (json.audio) {
                  if (!firstChunkCalculated) {
                    setTtfa(Math.round(performance.now() - startTime));
                    firstChunkCalculated = true;
                  }
                  await playChunk(json.audio);
                }
              } catch {}
            }
          }
        }
      } else {
        const response = await fetch(`/api/serve/tts/${versionId}/synthesize`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text }),
        });

        if (!response.ok) throw new Error(`${response.status}: ${await response.text()}`);
        setTtfa(Math.round(performance.now() - startTime));

        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        
        if (!audioRef.current) {
          audioRef.current = new Audio();
        }
        audioRef.current.src = url;
        audioRef.current.play();
        setPlaying(true);
        audioRef.current.onended = () => setPlaying(false);
      }
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function submitRating(starValue: number) {
    setRating(starValue);
    try {
      const res = await fetch(`/api/models/versions/${versionId}/mos`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ score: starValue }),
      });
      if (res.ok) {
        const updatedVersion = await res.json();
        setAverageMos(updatedVersion.metrics?.mos || null);
        setRatingsCount(updatedVersion.metrics?.mos_ratings?.length || 0);
      }
    } catch (e) {
      console.error("Submit MOS rating failed:", e);
    }
  }

  return (
    <Card className="relative overflow-hidden border border-border/80 bg-black/10 flex flex-col min-h-[45vh]">
      <CardTitle>Voice Serving Sandbox</CardTitle>
      <p className="text-[10px] text-muted mb-3">
        Synthesize text using this voice model version, measure latency (TTFA), and rate quality.
      </p>

      <div className="flex-1 space-y-3">
        <div>
          <label className="block text-[10px] text-muted mb-1">Synthesis Prompts</label>
          <textarea
            className="input text-xs w-full min-h-[80px]"
            style={{ marginBottom: 0 }}
            value={text}
            onChange={(e) => setText(e.target.value)}
            disabled={busy}
            placeholder="Type something to speak..."
          />
        </div>

        <div className="flex items-center justify-between bg-black/20 p-2 rounded border border-border/20">
          <div className="flex items-center gap-1.5">
            <input
              type="checkbox"
              id="streamToggle"
              checked={stream}
              onChange={(e) => setStream(e.target.checked)}
              className="w-3.5 h-3.5 cursor-pointer accent-accent"
              disabled={busy}
            />
            <label htmlFor="streamToggle" className="text-xs text-muted select-none cursor-pointer">
              Enable Low-Latency Sentence Streaming
            </label>
          </div>
          <button
            type="button"
            onClick={synthesize}
            disabled={busy || !text.trim()}
            className="px-4 py-1.5 bg-accent text-white rounded text-xs font-semibold hover:scale-105 disabled:opacity-50 transition"
          >
            {busy ? "Synthesizing…" : "Synthesize voice"}
          </button>
        </div>

        {/* Latency Gauge / Playing wave animation */}
        <div className="grid grid-cols-2 gap-3">
          <div className="bg-black/20 p-2.5 rounded border border-border/20 flex flex-col items-center justify-center">
            <span className="text-[9px] uppercase tracking-wide text-muted mb-1 font-semibold">First-Audio Latency (TTFA)</span>
            {ttfa !== null ? (
              <div className="flex items-baseline gap-0.5">
                <span className="text-xl font-bold font-mono text-accent animate-pulse">{ttfa}</span>
                <span className="text-[10px] text-muted">ms</span>
              </div>
            ) : (
              <span className="text-xs text-muted/50 italic">Awaiting playback...</span>
            )}
          </div>

          <div className="bg-black/20 p-2.5 rounded border border-border/20 flex flex-col items-center justify-center">
            <span className="text-[9px] uppercase tracking-wide text-muted mb-1 font-semibold">Playback Status</span>
            {playing ? (
              <div className="flex gap-1 items-center py-1">
                <span className="w-1 h-3 bg-accent rounded-full animate-bounce" style={{ animationDelay: "0s" }} />
                <span className="w-1 h-4 bg-accent rounded-full animate-bounce" style={{ animationDelay: "0.1s" }} />
                <span className="w-1 h-2 bg-accent rounded-full animate-bounce" style={{ animationDelay: "0.2s" }} />
                <span className="w-1 h-4 bg-accent rounded-full animate-bounce" style={{ animationDelay: "0.3s" }} />
                <span className="w-1 h-1 bg-accent rounded-full animate-bounce" style={{ animationDelay: "0.4s" }} />
              </div>
            ) : (
              <span className="text-xs text-muted/50">Idle</span>
            )}
          </div>
        </div>

        {err && <p className="text-red-400 text-[10px] mb-2 border border-red-500/10 p-1 bg-red-500/5 rounded font-mono">{err}</p>}

        {/* MOS Star Ratings Panel */}
        <div className="border border-border/40 p-2.5 bg-black/20 rounded-md">
          <div className="flex items-center justify-between mb-1.5">
            <span className="text-[10px] uppercase font-bold text-muted tracking-wide">Mean Opinion Score (MOS) Rating</span>
            {averageMos !== null && (
              <span className="text-[10px] text-accent font-semibold">
                ★ {averageMos.toFixed(2)} ({ratingsCount} ratings)
              </span>
            )}
          </div>
          <p className="text-[9px] text-muted mb-2 leading-tight">
            Help evaluate naturalness and pronunciation quality by grading the synthesized voice (1 = bad, 5 = perfect).
          </p>

          <div className="flex gap-1 items-center justify-center py-1">
            {[1, 2, 3, 4, 5].map((val) => {
              const active = hoverRating ? val <= hoverRating : val <= rating;
              return (
                <button
                  key={val}
                  type="button"
                  onMouseEnter={() => setHoverRating(val)}
                  onMouseLeave={() => setHoverRating(0)}
                  onClick={() => submitRating(val)}
                  className="p-1 hover:scale-125 transition"
                >
                  <svg
                    className={`w-5 h-5 ${active ? "text-yellow-400 fill-yellow-400" : "text-muted/40"}`}
                    xmlns="http://www.w3.org/2000/svg"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  >
                    <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2" />
                  </svg>
                </button>
              );
            })}
          </div>
          {rating > 0 && (
            <p className="text-[9px] text-accent text-center mt-1 font-semibold">
              Thank you! You rated this voice: {rating} ★
            </p>
          )}
        </div>
      </div>
    </Card>
  );
}

