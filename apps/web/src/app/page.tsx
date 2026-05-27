"use client";

import useSWR from "swr";
import Link from "next/link";
import { api } from "@/lib/api";
import { PageHeader } from "@/components/PageHeader";
import { Card, CardTitle } from "@/components/Card";

export default function Dashboard() {
  const info = useSWR("info", () => api.system.info());
  const projects = useSWR("projects", () => api.projects.list());
  const jobs = useSWR("jobs", () => api.jobs.list());

  return (
    <>
      <PageHeader
        title="Welcome to Open Audio Studio"
        subtitle="Fine-tune ASR, LLM, TTS, and speech-to-speech models — all from one place."
      />

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
        <Card>
          <CardTitle>System</CardTitle>
          {info.error ? (
            <p className="text-red-400 text-sm">API unreachable</p>
          ) : info.isLoading ? (
            <p className="text-muted text-sm">loading…</p>
          ) : (
            <div className="space-y-1 text-sm">
              <div>
                <span className="text-muted">Version:</span>{" "}
                {(info.data as { version: string }).version}
              </div>
              <div>
                <span className="text-muted">Python:</span>{" "}
                {(info.data as { python: string }).python}
              </div>
              <div>
                <span className="text-muted">GPUs:</span>{" "}
                {(info.data as { gpus: unknown[] }).gpus.length}
              </div>
              <div>
                <span className="text-muted">Handlers:</span>{" "}
                {(info.data as { handlers: string[] }).handlers.join(", ") || "none"}
              </div>
            </div>
          )}
        </Card>

        <Card>
          <CardTitle>Projects</CardTitle>
          <div className="text-2xl font-semibold">
            {Array.isArray(projects.data) ? projects.data.length : "—"}
          </div>
          <Link href="/projects" className="text-xs text-accent mt-2 inline-block">
            Manage projects →
          </Link>
        </Card>

        <Card>
          <CardTitle>Jobs in last batch</CardTitle>
          <div className="text-2xl font-semibold">
            {Array.isArray(jobs.data) ? jobs.data.length : "—"}
          </div>
          <Link href="/jobs" className="text-xs text-accent mt-2 inline-block">
            Open job queue →
          </Link>
        </Card>
      </div>

      <h2 className="text-sm font-medium text-muted mb-2">Start somewhere</h2>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {[
          { href: "/datasets", title: "Datasets", body: "Pull from HF or build custom ASR / TTS / LLM / S2S datasets." },
          { href: "/asr", title: "ASR Studio", body: "Fine-tune Whisper, Parakeet, wav2vec2. Stream partial transcripts." },
          { href: "/llm", title: "LLM Studio", body: "LoRA / QLoRA / full fine-tune. Tool use, spoken-style training." },
          { href: "/tts", title: "TTS Studio", body: "Fine-tune Piper, XTTS, StyleTTS2, F5-TTS. Voice management." },
          { href: "/s2s", title: "S2S Studio", body: "Pipeline (ASR+LLM+TTS) and native audio models. Realtime playground." },
          { href: "/lab", title: "Architecture Lab", body: "PyTorch + JAX. Custom CUDA / Triton / Pallas kernels." },
        ].map((c) => (
          <Link
            key={c.href}
            href={c.href}
            className="rounded-lg border border-border bg-card p-4 hover:border-accent transition"
          >
            <div className="font-medium">{c.title}</div>
            <div className="text-sm text-muted mt-1">{c.body}</div>
          </Link>
        ))}
      </div>
    </>
  );
}
