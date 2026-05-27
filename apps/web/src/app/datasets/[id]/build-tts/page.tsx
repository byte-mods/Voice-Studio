"use client";

import { use, useRef, useState } from "react";
import Link from "next/link";
import { PageHeader } from "@/components/PageHeader";
import { Card, CardTitle } from "@/components/Card";
import { VersionPicker, appendSample } from "@/components/VersionPicker";

type LineRow = {
  text: string;
  audioUri?: string;
  audioDuration?: number;
  audioSr?: number;
  status: "pending" | "uploaded" | "saved" | "error";
  error?: string;
};

export default function TTSBuilder({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const [versionId, setVersionId] = useState<string | null>(null);
  const [speakerId, setSpeakerId] = useState("");
  const [language, setLanguage] = useState("en");
  const [licenseSpdx, setLicenseSpdx] = useState("CC-BY-4.0");
  const [consent, setConsent] = useState({
    consent_id: "",
    granted_at: new Date().toISOString().slice(0, 10),
    document_uri: "",
  });
  const [style, setStyle] = useState("");
  const [emotion, setEmotion] = useState("");

  const [lines, setLines] = useState<LineRow[]>([{ text: "", status: "pending" }]);
  const [savedCount, setSavedCount] = useState(0);
  const [err, setErr] = useState<string | null>(null);

  function updateLine(i: number, patch: Partial<LineRow>) {
    setLines((cur) => cur.map((l, j) => (j === i ? { ...l, ...patch } : l)));
  }

  function addLine() {
    setLines([...lines, { text: "", status: "pending" }]);
  }

  function removeLine(i: number) {
    setLines(lines.filter((_, j) => j !== i));
  }

  async function uploadAudioFor(i: number, file: File) {
    updateLine(i, { status: "pending" });
    const form = new FormData();
    form.append("file", file);
    const r = await fetch("/api/uploads?prefix=tts_dataset", { method: "POST", body: form });
    if (!r.ok) {
      updateLine(i, { status: "error", error: `${r.status}: ${await r.text()}` });
      return;
    }
    const body = await r.json();
    updateLine(i, {
      audioUri: body.uri,
      audioDuration: body.audio?.duration_s,
      audioSr: body.audio?.sample_rate,
      status: "uploaded",
    });
  }

  async function saveLine(i: number) {
    setErr(null);
    if (!versionId) return setErr("pick or create a version first");
    if (!speakerId) return setErr("set a speaker id");
    const row = lines[i];
    if (!row.text.trim() || !row.audioUri) {
      return updateLine(i, { status: "error", error: "text and audio required" });
    }
    try {
      const sample = {
        modality: "tts",
        license: { spdx: licenseSpdx },
        language,
        text: row.text,
        speaker_id: speakerId,
        style: style || null,
        emotion: emotion || null,
        audio: {
          uri: row.audioUri,
          sample_rate: row.audioSr ?? 16000,
          channels: 1,
          duration_s: row.audioDuration ?? 0,
        },
        consent: consent.consent_id
          ? {
              consent_id: consent.consent_id,
              speaker_id: speakerId,
              granted_at: new Date(consent.granted_at + "T00:00:00Z").toISOString(),
              scope: ["tts_clone", "asr_training"],
              document_uri: consent.document_uri || null,
            }
          : null,
      };
      await appendSample(versionId, sample);
      updateLine(i, { status: "saved" });
      setSavedCount((n) => n + 1);
    } catch (e) {
      updateLine(i, { status: "error", error: (e as Error).message });
    }
  }

  async function saveAll() {
    for (let i = 0; i < lines.length; i++) {
      if (lines[i].status !== "saved") await saveLine(i);
    }
  }

  return (
    <>
      <PageHeader
        title="Build TTS dataset"
        subtitle="Voice intake with consent + paired text/audio lines."
        actions={
          <Link href={`/datasets/${id}`} className="px-3 py-1.5 rounded-md border border-border text-sm">
            ← Dataset
          </Link>
        }
      />

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="md:col-span-2 space-y-3">
          <VersionPicker datasetId={id} value={versionId} onChange={setVersionId} />

          <Card>
            <CardTitle>Speaker + voice metadata</CardTitle>
            <div className="grid grid-cols-3 gap-3">
              <Field label="Speaker id">
                <input className="input" value={speakerId} onChange={(e) => setSpeakerId(e.target.value)} required />
              </Field>
              <Field label="Language">
                <input className="input" value={language} onChange={(e) => setLanguage(e.target.value)} />
              </Field>
              <Field label="License (SPDX)">
                <input className="input" value={licenseSpdx} onChange={(e) => setLicenseSpdx(e.target.value)} />
              </Field>
              <Field label="Style">
                <input className="input" value={style} onChange={(e) => setStyle(e.target.value)} placeholder="neutral / cheerful / ..." />
              </Field>
              <Field label="Emotion">
                <input className="input" value={emotion} onChange={(e) => setEmotion(e.target.value)} />
              </Field>
              <div />
            </div>
          </Card>

          <Card>
            <CardTitle>Consent record (required for voice cloning)</CardTitle>
            <div className="grid grid-cols-3 gap-3">
              <Field label="Consent ID">
                <input className="input" value={consent.consent_id} onChange={(e) => setConsent({ ...consent, consent_id: e.target.value })} />
              </Field>
              <Field label="Granted on">
                <input type="date" className="input" value={consent.granted_at} onChange={(e) => setConsent({ ...consent, granted_at: e.target.value })} />
              </Field>
              <Field label="Document URI (optional)">
                <input className="input" placeholder="file:// or s3://" value={consent.document_uri} onChange={(e) => setConsent({ ...consent, document_uri: e.target.value })} />
              </Field>
            </div>
          </Card>

          <Card>
            <CardTitle>Lines</CardTitle>
            <div className="space-y-2">
              {lines.map((line, i) => (
                <LineRowEditor
                  key={i}
                  i={i}
                  line={line}
                  onText={(t) => updateLine(i, { text: t })}
                  onAudio={(f) => uploadAudioFor(i, f)}
                  onSave={() => saveLine(i)}
                  onRemove={() => removeLine(i)}
                />
              ))}
            </div>
            <div className="flex justify-between items-center mt-3">
              <button type="button" onClick={addLine} className="text-xs text-accent">
                + add line
              </button>
              <button onClick={saveAll} className="px-3 py-1.5 rounded-md bg-accent text-white text-sm font-medium">
                Save all unsaved
              </button>
            </div>
          </Card>

          {err && <p className="text-red-400 text-sm">{err}</p>}
          <p className="text-xs text-muted">{savedCount} sample(s) appended this session</p>
        </div>

        <Card>
          <CardTitle>Tips</CardTitle>
          <ul className="text-sm text-muted list-disc list-inside space-y-1">
            <li>Use the same <span className="font-mono">speaker_id</span> across rows for a single voice.</li>
            <li>Consent is enforced when fine-tuning a cloning voice.</li>
            <li>Audio is resampled to the model&apos;s rate at training time; any sane format works.</li>
            <li>Aim for 100+ clean clips per voice for a usable Piper model.</li>
          </ul>
        </Card>
      </div>

      <style jsx global>{`
        .input { width: 100%; background: rgb(var(--bg)); border: 1px solid rgb(var(--border)); border-radius: 6px; padding: 6px 10px; font-size: 14px; }
      `}</style>
    </>
  );
}

function LineRowEditor({
  i,
  line,
  onText,
  onAudio,
  onSave,
  onRemove,
}: {
  i: number;
  line: LineRow;
  onText: (t: string) => void;
  onAudio: (f: File) => void;
  onSave: () => void;
  onRemove: () => void;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const color = {
    pending: "text-muted",
    uploaded: "text-amber-400",
    saved: "text-emerald-400",
    error: "text-red-400",
  }[line.status];

  return (
    <div className="border border-border rounded p-2">
      <div className="flex items-center justify-between mb-1 text-xs">
        <span>#{i + 1}</span>
        <span className={color}>{line.status}{line.error ? `: ${line.error}` : ""}</span>
        <div className="flex gap-2">
          <button type="button" onClick={onRemove} className="text-xs text-red-400">remove</button>
          <button type="button" onClick={onSave} className="text-xs text-accent">save</button>
        </div>
      </div>
      <textarea
        className="input text-sm mb-2"
        rows={2}
        value={line.text}
        onChange={(e) => onText(e.target.value)}
        placeholder="Text to be spoken…"
      />
      <div className="flex items-center gap-3">
        <button
          type="button"
          onClick={() => inputRef.current?.click()}
          className="text-xs px-2 py-1 border border-border rounded"
        >
          {line.audioUri ? "Replace audio" : "Upload audio"}
        </button>
        {line.audioUri && (
          <audio controls className="h-8" src={`/api/uploads/file?uri=${encodeURIComponent(line.audioUri)}`} />
        )}
        {line.audioDuration && (
          <span className="text-[11px] text-muted">{line.audioDuration.toFixed(2)}s @ {line.audioSr}Hz</span>
        )}
        <input
          ref={inputRef}
          type="file"
          accept="audio/*"
          className="hidden"
          onChange={(e) => e.target.files?.[0] && onAudio(e.target.files[0])}
        />
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
