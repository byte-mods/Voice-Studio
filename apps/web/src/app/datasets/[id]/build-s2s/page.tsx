"use client";

import { use, useRef, useState } from "react";
import Link from "next/link";
import { PageHeader } from "@/components/PageHeader";
import { Card, CardTitle } from "@/components/Card";
import { VersionPicker, appendSample } from "@/components/VersionPicker";

type Role = "user" | "assistant";

type Turn = {
  role: Role;
  text: string;
  audioUri?: string;
  duration?: number;
};

export default function S2SBuilder({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const [versionId, setVersionId] = useState<string | null>(null);
  const [language, setLanguage] = useState("en");
  const [licenseSpdx, setLicenseSpdx] = useState("CC-BY-4.0");
  const [turns, setTurns] = useState<Turn[]>([]);
  const [savedCount, setSavedCount] = useState(0);
  const [err, setErr] = useState<string | null>(null);
  const [recording, setRecording] = useState<Role | null>(null);

  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const streamRef = useRef<MediaStream | null>(null);

  async function startRecording(role: Role) {
    setErr(null);
    chunksRef.current = [];
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
      });
      streamRef.current = stream;
      const mime = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
        ? "audio/webm;codecs=opus"
        : MediaRecorder.isTypeSupported("audio/webm")
          ? "audio/webm"
          : "";
      const rec = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined);
      rec.ondataavailable = (e) => e.data.size > 0 && chunksRef.current.push(e.data);
      rec.onstop = async () => {
        const blob = new Blob(chunksRef.current, { type: rec.mimeType || "audio/webm" });
        await finalizeRecording(role, blob);
        streamRef.current?.getTracks().forEach((t) => t.stop());
        streamRef.current = null;
      };
      rec.start();
      recorderRef.current = rec;
      setRecording(role);
    } catch (e) {
      setErr((e as Error).message);
    }
  }

  function stopRecording() {
    recorderRef.current?.stop();
    recorderRef.current = null;
    setRecording(null);
  }

  async function finalizeRecording(role: Role, blob: Blob) {
    const form = new FormData();
    const ext = blob.type.includes("ogg") ? "ogg" : "webm";
    form.append("file", new File([blob], `${role}-${Date.now()}.${ext}`, { type: blob.type }));
    const r = await fetch("/api/uploads?prefix=s2s_turns", { method: "POST", body: form });
    if (!r.ok) {
      setErr(`upload failed: ${r.status}`);
      return;
    }
    const body = await r.json();
    setTurns((cur) => [
      ...cur,
      { role, text: "", audioUri: body.uri, duration: body.audio?.duration_s },
    ]);
  }

  function updateTurn(i: number, patch: Partial<Turn>) {
    setTurns((cur) => cur.map((t, j) => (j === i ? { ...t, ...patch } : t)));
  }

  function removeTurn(i: number) {
    setTurns(turns.filter((_, j) => j !== i));
  }

  function resetConversation() {
    setTurns([]);
  }

  async function saveConversation() {
    setErr(null);
    if (!versionId) return setErr("pick or create a version first");
    if (turns.length === 0) return setErr("record at least one turn");
    try {
      const sample = {
        modality: "s2s",
        license: { spdx: licenseSpdx },
        language,
        turns: turns.map((t) => ({
          role: t.role,
          text: t.text || null,
          audio: t.audioUri
            ? {
                uri: t.audioUri,
                sample_rate: 16000,
                channels: 1,
                duration_s: t.duration ?? 0,
              }
            : null,
        })),
      };
      await appendSample(versionId, sample);
      setSavedCount((n) => n + 1);
      resetConversation();
    } catch (e) {
      setErr((e as Error).message);
    }
  }

  const nextRole: Role = turns.length === 0 || turns[turns.length - 1].role === "assistant" ? "user" : "assistant";

  return (
    <>
      <PageHeader
        title="Build S2S dataset"
        subtitle="Record conversations turn by turn. Browser mic. One save appends one S2SSample."
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
            <CardTitle>Conversation metadata</CardTitle>
            <div className="grid grid-cols-3 gap-3">
              <Field label="Preset Language">
                <select
                  className="input"
                  value={language}
                  onChange={(e) => setLanguage(e.target.value)}
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
              <Field label="Or Custom ISO Code (e.g. mr, te, hi-IN)">
                <input className="input" placeholder="e.g. hi-IN, mr" value={language} onChange={(e) => setLanguage(e.target.value)} />
              </Field>
              <Field label="License (SPDX)">
                <input className="input" value={licenseSpdx} onChange={(e) => setLicenseSpdx(e.target.value)} />
              </Field>
            </div>
          </Card>

          <Card>
            <CardTitle>Turns</CardTitle>
            {turns.length === 0 ? (
              <p className="text-sm text-muted">No turns yet. Hit a record button below.</p>
            ) : (
              <div className="space-y-2">
                {turns.map((t, i) => (
                  <TurnRow
                    key={i}
                    i={i}
                    turn={t}
                    onText={(text) => updateTurn(i, { text })}
                    onRemove={() => removeTurn(i)}
                  />
                ))}
              </div>
            )}

            <div className="flex items-center gap-3 mt-4">
              {recording ? (
                <button
                  onClick={stopRecording}
                  className="px-4 py-2 rounded-md bg-red-500 text-white text-sm font-medium animate-pulse"
                >
                  Stop ({recording})
                </button>
              ) : (
                <>
                  <button
                    onClick={() => startRecording(nextRole)}
                    className="px-4 py-2 rounded-md bg-accent text-white text-sm font-medium"
                  >
                    Record {nextRole}
                  </button>
                  <button
                    onClick={() => startRecording("user")}
                    className="px-3 py-1.5 rounded-md border border-border text-xs"
                  >
                    user
                  </button>
                  <button
                    onClick={() => startRecording("assistant")}
                    className="px-3 py-1.5 rounded-md border border-border text-xs"
                  >
                    assistant
                  </button>
                </>
              )}
            </div>
          </Card>

          {err && <p className="text-red-400 text-sm">{err}</p>}

          <div className="flex items-center justify-between">
            <span className="text-xs text-muted">{savedCount} conversation(s) appended this session</span>
            <div className="flex gap-2">
              <button onClick={resetConversation} className="px-3 py-1.5 rounded-md border border-border text-sm">
                Reset
              </button>
              <button
                onClick={saveConversation}
                disabled={!versionId || turns.length === 0}
                className="px-4 py-2 rounded-md bg-accent text-white text-sm font-medium disabled:opacity-50"
              >
                Save conversation
              </button>
            </div>
          </div>
        </div>

        <Card>
          <CardTitle>How it works</CardTitle>
          <ul className="text-sm text-muted list-disc list-inside space-y-1">
            <li>Each click records one turn from your mic.</li>
            <li>Buttons alternate user → assistant by default.</li>
            <li>Transcripts are optional — pure-audio S2S works too.</li>
            <li>One <em>Save conversation</em> = one S2SSample row in the manifest.</li>
          </ul>
        </Card>
      </div>

      <style jsx global>{`
        .input { width: 100%; background: rgb(var(--bg)); border: 1px solid rgb(var(--border)); border-radius: 6px; padding: 6px 10px; font-size: 14px; }
      `}</style>
    </>
  );
}

function TurnRow({
  i,
  turn,
  onText,
  onRemove,
}: {
  i: number;
  turn: Turn;
  onText: (t: string) => void;
  onRemove: () => void;
}) {
  return (
    <div className="border border-border rounded p-2">
      <div className="flex items-center justify-between mb-1 text-xs">
        <span className={turn.role === "user" ? "text-accent" : "text-emerald-400"}>
          #{i + 1} · {turn.role}
        </span>
        {turn.duration && (
          <span className="text-muted">{turn.duration.toFixed(2)}s</span>
        )}
        <button onClick={onRemove} className="text-red-400">remove</button>
      </div>
      {turn.audioUri && (
        <audio controls className="w-full h-8 mb-1" src={`/api/uploads/file?uri=${encodeURIComponent(turn.audioUri)}`} />
      )}
      <input
        className="input text-sm"
        placeholder="Optional transcript…"
        value={turn.text}
        onChange={(e) => onText(e.target.value)}
      />
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
