"use client";

import { use, useRef, useState } from "react";
import Link from "next/link";
import { PageHeader } from "@/components/PageHeader";
import { Card, CardTitle } from "@/components/Card";
import { VersionPicker, appendSample } from "@/components/VersionPicker";
import { LANGUAGES } from "@/lib/languages";

type Role = "system" | "user" | "assistant" | "tool";

type Turn = {
  role: Role;
  text: string;
  audioUri?: string;
  duration?: number;
  toolCallsJson?: string;
  toolCallId?: string;
  toolResultsJson?: string;
};

export default function S2SBuilder({ params }: { params: any }) {
  const resolvedParams = params && typeof params.then === "function" ? use(params) : params;
  const { id } = resolvedParams;
  const [versionId, setVersionId] = useState<string | null>(null);
  const [language, setLanguage] = useState("en-US");
  const [licenseSpdx, setLicenseSpdx] = useState("CC-BY-4.0");
  const [systemPrompt, setSystemPrompt] = useState("");
  const [toolsSchema, setToolsSchema] = useState("[]");
  const [turns, setTurns] = useState<Turn[]>([]);
  const [savedCount, setSavedCount] = useState(0);
  const [err, setErr] = useState<string | null>(null);
  const [recording, setRecording] = useState<Role | null>(null);
  const [recordingIndex, setRecordingIndex] = useState<number | null>(null);

  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const streamRef = useRef<MediaStream | null>(null);

  async function startRecording(indexOrRole: number | Role) {
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
        await finalizeRecording(indexOrRole, blob);
        streamRef.current?.getTracks().forEach((t) => t.stop());
        streamRef.current = null;
      };
      rec.start();
      recorderRef.current = rec;
      if (typeof indexOrRole === "number") {
        setRecordingIndex(indexOrRole);
      } else {
        setRecording(indexOrRole);
      }
    } catch (e) {
      setErr((e as Error).message);
    }
  }

  function stopRecording() {
    recorderRef.current?.stop();
    recorderRef.current = null;
    setRecording(null);
    setRecordingIndex(null);
  }

  async function finalizeRecording(indexOrRole: number | Role, blob: Blob) {
    const form = new FormData();
    const ext = blob.type.includes("ogg") ? "ogg" : "webm";
    const prefix = typeof indexOrRole === "number" ? `s2s-turn-${indexOrRole}` : `s2s-${indexOrRole}`;
    form.append("file", new File([blob], `${prefix}-${Date.now()}.${ext}`, { type: blob.type }));
    const r = await fetch("/api/uploads?prefix=s2s_turns", { method: "POST", body: form });
    if (!r.ok) {
      setErr(`upload failed: ${r.status}`);
      return;
    }
    const body = await r.json();
    
    if (typeof indexOrRole === "number") {
      updateTurn(indexOrRole, { audioUri: body.uri, duration: body.audio?.duration_s });
    } else {
      setTurns((cur) => [
        ...cur,
        { role: indexOrRole, text: "", audioUri: body.uri, duration: body.audio?.duration_s },
      ]);
    }
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

  function addManualTurn() {
    setTurns((cur) => [
      ...cur,
      { role: "user", text: "" },
    ]);
  }

  async function saveConversation() {
    setErr(null);
    if (!versionId) return setErr("pick or create a version first");
    if (turns.length === 0) return setErr("add at least one turn");
    try {
      let tools: unknown[] = [];
      try {
        tools = JSON.parse(toolsSchema || "[]");
      } catch (e) {
        throw new Error(`tools schema isn't valid JSON: ${(e as Error).message}`);
      }

      const formattedTurns = turns.map((t, idx) => {
        let tool_calls: any[] = [];
        if (t.role === "assistant" && t.toolCallsJson?.trim()) {
          try {
            tool_calls = JSON.parse(t.toolCallsJson);
            if (!Array.isArray(tool_calls)) {
              throw new Error("Tool calls must be a JSON array");
            }
          } catch (e) {
            throw new Error(`Turn #${idx + 1} tool calls are not valid JSON: ${(e as Error).message}`);
          }
        }

        let tool_results: any[] = [];
        if (t.role === "tool") {
          if (!t.toolCallId?.trim()) {
            throw new Error(`Turn #${idx + 1} role is "tool" but tool_call_id is missing.`);
          }
          let contentVal: any = t.toolResultsJson || "";
          try {
            contentVal = JSON.parse(t.toolResultsJson || "{}");
          } catch (e) {
            // raw string fallback is allowed
          }
          tool_results = [
            {
              tool_call_id: t.toolCallId,
              content: contentVal,
              is_error: false
            }
          ];
        }

        return {
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
          tool_calls,
          tool_results,
        };
      });

      const sample = {
        modality: "s2s",
        license: { spdx: licenseSpdx },
        language,
        system_prompt: systemPrompt || null,
        tools_schema: tools,
        turns: formattedTurns,
      };

      await appendSample(versionId, sample);
      setSavedCount((n) => n + 1);
      resetConversation();
    } catch (e) {
      setErr((e as Error).message);
    }
  }

  const lastActiveRole = turns.filter(t => t.role === "user" || t.role === "assistant").slice(-1)[0]?.role;
  const nextRole: Role = !lastActiveRole || lastActiveRole === "assistant" ? "user" : "assistant";

  return (
    <>
      <PageHeader
        title="Build S2S dataset"
        subtitle="Record conversations turn by turn with support for speech, text, tool calls, and RAG context."
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
                  {LANGUAGES.map((l) => (
                    <option key={l.value} value={l.value}>
                      {l.label} ({l.value})
                    </option>
                  ))}
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
            <CardTitle>System prompt (Optional)</CardTitle>
            <textarea
              className="input font-mono text-xs"
              rows={3}
              placeholder="You are a multimodal assistant capable of calling weather tools..."
              value={systemPrompt}
              onChange={(e) => setSystemPrompt(e.target.value)}
            />
          </Card>

          <Card>
            <CardTitle>Tools (JSON Schema array, optional)</CardTitle>
            <textarea
              className="input font-mono text-xs"
              rows={4}
              placeholder='[{"name":"get_weather","description":"Get current weather","parameters":{...}}]'
              value={toolsSchema}
              onChange={(e) => setToolsSchema(e.target.value)}
            />
          </Card>

          <Card>
            <div className="flex items-center justify-between mb-3">
              <CardTitle>Turns</CardTitle>
              <button
                type="button"
                onClick={addManualTurn}
                className="px-2 py-1 rounded border border-border text-[11px] hover:bg-border/30 transition text-accent font-medium"
              >
                + Add manual turn
              </button>
            </div>
            {turns.length === 0 ? (
              <p className="text-sm text-muted">No turns yet. Hit a record button below or add a manual turn.</p>
            ) : (
              <div className="space-y-2">
                {turns.map((t, i) => (
                  <TurnRow
                    key={i}
                    i={i}
                    turn={t}
                    recordingIndex={recordingIndex}
                    onText={(text) => updateTurn(i, { text })}
                    onRole={(role) => updateTurn(i, { role })}
                    onToolCalls={(toolCallsJson) => updateTurn(i, { toolCallsJson })}
                    onToolCallId={(toolCallId) => updateTurn(i, { toolCallId })}
                    onToolResults={(toolResultsJson) => updateTurn(i, { toolResultsJson })}
                    onRemove={() => removeTurn(i)}
                    startRecording={startRecording}
                    stopRecording={stopRecording}
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
                    className="px-4 py-2 rounded-md bg-accent text-white text-sm font-medium hover:bg-accent/90 transition shadow"
                  >
                    🎤 Record {nextRole}
                  </button>
                  <button
                    onClick={() => startRecording("user")}
                    className="px-3 py-1.5 rounded-md border border-border hover:bg-border/20 text-xs transition"
                  >
                    Record User
                  </button>
                  <button
                    onClick={() => startRecording("assistant")}
                    className="px-3 py-1.5 rounded-md border border-border hover:bg-border/20 text-xs transition"
                  >
                    Record Assistant
                  </button>
                </>
              )}
            </div>
          </Card>

          {err && <p className="text-red-400 text-sm font-mono p-2 bg-red-400/10 border border-red-400/20 rounded">{err}</p>}

          <div className="flex items-center justify-between">
            <span className="text-xs text-muted">{savedCount} conversation(s) appended this session</span>
            <div className="flex gap-2">
              <button onClick={resetConversation} className="px-3 py-1.5 rounded-md border border-border hover:bg-border/20 transition text-sm">
                Reset
              </button>
              <button
                onClick={saveConversation}
                disabled={!versionId || turns.length === 0}
                className="px-4 py-2 rounded-md bg-accent text-white text-sm font-medium disabled:opacity-50 hover:bg-accent/90 transition"
              >
                Save conversation
              </button>
            </div>
          </div>
        </div>

        <Card>
          <CardTitle>Multimodal S2S Builder</CardTitle>
          <ul className="text-xs text-muted list-disc list-inside space-y-2 leading-relaxed">
            <li><strong>Audio Recording</strong>: Each click records from your mic. Default alternates user ↔ assistant.</li>
            <li><strong>System Instructions</strong>: Add system prompts to instruct Qwen-Omni on style, personality, or RAG constraints.</li>
            <li><strong>JSON Tool Schemas</strong>: Declare tools available to the model (in standard JSON-Schema format).</li>
            <li><strong>API/Tool Simulators</strong>: Use <code className="font-mono text-accent">system</code> or <code className="font-mono text-accent">tool</code> roles to feed retrieved search results, weather response documents, or vector embeddings back to the turn sequence!</li>
            <li>One <em>Save conversation</em> = one complete S2SSample row in the version manifest.</li>
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
  recordingIndex,
  onText,
  onRole,
  onToolCalls,
  onToolCallId,
  onToolResults,
  onRemove,
  startRecording,
  stopRecording,
}: {
  i: number;
  turn: Turn;
  recordingIndex: number | null;
  onText: (t: string) => void;
  onRole: (r: Role) => void;
  onToolCalls: (t: string) => void;
  onToolCallId: (id: string) => void;
  onToolResults: (t: string) => void;
  onRemove: () => void;
  startRecording: (idx: number) => void;
  stopRecording: () => void;
}) {
  return (
    <div className="border border-border rounded-lg p-3 bg-card/60 backdrop-blur-sm space-y-2">
      <div className="flex items-center justify-between text-xs">
        <div className="flex items-center gap-2">
          <span className="font-mono text-muted">#{i + 1}</span>
          <select
            className="input w-28 text-xs font-semibold py-0.5"
            value={turn.role}
            onChange={(e) => onRole(e.target.value as Role)}
          >
            <option value="system">system</option>
            <option value="user">user</option>
            <option value="assistant">assistant</option>
            <option value="tool">tool</option>
          </select>
        </div>
        <div className="flex items-center gap-3">
          {turn.duration && (
            <span className="text-muted text-[11px] font-mono">{turn.duration.toFixed(2)}s</span>
          )}
          <button onClick={onRemove} className="text-red-400 hover:text-red-300 text-xs">
            remove
          </button>
        </div>
      </div>

      {["user", "assistant"].includes(turn.role) && (
        <div className="flex items-center gap-2">
          {turn.audioUri ? (
            <audio
              controls
              className="w-full h-8"
              src={`/api/uploads/file?uri=${encodeURIComponent(turn.audioUri)}`}
            />
          ) : (
            <div className="text-xs text-muted font-mono italic">No audio recorded yet.</div>
          )}
          
          {recordingIndex === i ? (
            <button
              onClick={stopRecording}
              className="px-2.5 py-1 bg-red-500 text-white rounded text-[11px] font-medium animate-pulse shrink-0"
            >
              🛑 Stop...
            </button>
          ) : (
            <button
              onClick={() => startRecording(i)}
              className="px-2.5 py-1 border border-border hover:bg-border/30 rounded text-[11px] shrink-0 font-medium transition"
            >
              🎤 {turn.audioUri ? "Re-record" : "Record mic"}
            </button>
          )}
        </div>
      )}

      {turn.role !== "tool" && (
        <textarea
          className="input text-sm"
          rows={2}
          placeholder={
            turn.role === "system"
              ? "System instructions or RAG retrieval context..."
              : "Optional transcript text..."
          }
          value={turn.text}
          onChange={(e) => onText(e.target.value)}
        />
      )}

      {turn.role === "assistant" && (
        <div className="text-xs space-y-1 bg-border/20 p-2 rounded">
          <span className="text-muted font-medium block">⚙️ Tool Calls (JSON array, optional)</span>
          <textarea
            className="input font-mono text-[11px]"
            rows={2}
            placeholder='[{"id":"call_weather","name":"get_weather","arguments":{"city":"Delhi"}}]'
            value={turn.toolCallsJson ?? ""}
            onChange={(e) => onToolCalls(e.target.value)}
          />
        </div>
      )}

      {turn.role === "tool" && (
        <div className="text-xs space-y-2 bg-border/20 p-2 rounded">
          <div>
            <span className="text-muted font-medium block mb-1">🔑 Tool Call ID (matching call id)</span>
            <input
              className="input text-xs font-mono"
              placeholder="call_weather"
              value={turn.toolCallId ?? ""}
              onChange={(e) => onToolCallId(e.target.value)}
            />
          </div>
          <div>
            <span className="text-muted font-medium block mb-1">📦 Tool Output (JSON or raw text)</span>
            <textarea
              className="input font-mono text-[11px]"
              rows={2}
              placeholder='{"weather": "Delhi is 38C and sunny"}'
              value={turn.toolResultsJson ?? ""}
              onChange={(e) => onToolResults(e.target.value)}
            />
          </div>
        </div>
      )}
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
