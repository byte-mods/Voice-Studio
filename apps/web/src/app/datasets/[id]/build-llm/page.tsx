"use client";

import { use, useState } from "react";
import Link from "next/link";
import { PageHeader } from "@/components/PageHeader";
import { Card, CardTitle } from "@/components/Card";
import { VersionPicker, appendSample } from "@/components/VersionPicker";
import { LANGUAGES } from "@/lib/languages";
import { PromptDictionary } from "@/components/PromptDictionary";

type Role = "system" | "user" | "assistant" | "tool";

type Turn = { role: Role; text: string };

export default function LLMBuilder({ params }: { params: any }) {
  const resolvedParams = params && typeof params.then === "function" ? use(params) : params;
  const { id } = resolvedParams;
  const [versionId, setVersionId] = useState<string | null>(null);
  const [systemPrompt, setSystemPrompt] = useState("");
  const [toolsSchema, setToolsSchema] = useState("[]");
  const [language, setLanguage] = useState("en-US");
  const [licenseSpdx, setLicenseSpdx] = useState("CC-BY-4.0");
  const [turns, setTurns] = useState<Turn[]>([
    { role: "user", text: "" },
    { role: "assistant", text: "" },
  ]);
  const [savedCount, setSavedCount] = useState(0);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<"tips" | "dict">("dict");

  function handleSelectPrompt(text: string, _emo: string) {
    setTurns((cur) => {
      if (cur.length === 0) {
        return [{ role: "user", text }];
      }
      const last = cur[cur.length - 1];
      if (!last.text.trim()) {
        return cur.map((t, idx) => (idx === cur.length - 1 ? { ...t, text } : t));
      }
      const nextRole: Role = last.role === "user" ? "assistant" : "user";
      return [...cur, { role: nextRole, text }];
    });
  }

  function updateTurn(i: number, patch: Partial<Turn>) {
    setTurns((cur) => cur.map((t, j) => (j === i ? { ...t, ...patch } : t)));
  }

  function addTurn() {
    const next: Role = turns[turns.length - 1].role === "user" ? "assistant" : "user";
    setTurns([...turns, { role: next, text: "" }]);
  }

  function removeTurn(i: number) {
    setTurns(turns.filter((_, j) => j !== i));
  }

  async function save() {
    setErr(null);
    if (!versionId) return setErr("pick or create a version first");
    setBusy(true);
    try {
      let tools: unknown[] = [];
      try {
        tools = JSON.parse(toolsSchema || "[]");
      } catch (e) {
        throw new Error(`tools schema isn't valid JSON: ${(e as Error).message}`);
      }
      const sample = {
        modality: "llm",
        license: { spdx: licenseSpdx },
        language: language,
        system_prompt: systemPrompt || null,
        tools_schema: tools,
        turns: turns
          .filter((t) => t.text.trim())
          .map((t) => ({ role: t.role, text: t.text })),
      };
      await appendSample(versionId, sample);
      setSavedCount((n) => n + 1);
      // Reset only the conversation; keep system prompt and tools sticky.
      setTurns([
        { role: "user", text: "" },
        { role: "assistant", text: "" },
      ]);
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      <PageHeader
        title="Build LLM dataset"
        subtitle="Multi-turn conversation editor. Each save appends one LLMSample."
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
            <CardTitle>Dataset metadata</CardTitle>
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
              <Field label="Or Custom ISO Code">
                <input className="input" placeholder="e.g. hi-IN, mr" value={language} onChange={(e) => setLanguage(e.target.value)} />
              </Field>
              <Field label="License (SPDX)">
                <input className="input" value={licenseSpdx} onChange={(e) => setLicenseSpdx(e.target.value)} />
              </Field>
            </div>
          </Card>

          <Card>
            <CardTitle>System prompt</CardTitle>
            <textarea
              className="input font-mono text-xs"
              rows={3}
              placeholder="You are a helpful assistant…"
              value={systemPrompt}
              onChange={(e) => setSystemPrompt(e.target.value)}
            />
          </Card>

          <Card>
            <CardTitle>Tools (JSON Schema array, optional)</CardTitle>
            <textarea
              className="input font-mono text-xs"
              rows={5}
              placeholder='[{"name":"lookup_weather","parameters":{...}}]'
              value={toolsSchema}
              onChange={(e) => setToolsSchema(e.target.value)}
            />
          </Card>

          <Card>
            <CardTitle>Conversation</CardTitle>
            <div className="space-y-2">
              {turns.map((t, i) => (
                <div key={i} className="border border-border rounded p-2">
                  <div className="flex items-center justify-between mb-1">
                    <select
                      className="input w-32 text-xs"
                      value={t.role}
                      onChange={(e) => updateTurn(i, { role: e.target.value as Role })}
                    >
                      <option value="system">system</option>
                      <option value="user">user</option>
                      <option value="assistant">assistant</option>
                      <option value="tool">tool</option>
                    </select>
                    <button
                      type="button"
                      onClick={() => removeTurn(i)}
                      className="text-xs text-red-400 hover:underline"
                    >
                      remove
                    </button>
                  </div>
                  <textarea
                    className="input text-sm"
                    rows={3}
                    value={t.text}
                    onChange={(e) => updateTurn(i, { text: e.target.value })}
                  />
                </div>
              ))}
            </div>
            <button
              type="button"
              onClick={addTurn}
              className="mt-2 text-xs text-accent hover:underline"
            >
              + add turn
            </button>
          </Card>

          {err && <p className="text-red-400 text-sm">{err}</p>}
          <div className="flex justify-between items-center">
            <span className="text-xs text-muted">{savedCount} sample(s) appended this session</span>
            <button
              onClick={save}
              disabled={busy || !versionId}
              className="px-4 py-2 rounded-md bg-accent text-white text-sm font-medium disabled:opacity-50"
            >
              {busy ? "Saving…" : "Save sample"}
            </button>
          </div>
        </div>

        <div className="space-y-4">
          <Card>
            <div className="flex bg-card/60 p-1 rounded-md gap-1 mb-3 border border-border">
              <button
                type="button"
                onClick={() => setActiveTab("dict")}
                className={`flex-1 text-center py-1 rounded text-xs font-semibold transition ${
                  activeTab === "dict" ? "bg-accent text-white" : "text-muted hover:text-white"
                }`}
              >
                📋 Prompt Dictionary
              </button>
              <button
                type="button"
                onClick={() => setActiveTab("tips")}
                className={`flex-1 text-center py-1 rounded text-xs font-semibold transition ${
                  activeTab === "tips" ? "bg-accent text-white" : "text-muted hover:text-white"
                }`}
              >
                📖 Guidelines
              </button>
            </div>

            {activeTab === "tips" ? (
              <>
                <CardTitle>LLM Dataset Builder</CardTitle>
                <ul className="text-xs text-muted list-disc list-inside space-y-2 leading-relaxed">
                  <li><strong>Manual Editing</strong>: Type user questions and expected assistant answers directly.</li>
                  <li><strong>System Prompt</strong>: Dictate the assistant persona or behavior (e.g. helpful customer service agent).</li>
                  <li><strong>API Integration</strong>: Declare schemas of tools that the model can choose to call.</li>
                  <li><strong>Synthetic Generator</strong>: Use a registered LLM model on the right to bootstrap synthetic dataset samples.</li>
                </ul>
              </>
            ) : (
              <>
                <CardTitle>Prompt Sheet</CardTitle>
                <p className="text-xs text-muted mb-3">
                  Click any sentence to instantly append or fill active dialogue turns.
                </p>
                <PromptDictionary onSelectSentence={handleSelectPrompt} />
              </>
            )}
          </Card>

          <SyntheticPanel
            versionId={versionId}
            systemPrompt={systemPrompt}
            onAppended={() => setSavedCount((n) => n + 1)}
          />
        </div>
      </div>

      <style jsx global>{`
        .input { width: 100%; background: rgb(var(--bg)); border: 1px solid rgb(var(--border)); border-radius: 6px; padding: 6px 10px; font-size: 14px; }
      `}</style>
    </>
  );
}

function SyntheticPanel({
  versionId,
  systemPrompt,
  onAppended,
}: {
  versionId: string | null;
  systemPrompt: string;
  onAppended: () => void;
}) {
  const [llmVersionId, setLlmVersionId] = useState("");
  const [baseModel, setBaseModel] = useState("Qwen/Qwen2.5-0.5B-Instruct");
  const [prompt, setPrompt] = useState(
    "Generate ONE diverse user question for the system above. Reply with only the question text.",
  );
  const [n, setN] = useState(5);
  const [busy, setBusy] = useState(false);
  const [log, setLog] = useState<string[]>([]);
  const [err, setErr] = useState<string | null>(null);

  async function run() {
    setErr(null);
    if (!versionId) return setErr("pick a target version on the left");
    if (!llmVersionId) return setErr("paste a serve-able LLM ModelVersion id");
    setBusy(true);
    setLog([]);
    try {
      for (let i = 0; i < n; i++) {
        const body: Record<string, unknown> = {
          messages: [
            { role: "system", content: systemPrompt || "You are a synthetic data generator." },
            { role: "user", content: prompt },
          ],
          max_tokens: 200,
          stream: false,
        };
        if (baseModel) body.base_model = baseModel;
        const r = await fetch(`/api/serve/llm/${llmVersionId}/v1/chat/completions`, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify(body),
        });
        if (!r.ok) throw new Error(`${r.status}: ${await r.text()}`);
        const json = await r.json();
        const userTurn = json.choices?.[0]?.message?.content?.trim() ?? "";
        if (!userTurn) {
          setLog((l) => [...l, `[${i + 1}/${n}] empty completion, skipped`]);
          continue;
        }
        const sample = {
          modality: "llm",
          license: { spdx: "CC-BY-4.0" },
          language: "en",
          system_prompt: systemPrompt || null,
          turns: [
            { role: "user", text: userTurn },
            { role: "assistant", text: "" },
          ],
          metadata: { synthetic: true, generator: llmVersionId },
        };
        await appendSample(versionId, sample);
        onAppended();
        setLog((l) => [...l, `[${i + 1}/${n}] ${userTurn.slice(0, 60)}…`]);
      }
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <Card>
      <CardTitle>Synthesize (optional)</CardTitle>
      <p className="text-xs text-muted mb-3">
        Generates N user questions via a registered LLM ModelVersion. Each becomes a fresh sample
        with an empty assistant turn for you to fill in later.
      </p>
      <label className="block text-xs text-muted mb-1">LLM ModelVersion id</label>
      <input className="input mb-2" placeholder="mv_…" value={llmVersionId} onChange={(e) => setLlmVersionId(e.target.value)} />
      <label className="block text-xs text-muted mb-1">Base model (PEFT adapters)</label>
      <input className="input mb-2" value={baseModel} onChange={(e) => setBaseModel(e.target.value)} />
      <label className="block text-xs text-muted mb-1">Prompt</label>
      <textarea className="input mb-2" rows={3} value={prompt} onChange={(e) => setPrompt(e.target.value)} />
      <label className="block text-xs text-muted mb-1">N</label>
      <input type="number" min={1} max={50} className="input mb-2" value={n} onChange={(e) => setN(Number(e.target.value))} />
      {err && <p className="text-xs text-red-400 mb-1">{err}</p>}
      <button
        onClick={run}
        disabled={busy}
        className="w-full px-3 py-1.5 rounded-md bg-accent text-white text-sm font-medium disabled:opacity-50"
      >
        {busy ? "Generating…" : `Generate ${n}`}
      </button>
      {log.length > 0 && (
        <pre className="mt-3 text-[10px] font-mono bg-bg p-2 rounded max-h-40 overflow-y-auto">
          {log.join("\n")}
        </pre>
      )}
    </Card>
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
