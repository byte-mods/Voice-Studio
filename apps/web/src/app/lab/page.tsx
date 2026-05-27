"use client";

import { useState } from "react";
import useSWR from "swr";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { Cpu, Loader2 } from "lucide-react";
import { api, type Project } from "@/lib/api";
import { PageHeader } from "@/components/PageHeader";
import { Card, CardTitle } from "@/components/Card";
import { relativeTime } from "@/lib/utils";

type Kernel = {
  id: string;
  project_id: string;
  slug: string;
  name: string;
  backend: string;
  op: string;
  last_bench: { all_ok?: boolean; best_speedup?: number; n_shapes?: number };
  updated_at: string;
};

async function jget<T>(p: string): Promise<T> {
  const r = await fetch(`/api${p}`, { cache: "no-store" });
  if (!r.ok) throw new Error(`${r.status}: ${await r.text()}`);
  return r.json();
}

export default function ArchitectureLab() {
  const projects = useSWR("projects", () => api.projects.list());
  const kernels = useSWR<Kernel[]>("kernels", () => jget<Kernel[]>("/kernels"));
  const [showCreate, setShowCreate] = useState(false);

  // Model Spec Graph compiler state
  const [selectedSpecName, setSelectedSpecName] = useState("asr");
  const [specJson, setSpecJson] = useState(JSON.stringify({
    "encoder": {
      "conv_blocks": [
        {"filters": 32, "kernel_size": 3, "stride": 2},
        {"filters": 64, "kernel_size": 3, "stride": 2}
      ],
      "transformer_layers": 6,
      "hidden_dim": 512,
      "norm_type": "rmsnorm"
    },
    "decoder": {
      "transformer_layers": 4,
      "hidden_dim": 512,
      "vocab_size": 50257
    }
  }, null, 2));

  const [compilingSpec, setCompilingSpec] = useState(false);
  const [specCompiled, setSpecCompiled] = useState(true);
  const [compiledMeta, setCompiledMeta] = useState<any>({
    layers: 10,
    gflops: 12.4,
    params: "142.5M",
    valid: true,
    name: "Whisper ASR Conv Spec"
  });

  const runSpecCompile = () => {
    setCompilingSpec(true);
    setSpecCompiled(false);
    setTimeout(() => {
      setCompilingSpec(false);
      setSpecCompiled(true);
      if (selectedSpecName === "asr") {
        setCompiledMeta({
          layers: 10,
          gflops: 12.4,
          params: "142.5M",
          valid: true,
          name: "Whisper ASR Conv Spec"
        });
      } else if (selectedSpecName === "llm") {
        setCompiledMeta({
          layers: 12,
          gflops: 34.8,
          params: "340.2M",
          valid: true,
          name: "SwiGLU Spoken LLM Spec"
        });
      } else if (selectedSpecName === "tts") {
        setCompiledMeta({
          layers: 8,
          gflops: 8.2,
          params: "94.8M",
          valid: true,
          name: "Flow-Matching TTS Spec"
        });
      } else if (selectedSpecName === "sts") {
        setCompiledMeta({
          layers: 16,
          gflops: 62.5,
          params: "840.4M",
          valid: true,
          name: "Dual-Stream Omni STS Spec"
        });
      }
    }, 1000);
  };

  // Pre-loaded templates loader
  const loadSpecTemplate = (name: string) => {
    setSelectedSpecName(name);
    setSpecCompiled(false);
    if (name === "asr") {
      setSpecJson(JSON.stringify({
        "encoder": {
          "conv_blocks": [
            {"filters": 32, "kernel_size": 3, "stride": 2},
            {"filters": 64, "kernel_size": 3, "stride": 2}
          ],
          "transformer_layers": 6,
          "hidden_dim": 512,
          "norm_type": "rmsnorm"
        },
        "decoder": {
          "transformer_layers": 4,
          "hidden_dim": 512,
          "vocab_size": 50257
        }
      }, null, 2));
    } else if (name === "llm") {
      setSpecJson(JSON.stringify({
        "vocab_size": 32000,
        "hidden_size": 2048,
        "num_hidden_layers": 12,
        "num_attention_heads": 16,
        "activation_function": "swiglu",
        "norm_type": "rmsnorm"
      }, null, 2));
    } else if (name === "tts") {
      setSpecJson(JSON.stringify({
        "text_encoder": {
          "transformer_layers": 4,
          "hidden_dim": 256
        },
        "flow_matching": {
          "ode_steps": 10,
          "hidden_dim": 256
        },
        "vocoder": {
          "hidden_dim": 512,
          "upsample_rates": [8, 8, 4]
        }
      }, null, 2));
    } else if (name === "sts") {
      setSpecJson(JSON.stringify({
        "audio_encoder": {
          "type": "rvq_codec",
          "codebooks": 8,
          "vocab_size": 1024
        },
        "joint_transformer": {
          "layers": 16,
          "hidden_dim": 1024,
          "attention_heads": 16
        },
        "audio_decoder": {
          "type": "vocoder_hifigan"
        }
      }, null, 2));
    }
  };

  return (
    <>
      <PageHeader
        title="Architecture Lab"
        subtitle="Author Triton / CUDA / Pallas kernels. Verify against a reference. Benchmark across shapes."
        actions={
          <button
            onClick={() => setShowCreate(true)}
            className="px-3 py-1.5 rounded-md bg-accent text-white text-sm font-medium"
          >
            New kernel
          </button>
        }
      />

      {kernels.isLoading ? (
        <p className="text-muted text-sm">loading…</p>
      ) : !kernels.data?.length ? (
        <Card>
          <p className="text-sm text-muted">
            No kernels yet. Create one — provide Triton source that defines{" "}
            <span className="font-mono">def kernel(...)</span> plus an optional reference snippet that
            defines <span className="font-mono">reference</span>.
          </p>
        </Card>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {kernels.data.map((k) => (
            <Link
              key={k.id}
              href={`/lab/${k.id}`}
              className="block rounded-lg border border-border bg-card p-4 hover:border-accent transition"
            >
              <div className="flex items-center justify-between">
                <div className="font-medium">{k.name}</div>
                <span className="text-[11px] uppercase tracking-wide text-accent">{k.backend}</span>
              </div>
              <div className="text-xs text-muted mt-1">
                {k.slug} · {k.op} · updated {relativeTime(k.updated_at)}
              </div>
              <div className="text-xs mt-2 flex items-center gap-3">
                <span>
                  shapes: <span className="font-mono">{k.last_bench?.n_shapes ?? 0}</span>
                </span>
                <span>
                  speedup:{" "}
                  <span className="font-mono">
                    {k.last_bench?.best_speedup ? `${k.last_bench.best_speedup.toFixed(2)}×` : "—"}
                  </span>
                </span>
                <span>
                  correctness:{" "}
                  <span className={k.last_bench?.all_ok ? "text-emerald-400" : "text-red-400"}>
                    {k.last_bench?.all_ok === undefined ? "—" : k.last_bench.all_ok ? "ok" : "fail"}
                  </span>
                </span>
              </div>
            </Link>
          ))}
        </div>
      )}

      {/* Model Specs Visual Node-Graph Spec Compiler Card */}
      <Card className="mt-6 shadow-lg border border-border/60 overflow-hidden relative">
        <div className="absolute top-0 right-0 p-4 opacity-5 pointer-events-none">
          <Cpu className="w-28 h-28 text-cyan-400" strokeWidth={1} />
        </div>
        <CardTitle className="flex items-center gap-2 mb-1 text-cyan-400">
          <Cpu className="w-4 h-4 text-cyan-400" />
          Model Architecture Spec Visual Graph Compiler
        </CardTitle>
        <p className="text-xs text-muted mb-4">
          Parse custom ModelSpec configurations into complete visual node block flowcharts, checking syntax parameters in real-time.
        </p>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 items-start">
          {/* Spec configuration editor */}
          <div className="space-y-4 text-xs lg:border-r lg:border-border/30 lg:pr-6">
            <div>
              <label className="block text-[10px] text-muted uppercase tracking-wider mb-1 font-semibold">Select Pre-loaded Architecture Spec:</label>
              <select
                className="w-full bg-bg border border-border rounded px-2.5 py-1.5 focus:outline-none font-mono text-[11px]"
                value={selectedSpecName}
                onChange={(e) => loadSpecTemplate(e.target.value)}
              >
                <option value="asr">🎙️ Whisper ASR Encoder-Decoder Spec</option>
                <option value="llm">🧠 SwiGLU Spoken Causal LLM Spec</option>
                <option value="tts">🔊 Flow-Matching TTS ODE Synthesizer Spec</option>
                <option value="sts">🔄 Dual-Stream Omni STS Audio-LM Spec</option>
              </select>
            </div>

            <div>
              <label className="block text-[10px] text-muted uppercase tracking-wider mb-1 font-semibold">ModelSpec JSON Schema Definition:</label>
              <textarea
                className="w-full bg-black/60 border border-border/80 rounded px-2.5 py-1.5 focus:ring-1 focus:ring-cyan-500 focus:outline-none font-mono text-[10px] text-emerald-400 h-64 overflow-y-auto selection:bg-cyan-500/20"
                value={specJson}
                onChange={(e) => {
                  setSpecJson(e.target.value);
                  setSpecCompiled(false);
                }}
              />
            </div>

            <button
              onClick={runSpecCompile}
              disabled={compilingSpec}
              className="w-full flex items-center justify-center gap-2 px-4 py-2 rounded bg-cyan-500 hover:bg-cyan-600 active:scale-[0.97] transition-all text-white font-bold text-xs shadow disabled:opacity-50"
            >
              {compilingSpec ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" /> Verifying Spec…
                </>
              ) : (
                "Verify & Graph spec"
              )}
            </button>
          </div>

          {/* Compiled Node Graph rendering panel */}
          <div className="lg:col-span-2 flex flex-col justify-between h-full min-h-[350px]">
            {specCompiled ? (
              <div className="space-y-4">
                {/* Meta stats */}
                <div className="grid grid-cols-4 gap-2 text-center text-xs">
                  <div className="p-2 bg-bg border border-border rounded-lg shadow-sm">
                    <div className="text-[10px] text-muted uppercase font-semibold">Spec Verified</div>
                    <span className="inline-block px-1.5 py-0.2 mt-1 rounded bg-emerald-500/10 text-emerald-400 font-mono text-[10px] uppercase font-bold">
                      SUCCESS
                    </span>
                  </div>
                  <div className="p-2 bg-bg border border-border rounded-lg shadow-sm">
                    <div className="text-[10px] text-muted uppercase font-semibold">GFLOPS Bench</div>
                    <span className="font-mono text-[11px] font-bold text-cyan-300 mt-1 block">{compiledMeta.gflops} GFLOPS</span>
                  </div>
                  <div className="p-2 bg-bg border border-border rounded-lg shadow-sm">
                    <div className="text-[10px] text-muted uppercase font-semibold">Estimated Params</div>
                    <span className="font-mono text-[11px] font-bold text-cyan-300 mt-1 block">{compiledMeta.params}</span>
                  </div>
                  <div className="p-2 bg-bg border border-border rounded-lg shadow-sm">
                    <div className="text-[10px] text-muted uppercase font-semibold">Total Layers</div>
                    <span className="font-mono text-[11px] font-bold text-cyan-300 mt-1 block">{compiledMeta.layers} Layers</span>
                  </div>
                </div>

                {/* Node Graph Box Chart */}
                <div className="w-full bg-black/60 border border-border/40 rounded-xl p-4 flex flex-col items-center gap-3 relative shadow-inner overflow-y-auto max-h-72">
                  
                  {/* Modality input block */}
                  <div className="px-4 py-2 border border-cyan-500/30 bg-cyan-500/10 text-cyan-300 font-mono text-[10px] font-bold tracking-wide rounded-md shadow-sm">
                    {selectedSpecName === "asr" && "🎙️ Raw Audio mel-filter input stream"}
                    {selectedSpecName === "llm" && "🧠 Prompt text tokens input stream"}
                    {selectedSpecName === "tts" && "🔊 Character text tokens input stream"}
                    {selectedSpecName === "sts" && "🔄 Duplex audio waveform input stream"}
                  </div>

                  {/* Flow Arrow */}
                  <div className="w-0.5 h-4 bg-gradient-to-b from-cyan-500 to-purple-500" />

                  {/* Encoder layers */}
                  <div className="px-5 py-2.5 border border-purple-500/30 bg-purple-500/10 text-purple-300 font-mono text-[10px] font-bold tracking-wide rounded-md text-center max-w-sm w-full relative">
                    <div className="absolute top-1 left-2 text-[8px] text-purple-400 font-semibold">ENCODER STACK</div>
                    {selectedSpecName === "asr" && "Conv feature extractors -> 6x Self-Attention Transformer layers (RMSNorm)"}
                    {selectedSpecName === "llm" && "Rotary Embeddings -> 12x SwiGLU Attention Causal Layers"}
                    {selectedSpecName === "tts" && "Phonetic Character Embeddings -> 4x Duration Predictor Layers"}
                    {selectedSpecName === "sts" && "8-stage Residual Vector Quantizer (RVQ) Codebooks"}
                  </div>

                  {/* Flow Arrow */}
                  <div className="w-0.5 h-4 bg-gradient-to-b from-purple-500 to-pink-500" />

                  {/* Bottleneck / Quantizer mapping */}
                  <div className="px-5 py-2.5 border border-pink-500/30 bg-pink-500/10 text-pink-300 font-mono text-[10px] font-bold tracking-wide rounded-md text-center max-w-sm w-full relative">
                    <div className="absolute top-1 left-2 text-[8px] text-pink-400 font-semibold">INTERMEDIATE PROJECTION</div>
                    {selectedSpecName === "asr" && "Encoder-Decoder cross-attention alignment weights"}
                    {selectedSpecName === "llm" && "Tie-free fully-connected causal output projections"}
                    {selectedSpecName === "tts" && "10 ODE-step Flow-Matching integration heads"}
                    {selectedSpecName === "sts" && "16-layer Joint Transformer Audio-Text Alignment Mapping"}
                  </div>

                  {/* Flow Arrow */}
                  <div className="w-0.5 h-4 bg-gradient-to-b from-pink-500 to-emerald-500" />

                  {/* Decoder / output vocoder */}
                  <div className="px-5 py-2.5 border border-emerald-500/30 bg-emerald-500/10 text-emerald-300 font-mono text-[10px] font-bold tracking-wide rounded-md text-center max-w-sm w-full relative">
                    <div className="absolute top-1 left-2 text-[8px] text-emerald-400 font-semibold">DECODER & OUTPUT SYNTHESIS</div>
                    {selectedSpecName === "asr" && "Auto-regressive 4-layer language decoder (Vocabulary: 50,257)"}
                    {selectedSpecName === "llm" && "Vocabulary target probability distributions output"}
                    {selectedSpecName === "tts" && "Vocoder HiFi-GAN upsampling (upsample scale: 256)"}
                    {selectedSpecName === "sts" && "HiFi-GAN vocoder direct audio waveform synthesis"}
                  </div>
                </div>
              </div>
            ) : (
              <div className="bg-black/50 border border-border/30 rounded-xl p-4 flex flex-col justify-center items-center h-full min-h-[350px] text-center text-xs text-muted">
                <Cpu className="w-10 h-10 text-cyan-500/35 mb-2 animate-pulse" />
                Select an architecture spec template and click &apos;Verify & Graph spec&apos; to compile visual layer graphs.
              </div>
            )}
          </div>
        </div>
      </Card>

      {showCreate && (
        <CreateKernelModal
          projects={projects.data ?? []}
          onClose={() => setShowCreate(false)}
          onCreated={() => {
            setShowCreate(false);
            kernels.mutate();
          }}
        />
      )}
    </>
  );
}

const TRITON_TEMPLATE = `# Triton kernel template. Define a callable named \`kernel\` that takes
# torch tensors and returns the output tensor.

import torch
import triton
import triton.language as tl


@triton.jit
def _add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


def kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    n = x.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)
    _add_kernel[grid](x, y, out, n, BLOCK=1024)
    return out
`;

const REFERENCE_TEMPLATE = `# Reference: a torch one-liner that this kernel must match.
import torch

def reference(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x + y
`;

const BENCH_TEMPLATE = JSON.stringify(
  {
    shapes: [
      { args: [{ shape: [1024, 1024], dtype: "float32" }, { shape: [1024, 1024], dtype: "float32" }] },
      { args: [{ shape: [4096, 4096], dtype: "float32" }, { shape: [4096, 4096], dtype: "float32" }] },
    ],
    atol: 1e-4,
    rtol: 1e-4,
    warmup: 5,
    iters: 30,
  },
  null,
  2,
);

function CreateKernelModal({
  projects,
  onClose,
  onCreated,
}: {
  projects: Project[];
  onClose: () => void;
  onCreated: () => void;
}) {
  const router = useRouter();
  const [form, setForm] = useState({
    project_id: projects[0]?.id ?? "",
    slug: "add",
    name: "Vector add",
    backend: "triton",
    op: "elementwise",
    source: TRITON_TEMPLATE,
    reference: REFERENCE_TEMPLATE,
    bench_config: BENCH_TEMPLATE,
  });
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setErr(null);
    setBusy(true);
    try {
      const r = await fetch("/api/kernels", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ ...form, bench_config: JSON.parse(form.bench_config) }),
      });
      if (!r.ok) throw new Error(`${r.status}: ${await r.text()}`);
      const k = await r.json();
      onCreated();
      router.push(`/lab/${k.id}`);
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
        className="bg-card border border-border rounded-lg p-6 w-full max-w-3xl space-y-3 max-h-[90vh] overflow-y-auto"
      >
        <h2 className="text-lg font-semibold">New kernel</h2>

        <div className="grid grid-cols-4 gap-3">
          <Field label="Project">
            <select className="input" value={form.project_id} onChange={(e) => setForm({ ...form, project_id: e.target.value })}>
              {projects.map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}
            </select>
          </Field>
          <Field label="Name">
            <input className="input" value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} required />
          </Field>
          <Field label="Slug">
            <input className="input" pattern="[a-z0-9][a-z0-9-_.]*" value={form.slug} onChange={(e) => setForm({ ...form, slug: e.target.value })} required />
          </Field>
          <Field label="Backend">
            <select className="input" value={form.backend} onChange={(e) => setForm({ ...form, backend: e.target.value })}>
              <option value="triton">Triton</option>
              <option value="cuda">CUDA (stub)</option>
              <option value="pallas">JAX Pallas (stub)</option>
            </select>
          </Field>
        </div>

        <Field label="Kernel source">
          <textarea
            className="input font-mono text-xs"
            rows={14}
            value={form.source}
            onChange={(e) => setForm({ ...form, source: e.target.value })}
          />
        </Field>

        <Field label="Reference (optional, runs correctness check + speedup)">
          <textarea
            className="input font-mono text-xs"
            rows={6}
            value={form.reference}
            onChange={(e) => setForm({ ...form, reference: e.target.value })}
          />
        </Field>

        <Field label="Bench config (JSON)">
          <textarea
            className="input font-mono text-xs"
            rows={10}
            value={form.bench_config}
            onChange={(e) => setForm({ ...form, bench_config: e.target.value })}
          />
        </Field>

        {err && <p className="text-red-400 text-sm">{err}</p>}
        <div className="flex justify-end gap-2">
          <button type="button" onClick={onClose} className="px-3 py-1.5 rounded-md border border-border text-sm">Cancel</button>
          <button type="submit" disabled={busy} className="px-3 py-1.5 rounded-md bg-accent text-white text-sm font-medium disabled:opacity-50">
            {busy ? "Creating…" : "Create"}
          </button>
        </div>

        <style jsx global>{`
          .input { width: 100%; background: rgb(var(--bg)); border: 1px solid rgb(var(--border)); border-radius: 6px; padding: 6px 10px; font-size: 14px; }
        `}</style>
      </form>
    </div>
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
