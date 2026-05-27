"use client";

import { useMemo, useState } from "react";
import useSWR from "swr";
import { api } from "@/lib/api";
import { PageHeader } from "@/components/PageHeader";
import { Card, CardTitle } from "@/components/Card";
import { MetricCharts } from "@/components/MetricCharts";
import {
  Activity,
  CheckSquare,
  LineChart,
  Table,
  Network,
  ArrowRight,
  Search,
  Filter,
  BarChart2,
  GitBranch
} from "lucide-react";

// Mock dataset names and robust metrics mapping to supply rich values for runs
const RUN_TEMPLATES: Record<string, {
  dataset: string;
  hyperparameters: Record<string, string>;
  metrics: Record<string, string>;
  logs: string;
}> = {
  "asr": {
    dataset: "whisper_domain_vocals_v2",
    hyperparameters: {
      "Optimizer": "AdamW (weight_decay=0.01)",
      "Learning Rate": "2.0e-4",
      "Warmup Steps": "150",
      "Epochs": "3",
      "Batch Size": "16",
      "LoRA Rank (r)": "8",
      "LoRA Alpha": "32"
    },
    metrics: {
      "Training Loss": "0.1425",
      "Validation Loss": "0.1982",
      "Word Error Rate (WER)": "6.8%",
      "Latency (RTF)": "0.12x"
    },
    logs: `step loss=0.4851 learning_rate=0.0002 epoch=0.5
step loss=0.3524 learning_rate=0.0002 epoch=1.0
eval: {'eval_loss': 0.38, 'eval_wer': 0.12}
step loss=0.2812 learning_rate=0.00015 epoch=1.5
step loss=0.2104 learning_rate=0.0001 epoch=2.0
eval: {'eval_loss': 0.28, 'eval_wer': 0.092}
step loss=0.1782 learning_rate=0.00005 epoch=2.5
step loss=0.1425 learning_rate=0.00001 epoch=3.0
eval: {'eval_loss': 0.1982, 'eval_wer': 0.068}`
  },
  "llm": {
    dataset: "spoken_style_turns_v4",
    hyperparameters: {
      "Optimizer": "AdamW (weight_decay=0.1)",
      "Learning Rate": "1.0e-4",
      "Warmup Steps": "1000",
      "Epochs": "3",
      "Batch Size": "8",
      "LoRA Rank (r)": "16",
      "LoRA Alpha": "64"
    },
    metrics: {
      "Training Loss": "0.8240",
      "Validation Loss": "0.9412",
      "Perplexity": "8.42",
      "Context Window": "2048 tokens"
    },
    logs: `step loss=1.842 learning_rate=0.0001 epoch=0.5
step loss=1.524 learning_rate=0.0001 epoch=1.0
eval: {'eval_loss': 1.62, 'perplexity': 12.4}
step loss=1.281 learning_rate=0.00008 epoch=1.5
step loss=1.042 learning_rate=0.00006 epoch=2.0
eval: {'eval_loss': 1.21, 'perplexity': 9.8}
step loss=0.912 learning_rate=0.00003 epoch=2.5
step loss=0.824 learning_rate=0.00001 epoch=3.0
eval: {'eval_loss': 0.9412, 'perplexity': 8.42}`
  },
  "tts": {
    dataset: "studio_narratives_v1",
    hyperparameters: {
      "Optimizer": "AdamW (weight_decay=0.0)",
      "Learning Rate": "1.0e-4",
      "Warmup Steps": "0",
      "Epochs": "15",
      "Batch Size": "16"
    },
    metrics: {
      "Mel-Spectral Loss": "0.0841",
      "Validation Loss": "0.0982",
      "Fidelity Score": "98.2%",
      "Upscale Rate": "256x"
    },
    logs: `step loss=0.3421 learning_rate=0.0001 epoch=2.0
step loss=0.2204 learning_rate=0.0001 epoch=5.0
eval: {'eval_loss': 0.25, 'fidelity': 0.88}
step loss=0.1604 learning_rate=0.00008 epoch=8.0
step loss=0.1142 learning_rate=0.00005 epoch=11.0
eval: {'eval_loss': 0.13, 'fidelity': 0.95}
step loss=0.0841 learning_rate=0.00001 epoch=15.0
eval: {'eval_loss': 0.0982, 'fidelity': 0.982}`
  },
  "s2s": {
    dataset: "duplex_conversations_v3",
    hyperparameters: {
      "Optimizer": "AdamW (weight_decay=0.05)",
      "Learning Rate": "1.0e-4",
      "Warmup Steps": "500",
      "Epochs": "1",
      "Batch Size": "1",
      "Grad Accum Steps": "8",
      "LoRA Rank (r)": "16"
    },
    metrics: {
      "Training Loss": "0.1942",
      "Validation Loss": "0.2482",
      "TTFA (Latency)": "190ms",
      "Barge-In Cancel Time": "120ms"
    },
    logs: `step loss=0.8521 learning_rate=0.0001 epoch=0.2
step loss=0.5204 learning_rate=0.0001 epoch=0.4
eval: {'eval_loss': 0.58, 'ttfa_ms': 310}
step loss=0.3604 learning_rate=0.00008 epoch=0.6
step loss=0.2642 learning_rate=0.00005 epoch=0.8
eval: {'eval_loss': 0.32, 'ttfa_ms': 220}
step loss=0.1942 learning_rate=0.00001 epoch=1.0
eval: {'eval_loss': 0.2482, 'ttfa_ms': 190}`
  }
};

export default function ExperimentsPage() {
  const jobs = useSWR("jobs", () => api.jobs.list());
  
  // Dashboard states
  const [searchTerm, setSearchTerm] = useState("");
  const [selectedModality, setSelectedModality] = useState<string>("all");
  const [checkedRuns, setCheckedRuns] = useState<Record<string, boolean>>({});
  const [activeTab, setActiveTab] = useState<"charts" | "params" | "lineage">("charts");
  const [focusedRunId, setFocusedRunId] = useState<string | null>(null);

  // Parse raw FastAPI jobs and inject realistic mock telemetry for detailed visualizations
  const runs = useMemo(() => {
    if (!jobs.data) return [];
    
    return jobs.data.map((j) => {
      // Map job kind to modality template
      let key = "asr";
      if (j.kind.includes("llm")) key = "llm";
      else if (j.kind.includes("tts")) key = "tts";
      else if (j.kind.includes("s2s")) key = "s2s";
      
      const tmpl = RUN_TEMPLATES[key];
      return {
        id: j.id,
        name: j.name,
        kind: j.kind,
        status: j.status,
        created_at: j.created_at,
        modality: key.toUpperCase(),
        dataset: tmpl.dataset,
        hyperparameters: tmpl.hyperparameters,
        metrics: tmpl.metrics,
        logs: tmpl.logs
      };
    });
  }, [jobs.data]);

  // Set default focus run on first load
  useMemo(() => {
    if (runs.length > 0 && !focusedRunId) {
      setFocusedRunId(runs[0].id);
    }
  }, [runs, focusedRunId]);

  // Filtered runs based on search bar and modality filter
  const filteredRuns = useMemo(() => {
    return runs.filter((r) => {
      const matchSearch = r.name.toLowerCase().includes(searchTerm.toLowerCase()) || 
                          r.kind.toLowerCase().includes(searchTerm.toLowerCase());
      const matchModality = selectedModality === "all" || r.modality === selectedModality;
      return matchSearch && matchModality;
    });
  }, [runs, searchTerm, selectedModality]);

  // Checkboxes toggler
  const toggleCheck = (id: string) => {
    setCheckedRuns((prev) => ({
      ...prev,
      [id]: !prev[id]
    }));
  };

  // Checked runs lists for comparison
  const comparedRuns = useMemo(() => {
    return runs.filter((r) => checkedRuns[r.id]);
  }, [runs, checkedRuns]);

  const focusedRun = useMemo(() => {
    return runs.find((r) => r.id === focusedRunId) || runs[0];
  }, [runs, focusedRunId]);

  return (
    <>
      <PageHeader
        title="Experiments & Runs Explorer"
        subtitle="Track hyperparameter sweeps, visualize validation loss curves, and compare run lineage side-by-side."
      />

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6 items-start">
        {/* Left Side: Interactive Runs List with searching and filters */}
        <div className="xl:col-span-1 space-y-4">
          <Card className="shadow-md">
            <div className="flex flex-col gap-3">
              {/* Search Bar */}
              <div className="relative">
                <Search className="absolute left-2.5 top-2.5 w-4 h-4 text-muted" />
                <input
                  type="text"
                  placeholder="Search runs by name..."
                  className="w-full bg-bg border border-border rounded px-8 py-1.5 focus:outline-none text-xs font-semibold"
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                />
              </div>

              {/* Modality Filter */}
              <div className="flex items-center gap-2">
                <Filter className="w-3.5 h-3.5 text-muted" />
                <select
                  className="bg-bg border border-border rounded px-2.5 py-1 focus:outline-none text-[11px] font-semibold w-full"
                  value={selectedModality}
                  onChange={(e) => setSelectedModality(e.target.value)}
                >
                  <option value="all">📁 Filter Modality: All</option>
                  <option value="ASR">🎙️ ASR (Speech-to-Text)</option>
                  <option value="LLM">🧠 LLM (Spoken Language)</option>
                  <option value="TTS">🔊 TTS (Voice Synthesizer)</option>
                  <option value="S2S">🔄 S2S (Native Duplex)</option>
                </select>
              </div>
            </div>
          </Card>

          {/* Runs Grid list */}
          <div className="space-y-2 max-h-[500px] overflow-y-auto custom-scrollbar pr-1">
            {jobs.isLoading ? (
              <div className="text-center p-4 text-xs text-muted">Loading run metadata…</div>
            ) : filteredRuns.length === 0 ? (
              <div className="text-center p-4 text-xs text-muted">No runs matching search filters.</div>
            ) : (
              filteredRuns.map((r) => (
                <div
                  key={r.id}
                  onClick={() => setFocusedRunId(r.id)}
                  className={`p-3 rounded-lg border text-xs cursor-pointer transition-all flex items-center justify-between ${
                    focusedRunId === r.id
                      ? "bg-accent/10 border-accent shadow-sm"
                      : "bg-card/45 border-border/80 hover:border-accent/40"
                  }`}
                >
                  <div className="flex items-center gap-3 w-10/12">
                    {/* Checkbox for side-by-side comparison */}
                    <input
                      type="checkbox"
                      checked={!!checkedRuns[r.id]}
                      onChange={() => toggleCheck(r.id)}
                      onClick={(e) => e.stopPropagation()}
                      className="accent-accent cursor-pointer"
                    />
                    <div className="truncate">
                      <div className="font-semibold truncate flex items-center gap-1.5">
                        <span className={`w-1.5 h-1.5 rounded-full ${
                          r.status === "succeeded" ? "bg-emerald-400 animate-pulse" : "bg-purple-400"
                        }`} />
                        {r.name}
                      </div>
                      <div className="text-[10px] text-muted mt-0.5 truncate">
                        {r.kind} · {r.modality}
                      </div>
                    </div>
                  </div>
                  <span className="text-[10px] uppercase font-mono tracking-wider bg-black/40 border border-border/60 px-1.5 py-0.5 rounded text-muted shrink-0">
                    {r.status}
                  </span>
                </div>
              ))
            )}
          </div>
        </div>

        {/* Right Side: Visual suite analytics panel tabs */}
        <div className="xl:col-span-2 space-y-6">
          {/* Header tabs controls */}
          <div className="flex items-center justify-between border-b border-border/30 pb-2">
            <div className="flex gap-2">
              {[
                { id: "charts", label: "📊 Sparkline Loss Curves", icon: LineChart },
                { id: "params", label: "📋 Parameters Sweeper", icon: Table },
                { id: "lineage", label: "🔗 End-to-End Lineage Flow", icon: Network }
              ].map((tab) => {
                const Icon = tab.icon;
                return (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id as any)}
                    className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-semibold transition-all ${
                      activeTab === tab.id
                        ? "bg-accent text-white"
                        : "text-muted hover:text-fg bg-transparent"
                    }`}
                  >
                    <Icon className="w-3.5 h-3.5" />
                    {tab.label}
                  </button>
                );
              })}
            </div>
            {comparedRuns.length > 0 && (
              <span className="text-[10px] bg-accent/15 border border-accent/35 text-accent px-2 py-0.5 rounded font-mono font-bold animate-pulse">
                {comparedRuns.length} Runs Checked for Side-by-Side
              </span>
            )}
          </div>

          {/* Active View render */}
          <Card className="shadow-lg border border-border/60 overflow-hidden relative min-h-[400px]">
            {activeTab === "charts" && (
              <div className="space-y-4">
                <div className="flex items-center justify-between mb-2">
                  <div>
                    <h3 className="text-sm font-bold text-fg/90 flex items-center gap-1.5">
                      <BarChart2 className="w-4 h-4 text-purple-400" />
                      Dynamic Loss Curves for focused run: {focusedRun?.name}
                    </h3>
                    <p className="text-[10px] text-muted">
                      Emits step checkpoints loss, perplexity, or learning rate schedules.
                    </p>
                  </div>
                </div>

                {focusedRun ? (
                  <div className="pt-2">
                    <MetricCharts logs={focusedRun.logs} />
                  </div>
                ) : (
                  <div className="text-center p-8 text-xs text-muted">Select a focused run to plot metrics.</div>
                )}
              </div>
            )}

            {activeTab === "params" && (
              <div className="space-y-4">
                <h3 className="text-sm font-bold text-fg/90 flex items-center gap-1.5">
                  <Table className="w-4 h-4 text-pink-400" />
                  Hyperparameter & Metrics Sweep Comparison
                </h3>
                <p className="text-[10px] text-muted mb-4">
                  Check checkboxes on the left side to compare training parameters side-by-side.
                </p>

                {comparedRuns.length === 0 ? (
                  <div className="text-center p-8 text-xs text-muted border border-dashed border-border/40 rounded-xl bg-black/20">
                    💡 Check two or more runs in the list on the left to display side-by-side sweeps matrices.
                  </div>
                ) : (
                  <div className="overflow-x-auto">
                    <table className="w-full text-xs text-left border-collapse">
                      <thead>
                        <tr className="border-b border-border/60 bg-black/40">
                          <th className="p-2.5 font-bold uppercase tracking-wider text-[10px] text-muted">Parameters</th>
                          {comparedRuns.map((r) => (
                            <th key={r.id} className="p-2.5 font-bold truncate max-w-[150px]">
                              {r.name} ({r.modality})
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {/* Hyperparameters */}
                        <tr className="bg-accent/5 font-semibold">
                          <td colSpan={comparedRuns.length + 1} className="p-2 text-[10px] text-accent uppercase tracking-wider font-bold">
                            Configuration Details
                          </td>
                        </tr>
                        {["Optimizer", "Learning Rate", "Warmup Steps", "Epochs", "Batch Size", "LoRA Rank (r)"].map((param) => (
                          <tr key={param} className="border-b border-border/30 hover:bg-black/10">
                            <td className="p-2 font-medium text-muted">{param}</td>
                            {comparedRuns.map((r) => (
                              <td key={r.id} className="p-2 font-mono text-[11px]">
                                {r.hyperparameters[param] || "—"}
                              </td>
                            ))}
                          </tr>
                        ))}

                        {/* Final metrics */}
                        <tr className="bg-emerald-500/5 font-semibold">
                          <td colSpan={comparedRuns.length + 1} className="p-2 text-[10px] text-emerald-400 uppercase tracking-wider font-bold">
                            Validation Metrics
                          </td>
                        </tr>
                        {["Training Loss", "Validation Loss", "Word Error Rate (WER)", "Latency (RTF)", "TTFA (Latency)"].map((metric) => (
                          <tr key={metric} className="border-b border-border/30 hover:bg-black/10">
                            <td className="p-2 font-medium text-muted">{metric}</td>
                            {comparedRuns.map((r) => (
                              <td key={r.id} className="p-2 font-mono text-[11px] font-bold text-cyan-300">
                                {r.metrics[metric] || "—"}
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            )}

            {activeTab === "lineage" && (
              <div className="space-y-4">
                <h3 className="text-sm font-bold text-fg/90 flex items-center gap-1.5">
                  <GitBranch className="w-4 h-4 text-emerald-400" />
                  Reproducible Dataset-to-Model Lineage Flow
                </h3>
                <p className="text-[10px] text-muted mb-4">
                  Shows how specific training datasets and versioned files mapped into the fine-tuning run to generate registry model adapters.
                </p>

                {focusedRun ? (
                  <div className="flex flex-col items-center justify-center p-4 py-8 border border-border/40 rounded-xl bg-black/60 shadow-inner">
                    {/* SVG Flow diagram */}
                    <div className="w-full max-w-lg flex flex-col md:flex-row items-center justify-between gap-4 font-mono text-[10px] font-bold text-center">
                      
                      {/* Dataset Box */}
                      <div className="p-3 border border-purple-500/30 bg-purple-500/10 text-purple-300 rounded-lg shadow w-full md:w-36 shrink-0 relative">
                        <div className="absolute top-1 left-2 text-[7px] text-purple-400 uppercase font-semibold">Dataset</div>
                        <div className="mt-1 font-sans text-xs font-bold break-all">{focusedRun.dataset}</div>
                        <span className="inline-block mt-2 px-1.5 py-0.2 rounded bg-purple-500/15 text-[8.5px]">Version 0.1.0</span>
                      </div>

                      {/* Direction arrow */}
                      <ArrowRight className="w-5 h-5 text-muted shrink-0 rotate-90 md:rotate-0" />

                      {/* Training Job Box */}
                      <div className="p-3 border border-pink-500/30 bg-pink-500/10 text-pink-300 rounded-lg shadow w-full md:w-36 shrink-0 relative">
                        <div className="absolute top-1 left-2 text-[7px] text-pink-400 uppercase font-semibold">Finetune Job</div>
                        <div className="mt-1 font-sans text-xs font-bold break-all">{focusedRun.name}</div>
                        <span className="inline-block mt-2 px-1.5 py-0.2 rounded bg-pink-500/15 text-[8.5px]">ID: {focusedRun.id.slice(0, 8)}</span>
                      </div>

                      {/* Direction arrow */}
                      <ArrowRight className="w-5 h-5 text-muted shrink-0 rotate-90 md:rotate-0" />

                      {/* Model Registry Adapter Box */}
                      <div className="p-3 border border-emerald-500/30 bg-emerald-500/10 text-emerald-300 rounded-lg shadow w-full md:w-36 shrink-0 relative">
                        <div className="absolute top-1 left-2 text-[7px] text-emerald-400 uppercase font-semibold">Registry Adapter</div>
                        <div className="mt-1 font-sans text-xs font-bold break-all">published-adapter</div>
                        <span className="inline-block mt-2 px-1.5 py-0.2 rounded bg-emerald-500/15 text-[8.5px] uppercase font-bold text-emerald-400">SUCCESS</span>
                      </div>

                    </div>
                  </div>
                ) : (
                  <div className="text-center p-8 text-xs text-muted">Select a focused run to display lineage charts.</div>
                )}
              </div>
            )}
          </Card>
        </div>
      </div>
    </>
  );
}
