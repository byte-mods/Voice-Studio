"use client";

import { useMemo, useState } from "react";
import useSWR from "swr";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { api } from "@/lib/api";
import { PageHeader } from "@/components/PageHeader";
import { Card, CardTitle } from "@/components/Card";
import { 
  Search, 
  Download, 
  ThumbsUp, 
  Calendar, 
  Cpu, 
  Database, 
  Sparkles, 
  Plus, 
  X, 
  Loader2 
} from "lucide-react";

type ModelVersion = {
  id: string;
  model_id: string;
  version: string;
  stage: string;
  artifact_uri: string;
  format: string;
  size_bytes: number;
  config: Record<string, any>;
  created_at: string;
};

type HFSearchResult = {
  id: string;
  downloads: number;
  likes: number;
  pipeline_tag: string | null;
  tags: string[];
  last_modified: string | null;
};

export default function ModelsPage() {
  const router = useRouter();
  const projects = useSWR("projects", () => api.projects.list());
  const firstProject = projects.data?.[0];
  
  // Local models
  const models = useSWR(firstProject ? ["models", firstProject.id] : null, ([, pid]) =>
    api.models.list(pid as string)
  );

  // Tabs: "local" or "hf"
  const [activeTab, setActiveTab] = useState<"local" | "hf">("local");

  // Search state for HF Hub
  const [searchQuery, setSearchQuery] = useState("");
  const [searchModality, setSearchModality] = useState<string>("all");
  const [searching, setSearching] = useState(false);
  const [searchResults, setSearchResults] = useState<HFSearchResult[]>([]);
  const [searchError, setSearchError] = useState<string | null>(null);

  // Download Dialog State
  const [showDownloadModal, setShowDownloadModal] = useState(false);
  const [selectedHFId, setSelectedHFId] = useState("");
  const [downloadForm, setDownloadForm] = useState({
    modality: "llm",
    version: "1.0.0",
    family: "qwen",
  });
  const [submittingJob, setSubmittingJob] = useState(false);

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    setSearching(true);
    setSearchError(null);
    try {
      const modalityParam = searchModality === "all" ? undefined : searchModality;
      const res = await api.models.hfSearch(searchQuery, modalityParam);
      setSearchResults(res || []);
    } catch (err) {
      setSearchError((err as Error).message || "Failed to search Hugging Face Hub");
    } finally {
      setSearching(false);
    }
  };

  const openDownloadModal = (item: HFSearchResult) => {
    setSelectedHFId(item.id);
    
    // Auto-detect modality from pipeline tag
    let defaultModality = "llm";
    if (item.pipeline_tag === "automatic-speech-recognition") {
      defaultModality = "asr";
    } else if (item.pipeline_tag === "text-to-speech") {
      defaultModality = "tts";
    } else if (item.pipeline_tag === "audio-to-audio") {
      defaultModality = "s2s";
    }

    // Auto-detect family
    let defaultFamily = "custom";
    const lowerId = item.id.toLowerCase();
    if (lowerId.includes("qwen")) {
      defaultFamily = "qwen";
    } else if (lowerId.includes("whisper")) {
      defaultFamily = "whisper";
    } else if (lowerId.includes("llama")) {
      defaultFamily = "llama";
    } else if (lowerId.includes("gemma")) {
      defaultFamily = "gemma";
    } else if (lowerId.includes("xtts") || lowerId.includes("coqui")) {
      defaultFamily = "xtts";
    }

    setDownloadForm({
      modality: defaultModality,
      version: "1.0.0",
      family: defaultFamily,
    });
    setShowDownloadModal(true);
  };

  const handleTriggerDownload = async () => {
    if (!firstProject || !selectedHFId) return;
    setSubmittingJob(true);
    try {
      const job = await api.jobs.submit({
        project_id: firstProject.id,
        kind: "hf_model_download",
        name: `HF Download: ${selectedHFId.split("/").pop()}`,
        config: {
          project_id: firstProject.id,
          hf_id: selectedHFId,
          modality: downloadForm.modality,
          version: downloadForm.version,
          family: downloadForm.family === "custom" ? undefined : downloadForm.family,
        },
      });
      router.push(`/jobs/${job.id}`);
    } catch (err) {
      alert((err as Error).message || "Failed to submit download job");
    } finally {
      setSubmittingJob(false);
      setShowDownloadModal(false);
    }
  };

  return (
    <>
      <PageHeader
        title="Model Hub & Registry"
        subtitle="Browse local versioned model checkpoints or download directly from Hugging Face."
      />

      <div className="flex gap-2 mb-6 border-b border-border/40 pb-px">
        <button
          onClick={() => setActiveTab("local")}
          className={`px-4 py-2 text-sm font-medium border-b-2 transition ${
            activeTab === "local"
              ? "border-accent text-accent"
              : "border-transparent text-muted hover:text-fg"
          }`}
        >
          <div className="flex items-center gap-1.5">
            <Database className="w-4 h-4" />
            <span>Local Registry</span>
          </div>
        </button>
        <button
          onClick={() => setActiveTab("hf")}
          className={`px-4 py-2 text-sm font-medium border-b-2 transition ${
            activeTab === "hf"
              ? "border-accent text-accent"
              : "border-transparent text-muted hover:text-fg"
          }`}
        >
          <div className="flex items-center gap-1.5">
            <Search className="w-4 h-4" />
            <span>Hugging Face Model Hub</span>
          </div>
        </button>
      </div>

      {!firstProject ? (
        <Card>
          <p className="text-sm text-muted">Create a project to start registering models.</p>
        </Card>
      ) : activeTab === "local" ? (
        // --- Tab 1: Local Model Registry ---
        models.isLoading ? (
          <p className="text-muted text-sm flex items-center gap-2">
            <Loader2 className="w-4 h-4 animate-spin text-accent" />
            <span>Loading registry models…</span>
          </p>
        ) : !models.data?.length ? (
          <Card className="text-center py-8">
            <Database className="w-10 h-10 text-muted mx-auto mb-2 opacity-50" />
            <p className="text-sm text-muted">
              No models registered in <span className="font-medium text-fg">{firstProject.name}</span> yet.
            </p>
            <p className="text-xs text-muted/80 mt-1">
              Search the Hugging Face Model Hub tab to download and register base models.
            </p>
          </Card>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {models.data.map((m) => (
              <Link key={m.id} href={`/models/${m.id}`} className="block group">
                <Card className="hover:border-accent border border-border/40 bg-glass backdrop-blur-md transition-all duration-300 group-hover:shadow-[0_0_15px_rgba(var(--accent-rgb),0.1)] h-full flex flex-col justify-between">
                  <div>
                    <div className="flex items-start justify-between">
                      <div>
                        <div className="font-semibold text-lg text-fg group-hover:text-accent transition">
                          {m.name.split("/").pop()}
                        </div>
                        <div className="text-xs text-muted/70 mt-0.5">{m.name}</div>
                      </div>
                      <span className="text-[10px] font-bold uppercase tracking-wider text-accent-light bg-accent/10 px-2 py-0.5 rounded-full border border-accent/20">
                        {m.modality}
                      </span>
                    </div>

                    <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-muted mt-3">
                      <span className="font-mono bg-border/20 px-1.5 py-0.5 rounded text-[11px]">
                        {m.slug}
                      </span>
                      {m.family && (
                        <span className="flex items-center gap-1">
                          <Cpu className="w-3.5 h-3.5" />
                          {m.family}
                        </span>
                      )}
                    </div>

                    {m.description && (
                      <p className="text-sm text-fg/80 mt-3 line-clamp-2 italic font-serif">
                        &ldquo;{m.description}&rdquo;
                      </p>
                    )}
                  </div>
                  
                  <div className="text-[11px] text-muted/60 mt-4 border-t border-border/20 pt-2 flex items-center justify-between">
                    <span>Registered on {new Date(m.created_at).toLocaleDateString()}</span>
                    <span className="text-accent group-hover:underline">View Versions &rarr;</span>
                  </div>
                </Card>
              </Link>
            ))}
          </div>
        )
      ) : (
        // --- Tab 2: Hugging Face Model Hub ---
        <div className="space-y-4">
          <form onSubmit={handleSearch} className="flex flex-wrap gap-2 items-end bg-glass border border-border/40 p-4 rounded-xl backdrop-blur-md">
            <div className="flex-1 min-w-[200px]">
              <label htmlFor="search-input" className="block text-xs font-semibold text-muted mb-1 uppercase tracking-wider">
                Model search query
              </label>
              <div className="relative">
                <input
                  id="search-input"
                  type="text"
                  placeholder="e.g. whisper-small, Qwen2.5-Omni, Llama-3.2..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="input pl-9"
                  disabled={searching}
                />
                <Search className="w-4 h-4 text-muted/60 absolute left-3 top-1/2 -translate-y-1/2" />
              </div>
            </div>

            <div className="w-full sm:w-[180px]">
              <label htmlFor="modality-select" className="block text-xs font-semibold text-muted mb-1 uppercase tracking-wider">
                Modality
              </label>
              <select
                id="modality-select"
                className="input"
                value={searchModality}
                onChange={(e) => setSearchModality(e.target.value)}
                disabled={searching}
              >
                <option value="all">All Modalities</option>
                <option value="asr">Speech Recognition (ASR)</option>
                <option value="tts">Text-to-Speech (TTS)</option>
                <option value="llm">Text Generation (LLM)</option>
                <option value="s2s">Speech-to-Speech (S2S)</option>
              </select>
            </div>

            <button
              type="submit"
              className="btn btn-primary h-[38px] flex items-center justify-center gap-1.5 px-6 min-w-[120px]"
              disabled={searching}
            >
              {searching ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <span>Searching…</span>
                </>
              ) : (
                <>
                  <Search className="w-4 h-4" />
                  <span>Search HF</span>
                </>
              )}
            </button>
          </form>

          {searchError && (
            <div className="p-3 bg-red-950/40 border border-red-500/30 text-red-200 text-sm rounded-lg">
              {searchError}
            </div>
          )}

          {searching ? (
            <div className="text-center py-12">
              <Loader2 className="w-8 h-8 animate-spin text-accent mx-auto mb-2" />
              <p className="text-sm text-muted">Scouring the Hugging Face Hub indices...</p>
            </div>
          ) : searchResults.length > 0 ? (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {searchResults.map((item) => (
                <Card 
                  key={item.id}
                  className="border border-border/40 bg-glass/60 hover:bg-glass/80 backdrop-blur-md p-4 flex flex-col justify-between transition-all duration-200 hover:border-muted/60"
                >
                  <div>
                    <div className="flex items-start justify-between">
                      <div className="pr-4 min-w-0">
                        <div className="font-semibold text-fg truncate text-base hover:text-accent cursor-pointer" title={item.id}>
                          {item.id}
                        </div>
                        <span className="text-[10px] text-muted/70 bg-border/20 px-1.5 py-0.5 rounded font-mono mt-1 inline-block">
                          {item.pipeline_tag || "custom"}
                        </span>
                      </div>
                      
                      <button
                        onClick={() => openDownloadModal(item)}
                        className="btn btn-secondary py-1 px-3 text-xs flex items-center gap-1 rounded bg-accent/10 border-accent/20 hover:bg-accent/20 text-accent font-semibold transition"
                      >
                        <Download className="w-3.5 h-3.5" />
                        <span>Import</span>
                      </button>
                    </div>

                    {item.tags.length > 0 && (
                      <div className="flex flex-wrap gap-1 mt-3">
                        {item.tags.slice(0, 5).map((t) => (
                          <span key={t} className="text-[10px] text-muted/60 bg-border/10 px-1.5 py-0.2 rounded font-sans">
                            {t}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>

                  <div className="flex items-center gap-4 mt-4 text-[11px] text-muted border-t border-border/20 pt-2.5">
                    <span className="flex items-center gap-1">
                      <Download className="w-3.5 h-3.5 opacity-60" />
                      {item.downloads.toLocaleString()}
                    </span>
                    <span className="flex items-center gap-1">
                      <ThumbsUp className="w-3.5 h-3.5 opacity-60" />
                      {item.likes.toLocaleString()}
                    </span>
                    {item.last_modified && (
                      <span className="flex items-center gap-1 ml-auto">
                        <Calendar className="w-3.5 h-3.5 opacity-60" />
                        Updated {new Date(item.last_modified).toLocaleDateString()}
                      </span>
                    )}
                  </div>
                </Card>
              ))}
            </div>
          ) : searchQuery ? (
            <Card className="text-center py-12">
              <Sparkles className="w-8 h-8 text-muted mx-auto mb-2 opacity-50" />
              <p className="text-sm text-muted">No models matched &ldquo;{searchQuery}&rdquo;</p>
              <p className="text-xs text-muted/80 mt-1">Try a different search query or select and query other modalities.</p>
            </Card>
          ) : (
            <Card className="text-center py-12 border border-dashed border-border/40 bg-glass/20">
              <Search className="w-10 h-10 text-muted mx-auto mb-2 opacity-30" />
              <p className="text-sm text-muted">Enter a search query to search open-source models</p>
              <p className="text-xs text-muted/70 mt-1">Examples: whisper-medium, gemma-2, Qwen2.5-0.5B</p>
            </Card>
          )}
        </div>
      )}

      {/* --- Download Options Modal Dialog --- */}
      {showDownloadModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm animate-fadeIn">
          <div className="w-full max-w-md bg-glass border border-border/50 rounded-2xl shadow-2xl p-6 relative backdrop-blur-xl bg-slate-950/80">
            <button
              onClick={() => setShowDownloadModal(false)}
              className="absolute top-4 right-4 text-muted hover:text-fg transition"
            >
              <X className="w-5 h-5" />
            </button>

            <div className="flex items-center gap-2 mb-4">
              <Download className="w-5 h-5 text-accent" />
              <h3 className="font-bold text-lg text-fg">Download & Register Model</h3>
            </div>

            <p className="text-xs text-muted mb-4 truncate" title={selectedHFId}>
              Source HF Repo: <span className="font-mono text-accent font-semibold">{selectedHFId}</span>
            </p>

            <div className="space-y-4">
              <div>
                <label className="block text-xs font-bold text-muted uppercase tracking-wider mb-1">
                  Target Modality
                </label>
                <select
                  className="input"
                  value={downloadForm.modality}
                  onChange={(e) => setDownloadForm({ ...downloadForm, modality: e.target.value })}
                >
                  <option value="asr">ASR (Speech Recognition / Transcription)</option>
                  <option value="tts">TTS (Speech Synthesis / Voice Clone)</option>
                  <option value="llm">LLM (Text Generation / Chat Agent)</option>
                  <option value="s2s">S2S (Multimodal Speech-to-Speech)</option>
                </select>
              </div>

              <div>
                <label className="block text-xs font-bold text-muted uppercase tracking-wider mb-1">
                  Model Family / Architecture
                </label>
                <select
                  className="input"
                  value={downloadForm.family}
                  onChange={(e) => setDownloadForm({ ...downloadForm, family: e.target.value })}
                >
                  <option value="qwen">Qwen (Omni / CausalLM)</option>
                  <option value="whisper">Whisper (ASR Encoder-Decoder)</option>
                  <option value="llama">Llama (Meta Instruct)</option>
                  <option value="gemma">Gemma (Google Open weights)</option>
                  <option value="xtts">XTTS (Coqui Voice cloning)</option>
                  <option value="custom">Other / Custom</option>
                </select>
              </div>

              <div>
                <label className="block text-xs font-bold text-muted uppercase tracking-wider mb-1">
                  Model Version Tag
                </label>
                <input
                  type="text"
                  placeholder="e.g. 1.0.0, 0.1.0"
                  value={downloadForm.version}
                  onChange={(e) => setDownloadForm({ ...downloadForm, version: e.target.value })}
                  className="input font-mono"
                  required
                />
              </div>

              <div className="pt-2 flex justify-end gap-2">
                <button
                  onClick={() => setShowDownloadModal(false)}
                  className="btn btn-secondary px-4 py-2"
                  disabled={submittingJob}
                >
                  Cancel
                </button>
                <button
                  onClick={handleTriggerDownload}
                  className="btn btn-primary px-5 py-2 flex items-center gap-1.5"
                  disabled={submittingJob}
                >
                  {submittingJob ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin" />
                      <span>Queuing...</span>
                    </>
                  ) : (
                    <>
                      <Download className="w-4 h-4" />
                      <span>Download Model</span>
                    </>
                  )}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
