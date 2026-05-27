"use client";

import { useState } from "react";
import useSWR from "swr";
import { api } from "@/lib/api";
import { PageHeader } from "@/components/PageHeader";
import { Card, CardTitle } from "@/components/Card";
import {
  Blocks,
  Download,
  FolderPlus,
  Terminal,
  FileCode,
  CheckCircle2,
  AlertCircle,
  Play,
  HelpCircle,
  Loader2,
  RefreshCw,
  ArrowRight,
  Code
} from "lucide-react";

export default function PluginsPage() {
  const { data: plugins, isLoading, error, mutate } = useSWR("plugins_list", () => api.plugins.list());

  // Pip installation state
  const [source, setSource] = useState("");
  const [installing, setInstalling] = useState(false);
  const [installLogs, setInstallLogs] = useState<string>("");
  const [installSuccess, setInstallSuccess] = useState<boolean | null>(null);

  // Scaffolding state
  const [pluginName, setPluginName] = useState("");
  const [jobKind, setJobKind] = useState("");
  const [pluginDesc, setPluginDesc] = useState("");
  const [scaffolding, setScaffolding] = useState(false);
  const [scaffoldResult, setScaffoldResult] = useState<any | null>(null);

  // Active code snippet tab state
  const [activeSnippet, setActiveSnippet] = useState<"handler" | "scaffold">("handler");

  async function handleInstall(e: React.FormEvent) {
    e.preventDefault();
    if (!source) return;
    setInstalling(true);
    setInstallSuccess(null);
    setInstallLogs(">>> Initiating pip installation process...\n");
    try {
      const res = await api.plugins.install(source);
      setInstallSuccess(res.success);
      setInstallLogs(
        (prev) =>
          prev +
          `>>> PIP Command Execution complete (code: ${res.returncode})\n\n` +
          res.logs +
          (res.success ? "\n>>> Success! Newly discovered entrypoint job handlers hot-loaded into active memory." : "\n>>> Error: Check installation logs above.")
      );
      mutate(); // Reload list
    } catch (e) {
      setInstallSuccess(false);
      setInstallLogs((prev) => prev + `\n>>> Exception occurred: ${(e as Error).message}`);
    } finally {
      setInstalling(false);
    }
  }

  async function handleScaffold(e: React.FormEvent) {
    e.preventDefault();
    if (!pluginName || !jobKind) return;
    setScaffolding(true);
    setScaffoldResult(null);
    try {
      const res = await api.plugins.scaffold({
        name: pluginName,
        kind: jobKind,
        description: pluginDesc,
      });
      setScaffoldResult(res);
      setPluginName("");
      setJobKind("");
      setPluginDesc("");
      mutate();
    } catch (e) {
      setScaffoldResult({
        success: false,
        message: (e as Error).message || "Failed to bootstrap custom package layout.",
      });
    } finally {
      setScaffolding(false);
    }
  }

  const boilerplateCode = `# custom_plugin/handlers.py
import logging
from datetime import UTC, datetime
from oas_core.db import Job, session_scope

log = logging.getLogger(__name__)

def my_custom_handler(job_id: str) -> dict:
    \"\"\"Boilerplate Open Audio Studio job handler.

    Receives the database job_id of the active queue run.
    \"\"\"
    log.info("Processing custom job run: %s", job_id)
    
    with session_scope() as s:
        job = s.get(Job, job_id)
        config = job.config or {}
        log.info("Loaded custom configuration: %s", config)

    # ... Your Speech architecture training/eval logic ...
    
    return {
        "status": "success",
        "finished_at": datetime.now(UTC).isoformat(),
        "exported_checkpoint": "oas-custom-v1.0"
    }
`;

  return (
    <>
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-6">
        <PageHeader
          title="Plugins Registry & SDK"
          subtitle="Explore installed entrypoints, trigger dynamic hot-loads, and scaffold custom speech job plugins."
        />
        <button
          onClick={() => mutate()}
          className="shrink-0 flex items-center justify-center gap-1.5 px-3 py-1.5 rounded-lg border border-border bg-card/40 hover:bg-border/30 active:scale-[0.98] transition-all text-xs font-semibold text-fg/80 shadow-sm"
        >
          <RefreshCw className="w-3.5 h-3.5" /> Refresh Plugins
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
        {/* Left Side: Installed Plugins registry */}
        <div className="lg:col-span-2 space-y-6">
          <Card className="shadow-lg border border-border/60">
            <CardTitle className="flex items-center gap-2 mb-1 text-pink-400">
              <Blocks className="w-4 h-4 text-pink-400" />
              Active Plugins Registry
            </CardTitle>
            <p className="text-xs text-muted mb-4">
              Packages containing active Python entrypoints registered in group <code className="text-pink-300 font-mono">oas.handlers</code>.
            </p>

            {isLoading ? (
              <div className="flex h-32 items-center justify-center">
                <Loader2 className="w-6 h-6 animate-spin text-accent" />
                <span className="ml-2 text-muted text-xs">Scanning entrypoints…</span>
              </div>
            ) : error ? (
              <div className="p-4 bg-red-500/10 border border-red-500/30 rounded-lg text-xs text-red-400 flex items-center gap-2">
                <AlertCircle className="w-5 h-5 shrink-0" />
                Failed to discover active server plugins. Verify FastAPI server logs.
              </div>
            ) : !plugins || plugins.length === 0 ? (
              <div className="p-8 text-center text-xs text-muted border border-dashed rounded-lg">
                No custom plugins installed. Scaffold a package using the developer tool!
              </div>
            ) : (
              <div className="space-y-4">
                {plugins.map((plugin) => (
                  <div
                    key={plugin.name}
                    className="p-4 bg-bg/40 border border-border/50 rounded-xl relative group hover:border-pink-500/35 transition-all shadow-sm"
                  >
                    <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-border/30 pb-2 mb-3">
                      <div>
                        <span className="font-extrabold text-sm text-fg/90">{plugin.name}</span>
                        <span className="ml-2 inline-block px-1.5 py-0.2 rounded bg-pink-500/10 text-pink-400 text-[10px] font-mono font-bold">
                          v{plugin.version}
                        </span>
                      </div>
                      <div className="text-[10px] font-mono text-muted max-w-xs truncate" title={plugin.code_path}>
                        {plugin.code_path}
                      </div>
                    </div>
                    {plugin.description && <p className="text-xs text-fg/75 mb-3">{plugin.description}</p>}
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="text-[10px] text-muted font-semibold uppercase tracking-wider">Registers job kinds:</span>
                      {plugin.handlers.map((h: string) => (
                        <span
                          key={h}
                          className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full bg-accent/10 border border-accent/30 text-accent text-[11px] font-mono font-semibold"
                        >
                          <span className="inline-block w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                          {h}
                        </span>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </Card>

          {/* Pip Installer Form */}
          <Card className="shadow-lg border border-border/60">
            <CardTitle className="flex items-center gap-2 mb-1 text-pink-400">
              <Download className="w-4 h-4 text-pink-400" />
              PIP Dynamic Installer
            </CardTitle>
            <p className="text-xs text-muted mb-4">
              Provide a local folder, PyPI name, or Git source to install. Installs under <code className="text-pink-300 font-mono">.venv</code> and registers entrypoints instantly.
            </p>

            <form onSubmit={handleInstall} className="space-y-4">
              <div className="flex gap-2">
                <input
                  className="flex-1 bg-bg border border-border/80 rounded-lg px-3 py-1.5 text-xs font-mono focus:ring-1 focus:ring-pink-500 focus:outline-none"
                  placeholder="e.g. /home/sudeepignition/projects/speech_to_speech/plugins/example"
                  value={source}
                  onChange={(e) => setSource(e.target.value)}
                  required
                />
                <button
                  type="submit"
                  disabled={installing}
                  className="px-4 py-1.5 rounded-lg bg-pink-500 hover:bg-pink-600 active:scale-[0.98] transition-all text-white text-xs font-bold shadow flex items-center gap-1.5 disabled:opacity-50"
                >
                  {installing ? (
                    <>
                      <Loader2 className="w-3.5 h-3.5 animate-spin" /> Installing…
                    </>
                  ) : (
                    <>
                      <Play className="w-3 h-3 fill-white" /> Install & Hot-Reload
                    </>
                  )}
                </button>
              </div>
            </form>

            {/* CRT scrolling logs terminal */}
            {installLogs && (
              <div className="mt-4 border border-pink-500/30 rounded-lg overflow-hidden bg-black shadow-inner shadow-pink-500/5">
                <div className="bg-zinc-900 px-3 py-1.5 border-b border-pink-500/20 flex items-center justify-between text-[10px] text-zinc-400 font-mono uppercase tracking-wider">
                  <div className="flex items-center gap-1.5">
                    <Terminal className="w-3.5 h-3.5 text-pink-400" /> Installer process terminal output
                  </div>
                  {installSuccess !== null && (
                    <span
                      className={`font-bold ${installSuccess ? "text-emerald-400 animate-pulse" : "text-red-400"}`}
                    >
                      {installSuccess ? "success" : "failed"}
                    </span>
                  )}
                </div>
                <pre className="p-3 text-[10px] font-mono text-emerald-400 max-h-56 overflow-y-auto whitespace-pre-wrap leading-relaxed select-text custom-scrollbar">
                  {installLogs}
                </pre>
              </div>
            )}
          </Card>
        </div>

        {/* Right Side: SDK Scaffolder & Help template Code snippets */}
        <div className="space-y-6">
          <Card className="shadow-lg border border-border/60">
            <CardTitle className="flex items-center gap-2 mb-1 text-purple-400">
              <FolderPlus className="w-4 h-4 text-purple-400" />
              SDK Template Scaffolder
            </CardTitle>
            <p className="text-xs text-muted mb-4">
              Bootstrap a custom Open Audio Studio plugin layout directly under workspace <code className="text-purple-300 font-mono">plugins/</code> folder.
            </p>

            <form onSubmit={handleScaffold} className="space-y-3 text-xs">
              <div>
                <label className="block text-[10px] text-muted mb-1 font-medium">PLUGIN PACKAGE NAME</label>
                <input
                  className="w-full bg-bg border border-border/80 rounded px-2.5 py-1.5 focus:ring-1 focus:ring-purple-500 focus:outline-none"
                  placeholder="e.g. DAC Voice Augmentation"
                  value={pluginName}
                  onChange={(e) => setPluginName(e.target.value)}
                  required
                />
              </div>

              <div>
                <label className="block text-[10px] text-muted mb-1 font-medium">JOB KIND IDENTIFIER</label>
                <input
                  className="w-full bg-bg border border-border/80 rounded px-2.5 py-1.5 focus:ring-1 focus:ring-purple-500 focus:outline-none font-mono text-[11px]"
                  placeholder="e.g. dac_augment"
                  value={jobKind}
                  onChange={(e) => setJobKind(e.target.value)}
                  required
                />
              </div>

              <div>
                <label className="block text-[10px] text-muted mb-1 font-medium">PLUGIN PACKAGE SUMMARY</label>
                <textarea
                  className="w-full bg-bg border border-border/80 rounded px-2.5 py-1.5 focus:ring-1 focus:ring-purple-500 focus:outline-none"
                  rows={2}
                  placeholder="Summarize what this custom speech task accomplishes..."
                  value={pluginDesc}
                  onChange={(e) => setPluginDesc(e.target.value)}
                />
              </div>

              {scaffoldResult && (
                <div
                  className={`p-3 rounded text-[11px] border ${
                    scaffoldResult.success
                      ? "bg-emerald-500/10 border-emerald-500/30 text-emerald-400"
                      : "bg-red-500/10 border-red-500/30 text-red-400"
                  }`}
                >
                  {scaffoldResult.success ? (
                    <>
                      <div className="font-bold mb-1 flex items-center gap-1">
                        <CheckCircle2 className="w-3.5 h-3.5" /> Scaffold Success!
                      </div>
                      <div>{scaffoldResult.message}</div>
                      <div className="mt-1.5 text-[10px] text-muted select-all">
                        pip install -e {scaffoldResult.destination}
                      </div>
                    </>
                  ) : (
                    <>
                      <div className="font-bold mb-1 flex items-center gap-1">
                        <AlertCircle className="w-3.5 h-3.5" /> Scaffold Failed
                      </div>
                      <div>{scaffoldResult.message}</div>
                    </>
                  )}
                </div>
              )}

              <button
                type="submit"
                disabled={scaffolding}
                className="w-full flex items-center justify-center gap-2 px-3 py-2 rounded bg-purple-500 hover:bg-purple-600 active:scale-[0.98] transition-all text-white text-xs font-bold shadow disabled:opacity-50"
              >
                {scaffolding ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <>
                    <FolderPlus className="w-4 h-4" /> Scaffold Custom Plugin layout
                  </>
                )}
              </button>
            </form>
          </Card>

          {/* Boilerplate code snippets */}
          <Card className="shadow-lg border border-border/60">
            <div className="flex items-center justify-between border-b border-border/20 pb-2 mb-3">
              <CardTitle className="flex items-center gap-2 text-purple-400 mb-0">
                <Code className="w-4 h-4 text-purple-400" />
                Developer Reference SDK
              </CardTitle>
            </div>

            <div className="text-xs space-y-3">
              <p className="text-xs text-muted leading-relaxed">
                Writing a plugin is extremely simple. An Open Audio Studio plugin is a Python package declaring a callable entrypoint that runs your custom neural code.
              </p>

              <div>
                <div className="bg-bg border border-border rounded p-2.5 font-mono text-[9px] text-purple-300 max-h-60 overflow-y-auto whitespace-pre leading-relaxed custom-scrollbar">
                  {boilerplateCode}
                </div>
              </div>

              <div className="p-3 bg-bg/50 border border-border/40 rounded-lg flex items-start gap-2.5">
                <HelpCircle className="w-4 h-4 text-accent shrink-0 mt-0.5" />
                <div className="text-[11px] text-muted">
                  Need deep integration examples? Review the reference built-in layouts inside <code className="text-accent">apps/server/oas_server/jobs/</code>!
                </div>
              </div>
            </div>
          </Card>
        </div>
      </div>

      <style jsx global>{`
        .custom-scrollbar::-webkit-scrollbar {
          width: 4px;
          height: 4px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
          background: rgba(0, 0, 0, 0.2);
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: rgba(236, 72, 153, 0.3);
          border-radius: 2px;
        }
      `}</style>
    </>
  );
}
