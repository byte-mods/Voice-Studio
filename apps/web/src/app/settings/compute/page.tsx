"use client";

import { useEffect, useState } from "react";
import useSWR from "swr";
import { api } from "@/lib/api";
import { PageHeader } from "@/components/PageHeader";
import { Card, CardTitle } from "@/components/Card";
import {
  Cloud,
  Cpu,
  DollarSign,
  Clock,
  Play,
  CheckCircle2,
  AlertCircle,
  Terminal,
  Activity,
  ChevronRight,
  Shield,
  Loader2,
  RefreshCw
} from "lucide-react";

export default function ComputePage() {
  const { data: projects, isLoading: projectsLoading } = useSWR("projects_list", () => api.projects.list());
  const [selectedProjectId, setSelectedProjectId] = useState<string>("");

  // Auto-select first project
  useEffect(() => {
    if (projects && projects.length > 0 && !selectedProjectId) {
      setSelectedProjectId(projects[0].id);
    }
  }, [projects, selectedProjectId]);

  // Fetch project details (unredacted is stored on server, API returns redacted settings)
  const { data: project, mutate: mutateProject } = useSWR(
    selectedProjectId ? ["project_detail", selectedProjectId] : null,
    () => api.projects.get(selectedProjectId)
  );

  // Fetch telemetry
  const { data: telemetry, mutate: mutateTelemetry } = useSWR(
    selectedProjectId ? ["compute_telemetry", selectedProjectId] : null,
    () => api.compute.telemetry(selectedProjectId)
  );

  // Fetch scoped audit logs
  const { data: auditLogs, mutate: mutateAudit } = useSWR(
    selectedProjectId ? ["scoped_audit", selectedProjectId] : null,
    () => api.audit.list(selectedProjectId)
  );

  // Credentials form state
  const [modalTokenId, setModalTokenId] = useState("");
  const [modalTokenSecret, setModalTokenSecret] = useState("");
  const [runpodApiKey, setRunpodApiKey] = useState("");
  const [slurmHost, setSlurmHost] = useState("");
  const [slurmUser, setSlurmUser] = useState("");
  const [slurmSshKey, setSlurmSshKey] = useState("");

  const [saving, setSaving] = useState(false);
  const [saveStatus, setSaveStatus] = useState<"idle" | "success" | "error">("idle");

  // Sync form states with fetched settings
  useEffect(() => {
    if (project?.settings) {
      const providers = project.settings.cloud_providers || {};
      const modal = providers.modal || {};
      const runpod = providers.runpod || {};
      const slurm = providers.slurm || {};

      setModalTokenId(modal.token_id || "");
      setModalTokenSecret(modal.token_secret || "");
      setRunpodApiKey(runpod.api_key || "");
      setSlurmHost(slurm.host || "");
      setSlurmUser(slurm.username || "");
      setSlurmSshKey(slurm.ssh_key || "");
    } else {
      setModalTokenId("");
      setModalTokenSecret("");
      setRunpodApiKey("");
      setSlurmHost("");
      setSlurmUser("");
      setSlurmSshKey("");
    }
    setSaveStatus("idle");
  }, [project]);

  // Save settings
  async function saveCredentials(e: React.FormEvent) {
    e.preventDefault();
    if (!selectedProjectId) return;
    setSaving(true);
    setSaveStatus("idle");
    try {
      const settingsPayload = {
        ...(project?.settings || {}),
        cloud_providers: {
          modal: {
            token_id: modalTokenId,
            token_secret: modalTokenSecret,
          },
          runpod: {
            api_key: runpodApiKey,
          },
          slurm: {
            host: slurmHost,
            username: slurmUser,
            ssh_key: slurmSshKey,
          },
        },
      };

      await api.projects.update(selectedProjectId, {
        settings: settingsPayload,
      });

      setSaveStatus("success");
      mutateProject();
      mutateTelemetry();
      mutateAudit();
    } catch (e) {
      setSaveStatus("error");
      console.error(e);
    } finally {
      setSaving(false);
    }
  }

  // Capability Testing State
  const [testingProvider, setTestingProvider] = useState<string | null>(null);
  const [testResult, setTestResult] = useState<Record<string, any> | null>(null);

  async function testCapability(provider: string) {
    if (!selectedProjectId) return;
    setTestingProvider(provider);
    setTestResult(null);
    try {
      const res = await api.compute.test(selectedProjectId, provider);
      setTestResult({ provider, ...res });
      mutateAudit(); // Refresh action logs since testing is audited!
    } catch (e) {
      setTestResult({
        provider,
        status: "offline",
        latency_ms: 0,
        gpus: [],
        message: (e as Error).message || "Connection timed out.",
      });
    } finally {
      setTestingProvider(null);
    }
  }

  if (projectsLoading) {
    return (
      <div className="flex h-64 items-center justify-center">
        <Loader2 className="w-8 h-8 animate-spin text-accent" />
        <span className="ml-2 text-muted text-sm">Loading project context…</span>
      </div>
    );
  }

  if (!projects || projects.length === 0) {
    return (
      <>
        <PageHeader title="Compute & Scale" subtitle="Manage remote cloud runners and multi-user settings." />
        <Card className="p-8 text-center border-dashed">
          <Cloud className="w-12 h-12 text-muted mx-auto mb-4" />
          <h3 className="text-lg font-medium mb-1">No Projects Found</h3>
          <p className="text-sm text-muted mb-4 max-w-md mx-auto">
            You must create a project to configure secure cloud compute integrations, test remote GPU liveness, and monitor telemetry boards.
          </p>
        </Card>
      </>
    );
  }

  const currentProj = projects.find((p) => p.id === selectedProjectId) || projects[0];

  return (
    <>
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-6">
        <PageHeader
          title="Compute & Collaboration"
          subtitle="Register remote servers, audit active runs, and monitor cluster pricing dials."
        />
        <div className="shrink-0 flex items-center gap-2 bg-card/60 backdrop-blur border border-border rounded-lg px-3 py-1.5 shadow-sm">
          <label className="text-xs text-muted font-medium uppercase tracking-wide">Project Context:</label>
          <select
            className="bg-transparent text-sm font-semibold text-accent focus:outline-none border-none pr-6 cursor-pointer"
            value={selectedProjectId}
            onChange={(e) => {
              setSelectedProjectId(e.target.value);
              setTestResult(null);
            }}
          >
            {projects.map((p) => (
              <option key={p.id} value={p.id}>
                {p.name}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Grid: 1. Telemetry and cost dials */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
        <Card className="relative overflow-hidden group shadow-lg hover:shadow-accent/5 transition-all">
          <div className="absolute top-0 right-0 p-4 opacity-5 group-hover:opacity-10 transition-opacity">
            <Clock className="w-24 h-24 text-accent" />
          </div>
          <CardTitle className="flex items-center gap-1.5 text-xs text-muted uppercase tracking-wider mb-4">
            <Clock className="w-4 h-4 text-purple-400" />
            GPU Hours Consumed
          </CardTitle>
          <div className="flex items-baseline gap-2">
            <span className="text-4xl font-extrabold tracking-tight text-white glow-text">
              {telemetry?.total_gpu_hours ?? "0.0"}
            </span>
            <span className="text-xs text-muted">hrs</span>
          </div>
          <div className="mt-4">
            <div className="w-full bg-border/40 rounded-full h-1.5">
              <div
                className="bg-gradient-to-r from-purple-500 to-indigo-500 h-1.5 rounded-full"
                style={{
                  width: `${Math.min(
                    100,
                    ((telemetry?.total_gpu_hours || 0) / (telemetry?.billing_dials?.project_quota_limit || 500.0)) * 100
                  )}%`,
                }}
              />
            </div>
            <div className="flex justify-between text-[10px] text-muted mt-1.5">
              <span>Usage Progress</span>
              <span>Quota Limit: {telemetry?.billing_dials?.project_quota_limit ?? 500} hrs</span>
            </div>
          </div>
        </Card>

        <Card className="relative overflow-hidden group shadow-lg hover:shadow-accent/5 transition-all">
          <div className="absolute top-0 right-0 p-4 opacity-5 group-hover:opacity-10 transition-opacity">
            <DollarSign className="w-24 h-24 text-accent" />
          </div>
          <CardTitle className="flex items-center gap-1.5 text-xs text-muted uppercase tracking-wider mb-4">
            <DollarSign className="w-4 h-4 text-green-400" />
            USD Cost Telemetry
          </CardTitle>
          <div className="flex items-baseline gap-1">
            <span className="text-xs text-muted">$</span>
            <span className="text-4xl font-extrabold tracking-tight text-white text-emerald-400">
              {(telemetry?.total_cost_usd ?? 0).toLocaleString(undefined, { minimumFractionDigits: 2 })}
            </span>
          </div>
          <p className="text-[11px] text-muted mt-4 flex items-center gap-1">
            <Activity className="w-3.5 h-3.5 text-emerald-400" />
            Simulated burn rate: ${(telemetry?.billing_dials?.hourly_consumption_rate ?? 1.45).toFixed(2)}/hr
          </p>
        </Card>

        <Card className="relative overflow-hidden group shadow-lg hover:shadow-accent/5 transition-all">
          <div className="absolute top-0 right-0 p-4 opacity-5 group-hover:opacity-10 transition-opacity">
            <Cpu className="w-24 h-24 text-accent" />
          </div>
          <CardTitle className="flex items-center gap-1.5 text-xs text-muted uppercase tracking-wider mb-4">
            <Cpu className="w-4 h-4 text-cyan-400" />
            Active Remote Clusters
          </CardTitle>
          <div className="space-y-2 mt-1">
            {telemetry?.active_nodes && telemetry.active_nodes.length > 0 ? (
              telemetry.active_nodes.map((node, i) => (
                <div key={i} className="flex items-center justify-between border-b border-border/20 pb-1.5 last:border-0 last:pb-0 text-xs">
                  <div>
                    <div className="font-semibold text-fg/90">{node.name}</div>
                    <div className="text-[10px] text-muted">{node.gpus}</div>
                  </div>
                  <div className="text-right">
                    <span className="inline-block px-1.5 py-0.5 rounded bg-emerald-500/10 text-emerald-400 font-mono text-[9px] uppercase font-bold tracking-wider">
                      {node.status}
                    </span>
                    <div className="text-[10px] text-muted mt-0.5">{node.rate}</div>
                  </div>
                </div>
              ))
            ) : (
              <p className="text-xs text-muted">No active cloud nodes mapped.</p>
            )}
          </div>
        </Card>
      </div>

      {/* Grid: 2. Credentials Registration & GPU Liveness Testing */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
        {/* Registration Forms */}
        <div className="lg:col-span-2 space-y-6">
          <Card className="shadow-lg border border-border/60">
            <CardTitle className="flex items-center gap-2 mb-1">
              <Cloud className="w-4 h-4 text-accent" />
              Secure Credentials Registry
            </CardTitle>
            <p className="text-xs text-muted mb-4">
              Sensitive parameters like tokens or SSH keys are fully encrypted and redacted on client read.
            </p>

            <form onSubmit={saveCredentials} className="space-y-4 text-sm">
              {/* Modal config */}
              <div className="border-b border-border/30 pb-4">
                <h3 className="font-semibold text-xs uppercase text-accent tracking-wider mb-2">Modal.com Integration</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  <div>
                    <label className="block text-[11px] text-muted mb-1 font-medium">MODAL_TOKEN_ID</label>
                    <input
                      className="w-full bg-bg border border-border/80 rounded px-2.5 py-1.5 font-mono text-xs focus:ring-1 focus:ring-accent focus:outline-none"
                      placeholder="e.g. ak-XXXXXXXX"
                      value={modalTokenId}
                      onChange={(e) => setModalTokenId(e.target.value)}
                    />
                  </div>
                  <div>
                    <label className="block text-[11px] text-muted mb-1 font-medium">MODAL_TOKEN_SECRET</label>
                    <input
                      className="w-full bg-bg border border-border/80 rounded px-2.5 py-1.5 font-mono text-xs focus:ring-1 focus:ring-accent focus:outline-none"
                      type="password"
                      placeholder={modalTokenSecret ? "********" : "Enter Token Secret"}
                      value={modalTokenSecret}
                      onChange={(e) => setModalTokenSecret(e.target.value)}
                    />
                  </div>
                </div>
              </div>

              {/* RunPod config */}
              <div className="border-b border-border/30 pb-4">
                <h3 className="font-semibold text-xs uppercase text-accent tracking-wider mb-2">RunPod Serverless</h3>
                <div>
                  <label className="block text-[11px] text-muted mb-1 font-medium">RUNPOD_API_KEY</label>
                  <input
                    className="w-full bg-bg border border-border/80 rounded px-2.5 py-1.5 font-mono text-xs focus:ring-1 focus:ring-accent focus:outline-none"
                    type="password"
                    placeholder={runpodApiKey ? "********" : "Enter API Key"}
                    value={runpodApiKey}
                    onChange={(e) => setRunpodApiKey(e.target.value)}
                  />
                </div>
              </div>

              {/* Slurm configuration */}
              <div className="pb-2">
                <h3 className="font-semibold text-xs uppercase text-accent tracking-wider mb-2">Slurm SSH Cluster Node</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mb-3">
                  <div>
                    <label className="block text-[11px] text-muted mb-1 font-medium">CLUSTER_SSH_HOST</label>
                    <input
                      className="w-full bg-bg border border-border/80 rounded px-2.5 py-1.5 font-mono text-xs focus:ring-1 focus:ring-accent focus:outline-none"
                      placeholder="e.g. login.slurm.university.edu"
                      value={slurmHost}
                      onChange={(e) => setSlurmHost(e.target.value)}
                    />
                  </div>
                  <div>
                    <label className="block text-[11px] text-muted mb-1 font-medium">SSH_USERNAME</label>
                    <input
                      className="w-full bg-bg border border-border/80 rounded px-2.5 py-1.5 font-mono text-xs focus:ring-1 focus:ring-accent focus:outline-none"
                      placeholder="e.g. sudeepignition"
                      value={slurmUser}
                      onChange={(e) => setSlurmUser(e.target.value)}
                    />
                  </div>
                </div>
                <div>
                  <label className="block text-[11px] text-muted mb-1 font-medium">SSH_PRIVATE_KEY</label>
                  <textarea
                    className="w-full bg-bg border border-border/80 rounded px-2.5 py-1.5 font-mono text-[10px] h-16 focus:ring-1 focus:ring-accent focus:outline-none"
                    placeholder={slurmSshKey ? "********" : "-----BEGIN OPENSSH PRIVATE KEY-----"}
                    value={slurmSshKey}
                    onChange={(e) => setSlurmSshKey(e.target.value)}
                  />
                </div>
              </div>

              {saveStatus === "success" && (
                <div className="p-2.5 bg-emerald-500/10 border border-emerald-500/30 rounded text-xs text-emerald-400 flex items-center gap-1.5">
                  <CheckCircle2 className="w-4 h-4 shrink-0" />
                  Credentials updated securely. Places matching place keys are redacted.
                </div>
              )}
              {saveStatus === "error" && (
                <div className="p-2.5 bg-red-500/10 border border-red-500/30 rounded text-xs text-red-400 flex items-center gap-1.5">
                  <AlertCircle className="w-4 h-4 shrink-0" />
                  Failed to securely update credentials. Check server logs.
                </div>
              )}

              <button
                type="submit"
                disabled={saving}
                className="w-full flex items-center justify-center gap-2 px-4 py-2 rounded-md bg-accent text-white text-sm font-semibold shadow hover:bg-accent-hover active:scale-[0.98] transition-all disabled:opacity-50 disabled:scale-100"
              >
                {saving ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" /> Saving credentials…
                  </>
                ) : (
                  "Securely Save Providers Configuration"
                )}
              </button>
            </form>
          </Card>
        </div>

        {/* Liveness Test triggers */}
        <div>
          <Card className="shadow-lg h-full border border-border/60">
            <CardTitle className="flex items-center gap-2 mb-1">
              <Activity className="w-4 h-4 text-purple-400" />
              Capability Liveness Tests
            </CardTitle>
            <p className="text-xs text-muted mb-4">
              Test provider authentication and retrieve available hardware configurations on remote clusters.
            </p>

            <div className="space-y-3">
              {[
                { id: "modal", name: "Modal Serverless app", desc: "Remote app handshakes" },
                { id: "runpod", name: "RunPod Cloud GPU", desc: "API Endpoint validity" },
                { id: "slurm", name: "Slurm login node SSH", desc: "sinfo SSH parser test" },
              ].map((prov) => (
                <div key={prov.id} className="p-3 bg-bg/50 border border-border/40 rounded-lg flex items-center justify-between gap-3 text-xs">
                  <div>
                    <div className="font-semibold text-fg/90">{prov.name}</div>
                    <div className="text-[10px] text-muted">{prov.desc}</div>
                  </div>
                  <button
                    onClick={() => testCapability(prov.id)}
                    disabled={testingProvider !== null}
                    className="flex items-center gap-1 px-3 py-1 rounded bg-purple-500/10 border border-purple-500/30 text-purple-300 font-medium hover:bg-purple-500/20 active:scale-[0.97] transition-all disabled:opacity-50"
                  >
                    {testingProvider === prov.id ? (
                      <Loader2 className="w-3.5 h-3.5 animate-spin" />
                    ) : (
                      <>
                        <Play className="w-3 h-3 fill-purple-300" /> Test
                      </>
                    )}
                  </button>
                </div>
              ))}
            </div>

            {/* Test Results Console */}
            <div className="mt-6 border-t border-border/30 pt-4">
              <h4 className="text-[11px] uppercase font-bold tracking-wider text-muted mb-2 flex items-center gap-1">
                <Terminal className="w-3.5 h-3.5 text-cyan-400" /> Test Diagnostics Console
              </h4>
              {testResult ? (
                <div className="bg-bg border border-border rounded p-3 text-[11px] font-mono space-y-2 h-44 overflow-y-auto">
                  <div className="flex justify-between items-center">
                    <span className="text-accent uppercase font-bold">{testResult.provider} test</span>
                    <span
                      className={`inline-block px-1.5 py-0.2 rounded text-[9px] font-bold tracking-wide uppercase ${
                        testResult.status === "online" ? "bg-emerald-500/20 text-emerald-400" : "bg-red-500/20 text-red-400"
                      }`}
                    >
                      {testResult.status}
                    </span>
                  </div>
                  <div className="text-muted">Ping time: {testResult.latency_ms} ms</div>
                  <div className="text-fg/90 mt-1">{testResult.message}</div>
                  {testResult.gpus && testResult.gpus.length > 0 && (
                    <div className="mt-2 space-y-1">
                      <div className="text-[10px] text-accent border-b border-border/30 pb-0.5">Detected GPUs:</div>
                      {testResult.gpus.map((gpu: any, i: number) => (
                        <div key={i} className="flex justify-between text-fg/80">
                          <span>{gpu.count}x {gpu.name} ({gpu.vram})</span>
                          <span className="text-emerald-400 font-semibold">{gpu.status}</span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              ) : (
                <div className="bg-bg/40 border border-border/20 rounded p-4 text-center text-xs text-muted flex flex-col justify-center items-center h-44">
                  <Terminal className="w-6 h-6 text-border/60 mb-1" />
                  Trigger a liveness test above to view diagnostics logs.
                </div>
              )}
            </div>
          </Card>
        </div>
      </div>

      {/* Section 3: Historical Remote runs logs list */}
      <Card className="shadow-lg border border-border/60 mb-6">
        <CardTitle className="flex items-center gap-2 mb-1">
          <Activity className="w-4 h-4 text-accent" />
          Remote Job Execution Ledger
        </CardTitle>
        <p className="text-xs text-muted mb-4">
          Training and benchmark logs run on remote serverless clusters, incorporating stage-level cost calculations.
        </p>

        <div className="overflow-x-auto">
          <table className="w-full text-xs text-left">
            <thead className="bg-border/25 text-muted uppercase text-[9px] tracking-wider">
              <tr>
                <th className="px-4 py-2">Job Run ID</th>
                <th className="px-4 py-2">Name / Phase</th>
                <th className="px-4 py-2">GPU Node Hardware</th>
                <th className="px-4 py-2">Duration</th>
                <th className="px-4 py-2">Calculated Cost</th>
                <th className="px-4 py-2">Finished Time</th>
                <th className="px-4 py-2 text-right">Status</th>
              </tr>
            </thead>
            <tbody>
              {telemetry?.runs_history && telemetry.runs_history.length > 0 ? (
                telemetry.runs_history.map((run) => (
                  <tr key={run.id} className="border-t border-border/30 hover:bg-border/10">
                    <td className="px-4 py-3 font-mono text-[10px] text-muted">{run.id}</td>
                    <td className="px-4 py-3 font-medium text-fg/90">{run.job_name}</td>
                    <td className="px-4 py-3 text-muted">{run.gpu_type}</td>
                    <td className="px-4 py-3 font-mono text-muted">{run.hours} hrs</td>
                    <td className="px-4 py-3 font-bold text-emerald-400 font-mono">${run.cost.toFixed(2)}</td>
                    <td className="px-4 py-3 text-muted">{run.finished_at ? new Date(run.finished_at).toLocaleString() : "—"}</td>
                    <td className="px-4 py-3 text-right">
                      <span
                        className={`inline-block px-1.5 py-0.5 rounded text-[9px] uppercase font-bold tracking-wide ${
                          run.status === "succeeded"
                            ? "bg-emerald-500/10 text-emerald-400"
                            : run.status === "failed"
                            ? "bg-red-500/10 text-red-400"
                            : "bg-amber-500/10 text-amber-400"
                        }`}
                      >
                        {run.status}
                      </span>
                    </td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan={7} className="px-4 py-6 text-center text-muted">
                    No remote cluster jobs run history.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Section 4: Append-Only Scrollable Project Activity Ledger */}
      <Card className="shadow-lg border border-border/60">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-4">
          <div>
            <CardTitle className="flex items-center gap-2 mb-1">
              <Shield className="w-4 h-4 text-indigo-400" />
              Project Collaboration Activity Logs
            </CardTitle>
            <p className="text-xs text-muted">
              Append-only audit logs tracking actor events, REST method routes, and response outcomes for this project.
            </p>
          </div>
          <button
            onClick={() => mutateAudit()}
            className="flex items-center justify-center gap-1.5 px-3 py-1 rounded border border-border text-xs hover:bg-border/30 transition-all font-medium text-fg/80"
          >
            <RefreshCw className="w-3.5 h-3.5" /> Refresh Logs
          </button>
        </div>

        <div className="border border-border/40 rounded-lg overflow-hidden bg-bg/35">
          <div className="max-h-60 overflow-y-auto">
            {auditLogs && auditLogs.length > 0 ? (
              <div className="divide-y divide-border/20 text-xs font-mono">
                {auditLogs.map((logItem) => (
                  <div key={logItem.id} className="p-3 hover:bg-border/10 flex items-center justify-between gap-4">
                    <div className="flex items-center gap-3 min-w-0">
                      <span
                        className={`inline-block w-14 py-0.5 rounded text-center text-[9px] uppercase font-bold tracking-wider ${
                          logItem.method === "POST"
                            ? "bg-purple-500/20 text-purple-400"
                            : logItem.method === "PUT"
                            ? "bg-cyan-500/20 text-cyan-400"
                            : logItem.method === "DELETE"
                            ? "bg-red-500/20 text-red-400"
                            : "bg-border/40 text-muted"
                        }`}
                      >
                        {logItem.method}
                      </span>
                      <div className="min-w-0">
                        <div className="text-[11px] font-semibold text-fg/90 truncate">{logItem.path}</div>
                        <div className="text-[10px] text-muted mt-0.5 flex items-center gap-1.5 truncate">
                          <span className="font-semibold text-accent/80">{logItem.actor_email}</span>
                          <span>·</span>
                          <span>ID: {logItem.actor_user_id || "anonymous"}</span>
                        </div>
                      </div>
                    </div>
                    <div className="text-right shrink-0">
                      <span
                        className={`inline-block px-1.5 py-0.2 rounded text-[10px] font-bold ${
                          logItem.status_code >= 200 && logItem.status_code < 300
                            ? "bg-emerald-500/10 text-emerald-400"
                            : "bg-red-500/10 text-red-400"
                        }`}
                      >
                        {logItem.status_code}
                      </span>
                      <div className="text-[9px] text-muted mt-0.5">{new Date(logItem.created_at).toLocaleString()}</div>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="p-8 text-center text-xs text-muted">
                No activity logs recorded for this project context.
              </div>
            )}
          </div>
        </div>
      </Card>
    </>
  );
}
