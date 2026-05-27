"use client";

import { use, useEffect, useRef, useState } from "react";
import useSWR from "swr";
import Link from "next/link";
import { PageHeader } from "@/components/PageHeader";
import { Card, CardTitle } from "@/components/Card";
import { StatusPill } from "@/components/StatusPill";
import { MetricCharts } from "@/components/MetricCharts";
import { relativeTime } from "@/lib/utils";

type Run = {
  id: string;
  status: string;
  attempt: number;
  metrics: Record<string, unknown>;
  started_at: string | null;
  finished_at: string | null;
  error: string | null;
  logs_uri: string | null;
};
type Job = {
  id: string;
  kind: string;
  name: string;
  status: string;
  config: Record<string, unknown>;
  error: string | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  runs: Run[];
};

async function jget<T>(path: string): Promise<T> {
  const r = await fetch(`/api${path}`, { cache: "no-store" });
  if (!r.ok) throw new Error(`${r.status}: ${await r.text()}`);
  return r.json();
}

export default function JobDetailPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const { data, isLoading, mutate } = useSWR<Job>(["job", id], () => jget<Job>(`/jobs/${id}`), {
    refreshInterval: (latest) =>
      latest && ["succeeded", "failed", "canceled"].includes(latest.status) ? 0 : 1500,
  });

  const latestRun = data?.runs[data.runs.length - 1];
  const [logs, setLogs] = useState<string>("");

  async function cancel() {
    await fetch(`/api/jobs/${id}/cancel`, { method: "POST" });
    mutate();
  }

  return (
    <>
      <PageHeader
        title={data?.name ?? "Job"}
        subtitle={data ? `${data.kind} · created ${relativeTime(data.created_at)}` : ""}
        actions={
          <div className="flex items-center gap-2">
            {data && <StatusPill status={data.status} />}
            {data && ["queued", "running"].includes(data.status) && (
              <button onClick={cancel} className="px-3 py-1.5 rounded-md border border-border text-sm">
                Cancel
              </button>
            )}
            <Link href="/jobs" className="px-3 py-1.5 rounded-md border border-border text-sm">
              ← Jobs
            </Link>
          </div>
        }
      />

      {isLoading || !data ? (
        <p className="text-muted text-sm">loading…</p>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="md:col-span-1 space-y-3">
            <Card>
              <CardTitle>Runs</CardTitle>
              {data.runs.length === 0 && <p className="text-sm text-muted">no runs yet</p>}
              <ul className="space-y-1">
                {data.runs.map((r) => (
                  <li key={r.id} className="text-sm flex items-center justify-between">
                    <span className="font-mono text-xs">#{r.attempt}</span>
                    <StatusPill status={r.status} />
                  </li>
                ))}
              </ul>
            </Card>
            <Card>
              <CardTitle>Config</CardTitle>
              <pre className="text-xs font-mono overflow-x-auto bg-bg p-2 rounded">
                {JSON.stringify(data.config, null, 2)}
              </pre>
            </Card>
            {latestRun && Object.keys(latestRun.metrics).length > 0 && (
              <Card>
                <CardTitle>Final metrics</CardTitle>
                <pre className="text-xs font-mono overflow-x-auto bg-bg p-2 rounded">
                  {JSON.stringify(latestRun.metrics, null, 2)}
                </pre>
              </Card>
            )}
            {data.error && (
              <Card>
                <CardTitle>Error</CardTitle>
                <pre className="text-xs font-mono overflow-x-auto bg-bg p-2 rounded text-red-400 whitespace-pre-wrap">
                  {data.error}
                </pre>
              </Card>
            )}
          </div>

          <div className="md:col-span-2 space-y-4">
            <Card>
              <CardTitle>Live metrics</CardTitle>
              <MetricCharts logs={logs} />
            </Card>

            <Card className="p-0 overflow-hidden">
              <div className="px-4 py-2 border-b border-border flex items-center justify-between">
                <span className="text-sm font-medium">Logs</span>
                {latestRun && (
                  <span className="text-xs text-muted">
                    run {latestRun.id.slice(0, 6)} · {latestRun.status}
                  </span>
                )}
              </div>
              {latestRun ? (
                <LogStream
                  runId={latestRun.id}
                  terminal={["succeeded", "failed", "canceled"].includes(latestRun.status)}
                  onUpdate={setLogs}
                />
              ) : (
                <div className="p-4 text-sm text-muted">No run yet — logs appear as soon as the worker picks this up.</div>
              )}
            </Card>
          </div>
        </div>
      )}
    </>
  );
}

function LogStream({
  runId,
  terminal,
  onUpdate,
}: {
  runId: string;
  terminal: boolean;
  onUpdate?: (logs: string) => void;
}) {
  const [lines, setLines] = useState<string>("");
  const ref = useRef<HTMLPreElement>(null);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const apiBase = process.env.NEXT_PUBLIC_API_BASE || `${location.protocol}//${location.host}`;
    const wsBase = apiBase.replace(/^http/, "ws");
    const url = `${wsBase}/ws/runs/${runId}/logs`;
    let ws: WebSocket | null = null;
    try {
      ws = new WebSocket(url);
    } catch {
      return;
    }
    ws.onmessage = (e) => {
      setLines((prev) => {
        const next = prev + (typeof e.data === "string" ? e.data : "");
        onUpdate?.(next);
        return next;
      });
    };
    ws.onerror = () => {};
    return () => {
      ws?.close();
    };
  }, [runId, onUpdate]);

  useEffect(() => {
    if (ref.current) ref.current.scrollTop = ref.current.scrollHeight;
  }, [lines]);

  return (
    <pre
      ref={ref}
      className="bg-black/40 text-emerald-300 text-xs font-mono p-3 h-[40vh] overflow-y-auto whitespace-pre-wrap"
    >
      {lines || (terminal ? "(no logs captured)" : "Connecting…")}
    </pre>
  );
}
