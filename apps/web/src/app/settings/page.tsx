"use client";

import useSWR from "swr";
import { api } from "@/lib/api";
import { PageHeader } from "@/components/PageHeader";
import { Card, CardTitle } from "@/components/Card";

export default function SettingsPage() {
  const info = useSWR("info", () => api.system.info());
  const settings = useSWR("settings", () => api.system.settings());

  return (
    <>
      <PageHeader
        title="Settings"
        subtitle="Studio configuration and detected hardware. Edit via environment variables; .env.example documents every key."
      />

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Card>
          <CardTitle>Runtime</CardTitle>
          {settings.data ? (
            <Kv data={settings.data as Record<string, unknown>} />
          ) : (
            <p className="text-muted text-sm">loading…</p>
          )}
        </Card>
        <Card>
          <CardTitle>System</CardTitle>
          {info.data ? <Kv data={info.data as Record<string, unknown>} /> : null}
        </Card>
      </div>
    </>
  );
}

function Kv({ data }: { data: Record<string, unknown> }) {
  return (
    <dl className="text-sm space-y-1">
      {Object.entries(data).map(([k, v]) => (
        <div key={k} className="flex gap-2">
          <dt className="text-muted w-40 shrink-0">{k}</dt>
          <dd className="font-mono text-xs break-all">
            {typeof v === "object" ? JSON.stringify(v) : String(v)}
          </dd>
        </div>
      ))}
    </dl>
  );
}
