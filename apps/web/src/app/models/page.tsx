"use client";

import useSWR from "swr";
import Link from "next/link";
import { api } from "@/lib/api";
import { PageHeader } from "@/components/PageHeader";
import { Card } from "@/components/Card";

export default function ModelsPage() {
  const projects = useSWR("projects", () => api.projects.list());
  const firstProject = projects.data?.[0];
  const models = useSWR(firstProject ? ["models", firstProject.id] : null, ([, pid]) =>
    api.models.list(pid as string)
  );

  return (
    <>
      <PageHeader
        title="Model Registry"
        subtitle="Versioned, signed model artifacts with full lineage to runs and dataset versions."
      />

      {!firstProject ? (
        <Card>
          <p className="text-sm text-muted">Create a project to start registering models.</p>
        </Card>
      ) : models.isLoading ? (
        <p className="text-muted text-sm">Loading…</p>
      ) : !models.data?.length ? (
        <Card>
          <p className="text-sm text-muted">
            No models registered in <span className="font-medium">{firstProject.name}</span> yet.
          </p>
        </Card>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {models.data.map((m) => (
            <Link key={m.id} href={`/models/${m.id}`} className="block">
              <Card className="hover:border-accent transition">
                <div className="flex items-center justify-between">
                  <div className="font-medium">{m.name}</div>
                  <span className="text-[11px] uppercase tracking-wide text-accent">{m.modality}</span>
                </div>
                <div className="text-xs text-muted mt-1">{m.slug}{m.family ? ` · ${m.family}` : ""}</div>
                {m.description && <p className="text-sm text-fg/80 mt-2">{m.description}</p>}
              </Card>
            </Link>
          ))}
        </div>
      )}
    </>
  );
}
