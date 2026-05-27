"use client";

import useSWR from "swr";
import Link from "next/link";
import { api } from "@/lib/api";
import { PageHeader } from "@/components/PageHeader";
import { Card } from "@/components/Card";
import { StatusPill } from "@/components/StatusPill";
import { relativeTime } from "@/lib/utils";

export default function JobsPage() {
  const { data, isLoading, error, mutate } = useSWR("jobs", () => api.jobs.list(), {
    refreshInterval: 2000,
  });

  return (
    <>
      <PageHeader
        title="Jobs"
        subtitle="Every training run, evaluation, import, and export goes through the queue."
        actions={
          <button
            onClick={() => mutate()}
            className="px-3 py-1.5 rounded-md border border-border text-sm"
          >
            Refresh
          </button>
        }
      />

      {isLoading ? (
        <p className="text-muted text-sm">Loading…</p>
      ) : error ? (
        <p className="text-red-400 text-sm">Failed to load</p>
      ) : !data?.length ? (
        <Card>
          <p className="text-sm text-muted">No jobs yet. Submit one from the SDK or any studio section.</p>
        </Card>
      ) : (
        <Card className="overflow-hidden p-0">
          <table className="w-full text-sm">
            <thead className="bg-border/30 text-muted text-xs uppercase tracking-wide">
              <tr>
                <th className="text-left px-4 py-2">Name</th>
                <th className="text-left px-4 py-2">Kind</th>
                <th className="text-left px-4 py-2">Status</th>
                <th className="text-left px-4 py-2">Submitted</th>
                <th className="text-left px-4 py-2">Finished</th>
              </tr>
            </thead>
            <tbody>
              {data.map((j) => (
                <tr key={j.id} className="border-t border-border hover:bg-border/20">
                  <td className="px-4 py-2 font-medium">
                    <Link href={`/jobs/${j.id}`} className="hover:text-accent">{j.name}</Link>
                  </td>
                  <td className="px-4 py-2 text-muted">{j.kind}</td>
                  <td className="px-4 py-2">
                    <StatusPill status={j.status} />
                  </td>
                  <td className="px-4 py-2 text-muted">{relativeTime(j.created_at)}</td>
                  <td className="px-4 py-2 text-muted">
                    {j.finished_at ? relativeTime(j.finished_at) : "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      )}
    </>
  );
}
