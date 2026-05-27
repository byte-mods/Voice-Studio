"use client";

import { useState } from "react";
import useSWR from "swr";
import { api } from "@/lib/api";
import { PageHeader } from "@/components/PageHeader";
import { Card } from "@/components/Card";
import { relativeTime } from "@/lib/utils";

export default function ProjectsPage() {
  const { data, isLoading, error, mutate } = useSWR("projects", () => api.projects.list());
  const [form, setForm] = useState({ slug: "", name: "", description: "" });
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setErr(null);
    try {
      await api.projects.create(form);
      setForm({ slug: "", name: "", description: "" });
      mutate();
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      <PageHeader
        title="Projects"
        subtitle="A project groups datasets, models, jobs, and experiments."
      />

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="md:col-span-2 space-y-3">
          {isLoading ? (
            <p className="text-muted text-sm">Loading…</p>
          ) : error ? (
            <p className="text-red-400 text-sm">Failed to load</p>
          ) : !data?.length ? (
            <Card>
              <p className="text-sm text-muted">No projects yet. Create one on the right →</p>
            </Card>
          ) : (
            data.map((p) => (
              <Card key={p.id}>
                <div className="flex items-center justify-between">
                  <div>
                    <div className="font-medium">{p.name}</div>
                    <div className="text-xs text-muted">
                      {p.slug} · created {relativeTime(p.created_at)}
                    </div>
                  </div>
                </div>
                {p.description && <p className="text-sm text-fg/80 mt-2">{p.description}</p>}
              </Card>
            ))
          )}
        </div>

        <form onSubmit={submit} className="space-y-3">
          <Card>
            <h2 className="text-sm font-medium mb-3">New project</h2>
            <label className="block text-xs text-muted mb-1">Name</label>
            <input
              className="input mb-2"
              value={form.name}
              onChange={(e) => setForm({ ...form, name: e.target.value })}
              required
            />
            <label className="block text-xs text-muted mb-1">Slug</label>
            <input
              className="input mb-2"
              pattern="[a-z0-9][a-z0-9-_]*"
              value={form.slug}
              onChange={(e) => setForm({ ...form, slug: e.target.value })}
              required
            />
            <label className="block text-xs text-muted mb-1">Description</label>
            <textarea
              className="input mb-2"
              rows={3}
              value={form.description}
              onChange={(e) => setForm({ ...form, description: e.target.value })}
            />
            {err && <p className="text-red-400 text-sm mb-2">{err}</p>}
            <button
              type="submit"
              disabled={busy}
              className="w-full px-3 py-1.5 rounded-md bg-accent text-white text-sm font-medium disabled:opacity-50"
            >
              {busy ? "Creating…" : "Create"}
            </button>
          </Card>
        </form>
      </div>

      <style jsx global>{`
        .input {
          width: 100%;
          background: rgb(var(--bg));
          border: 1px solid rgb(var(--border));
          border-radius: 6px;
          padding: 6px 10px;
          font-size: 14px;
        }
      `}</style>
    </>
  );
}
