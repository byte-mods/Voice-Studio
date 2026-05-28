"use client";

import { use, useState } from "react";
import useSWR from "swr";
import Link from "next/link";
import { PageHeader } from "@/components/PageHeader";
import { Card, CardTitle } from "@/components/Card";

type Project = {
  id: string;
  slug: string;
  name: string;
  description: string | null;
  created_at: string;
};

type Member = {
  id: string;
  user_id: string;
  email: string;
  role: "viewer" | "editor" | "admin";
};

async function jget<T>(p: string): Promise<T> {
  const r = await fetch(`/api${p}`, {
    cache: "no-store",
    headers: getAuth(),
  });
  if (!r.ok) throw new Error(`${r.status}: ${await r.text()}`);
  return r.json();
}

function getAuth(): Record<string, string> {
  if (typeof window === "undefined") return {};
  const t = window.localStorage.getItem("oas_token");
  return t ? { authorization: `Bearer ${t}` } : {};
}

export default function ProjectDetail({ params }: { params: any }) {
  const resolvedParams = params && typeof params.then === "function" ? use(params) : params;
  const { id } = resolvedParams;
  const proj = useSWR<Project>(["proj", id], () => jget<Project>(`/projects/${id}`));
  const members = useSWR<Member[]>(["members", id], () =>
    jget<Member[]>(`/auth/projects/${id}/members`),
  );

  return (
    <>
      <PageHeader
        title={proj.data?.name ?? "Project"}
        subtitle={proj.data?.slug}
        actions={<Link href="/projects" className="px-3 py-1.5 rounded-md border border-border text-sm">← Projects</Link>}
      />

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="md:col-span-2 space-y-3">
          <Card>
            <CardTitle>Description</CardTitle>
            <p className="text-sm">{proj.data?.description ?? "—"}</p>
          </Card>
          <Card>
            <CardTitle>Members</CardTitle>
            {!members.data?.length ? (
              <p className="text-sm text-muted">Anonymous mode (or no explicit members yet).</p>
            ) : (
              <table className="w-full text-sm">
                <thead className="text-xs text-muted uppercase">
                  <tr>
                    <th className="text-left py-1">Email</th>
                    <th className="text-left py-1">Role</th>
                    <th className="text-right py-1"></th>
                  </tr>
                </thead>
                <tbody>
                  {members.data.map((m) => (
                    <MemberRow key={m.id} projectId={id} member={m} onChange={() => members.mutate()} />
                  ))}
                </tbody>
              </table>
            )}
          </Card>
        </div>

        <div>
          <AddMember projectId={id} onAdded={() => members.mutate()} />
        </div>
      </div>
    </>
  );
}

function MemberRow({
  projectId,
  member,
  onChange,
}: {
  projectId: string;
  member: Member;
  onChange: () => void;
}) {
  const [role, setRole] = useState(member.role);
  const [busy, setBusy] = useState(false);

  async function update(next: typeof role) {
    setBusy(true);
    try {
      await fetch(`/api/auth/projects/${projectId}/members`, {
        method: "POST",
        headers: { "content-type": "application/json", ...getAuth() },
        body: JSON.stringify({ user_id: member.user_id, role: next }),
      });
      setRole(next);
      onChange();
    } finally {
      setBusy(false);
    }
  }

  async function remove() {
    setBusy(true);
    try {
      await fetch(`/api/auth/projects/${projectId}/members/${member.id}`, {
        method: "DELETE",
        headers: getAuth(),
      });
      onChange();
    } finally {
      setBusy(false);
    }
  }

  return (
    <tr className="border-t border-border">
      <td className="py-1">{member.email}</td>
      <td className="py-1">
        <select
          className="bg-bg border border-border rounded px-2 py-0.5 text-xs"
          value={role}
          onChange={(e) => update(e.target.value as typeof role)}
          disabled={busy}
        >
          <option value="viewer">viewer</option>
          <option value="editor">editor</option>
          <option value="admin">admin</option>
        </select>
      </td>
      <td className="py-1 text-right">
        <button onClick={remove} disabled={busy} className="text-xs text-red-400 hover:underline">
          remove
        </button>
      </td>
    </tr>
  );
}

function AddMember({ projectId, onAdded }: { projectId: string; onAdded: () => void }) {
  const [userId, setUserId] = useState("");
  const [role, setRole] = useState<"viewer" | "editor" | "admin">("editor");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setErr(null);
    try {
      const r = await fetch(`/api/auth/projects/${projectId}/members`, {
        method: "POST",
        headers: { "content-type": "application/json", ...getAuth() },
        body: JSON.stringify({ user_id: userId, role }),
      });
      if (!r.ok) throw new Error(`${r.status}: ${await r.text()}`);
      setUserId("");
      onAdded();
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <Card>
      <CardTitle>Add member</CardTitle>
      <form onSubmit={submit} className="space-y-2">
        <input
          className="w-full bg-bg border border-border rounded px-2 py-1.5 text-sm"
          placeholder="User ID"
          value={userId}
          onChange={(e) => setUserId(e.target.value)}
          required
        />
        <select
          className="w-full bg-bg border border-border rounded px-2 py-1.5 text-sm"
          value={role}
          onChange={(e) => setRole(e.target.value as typeof role)}
        >
          <option value="viewer">viewer</option>
          <option value="editor">editor</option>
          <option value="admin">admin</option>
        </select>
        {err && <p className="text-xs text-red-400">{err}</p>}
        <button
          type="submit"
          disabled={busy}
          className="w-full px-3 py-1.5 rounded-md bg-accent text-white text-sm font-medium disabled:opacity-50"
        >
          {busy ? "…" : "Add"}
        </button>
      </form>
      <p className="text-[11px] text-muted mt-2">
        For now the user id is required directly. A future iteration will offer a search-by-email
        picker.
      </p>
    </Card>
  );
}
