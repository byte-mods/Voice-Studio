"use client";

import { useState } from "react";
import useSWR from "swr";

type Version = { id: string; version: string; num_samples: number };

async function jget<T>(p: string): Promise<T> {
  const r = await fetch(`/api${p}`, { cache: "no-store" });
  if (!r.ok) throw new Error(`${r.status}: ${await r.text()}`);
  return r.json();
}

/**
 * Reusable widget for picking-or-creating a writable manifest version on a
 * dataset. Used by all four dataset builders.
 */
export function VersionPicker({
  datasetId,
  value,
  onChange,
}: {
  datasetId: string;
  value: string | null;
  onChange: (versionId: string | null) => void;
}) {
  const versions = useSWR<Version[]>(["versions", datasetId], () =>
    jget<Version[]>(`/datasets/${datasetId}/versions`),
  );
  const [newVersion, setNewVersion] = useState("0.1.0");
  const [licenseSpdx, setLicenseSpdx] = useState("CC-BY-4.0");
  const [creating, setCreating] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  async function init() {
    setCreating(true);
    setErr(null);
    try {
      const r = await fetch(`/api/datasets/${datasetId}/versions/init`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          version: newVersion,
          license: { spdx: licenseSpdx },
        }),
      });
      if (!r.ok) throw new Error(`${r.status}: ${await r.text()}`);
      const v = await r.json();
      await versions.mutate();
      onChange(v.id);
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setCreating(false);
    }
  }

  return (
    <div className="border border-border rounded-md p-3 mb-3 text-sm">
      <div className="text-xs uppercase tracking-wide text-muted mb-2">Target version</div>
      <div className="grid grid-cols-[1fr_auto] gap-2 items-end">
        <div>
          <label className="block text-xs text-muted mb-1">Existing version</label>
          <select
            className="input"
            value={value ?? ""}
            onChange={(e) => onChange(e.target.value || null)}
          >
            <option value="">— pick or create —</option>
            {versions.data?.map((v) => (
              <option key={v.id} value={v.id}>
                {v.version} ({v.num_samples} samples)
              </option>
            ))}
          </select>
        </div>
      </div>

      <details className="mt-2">
        <summary className="cursor-pointer text-xs text-accent">+ start a new empty version</summary>
        <div className="grid grid-cols-3 gap-2 mt-2 items-end">
          <div>
            <label className="block text-[11px] text-muted mb-1">version (semver)</label>
            <input
              className="input"
              pattern="\d+\.\d+\.\d+"
              value={newVersion}
              onChange={(e) => setNewVersion(e.target.value)}
            />
          </div>
          <div>
            <label className="block text-[11px] text-muted mb-1">license SPDX</label>
            <input
              className="input"
              value={licenseSpdx}
              onChange={(e) => setLicenseSpdx(e.target.value)}
            />
          </div>
          <button
            type="button"
            onClick={init}
            disabled={creating}
            className="px-3 py-1.5 rounded-md bg-accent text-white text-xs font-medium disabled:opacity-50"
          >
            {creating ? "…" : "Create empty"}
          </button>
        </div>
        {err && <p className="text-red-400 text-xs mt-2">{err}</p>}
      </details>

      <style jsx global>{`
        .input { width: 100%; background: rgb(var(--bg)); border: 1px solid rgb(var(--border)); border-radius: 6px; padding: 6px 10px; font-size: 13px; }
      `}</style>
    </div>
  );
}

export async function appendSample(versionId: string, sample: Record<string, unknown>): Promise<void> {
  const r = await fetch(`/api/datasets/versions/${versionId}/samples`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ sample }),
  });
  if (!r.ok) throw new Error(`${r.status}: ${await r.text()}`);
}
