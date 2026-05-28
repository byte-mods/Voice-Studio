// API client. All requests go through the Next.js /api/* rewrite, which
// forwards to the FastAPI backend. This avoids CORS in dev and keeps the
// origin consistent between client and server components.

export type Project = {
  id: string;
  slug: string;
  name: string;
  description: string | null;
  created_at: string;
  updated_at: string;
};

export type Dataset = {
  id: string;
  project_id: string;
  slug: string;
  name: string;
  modality: "asr" | "tts" | "llm" | "s2s" | "custom";
  description: string | null;
  source: string | null;
  tags: string[];
  created_at: string;
};

export type Job = {
  id: string;
  project_id: string;
  kind: string;
  name: string;
  status: "queued" | "running" | "succeeded" | "failed" | "canceled";
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  error: string | null;
};

export type Model = {
  id: string;
  project_id: string;
  slug: string;
  name: string;
  modality: string;
  family: string | null;
  description: string | null;
  created_at: string;
};

export type SystemInfo = {
  version: string;
  python: string;
  platform: string;
  handlers: string[];
  gpus: Array<{ name: string; memory_total: string; driver: string }>;
};

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  // Attach bearer token if logged in (deferred import to avoid SSR window access).
  let auth: Record<string, string> = {};
  if (typeof window !== "undefined") {
    const token = window.localStorage.getItem("oas_token");
    if (token) auth = { authorization: `Bearer ${token}` };
  }
  const r = await fetch(`/api${path}`, {
    ...init,
    headers: { "content-type": "application/json", ...auth, ...(init?.headers || {}) },
    cache: "no-store",
  });
  if (r.status === 401 && typeof window !== "undefined") {
    // Token rejected — clear it and bounce to login.
    window.localStorage.removeItem("oas_token");
    const here = window.location.pathname + window.location.search;
    if (!here.startsWith("/login")) {
      window.location.assign(`/login?next=${encodeURIComponent(here)}`);
    }
  }
  if (!r.ok) {
    const text = await r.text();
    throw new Error(`${r.status} ${r.statusText}: ${text}`);
  }
  if (r.status === 204) return undefined as T;
  return (await r.json()) as T;
}

export const api = {
  system: {
    info: () => request<SystemInfo>("/system/info"),
    settings: () => request<Record<string, unknown>>("/settings"),
  },
  plugins: {
    list: () => request<any[]>("/plugins"),
    install: (source: string) =>
      request<{ success: boolean; returncode: number; logs: string }>("/plugins/install", {
        method: "POST",
        body: JSON.stringify({ source }),
      }),
    scaffold: (body: { name: string; kind: string; description?: string }) =>
      request<{ success: boolean; slug: string; destination: string; message: string }>("/plugins/scaffold", {
        method: "POST",
        body: JSON.stringify(body),
      }),
  },
  projects: {
    list: () => request<Project[]>("/projects"),
    get: (id: string) => request<any>(`/projects/${id}`),
    create: (body: { slug: string; name: string; description?: string }) =>
      request<Project>("/projects", { method: "POST", body: JSON.stringify(body) }),
    update: (id: string, body: { name?: string; description?: string; settings?: Record<string, unknown> }) =>
      request<Project>(`/projects/${id}`, { method: "PUT", body: JSON.stringify(body) }),
  },
  compute: {
    test: (projectId: string, provider: string) =>
      request<{ status: string; latency_ms: number; gpus: any[]; message: string }>(
        `/projects/${projectId}/compute/test`,
        { method: "POST", body: JSON.stringify({ provider }) }
      ),
    telemetry: (projectId: string) =>
      request<{
        total_gpu_hours: number;
        total_cost_usd: number;
        active_nodes: any[];
        billing_dials: any;
        runs_history: any[];
      }>(`/projects/${projectId}/compute/telemetry`),
  },
  audit: {
    list: (projectId?: string) => {
      const q = projectId ? `?project_id=${projectId}` : "";
      return request<any[]>(`/audit${q}`);
    },
  },
  datasets: {
    list: (project_id?: string, modality?: string) => {
      const q = new URLSearchParams();
      if (project_id) q.set("project_id", project_id);
      if (modality) q.set("modality", modality);
      const qs = q.toString();
      return request<Dataset[]>(`/datasets${qs ? "?" + qs : ""}`);
    },
    create: (body: { project_id: string; slug: string; name: string; modality: string }) =>
      request<Dataset>("/datasets", { method: "POST", body: JSON.stringify(body) }),
  },
  jobs: {
    list: (project_id?: string) => {
      const q = project_id ? `?project_id=${project_id}` : "";
      return request<Job[]>(`/jobs${q}`);
    },
    submit: (body: { project_id: string; kind: string; name: string; config?: Record<string, unknown> }) =>
      request<Job>("/jobs", { method: "POST", body: JSON.stringify(body) }),
    kinds: () => request<string[]>("/jobs/handlers"),
  },
  models: {
    list: (project_id: string, modality?: string) => {
      const q = new URLSearchParams({ project_id });
      if (modality) q.set("modality", modality);
      return request<Model[]>(`/models?${q.toString()}`);
    },
    create: (body: { project_id: string; slug: string; name: string; modality: string; family?: string; description?: string }) =>
      request<Model>("/models", { method: "POST", body: JSON.stringify(body) }),
    listVersions: (modelId: string) =>
      request<any[]>(`/models/${modelId}/versions`),
    hfSearch: (query?: string, modality?: string) => {
      const q = new URLSearchParams();
      if (query) q.set("query", query);
      if (modality) q.set("modality", modality);
      return request<any[]>(`/models/hf/search?${q.toString()}`);
    },
  },
};
