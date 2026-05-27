"use client";

import { useState, Suspense } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { PageHeader } from "@/components/PageHeader";
import { Card } from "@/components/Card";
import { login, signup } from "@/lib/auth";

function LoginForm() {
  const router = useRouter();
  const params = useSearchParams();
  const next = params.get("next") || "/";
  const [mode, setMode] = useState<"login" | "signup">("login");
  const [form, setForm] = useState({ email: "", password: "", name: "" });
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setErr(null);
    try {
      if (mode === "login") await login(form.email, form.password);
      else await signup(form.email, form.password, form.name);
      router.push(next);
      router.refresh();
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      <PageHeader title="Open Audio Studio" subtitle={mode === "login" ? "Sign in" : "Create an account"} />
      <div className="max-w-md mx-auto">
        <Card>
          <div className="flex gap-2 mb-4">
            <button
              type="button"
              onClick={() => setMode("login")}
              className={`flex-1 px-3 py-1.5 rounded-md text-sm ${mode === "login" ? "bg-accent text-white" : "border border-border"}`}
            >
              Sign in
            </button>
            <button
              type="button"
              onClick={() => setMode("signup")}
              className={`flex-1 px-3 py-1.5 rounded-md text-sm ${mode === "signup" ? "bg-accent text-white" : "border border-border"}`}
            >
              Sign up
            </button>
          </div>

          <form onSubmit={submit} className="space-y-3">
            {mode === "signup" && (
              <Field label="Name (optional)">
                <input className="input" value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} />
              </Field>
            )}
            <Field label="Email">
              <input className="input" type="email" required value={form.email} onChange={(e) => setForm({ ...form, email: e.target.value })} />
            </Field>
            <Field label="Password">
              <input className="input" type="password" required minLength={8} value={form.password} onChange={(e) => setForm({ ...form, password: e.target.value })} />
            </Field>
            {err && <p className="text-red-400 text-sm">{err}</p>}
            <button type="submit" disabled={busy} className="w-full px-3 py-2 rounded-md bg-accent text-white text-sm font-medium disabled:opacity-50">
              {busy ? "…" : mode === "login" ? "Sign in" : "Create account"}
            </button>
            {mode === "signup" && (
              <p className="text-[11px] text-muted text-center">
                The first account created becomes the superuser.
              </p>
            )}
          </form>
        </Card>
      </div>

      <style jsx global>{`
        .input { width: 100%; background: rgb(var(--bg)); border: 1px solid rgb(var(--border)); border-radius: 6px; padding: 8px 10px; font-size: 14px; }
      `}</style>
    </>
  );
}

export default function LoginPage() {
  return (
    <Suspense fallback={<p className="text-muted text-sm text-center">Loading oas auth...</p>}>
      <LoginForm />
    </Suspense>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <label className="block text-xs text-muted mb-1">{label}</label>
      {children}
    </div>
  );
}
