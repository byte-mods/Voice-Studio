// Client-side auth state. Token is stored in localStorage so it survives
// reloads, and attached to /api requests via the fetch wrapper in lib/api.

const KEY = "oas_token";

export function getToken(): string | null {
  if (typeof window === "undefined") return null;
  return window.localStorage.getItem(KEY);
}

export function setToken(token: string | null): void {
  if (typeof window === "undefined") return;
  if (token) window.localStorage.setItem(KEY, token);
  else window.localStorage.removeItem(KEY);
}

export type AuthUser = {
  id: string;
  email: string;
  name: string | null;
  is_superuser: boolean;
};

export async function login(email: string, password: string): Promise<AuthUser> {
  const r = await fetch("/api/auth/login", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ email, password }),
  });
  if (!r.ok) throw new Error((await r.json()).detail ?? "login failed");
  const body = await r.json();
  setToken(body.access_token);
  return body.user as AuthUser;
}

export async function signup(email: string, password: string, name?: string): Promise<AuthUser> {
  const r = await fetch("/api/auth/signup", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ email, password, name }),
  });
  if (!r.ok) throw new Error((await r.json()).detail ?? "signup failed");
  const body = await r.json();
  setToken(body.access_token);
  return body.user as AuthUser;
}

export async function me(): Promise<AuthUser | null> {
  const token = getToken();
  if (!token) return null;
  const r = await fetch("/api/auth/me", { headers: { authorization: `Bearer ${token}` } });
  if (r.status === 401) {
    setToken(null);
    return null;
  }
  if (!r.ok) return null;
  return (await r.json()) as AuthUser;
}

export function logout(): void {
  setToken(null);
}
