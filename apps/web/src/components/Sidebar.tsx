"use client";

import Link from "next/link";
import { useRouter, usePathname } from "next/navigation";
import {
  Mic,
  Brain,
  Speaker,
  AudioWaveform,
  Database,
  FlaskConical,
  ListChecks,
  Boxes,
  Settings,
  Activity,
  LogOut,
  LogIn,
  Cpu,
  Blocks,
  BookOpen,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { useCurrentUser } from "@/lib/useCurrentUser";
import { logout } from "@/lib/auth";

const sections = [
  { href: "/asr", label: "ASR", icon: Mic },
  { href: "/llm", label: "LLM", icon: Brain },
  { href: "/tts", label: "TTS", icon: Speaker },
  { href: "/s2s", label: "Speech-to-Speech", icon: AudioWaveform },
  { href: "/datasets", label: "Datasets", icon: Database },
  { href: "/lab", label: "Architecture Lab", icon: FlaskConical },
];

const shared = [
  { href: "/jobs", label: "Jobs", icon: ListChecks },
  { href: "/models", label: "Model Registry", icon: Boxes },
  { href: "/experiments", label: "Experiments", icon: Activity },
  { href: "/settings/compute", label: "Compute Scale", icon: Cpu },
  { href: "/settings/plugins", label: "Plugins SDK", icon: Blocks },
  { href: "/settings/templates", label: "Templates & Docs", icon: BookOpen },
  { href: "/settings", label: "Settings", icon: Settings },
];

export function Sidebar() {
  const path = usePathname();
  const router = useRouter();
  const { user, reload } = useCurrentUser();

  return (
    <aside className="w-60 shrink-0 border-r border-border bg-card flex flex-col h-screen sticky top-0">
      <div className="p-4 border-b border-border">
        <Link href="/" className="flex items-center gap-2 font-semibold">
          <span className="inline-block w-2 h-2 rounded-full bg-accent" />
          <span>Open Audio Studio</span>
        </Link>
        <p className="text-xs text-muted mt-1">v0.1.0 · Phase 0</p>
      </div>

      <nav className="flex-1 overflow-y-auto p-2 text-sm">
        <Group title="Modalities" items={sections} active={path} />
        <Group title="Workspace" items={shared} active={path} />
      </nav>

      <div className="p-3 text-xs border-t border-border">
        {user && !user.email.startsWith("anonymous@") ? (
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <div className="w-6 h-6 rounded-full bg-accent/20 grid place-items-center text-[10px] font-semibold uppercase">
                {(user.name?.[0] ?? user.email[0])}
              </div>
              <div className="min-w-0">
                <div className="truncate text-fg" title={user.email}>
                  {user.name ?? user.email.split("@")[0]}
                </div>
                <div className="text-[10px] text-muted truncate">{user.email}</div>
              </div>
            </div>
            <button
              onClick={() => {
                logout();
                reload();
                router.push("/login");
              }}
              className="w-full flex items-center justify-center gap-1 px-2 py-1 rounded border border-border hover:bg-border/40"
            >
              <LogOut className="w-3 h-3" /> Sign out
            </button>
          </div>
        ) : (
          <Link
            href="/login"
            className="flex items-center justify-center gap-1 px-2 py-1.5 rounded border border-border hover:bg-border/40"
          >
            <LogIn className="w-3 h-3" /> Sign in / sign up
          </Link>
        )}
      </div>
    </aside>
  );
}

function Group({
  title,
  items,
  active,
}: {
  title: string;
  items: { href: string; label: string; icon: React.ComponentType<{ className?: string }> }[];
  active: string;
}) {
  return (
    <div className="mb-4">
      <div className="px-2 pb-1 text-[11px] uppercase tracking-wide text-muted">{title}</div>
      <ul className="space-y-0.5">
        {items.map(({ href, label, icon: Icon }) => {
          const isActive = active === href || active.startsWith(href + "/");
          return (
            <li key={href}>
              <Link
                href={href}
                className={cn(
                  "flex items-center gap-2 rounded-md px-2 py-1.5",
                  isActive
                    ? "bg-accent/10 text-accent"
                    : "text-fg/80 hover:text-fg hover:bg-border/40"
                )}
              >
                <Icon className="w-4 h-4" />
                <span>{label}</span>
              </Link>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
