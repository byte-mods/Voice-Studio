import { cn } from "@/lib/utils";

const colors: Record<string, string> = {
  queued: "bg-zinc-500/15 text-zinc-400",
  running: "bg-blue-500/15 text-blue-400",
  succeeded: "bg-emerald-500/15 text-emerald-400",
  failed: "bg-red-500/15 text-red-400",
  canceled: "bg-amber-500/15 text-amber-400",
  dev: "bg-zinc-500/15 text-zinc-400",
  staging: "bg-blue-500/15 text-blue-400",
  prod: "bg-emerald-500/15 text-emerald-400",
  archived: "bg-zinc-700/30 text-zinc-500",
};

export function StatusPill({ status }: { status: string }) {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full px-2 py-0.5 text-[11px] font-medium",
        colors[status] ?? "bg-zinc-500/15 text-zinc-400"
      )}
    >
      {status}
    </span>
  );
}
