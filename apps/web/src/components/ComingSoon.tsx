import { Card } from "@/components/Card";

export function PhaseBadge({ phase }: { phase: string }) {
  return (
    <span className="inline-flex items-center rounded-full bg-accent/10 text-accent px-2 py-0.5 text-[11px] font-medium">
      {phase}
    </span>
  );
}

export function FeatureList({ items }: { items: string[] }) {
  return (
    <ul className="grid grid-cols-1 md:grid-cols-2 gap-x-6 gap-y-1.5 text-sm">
      {items.map((i) => (
        <li key={i} className="flex items-start gap-2">
          <span className="mt-1.5 inline-block w-1 h-1 rounded-full bg-muted shrink-0" />
          <span>{i}</span>
        </li>
      ))}
    </ul>
  );
}

export function ComingSoon({
  phase,
  description,
  features,
}: {
  phase: string;
  description: string;
  features: string[];
}) {
  return (
    <Card>
      <div className="flex items-center gap-2 mb-3">
        <PhaseBadge phase={phase} />
        <span className="text-xs text-muted">scaffolded · not yet implemented</span>
      </div>
      <p className="text-sm mb-4">{description}</p>
      <div className="text-xs uppercase tracking-wide text-muted mb-2">Planned features</div>
      <FeatureList items={features} />
    </Card>
  );
}
