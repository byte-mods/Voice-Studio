import { PageHeader } from "@/components/PageHeader";
import { ComingSoon } from "@/components/ComingSoon";

export default function ExperimentsPage() {
  return (
    <>
      <PageHeader
        title="Experiments"
        subtitle="Named groups of runs for comparison, side-by-side metrics, and lineage."
      />
      <ComingSoon
        phase="Phase 0/3"
        description="The data layer is in place; the comparison UI lands with LLM Studio in Phase 3."
        features={[
          "Side-by-side run comparison",
          "Metric diff + sample-level diff",
          "Hyperparameter sweeps + ablation matrix",
          "Blind A/B voting for human eval",
          "Paper-mode report export",
        ]}
      />
    </>
  );
}
