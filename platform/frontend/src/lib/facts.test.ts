import { describe, expect, it } from "vitest";
import { chapters, fallbackOptimizationStates, replayFallback } from "./facts";

describe("platform foundation facts", () => {
  it("defines the complete nine-page journey", () => {
    expect(chapters.map((chapter) => chapter.path)).toEqual([
      "/",
      "/slo-metrics",
      "/data",
      "/preparation",
      "/main-inference",
      "/optimization",
      "/optimized-inference",
      "/comparison",
      "/conclusions"
    ]);
  });

  it("places pre-run SLO education before measured data pages", () => {
    const slo = chapters.find((chapter) => chapter.id === "slo-metrics");
    expect(slo?.resultType).toBe("planned");
    expect(slo?.sourceArtifacts).toContain("configs/slo_targets.yaml");
  });

  it("labels optimized and comparison routes as planned", () => {
    const optimized = chapters.find((chapter) => chapter.id === "optimized-inference");
    const comparison = chapters.find((chapter) => chapter.id === "comparison");
    expect(optimized?.resultType).toBe("planned");
    expect(comparison?.resultType).toBe("planned");
  });

  it("keeps blocked core strategies visible for education", () => {
    const quantization = fallbackOptimizationStates.find(
      (item) => item.optimization_id === "use_quantized_model"
    );
    expect(quantization?.state).toBe("blocked_by_negative_rule");
    expect(quantization?.negative_rule).toBe("quantization");
  });

  it("fallback replay ends at exact measured totals", () => {
    expect(replayFallback.at(-1)?.completed_requests).toBe(250000);
    expect(replayFallback.at(-1)?.failure_count).toBe(0);
  });
});
