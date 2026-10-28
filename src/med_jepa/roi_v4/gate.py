"""Prespecified development gate against persistence and both formal RRR baselines."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Comparator:
    name: str
    candidate_mae: float
    reference_mae: float
    improvement_ci_low: float


def evaluate_v4_gate(
    comparisons: list[Comparator], *, effective_rank: float, feature_dim: int = 1024
) -> dict[str, object]:
    required = {"persistence", "global_rrr", "coarse_spatial_rrr"}
    observed = {item.name for item in comparisons}
    if observed != required:
        raise ValueError(f"gate requires exactly {sorted(required)}")
    results = {
        item.name: {
            "candidate_mae": item.candidate_mae,
            "reference_mae": item.reference_mae,
            "absolute_improvement": item.reference_mae - item.candidate_mae,
            "paired_bootstrap_improvement_95ci_low": item.improvement_ci_low,
            "passed": item.reference_mae > item.candidate_mae and item.improvement_ci_low > 0,
        }
        for item in comparisons
    }
    rank_ratio = effective_rank / feature_dim
    passed = all(bool(value["passed"]) for value in results.values()) and rank_ratio >= 0.08
    return {
        "status": "passed" if passed else "failed",
        "development_gate_passed": passed,
        "comparisons": results,
        "effective_rank": effective_rank,
        "effective_rank_ratio": rank_ratio,
        "test_split_loaded": False,
        "confirmatory_claim_permitted": False,
    }


__all__ = ["Comparator", "evaluate_v4_gate"]
