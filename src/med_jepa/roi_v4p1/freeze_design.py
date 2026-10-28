"""Freeze the v4.1 source-anchor repair without training a model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


SCHEMA = "med-jepa-roi-v4p1-frozen-design-v1"
V4_SCHEMA = "med-jepa-roi-v4-development-training-v1"


def _sha(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    return value


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError("immutable v4.1 design report exists")
    training_path = Path(args.v4_training_report).expanduser().resolve()
    checkpoint_path = Path(args.v4_checkpoint).expanduser().resolve()
    mechanistic_path = Path(args.mechanistic_audit).expanduser().resolve()
    training = _json(training_path)
    mechanistic = _json(mechanistic_path)
    if (
        training.get("schema_version") != V4_SCHEMA
        or training.get("status") != "completed_development_training"
        or training.get("test_split_loaded") is not False
        or training.get("evidence", {}).get("checkpoint_sha256") != _sha(checkpoint_path)
        or mechanistic.get("status") != "completed_posthoc_development_mechanistic_audit"
        or mechanistic.get("test_split_loaded") is not False
    ):
        raise ValueError("v4.0 evidence chain is not approved for v4.1 repair")
    report = {
        "schema_version": SCHEMA,
        "status": "frozen_not_trained",
        "repository_namespace": "med_jepa.roi_v4p1",
        "scientific_target": "future_frozen_3dino_full_token_prediction",
        "v4p0_erratum": {
            "masked_residual_base_was_shared_mask_token": True,
            "masked_zero_velocity_was_not_source_persistence": True,
            "masked_and_semigroup_paths_used_different_residual_bases": True,
            "v4p0_checkpoint_remains_immutable": True,
        },
        "v4p1_contract": {
            "context_mask_affects_velocity_input_only": True,
            "residual_base_is_always_observed_source": True,
            "zero_velocity_is_exact_persistence_for_every_mask": True,
            "terminal_prediction_normalization": False,
            "mask_fraction_schedule": [0.0, 0.5, 0.75, 0.9, 1.0],
            "full_source_deployment_condition_trained": True,
            "fully_masked_context_has_explicit_probability_mass": True,
            "ordered_pairs_per_patient": 3,
            "semigroup_consistency": True,
            "variance_or_rank_matching_pressure": False,
            "ftv_encoder_gradient": False,
            "future_pixel_reconstruction": False,
        },
        "development_gate": {
            "primary_estimand": "patient_mean_t0_disease_bed_weighted_full_7cube_token_mae",
            "must_beat_with_positive_patient_bootstrap_ci": [
                "latent_persistence",
                "full_token_markov_matched_linear_ridge",
                "full_token_history_linear_ridge",
                "full_token_conditional_history_linear_ridge",
            ],
            "paired_bootstrap_replicates": 10_000,
            "fixed_effective_rank_threshold": None,
            "best_seed_selection_permitted": False,
        },
        "required_before_training_release": [
            "v4p0_anchor_calibration_audit",
            "full_token_linear_baseline",
            "frozen_3dino_noise_floor_audit",
        ],
        "training_started": False,
        "test_split_loaded": False,
        "evidence": {
            "v4_training_report_sha256": _sha(training_path),
            "v4_checkpoint_sha256": _sha(checkpoint_path),
            "mechanistic_audit_sha256": _sha(mechanistic_path),
            "design_source_sha256": _sha(Path(__file__)),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.chmod(output, 0o600)
    print(json.dumps({"status": report["status"], "test_split_loaded": False}, sort_keys=True))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v4-training-report", required=True)
    parser.add_argument("--v4-checkpoint", required=True)
    parser.add_argument("--mechanistic-audit", required=True)
    parser.add_argument("--output", required=True)
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
