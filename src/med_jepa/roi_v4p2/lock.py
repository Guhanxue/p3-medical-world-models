"""Lock the v4.2 training protocol after train-only feature triage."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import torch

from med_jepa.roi_v4p2.config import V42TrainingConfig
from med_jepa.roi_v4p2.feature_triage import (
    PRIVATE_SCHEMA as TRIAGE_PRIVATE_SCHEMA,
    SCHEMA as TRIAGE_SCHEMA,
)
from med_jepa.roi_v4p2.freeze_design import SCHEMA as DESIGN_SCHEMA


SCHEMA = "med-jepa-roi-v4p2-training-protocol-lock-v1"
SELECTED_VARIANT = "t0_anchored_local_magnification_64_to_112"


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


def validate_training_protocol_inputs(
    *,
    config_path: Path,
    design_path: Path,
    triage_report_path: Path,
    triage_private_index_path: Path,
) -> tuple[V42TrainingConfig, dict[str, Any], dict[str, Any], Mapping[str, Any]]:
    config = V42TrainingConfig.load(config_path)
    design = _json(design_path)
    triage = _json(triage_report_path)
    private = torch.load(
        triage_private_index_path, map_location="cpu", weights_only=True
    )
    eligibility = triage.get("eligibility", {}).get(SELECTED_VARIANT, {})
    selection = triage.get("selection", {})
    design_contract = design.get("training_objective_contract", {})
    if (
        design.get("schema_version") != DESIGN_SCHEMA
        or design.get("status") != "frozen_feature_triage_required_training_prohibited"
        or design.get("training_started") is not False
        or design.get("validation_88_loaded_by_freeze") is not False
        or design.get("test_split_loaded") is not False
        or triage.get("schema_version") != TRIAGE_SCHEMA
        or triage.get("status") != "completed_train_only_feature_triage"
        or triage.get("patients")
        != {"train_loaded": 409, "validation_loaded": 0, "test_loaded": 0}
        or triage.get("training_performed") is not False
        or triage.get("validation_88_loaded") is not False
        or triage.get("test_split_loaded") is not False
        or selection.get("selected_variant") != SELECTED_VARIANT
        or selection.get("decision")
        != "intervention_locked_from_train_only_nested_CV"
        or selection.get("validation_88_used_for_selection") is not False
        or eligibility.get("passed") is not True
        or not all(eligibility.get("checks", {}).values())
        or triage.get("evidence", {}).get("frozen_design_sha256")
        != _sha(design_path)
        or triage.get("evidence", {}).get("private_index_sha256")
        != _sha(triage_private_index_path)
        or not isinstance(private, Mapping)
        or private.get("schema_version") != TRIAGE_PRIVATE_SCHEMA
        or private.get("selected_variant") != SELECTED_VARIANT
        or len(private.get("patient_ids", ())) != 409
        or len(set(private.get("patient_ids", ()))) != 409
        or len(private.get("patient_files", ())) != 409
        or design_contract.get("lambda_candidates_train_only")
        != list(config.occupancy_lambda_candidates)
        or design_contract.get("mask_fraction_schedule")
        != list(config.mask_fractions)
        or design_contract.get("mask_sampling")
        != "uniform_spatial_block_sampling_independent_of_occupancy"
        or design_contract.get("variance_or_rank_matching_pressure") is not False
        or design_contract.get("future_pixel_reconstruction") is not False
    ):
        raise ValueError("v4.2 training protocol inputs are not an approved chain")
    return config, design, triage, private


def run(args: argparse.Namespace) -> dict[str, Any]:
    config_path = Path(args.config).expanduser().resolve()
    design_path = Path(args.frozen_design).expanduser().resolve()
    triage_report_path = Path(args.feature_triage_report).expanduser().resolve()
    triage_private_index_path = Path(args.feature_triage_private_index).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError("immutable v4.2 training protocol lock exists")
    config, design, triage, _ = validate_training_protocol_inputs(
        config_path=config_path,
        design_path=design_path,
        triage_report_path=triage_report_path,
        triage_private_index_path=triage_private_index_path,
    )
    report = {
        "schema_version": SCHEMA,
        "status": "locked_train_only_cv_required_before_validation",
        "selected_feature_variant": SELECTED_VARIANT,
        "feature_interpretation": (
            "native_2p1mm_T0_anchored_64cube_FOV_resampled_to_112cube_"
            "not_true_high_resolution"
        ),
        "config": asdict(config),
        "selection_protocol": {
            "patients": 409,
            "grouped_folds": config.grouped_cv_folds,
            "selectable_hyperparameters": [
                "occupancy_uniform_floor_lambda",
                "final_epoch_from_selected_lambda_fold_median",
                "pair_specific_train_only_shrinkage_alpha",
            ],
            "validation_may_be_loaded_before_selection_release": False,
            "primary_metric": "t0_disease_bed_MAE_relative_to_fold_local_shrinkage",
            "tiebreakers": [
                "disease_minus_background_relative_improvement",
                "lower_full_shared_valid_grid_MAE",
                "larger_occupancy_lambda",
            ],
        },
        "validation_contract": {
            "patients": 88,
            "loaded_once_after_train_only_selection_release": True,
            "checkpoint_selection_or_early_stopping": False,
            "primary_reference": "train_only_shrinkage_to_mean_t3",
            "gate": "med_jepa.roi_v4p2.gate.evaluate_v4p2_gate",
        },
        "future_pixel_reconstruction": False,
        "training_started": False,
        "validation_88_loaded": False,
        "test_split_loaded": False,
        "patient_identifiers_emitted": False,
        "evidence": {
            "config_sha256": _sha(config_path),
            "frozen_design_sha256": _sha(design_path),
            "feature_triage_report_sha256": _sha(triage_report_path),
            "feature_triage_private_index_sha256": _sha(
                triage_private_index_path
            ),
            "lock_source_sha256": _sha(Path(__file__)),
            "design_source_sha256": design.get("evidence", {}).get(
                "design_source_sha256"
            ),
            "triage_source_sha256": triage.get("evidence", {}).get(
                "source_sha256"
            ),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.chmod(output, 0o600)
    print(
        json.dumps(
            {
                "status": report["status"],
                "selected_feature_variant": SELECTED_VARIANT,
                "validation_88_loaded": False,
                "test_split_loaded": False,
            },
            sort_keys=True,
        )
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--frozen-design", required=True)
    parser.add_argument("--feature-triage-report", required=True)
    parser.add_argument("--feature-triage-private-index", required=True)
    parser.add_argument("--output", required=True)
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "SCHEMA",
    "SELECTED_VARIANT",
    "run",
    "validate_training_protocol_inputs",
]
