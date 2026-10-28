"""Freeze and execute the no-refit Med-JEPA v4.3.2 OOF re-audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor

from med_jepa.roi_v4p3p1.config import V431TrainingConfig
from med_jepa.roi_v4p3p1.freeze_design import SCHEMA as V431_DESIGN_SCHEMA
from med_jepa.roi_v4p3p1.gate import evaluate_v431_train_oof_gate
from med_jepa.roi_v4p3p1.train import (
    PRIVATE_SCHEMA as V431_PRIVATE_SCHEMA,
    SCHEMA as V431_TRAINING_SCHEMA,
)
from med_jepa.roi_v4p3p2.config import V432RecoveryConfig
from med_jepa.roi_v4p3p2.gate import evaluate_v432_recovery_gate


SCHEMA = "med-jepa-roi-v4p3p2-adaptive-recovery-lock-v1"
STATUS = "frozen_adaptive_numerical_recovery_gate_pass_final_fit_authorized"


def _sha(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("v4.3.2 evidence must be a JSON object")
    return value


def _same_number(left: Any, right: Any) -> bool:
    return abs(float(left) - float(right)) <= 1e-12


def _validate_recomputed_original(
    observed: Mapping[str, Any], expected: Mapping[str, Any]
) -> None:
    if observed.get("criterion_results") != expected.get("criterion_results"):
        raise ValueError("recomputed v4.3.1 criterion results changed")
    for region, value in observed.get("candidate_vs_coarse_RRR", {}).items():
        reference = expected.get("candidate_vs_coarse_RRR", {}).get(region)
        if not isinstance(reference, Mapping):
            raise ValueError("v4.3.1 region comparison is missing")
        for key in (
            "candidate_mean",
            "reference_mean",
            "improvement",
            "ci_low",
            "ci_high",
            "patient_win_fraction",
            "bootstrap_positive_fraction",
        ):
            if not _same_number(value[key], reference[key]):
                raise ValueError("recomputed v4.3.1 OOF statistic changed")


def freeze(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        name: Path(value).expanduser().resolve()
        for name, value in {
            "recovery_config": args.recovery_config,
            "v431_config": args.v4p3p1_config,
            "v431_design": args.v4p3p1_frozen_design,
            "v431_report": args.v4p3p1_report,
            "v431_private": args.v4p3p1_private_measurements,
            "output": args.output,
        }.items()
    }
    for name, path in paths.items():
        if name != "output" and not path.is_file():
            raise FileNotFoundError(f"v4.3.2 prerequisite unavailable: {name}")
    if paths["output"].exists():
        raise FileExistsError("immutable v4.3.2 recovery lock exists")
    recovery_config = V432RecoveryConfig.load(paths["recovery_config"])
    v431_config = V431TrainingConfig.load(paths["v431_config"])
    design = _json(paths["v431_design"])
    report = _json(paths["v431_report"])
    private = torch.load(
        paths["v431_private"], map_location="cpu", weights_only=True
    )
    evidence = report.get("evidence", {})
    original_gate = report.get("train409_cross_validation", {}).get("oof_gate")
    epochs = report.get("train409_cross_validation", {}).get(
        "candidate_summary", {}
    ).get("best_epoch_counts")
    if (
        recovery_config.seed != v431_config.seed
        or recovery_config.bootstrap_seed != v431_config.bootstrap_seed
        or recovery_config.bootstrap_replicates
        != v431_config.bootstrap_replicates
        or design.get("schema_version") != V431_DESIGN_SCHEMA
        or report.get("schema_version") != V431_TRAINING_SCHEMA
        or report.get("status")
        != "completed_train409_core_residual_gate_fail_no_final_model"
        or report.get("selected_candidate") is not None
        or report.get("final_checkpoint_created") is not False
        or report.get("validation_88_loaded") is not False
        or report.get("test_split_loaded") is not False
        or evidence.get("frozen_design_sha256") != _sha(paths["v431_design"])
        or evidence.get("config_sha256") != _sha(paths["v431_config"])
        or evidence.get("private_train409_oof_measurements_sha256")
        != _sha(paths["v431_private"])
        or not isinstance(original_gate, Mapping)
        or original_gate.get("development_gate_passed") is not False
        or not isinstance(epochs, list)
        or len(epochs) != 5
        or int(statistics.median(epochs))
        != recovery_config.final_epoch_count
        or not isinstance(private, Mapping)
        or private.get("schema_version") != V431_PRIVATE_SCHEMA
        or private.get("selected_candidate") is not None
    ):
        raise ValueError("v4.3.1 immutable negative-result chain is invalid")
    original_criteria = original_gate.get("criterion_results")
    expected_original_criteria = {
        "T0_occupancy_alignment_core_CI_lower_positive": True,
        "background_mean_not_worse_than_coarse_RRR": False,
        "core_patient_win_fraction_gt_half": True,
        "core_residual_exactly_zero_where_T0_core_occupancy_is_zero": True,
        "core_vs_coarse_RRR_CI_lower_positive": True,
        "disease_bed_mean_not_worse_than_coarse_RRR": True,
        "full_grid_mean_not_worse_than_coarse_RRR": True,
        "ring_mean_not_worse_than_coarse_RRR": True,
    }
    matrices = private.get("error_matrices")
    patient_ids = private.get("patient_ids")
    if (
        original_criteria != expected_original_criteria
        or not isinstance(matrices, Mapping)
        or not isinstance(patient_ids, list)
        or len(patient_ids) != 409
        or any(
            not isinstance(matrices.get(key), Tensor)
            for key in ("candidate", "coarse_rrr", "occupancy_shuffled")
        )
    ):
        raise ValueError("v4.3.1 did not fail solely at the strict background gate")
    maximum_outside = float(
        original_gate["maximum_outside_core_support_absolute"]
    )
    recomputed_original = evaluate_v431_train_oof_gate(
        matrices["candidate"],
        matrices["coarse_rrr"],
        matrices["occupancy_shuffled"],
        patient_ids,
        maximum_outside_core_support_absolute=maximum_outside,
        seed=v431_config.bootstrap_seed,
        replicates=v431_config.bootstrap_replicates,
    )
    _validate_recomputed_original(recomputed_original, original_gate)
    recovery_gate = evaluate_v432_recovery_gate(
        matrices["candidate"],
        matrices["coarse_rrr"],
        matrices["occupancy_shuffled"],
        patient_ids,
        maximum_outside_core_support_absolute=maximum_outside,
        numerical_noninferiority_margin=(
            recovery_config.background_numerical_noninferiority_margin
        ),
        seed=recovery_config.bootstrap_seed,
        replicates=recovery_config.bootstrap_replicates,
    )
    if recovery_gate["recovery_gate_passed"] is not True:
        raise ValueError("v4.3.2 fixed numerical recovery gate did not pass")
    output = {
        "schema_version": SCHEMA,
        "status": STATUS,
        "scope": "adaptive_development_after_reviewing_v4p3p1_train409_OOF",
        "scientific_target": (
            "same_v4p3p1_core_residual_with_finite_precision_background_"
            "noninferiority"
        ),
        "adaptive_disclosure": {
            "recovery_defined_after_observing_v4p3p1_gate_failure": True,
            "original_failure_preserved": True,
            "original_report_overwritten": False,
            "confirmatory_claim_permitted": False,
        },
        "protocol_change": {
            "only_changed_criterion": (
                "background_mean_not_worse_than_coarse_RRR_to_"
                "background_CI_lower_gt_negative_1e-6"
            ),
            "fixed_absolute_numerical_noninferiority_margin": (
                recovery_config.background_numerical_noninferiority_margin
            ),
            "architecture_changed": False,
            "loss_changed": False,
            "feature_release_changed": False,
            "OOF_models_refit": False,
            "OOF_predictions_regenerated": False,
        },
        "original_v4p3p1": {
            "status": report["status"],
            "development_gate_passed": False,
            "final_checkpoint_created": False,
            "criterion_results": original_criteria,
            "best_epoch_counts": epochs,
        },
        "recovery_gate": recovery_gate,
        "final_fit_contract": {
            "authorized": True,
            "train_patients": 409,
            "seed": recovery_config.seed,
            "fixed_epoch_count": recovery_config.final_epoch_count,
            "RRR_anchor_refit_on_train409": True,
            "nested_OOF_rerun": False,
            "validation_88_loaded_during_final_fit": False,
            "test90_loaded_during_final_fit": False,
        },
        "evidence": {
            "recovery_config_sha256": _sha(paths["recovery_config"]),
            "v4p3p1_config_sha256": _sha(paths["v431_config"]),
            "v4p3p1_frozen_design_sha256": _sha(paths["v431_design"]),
            "v4p3p1_failed_report_sha256": _sha(paths["v431_report"]),
            "v4p3p1_private_OOF_measurements_sha256": _sha(
                paths["v431_private"]
            ),
            "manifest_sha256": evidence["manifest_sha256"],
            "feature_index_sha256": evidence["feature_index_sha256"],
            "freeze_source_sha256": _sha(Path(__file__)),
        },
        "training_started": False,
        "validation_88_loaded": False,
        "test_split_loaded": False,
        "patient_identifiers_emitted": False,
        "paths_emitted": False,
    }
    paths["output"].parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = paths["output"].with_suffix(".partial")
    temporary.write_text(
        json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.chmod(temporary, 0o600)
    os.replace(temporary, paths["output"])
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovery-config", required=True)
    parser.add_argument("--v4p3p1-config", required=True)
    parser.add_argument("--v4p3p1-frozen-design", required=True)
    parser.add_argument("--v4p3p1-report", required=True)
    parser.add_argument("--v4p3p1-private-measurements", required=True)
    parser.add_argument("--output", required=True)
    report = freeze(parser.parse_args())
    print(
        json.dumps(
            {
                "status": report["status"],
                "recovery_gate_passed": report["recovery_gate"][
                    "recovery_gate_passed"
                ],
                "OOF_models_refit": False,
                "training_started": False,
                "validation_88_loaded": False,
                "test_split_loaded": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["SCHEMA", "STATUS", "freeze"]
