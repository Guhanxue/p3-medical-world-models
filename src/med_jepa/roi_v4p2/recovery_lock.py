"""Freeze the post-hoc v4.2r selection-policy recovery before retraining."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
from pathlib import Path
from typing import Any, Mapping

from med_jepa.roi_v4p2.config import V42TrainingConfig
from med_jepa.roi_v4p2.lock import SCHEMA as PARENT_LOCK_SCHEMA
from med_jepa.roi_v4p2.train import SELECTION_SCHEMA as PARENT_SELECTION_SCHEMA


SCHEMA = "med-jepa-roi-v4p2r-selection-recovery-lock-v1"
STATUS = "locked_posthoc_gate_constrained_selection_recovery"
SCOPE = "posthoc_selection_policy_recovery_after_v4p2_gate_failure"


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


def select_gate_constrained_candidate(
    parent_selection: Mapping[str, Any],
) -> dict[str, Any]:
    """Select the lowest disease MAE among train-only OOF gate passers."""

    if (
        parent_selection.get("schema_version") != PARENT_SELECTION_SCHEMA
        or parent_selection.get("status")
        != "locked_train_only_hyperparameters_before_validation"
        or parent_selection.get("patients")
        != {"train": 409, "validation_loaded": 0, "test_loaded": 0}
        or parent_selection.get("validation_88_loaded") is not False
        or parent_selection.get("test_split_loaded") is not False
    ):
        raise ValueError("parent selection is not a sealed train-only release")
    cv = parent_selection.get("train_only_cross_validation")
    if not isinstance(cv, Mapping):
        raise ValueError("parent selection lacks train-only cross-validation")
    summaries = cv.get("candidate_summaries")
    gates = cv.get("candidate_oof_gates")
    epochs = cv.get("best_epoch_count_by_lambda")
    if (
        not isinstance(summaries, Mapping)
        or not isinstance(gates, Mapping)
        or not isinstance(epochs, Mapping)
        or set(summaries) != set(gates)
        or set(summaries) != set(epochs)
    ):
        raise ValueError("parent OOF candidate evidence is incomplete")

    eligible: list[str] = []
    for key in summaries:
        summary = summaries[key]
        gate = gates[key]
        criteria = gate.get("criterion_results", {})
        if (
            gate.get("development_gate_passed") is True
            and criteria.get(
                "disease_bed_candidate_vs_shrinkage_CI_lower_positive"
            )
            is True
            and criteria.get(
                "disease_minus_background_relative_improvement_CI_lower_positive"
            )
            is True
        ):
            eligible.append(key)
        if not isinstance(summary, Mapping):
            raise ValueError("parent candidate summary is malformed")
    if not eligible:
        raise ValueError("no train-only OOF candidate satisfies the recovery gate")

    selected_key = min(
        eligible,
        key=lambda key: (
            float(summaries[key]["disease_candidate_to_shrinkage_MAE_ratio"]),
            -float(
                summaries[key]["disease_minus_background_relative_improvement"]
            ),
            float(summaries[key]["full_grid_candidate_MAE"]),
            -float(key),
        ),
    )
    selected_epochs = epochs[selected_key]
    if (
        not isinstance(selected_epochs, list)
        or len(selected_epochs) != 5
        or any(isinstance(value, bool) or int(value) < 1 for value in selected_epochs)
    ):
        raise ValueError("selected recovery epoch evidence is invalid")
    final_epochs = int(round(statistics.median(int(value) for value in selected_epochs)))
    alphas = parent_selection.get("selected_pair_specific_shrinkage_alphas")
    if (
        not isinstance(alphas, Mapping)
        or set(alphas) != {"T0_to_T1", "T0_to_T3", "T1_to_T3"}
        or any(not 0.0 <= float(value) <= 1.0 for value in alphas.values())
    ):
        raise ValueError("parent train-only shrinkage alphas are invalid")
    parent_selected_key = f"{float(parent_selection['selected_occupancy_lambda']):.4f}"
    if (
        parent_selected_key not in gates
        or gates[parent_selected_key].get("development_gate_passed") is not False
        or selected_key == parent_selected_key
    ):
        raise ValueError("parent run does not exhibit the selection-policy failure")
    return {
        "selected_occupancy_lambda": float(selected_key),
        "selected_final_epoch_count": final_epochs,
        "selected_pair_specific_shrinkage_alphas": {
            key: float(value) for key, value in alphas.items()
        },
        "eligible_candidate_keys": sorted(eligible, key=float),
        "selected_candidate_summary": dict(summaries[selected_key]),
        "selected_candidate_oof_gate": dict(gates[selected_key]),
        "parent_selected_occupancy_lambda": float(
            parent_selection["selected_occupancy_lambda"]
        ),
        "parent_selected_candidate_oof_gate": dict(gates[parent_selected_key]),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    config_path = Path(args.config).expanduser().resolve()
    parent_lock_path = Path(args.parent_protocol_lock).expanduser().resolve()
    parent_selection_path = Path(args.parent_selection).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError("immutable v4.2r recovery lock exists")
    config = V42TrainingConfig.load(config_path)
    parent_lock = _json(parent_lock_path)
    parent_selection = _json(parent_selection_path)
    if (
        parent_lock.get("schema_version") != PARENT_LOCK_SCHEMA
        or parent_lock.get("status")
        != "locked_train_only_cv_required_before_validation"
        or parent_lock.get("evidence", {}).get("config_sha256")
        != _sha(config_path)
        or parent_selection.get("evidence", {}).get("protocol_lock_sha256")
        != _sha(parent_lock_path)
        or config.seed != 2026
    ):
        raise ValueError("v4.2r parent protocol chain is invalid")
    selection = select_gate_constrained_candidate(parent_selection)
    report = {
        "schema_version": SCHEMA,
        "status": STATUS,
        "scope": SCOPE,
        "seed": config.seed,
        "selection_policy": {
            "evidence_scope": "parent_immutable_train409_OOF_only",
            "eligibility": [
                "disease_bed_candidate_vs_shrinkage_CI_lower_positive",
                "disease_minus_background_relative_improvement_CI_lower_positive",
            ],
            "ordering_within_eligible_set": [
                "lowest_disease_candidate_to_shrinkage_MAE_ratio",
                "largest_disease_minus_background_relative_improvement",
                "lowest_full_shared_valid_grid_MAE",
                "larger_occupancy_lambda",
            ],
            "validation_values_read_by_recovery_lock": False,
            "test_values_read_by_recovery_lock": False,
        },
        "recovery_selection": selection,
        "validation_contract": {
            "reuse_parent_validation_features": True,
            "analysis_role": "posthoc_mechanistic_selection_policy_recovery",
            "independent_validation_claim_permitted": False,
            "confirmatory_claim_permitted": False,
        },
        "architecture_changed": False,
        "loss_changed": False,
        "feature_variant_changed": False,
        "test_split_loaded": False,
        "eligible_for_test_evaluation": False,
        "patient_identifiers_emitted": False,
        "evidence": {
            "config_sha256": _sha(config_path),
            "parent_protocol_lock_sha256": _sha(parent_lock_path),
            "parent_train_only_selection_sha256": _sha(parent_selection_path),
            "recovery_lock_source_sha256": _sha(Path(__file__)),
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
                "status": STATUS,
                "selected_occupancy_lambda": selection[
                    "selected_occupancy_lambda"
                ],
                "selected_final_epoch_count": selection[
                    "selected_final_epoch_count"
                ],
                "validation_values_read_by_recovery_lock": False,
                "test_split_loaded": False,
            },
            sort_keys=True,
        )
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--parent-protocol-lock", required=True)
    parser.add_argument("--parent-selection", required=True)
    parser.add_argument("--output", required=True)
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "SCHEMA",
    "SCOPE",
    "STATUS",
    "run",
    "select_gate_constrained_candidate",
]
