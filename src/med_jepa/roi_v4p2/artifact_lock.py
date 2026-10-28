"""Freeze train409-only token-artifact and tumour-locality audit parameters."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import torch

from med_jepa.roi_v4.audit_mechanistic import REGIONS
from med_jepa.roi_v4p2.data import load_selected_train_release
from med_jepa.roi_v4p2.lock import SELECTED_VARIANT
from med_jepa.roi_v4p2.recovery import SCHEMA as RECOVERY_SCHEMA
from med_jepa.roi_v4p2.token_artifact import (
    freeze_source_norm_rule,
    token_l2_norm,
    train_grouped_locality_probe,
)


SCHEMA = "med-jepa-roi-v4p2r-token-artifact-lock-v1"
PRIVATE_SCHEMA = "med-jepa-roi-v4p2r-private-token-artifact-lock-v1"


def _sha(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    return value


def _region_outlier_prevalence(outlier: torch.Tensor, weights: torch.Tensor) -> dict[str, float]:
    result: dict[str, float] = {}
    for index, region in enumerate(REGIONS):
        weight = weights[:, index].float()
        denominator = weight.sum()
        if float(denominator) <= 0:
            raise ValueError(f"train region has no support: {region}")
        result[region] = float((weight * outlier.float()).sum() / denominator)
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        name: Path(value).expanduser().resolve()
        for name, value in {
            "manifest": args.manifest,
            "triage_private": args.feature_triage_private_root,
            "triage_index": args.feature_triage_private_index,
            "recovery_report": args.recovery_report,
            "output": args.output_dir,
            "private_output": args.private_output_dir,
        }.items()
    }
    final_output = paths["output"]
    final_private = paths["private_output"]
    if final_output.exists() or final_private.exists():
        raise FileExistsError("immutable token-artifact lock output exists")
    staging_output = final_output.parent / f".{final_output.name}.staging"
    staging_private = final_private.parent / f".{final_private.name}.staging"
    if staging_output.exists() or staging_private.exists():
        raise FileExistsError("stale token-artifact lock staging output exists")
    staging_output.mkdir(parents=True, mode=0o700)
    staging_private.mkdir(parents=True, mode=0o700)

    recovery = _json(paths["recovery_report"])
    evidence = recovery.get("evidence", {})
    if (
        recovery.get("schema_version") != RECOVERY_SCHEMA
        or recovery.get("status") not in {
            "completed_posthoc_selection_recovery_gate_pass",
            "completed_posthoc_selection_recovery_gate_fail",
        }
        or recovery.get("posthoc_exploratory") is not True
        or recovery.get("test_split_loaded") is not False
        or recovery.get("selected_feature_variant") != SELECTED_VARIANT
        or recovery.get("patient_counts") != {"train": 409, "val_reused": 88, "test": 0}
        or not isinstance(evidence, Mapping)
        or evidence.get("feature_triage_private_index_sha256")
        != _sha(paths["triage_index"])
        or evidence.get("manifest_sha256") != _sha(paths["manifest"])
    ):
        raise ValueError("v4.2r recovery release chain is invalid")

    records, weights, train_merkle = load_selected_train_release(
        paths["manifest"],
        paths["triage_private"],
        paths["triage_index"],
        expected_index_sha256=_sha(paths["triage_index"]),
    )
    if train_merkle != evidence.get("train_feature_payload_merkle_sha256"):
        raise ValueError("train409 selected-feature release changed")

    source_norms = torch.stack(
        [token_l2_norm(record.features[1]) for record in records]
    )
    norm_rule = freeze_source_norm_rule(source_norms)
    source_outlier = source_norms > float(norm_rule["threshold"])
    locality_report, locality_model = train_grouped_locality_probe(records, weights)
    private_path = staging_private / "lock_payload.pt"
    torch.save(
        {
            "schema_version": PRIVATE_SCHEMA,
            "selected_feature_variant": SELECTED_VARIANT,
            "source_visit_for_outlier_definition": "T1",
            "norm_rule": norm_rule,
            "locality_probe": locality_model,
            "train_feature_payload_merkle_sha256": train_merkle,
        },
        private_path,
    )
    os.chmod(private_path, 0o600)

    report = {
        "schema_version": SCHEMA,
        "status": "locked_train409_token_artifact_audit",
        "scope": "posthoc_train_locked_validation_sensitivity_analysis",
        "patient_counts_loaded": {"train": 409, "val": 0, "test": 0},
        "selected_feature_variant": SELECTED_VARIANT,
        "feature_layer_contract": (
            "final_normalized_3DINO_patch_tokens_from_get_intermediate_layers_n1"
        ),
        "source_visit_for_primary_outlier_definition": "T1",
        "norm_rule": norm_rule,
        "train_source_outlier_prevalence_by_region": _region_outlier_prevalence(
            source_outlier, weights
        ),
        "train_only_grouped_OOF_tumour_locality_probe": locality_report,
        "primary_validation_estimand": (
            "source_T1_defined_normal_token_sensitivity_vs_train_only_shrinkage"
        ),
        "target_T3_defined_exclusion_is_primary": False,
        "target_T3_defined_exclusion_permitted_as_descriptive_sensitivity": True,
        "formal_development_gate": False,
        "confirmatory_claim_permitted": False,
        "eligible_for_test_evaluation": False,
        "validation_loaded_while_freezing_parameters": False,
        "test_split_loaded": False,
        "patient_identifiers_emitted": False,
        "interpretation_boundary": (
            "absence_of_a_final_norm_tail_does_not_exclude_hidden_layer_register_neurons"
        ),
        "evidence": {
            "manifest_sha256": _sha(paths["manifest"]),
            "feature_triage_private_index_sha256": _sha(paths["triage_index"]),
            "train_feature_payload_merkle_sha256": train_merkle,
            "recovery_report_sha256": _sha(paths["recovery_report"]),
            "recovery_private_validation_measurements_sha256": evidence.get(
                "private_validation_measurements_sha256"
            ),
            "private_lock_payload_sha256": _sha(private_path),
            "artifact_lock_source_sha256": _sha(Path(__file__)),
            "token_artifact_source_sha256": _sha(
                Path(__file__).with_name("token_artifact.py")
            ),
        },
    }
    report_path = staging_output / "aggregate_report.json"
    with report_path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.chmod(report_path, 0o600)
    os.replace(staging_private, final_private)
    os.replace(staging_output, final_output)
    print(
        json.dumps(
            {
                "status": report["status"],
                "final_norm_sparse_heavy_tail_signature": norm_rule[
                    "final_norm_sparse_heavy_tail_signature"
                ],
                "train_locality_AUPRC": locality_report["auprc"],
                "validation_loaded": False,
                "test_split_loaded": False,
            },
            sort_keys=True,
        )
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--feature-triage-private-root", required=True)
    parser.add_argument("--feature-triage-private-index", required=True)
    parser.add_argument("--recovery-report", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--private-output-dir", required=True)
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["PRIVATE_SCHEMA", "SCHEMA", "run"]
