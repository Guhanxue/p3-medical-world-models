"""Run the fixed-candidate Med-JEPA v4.2r post-hoc recovery analysis."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import torch

from med_jepa.roi_v4p2.config import V42TrainingConfig
from med_jepa.roi_v4p2.data import (
    load_existing_selected_validation_release,
    load_selected_train_release,
)
from med_jepa.roi_v4p2.lock import (
    SCHEMA as PARENT_LOCK_SCHEMA,
    SELECTED_VARIANT,
)
from med_jepa.roi_v4p2.model import ShrinkageAnchoredTokenDynamicsJEPA
from med_jepa.roi_v4p2.recovery_lock import (
    SCHEMA as RECOVERY_LOCK_SCHEMA,
    SCOPE,
    STATUS as RECOVERY_LOCK_STATUS,
)
from med_jepa.roi_v4p2.train import (
    SCHEMA as PARENT_TRAINING_SCHEMA,
    SELECTION_SCHEMA as PARENT_SELECTION_SCHEMA,
    _fit_anchors,
    _record_features,
    _train_final,
    _validation_evaluation,
)


SCHEMA = "med-jepa-roi-v4p2r-posthoc-selection-recovery-v1"
PRIVATE_SCHEMA = "med-jepa-roi-v4p2r-private-validation-measurements-v1"


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


def _same_float_mapping(
    observed: Mapping[str, Any], expected: Mapping[str, Any]
) -> bool:
    return set(observed) == set(expected) and all(
        float(observed[key]) == float(expected[key]) for key in expected
    )


def _validate_parent_chain(
    *,
    paths: Mapping[str, Path],
    config: V42TrainingConfig,
    recovery_lock: Mapping[str, Any],
    parent_lock: Mapping[str, Any],
    parent_selection: Mapping[str, Any],
    parent_report: Mapping[str, Any],
) -> tuple[float, int, dict[str, float]]:
    evidence = recovery_lock.get("evidence", {})
    parent_evidence = parent_report.get("evidence", {})
    recovery = recovery_lock.get("recovery_selection", {})
    parent_gate = parent_report.get("validation", {}).get("development_gate", {})
    if (
        recovery_lock.get("schema_version") != RECOVERY_LOCK_SCHEMA
        or recovery_lock.get("status") != RECOVERY_LOCK_STATUS
        or recovery_lock.get("scope") != SCOPE
        or recovery_lock.get("architecture_changed") is not False
        or recovery_lock.get("loss_changed") is not False
        or recovery_lock.get("feature_variant_changed") is not False
        or recovery_lock.get("test_split_loaded") is not False
        or recovery_lock.get("eligible_for_test_evaluation") is not False
        or evidence.get("config_sha256") != _sha(paths["config"])
        or evidence.get("parent_protocol_lock_sha256")
        != _sha(paths["parent_lock"])
        or evidence.get("parent_train_only_selection_sha256")
        != _sha(paths["parent_selection"])
        or parent_lock.get("schema_version") != PARENT_LOCK_SCHEMA
        or parent_selection.get("schema_version") != PARENT_SELECTION_SCHEMA
        or parent_report.get("schema_version") != PARENT_TRAINING_SCHEMA
        or parent_report.get("status")
        != "completed_development_training_gate_fail"
        or parent_report.get("selected_feature_variant") != SELECTED_VARIANT
        or float(parent_report.get("selected_occupancy_lambda")) != 0.25
        or parent_report.get("validation_88_loaded_once_after_lock") is not True
        or parent_report.get("test_split_loaded") is not False
        or parent_report.get("confirmatory_claim_permitted") is not False
        or parent_gate.get("development_gate_passed") is not False
        or parent_report.get("training_selection", {}).get(
            "selection_report_sha256"
        )
        != _sha(paths["parent_selection"])
        or parent_evidence.get("config_sha256") != _sha(paths["config"])
        or parent_evidence.get("protocol_lock_sha256")
        != _sha(paths["parent_lock"])
        or parent_evidence.get("feature_triage_private_index_sha256")
        != _sha(paths["triage_private_index"])
        or parent_evidence.get("manifest_sha256") != _sha(paths["manifest"])
        or config.seed != 2026
    ):
        raise ValueError("v4.2r parent release chain is invalid")
    occupancy_lambda = float(recovery.get("selected_occupancy_lambda"))
    final_epochs = int(recovery.get("selected_final_epoch_count"))
    alphas = recovery.get("selected_pair_specific_shrinkage_alphas")
    selected_gate = recovery.get("selected_candidate_oof_gate", {})
    if (
        occupancy_lambda != 0.05
        or final_epochs != 45
        or not isinstance(alphas, Mapping)
        or selected_gate.get("development_gate_passed") is not True
        or recovery.get("eligible_candidate_keys") != ["0.0100", "0.0500", "0.1000"]
    ):
        raise ValueError("v4.2r fixed recovery candidate changed")
    return (
        occupancy_lambda,
        final_epochs,
        {key: float(value) for key, value in alphas.items()},
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("v4.2r recovery requires exactly one CUDA GPU")
    paths = {
        name: Path(value).expanduser().resolve()
        for name, value in {
            "config": args.config,
            "recovery_lock": args.recovery_lock,
            "parent_lock": args.parent_protocol_lock,
            "parent_selection": args.parent_selection,
            "parent_report": args.parent_training_report,
            "parent_private": args.parent_private_root,
            "triage_private": args.feature_triage_private_root,
            "triage_private_index": args.feature_triage_private_index,
            "manifest": args.manifest,
            "output": args.output_dir,
            "private_output": args.private_output_dir,
        }.items()
    }
    final_output = paths["output"]
    final_private = paths["private_output"]
    if final_output.exists() or final_private.exists():
        raise FileExistsError("immutable v4.2r output exists")
    paths["output"] = final_output.parent / f".{final_output.name}.staging"
    paths["private_output"] = final_private.parent / f".{final_private.name}.staging"
    config = V42TrainingConfig.load(paths["config"])
    recovery_lock = _json(paths["recovery_lock"])
    parent_lock = _json(paths["parent_lock"])
    parent_selection = _json(paths["parent_selection"])
    parent_report = _json(paths["parent_report"])
    occupancy_lambda, final_epochs, locked_alphas = _validate_parent_chain(
        paths=paths,
        config=config,
        recovery_lock=recovery_lock,
        parent_lock=parent_lock,
        parent_selection=parent_selection,
        parent_report=parent_report,
    )
    paths["output"].mkdir(parents=True, exist_ok=True, mode=0o700)
    paths["private_output"].mkdir(parents=True, exist_ok=True, mode=0o700)
    device = torch.device("cuda")
    train, train_weights, train_merkle = load_selected_train_release(
        paths["manifest"],
        paths["triage_private"],
        paths["triage_private_index"],
        expected_index_sha256=_sha(paths["triage_private_index"]),
    )
    if train_merkle != parent_report.get("evidence", {}).get(
        "train_feature_payload_merkle_sha256"
    ):
        raise ValueError("parent train feature payload release changed")
    features = _record_features(train)
    observed_alphas, alpha_scores, means = _fit_anchors(
        features,
        train_weights,
        torch.arange(409),
        device=device,
    )
    if not _same_float_mapping(observed_alphas, locked_alphas):
        raise ValueError("recomputed train-only shrinkage alphas changed")

    resume_contract = {
        "config_sha256": _sha(paths["config"]),
        "recovery_lock_sha256": _sha(paths["recovery_lock"]),
        "parent_train_only_selection_sha256": _sha(paths["parent_selection"]),
        "train_feature_payload_merkle_sha256": train_merkle,
        "recovery_source_sha256": _sha(Path(__file__)),
    }
    final_fit_cache = paths["private_output"] / "final_fit_state.pt"
    if final_fit_cache.is_file():
        fit = torch.load(final_fit_cache, map_location="cpu", weights_only=True)
        if (
            not isinstance(fit, Mapping)
            or fit.get("schema_version")
            != "med-jepa-roi-v4p2r-private-final-fit-v1"
            or fit.get("resume_contract") != resume_contract
            or not isinstance(fit.get("model"), Mapping)
            or not isinstance(fit.get("history"), list)
        ):
            raise ValueError("resumable v4.2r final fit does not match")
        state = dict(fit["model"])
        history = list(fit["history"])
        print(json.dumps({"stage": "v4p2r_final_train_resume"}), flush=True)
    else:
        state, history = _train_final(
            train,
            train_weights,
            config,
            occupancy_lambda,
            final_epochs,
            observed_alphas,
            means,
            device,
        )
        temporary = final_fit_cache.with_suffix(".partial")
        torch.save(
            {
                "schema_version": "med-jepa-roi-v4p2r-private-final-fit-v1",
                "resume_contract": resume_contract,
                "model": state,
                "history": history,
            },
            temporary,
        )
        os.chmod(temporary, 0o600)
        os.replace(temporary, final_fit_cache)

    parent_evidence = parent_report["evidence"]
    validation_root = paths["parent_private"] / "validation_features"
    val, val_weights, val_merkle, val_index, extraction_contract = (
        load_existing_selected_validation_release(
            manifest_path=paths["manifest"],
            validation_root=validation_root,
            expected_index_sha256=parent_evidence[
                "validation_feature_index_sha256"
            ],
            expected_payload_merkle_sha256=parent_evidence[
                "validation_feature_payload_merkle_sha256"
            ],
        )
    )
    gc.collect()
    torch.cuda.empty_cache()
    model = ShrinkageAnchoredTokenDynamicsJEPA(
        config.hidden_channels, config.blocks
    ).to(device)
    model.load_state_dict(state, strict=True)
    evaluation, private = _validation_evaluation(
        model,
        val,
        val_weights,
        config,
        observed_alphas,
        means,
        device,
    )
    evaluation.update(
        {
            "analysis_role": SCOPE,
            "validation_features_reused_from_parent": True,
            "validation_previously_opened_by_parent": True,
            "independent_validation_claim_permitted": False,
            "confirmatory_claim_permitted": False,
        }
    )
    private.update(
        {
            "schema_version": PRIVATE_SCHEMA,
            "analysis_role": SCOPE,
            "parent_validation_feature_index_sha256": _sha(val_index),
        }
    )
    passed = bool(evaluation["development_gate"]["development_gate_passed"])

    checkpoint_path = paths["output"] / "checkpoint_best.pt"
    torch.save(
        {
            "schema_version": SCHEMA,
            "model": state,
            "config": asdict(config),
            "scope": SCOPE,
            "selected_feature_variant": SELECTED_VARIANT,
            "occupancy_lambda": occupancy_lambda,
            "epochs": final_epochs,
            "pair_specific_shrinkage_alphas": observed_alphas,
            "population_target_means": {
                str(key): value.half() for key, value in means.items()
            },
        },
        checkpoint_path,
    )
    history_path = paths["output"] / "metrics.jsonl"
    with history_path.open("x", encoding="utf-8") as handle:
        for row in history:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    private_path = paths["private_output"] / "validation_measurements.pt"
    torch.save(private, private_path)
    report = {
        "schema_version": SCHEMA,
        "status": (
            "completed_posthoc_selection_recovery_gate_pass"
            if passed
            else "completed_posthoc_selection_recovery_gate_fail"
        ),
        "scope": SCOPE,
        "seed": config.seed,
        "patient_counts": {"train": 409, "val_reused": 88, "test": 0},
        "selected_feature_variant": SELECTED_VARIANT,
        "selected_occupancy_lambda": occupancy_lambda,
        "final_epoch_count": final_epochs,
        "pair_specific_shrinkage_alphas": observed_alphas,
        "pair_specific_shrinkage_alpha_scores": alpha_scores,
        "selection_policy_recovery": recovery_lock["selection_policy"],
        "parent_v4p2": {
            "status": parent_report["status"],
            "selected_occupancy_lambda": parent_report[
                "selected_occupancy_lambda"
            ],
            "development_gate": parent_report["validation"][
                "development_gate"
            ],
        },
        "validation": evaluation,
        "validation_features_reused_from_parent": True,
        "validation_extraction_contract": extraction_contract,
        "posthoc_exploratory": True,
        "independent_validation_claim_permitted": False,
        "confirmatory_claim_permitted": False,
        "eligible_for_test_evaluation": False,
        "future_pixel_reconstruction": False,
        "test_split_loaded": False,
        "patient_identifiers_emitted": False,
        "evidence": {
            "config_sha256": _sha(paths["config"]),
            "recovery_lock_sha256": _sha(paths["recovery_lock"]),
            "parent_protocol_lock_sha256": _sha(paths["parent_lock"]),
            "parent_train_only_selection_sha256": _sha(
                paths["parent_selection"]
            ),
            "parent_training_report_sha256": _sha(paths["parent_report"]),
            "feature_triage_private_index_sha256": _sha(
                paths["triage_private_index"]
            ),
            "train_feature_payload_merkle_sha256": train_merkle,
            "parent_validation_feature_index_sha256": _sha(val_index),
            "parent_validation_feature_payload_merkle_sha256": val_merkle,
            "manifest_sha256": _sha(paths["manifest"]),
            "checkpoint_sha256": _sha(checkpoint_path),
            "private_validation_measurements_sha256": _sha(private_path),
            "model_source_sha256": _sha(Path(__file__).with_name("model.py")),
            "data_source_sha256": _sha(Path(__file__).with_name("data.py")),
            "parent_training_source_sha256": _sha(
                Path(__file__).with_name("train.py")
            ),
            "recovery_source_sha256": _sha(Path(__file__)),
        },
    }
    report_path = paths["output"] / "aggregate_report.json"
    with report_path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    for path in checkpoint_path, history_path, private_path, report_path:
        os.chmod(path, 0o600)
    os.replace(paths["private_output"], final_private)
    os.replace(paths["output"], final_output)
    print(
        json.dumps(
            {
                "status": report["status"],
                "scope": SCOPE,
                "development_gate_passed": passed,
                "selected_occupancy_lambda": occupancy_lambda,
                "validation_features_reused_from_parent": True,
                "confirmatory_claim_permitted": False,
                "test_split_loaded": False,
            },
            sort_keys=True,
        )
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--recovery-lock", required=True)
    parser.add_argument("--parent-protocol-lock", required=True)
    parser.add_argument("--parent-selection", required=True)
    parser.add_argument("--parent-training-report", required=True)
    parser.add_argument("--parent-private-root", required=True)
    parser.add_argument("--feature-triage-private-root", required=True)
    parser.add_argument("--feature-triage-private-index", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--private-output-dir", required=True)
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["PRIVATE_SCHEMA", "SCHEMA", "run"]
