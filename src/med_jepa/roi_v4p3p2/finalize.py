"""Fit the authorized v4.3.2 train409 model without repeating nested OOF."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import torch

from med_jepa.roi_v4p2.train import _record_features
from med_jepa.roi_v4p3.data import load_v4p3_train409_release
from med_jepa.roi_v4p3p1.config import V431TrainingConfig
from med_jepa.roi_v4p3p1.train import _fit_anchor_bank, _fit_for_epochs
from med_jepa.roi_v4p3p2.config import V432RecoveryConfig
from med_jepa.roi_v4p3p2.freeze_recovery import (
    SCHEMA as RECOVERY_LOCK_SCHEMA,
    STATUS as RECOVERY_LOCK_STATUS,
)


SCHEMA = "med-jepa-roi-v4p3p2-train409-final-fit-v1"
PRIVATE_SCHEMA = "med-jepa-roi-v4p3p2-private-final-fit-v1"


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


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    rendered = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(rendered, encoding="utf-8")
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("v4.3.2 final fit requires exactly one CUDA GPU")
    paths = {
        name: Path(value).expanduser().resolve()
        for name, value in {
            "recovery_config": args.recovery_config,
            "v431_config": args.v4p3p1_config,
            "recovery_lock": args.recovery_lock,
            "v431_design": args.v4p3p1_frozen_design,
            "v431_report": args.v4p3p1_report,
            "v431_private": args.v4p3p1_private_measurements,
            "manifest": args.manifest,
            "manifest_report": args.manifest_build_report,
            "feature_root": args.feature_triage_private_root,
            "feature_index": args.feature_triage_private_index,
            "mask_root": args.context_mask_release,
            "mask_report": args.context_mask_report,
            "mask_index": args.context_mask_index,
            "output": args.output_dir,
            "private_output": args.private_output_dir,
        }.items()
    }
    for name in (
        "recovery_config",
        "v431_config",
        "recovery_lock",
        "v431_design",
        "v431_report",
        "v431_private",
        "manifest",
        "manifest_report",
        "feature_index",
        "mask_report",
        "mask_index",
    ):
        if not paths[name].is_file():
            raise FileNotFoundError(f"v4.3.2 final-fit input unavailable: {name}")
    if not paths["feature_root"].is_dir() or not paths["mask_root"].is_dir():
        raise FileNotFoundError("v4.3.2 governed release is unavailable")
    if paths["output"].exists() or paths["private_output"].exists():
        raise FileExistsError("immutable v4.3.2 final-fit output exists")
    output_staging = paths["output"].parent / f".{paths['output'].name}.staging"
    private_staging = (
        paths["private_output"].parent
        / f".{paths['private_output'].name}.staging"
    )
    output_staging.mkdir(parents=True, exist_ok=True, mode=0o700)
    private_staging.mkdir(parents=True, exist_ok=True, mode=0o700)

    recovery_config = V432RecoveryConfig.load(paths["recovery_config"])
    v431_config = V431TrainingConfig.load(paths["v431_config"])
    lock = _json(paths["recovery_lock"])
    parent = _json(paths["v431_report"])
    manifest_report = _json(paths["manifest_report"])
    lock_evidence = lock.get("evidence", {})
    parent_evidence = parent.get("evidence", {})
    if (
        lock.get("schema_version") != RECOVERY_LOCK_SCHEMA
        or lock.get("status") != RECOVERY_LOCK_STATUS
        or lock.get("recovery_gate", {}).get("recovery_gate_passed") is not True
        or lock.get("final_fit_contract", {}).get("fixed_epoch_count") != 18
        or lock.get("validation_88_loaded") is not False
        or lock.get("test_split_loaded") is not False
        or lock_evidence.get("recovery_config_sha256")
        != _sha(paths["recovery_config"])
        or lock_evidence.get("v4p3p1_config_sha256")
        != _sha(paths["v431_config"])
        or lock_evidence.get("v4p3p1_frozen_design_sha256")
        != _sha(paths["v431_design"])
        or lock_evidence.get("v4p3p1_failed_report_sha256")
        != _sha(paths["v431_report"])
        or lock_evidence.get("v4p3p1_private_OOF_measurements_sha256")
        != _sha(paths["v431_private"])
        or parent.get("status")
        != "completed_train409_core_residual_gate_fail_no_final_model"
        or parent_evidence.get("manifest_sha256") != _sha(paths["manifest"])
        or parent_evidence.get("feature_index_sha256")
        != _sha(paths["feature_index"])
        or manifest_report.get("status") != "ok"
        or manifest_report.get("manifest_sha256") != _sha(paths["manifest"])
        or recovery_config.final_epoch_count != 18
        or v431_config.seed != recovery_config.seed
    ):
        raise ValueError("v4.3.2 recovery release chain is invalid")

    records, evaluation_weights, input_occupancy, feature_merkle, occupancy_digest = (
        load_v4p3_train409_release(
            paths["manifest"],
            paths["feature_root"],
            paths["feature_index"],
            expected_feature_index_sha256=_sha(paths["feature_index"]),
            context_mask_root=paths["mask_root"],
            context_mask_report=paths["mask_report"],
            context_mask_index=paths["mask_index"],
        )
    )
    if (
        len(records) != 409
        or parent_evidence.get("feature_payload_merkle_sha256") != feature_merkle
        or parent_evidence.get("T0_only_input_occupancy_digest")
        != occupancy_digest
    ):
        raise ValueError("v4.3.2 train409 payload changed")
    resume_contract = {
        "recovery_lock_sha256": _sha(paths["recovery_lock"]),
        "v4p3p1_failed_report_sha256": _sha(paths["v431_report"]),
        "v4p3p1_private_OOF_measurements_sha256": _sha(paths["v431_private"]),
        "manifest_sha256": _sha(paths["manifest"]),
        "feature_index_sha256": _sha(paths["feature_index"]),
        "feature_payload_merkle_sha256": feature_merkle,
        "T0_only_input_occupancy_digest": occupancy_digest,
        "final_epoch_count": 18,
        "finalize_source_sha256": _sha(Path(__file__)),
    }
    cache_path = private_staging / "final_fit_state.pt"
    if cache_path.is_file():
        fit = torch.load(cache_path, map_location="cpu", weights_only=True)
        if (
            not isinstance(fit, Mapping)
            or fit.get("schema_version") != PRIVATE_SCHEMA
            or fit.get("resume_contract") != resume_contract
            or not isinstance(fit.get("model"), Mapping)
            or not isinstance(fit.get("history"), list)
            or not isinstance(fit.get("coarse_RRR_train409_fit_report"), Mapping)
        ):
            raise ValueError("resumable v4.3.2 final fit changed")
        state = dict(fit["model"])
        history = list(fit["history"])
        anchor_report = dict(fit["coarse_RRR_train409_fit_report"])
        print(json.dumps({"stage": "v4p3p2_final_fit_resume"}), flush=True)
    else:
        features = _record_features(records)
        indices = torch.arange(409)
        anchors, anchor_report = _fit_anchor_bank(
            records,
            features,
            evaluation_weights,
            input_occupancy,
            indices,
            indices,
            seed=recovery_config.seed + 900_001,
            device=torch.device("cuda"),
        )
        state, history = _fit_for_epochs(
            records,
            input_occupancy,
            anchors,
            indices,
            v431_config,
            epochs=18,
            seed=recovery_config.seed,
            device=torch.device("cuda"),
        )
        temporary = cache_path.with_suffix(".partial")
        torch.save(
            {
                "schema_version": PRIVATE_SCHEMA,
                "resume_contract": resume_contract,
                "model": state,
                "history": history,
                "coarse_RRR_train409_fit_report": anchor_report,
            },
            temporary,
        )
        os.chmod(temporary, 0o600)
        os.replace(temporary, cache_path)

    checkpoint_path = output_staging / "checkpoint_train409.pt"
    if not checkpoint_path.exists():
        torch.save(
            {
                "schema_version": SCHEMA,
                "model": state,
                "v4p3p1_model_config": asdict(v431_config),
                "recovery_config": asdict(recovery_config),
                "selected_candidate": (
                    "coarse_RRR_anchored_fractional_T0_core_residual"
                ),
                "final_epoch_count": 18,
                "coarse_RRR_train409_fit_report": anchor_report,
                "coarse_RRR_refit_required_from_bound_train409_release": True,
                "recovery_lock_sha256": _sha(paths["recovery_lock"]),
                "validation_88_loaded": False,
                "test_split_loaded": False,
            },
            checkpoint_path,
        )
        os.chmod(checkpoint_path, 0o600)
    history_path = output_staging / "metrics.jsonl"
    if not history_path.exists():
        with history_path.open("x", encoding="utf-8") as handle:
            for row in history:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        os.chmod(history_path, 0o600)
    report = {
        "schema_version": SCHEMA,
        "status": "completed_adaptive_recovery_train409_final_fit",
        "scope": "adaptive_development_train409_final_fit_after_OOF_reaudit",
        "seed": recovery_config.seed,
        "final_epoch_count": 18,
        "patient_counts": {"train_loaded": 409, "validation_loaded": 0, "test_loaded": 0},
        "selected_candidate": (
            "coarse_RRR_anchored_fractional_T0_core_residual"
        ),
        "nested_OOF_rerun": False,
        "coarse_RRR_train409_fit_report": anchor_report,
        "checkpoint_created": True,
        "validation_88_loaded": False,
        "test_split_loaded": False,
        "confirmatory_claim_permitted": False,
        "evidence": {
            **resume_contract,
            "checkpoint_sha256": _sha(checkpoint_path),
            "metrics_sha256": _sha(history_path),
            "private_final_fit_state_sha256": _sha(cache_path),
            "context_mask_report_sha256": _sha(paths["mask_report"]),
            "context_mask_index_sha256": _sha(paths["mask_index"]),
            "v4p3p1_model_source_sha256": _sha(
                Path(__file__).parents[1] / "roi_v4p3p1" / "model.py"
            ),
        },
        "patient_identifiers_emitted": False,
        "paths_emitted": False,
    }
    _write_json(output_staging / "aggregate_report.json", report)
    os.replace(private_staging, paths["private_output"])
    os.replace(output_staging, paths["output"])
    print(
        json.dumps(
            {
                "status": report["status"],
                "final_epoch_count": 18,
                "checkpoint_created": True,
                "nested_OOF_rerun": False,
                "validation_88_loaded": False,
                "test_split_loaded": False,
            },
            sort_keys=True,
        )
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovery-config", required=True)
    parser.add_argument("--v4p3p1-config", required=True)
    parser.add_argument("--recovery-lock", required=True)
    parser.add_argument("--v4p3p1-frozen-design", required=True)
    parser.add_argument("--v4p3p1-report", required=True)
    parser.add_argument("--v4p3p1-private-measurements", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manifest-build-report", required=True)
    parser.add_argument("--feature-triage-private-root", required=True)
    parser.add_argument("--feature-triage-private-index", required=True)
    parser.add_argument("--context-mask-release", required=True)
    parser.add_argument("--context-mask-report", required=True)
    parser.add_argument("--context-mask-index", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--private-output-dir", required=True)
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["PRIVATE_SCHEMA", "SCHEMA", "run"]
