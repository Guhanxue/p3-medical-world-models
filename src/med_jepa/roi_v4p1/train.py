"""Single-seed source-anchored Med-JEPA v4.1 development training."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor

from med_jepa.frozen_audit.statistics import TrainStandardizer
from med_jepa.roi_v4.audit_masked_cover import representation_rank_diagnostics
from med_jepa.roi_v4.audit_mechanistic import (
    REGIONS,
    _bootstrap_from_errors,
    _delta_diagnostics,
    _load_region_weights,
    _retrieval,
    _summary,
    _validate_mask_release,
    _weighted_pool,
    patient_region_mae,
)
from med_jepa.roi_v4.data import PatientStates, load_development_states, stack_batch
from med_jepa.roi_v4.losses import masked_latent_loss, semigroup_loss
from med_jepa.roi_v4.train import _record_tensor
from med_jepa.roi_v4p1.config import V41TrainingConfig
from med_jepa.roi_v4p1.freeze_design import SCHEMA as DESIGN_SCHEMA
from med_jepa.roi_v4p1.full_token_linear import (
    PRIVATE_SCHEMA as FULL_TOKEN_PRIVATE_SCHEMA,
    SCHEMA as FULL_TOKEN_SCHEMA,
)
from med_jepa.roi_v4p1.masking import scheduled_masks
from med_jepa.roi_v4p1.model import SourceAnchoredTokenDynamicsJEPA


SCHEMA = "med-jepa-roi-v4p1-development-training-v1"
PRIVATE_SCHEMA = "med-jepa-roi-v4p1-private-validation-measurements-v1"
CALIBRATION_SCHEMA = "med-jepa-roi-v4-anchor-calibration-audit-v1"
NOISE_SCHEMA = "med-jepa-roi-v4p1-frozen-3dino-noise-floor-v1"
BASELINE_NAMES = (
    "full_token_markov_matched_linear_ridge",
    "full_token_history_linear_ridge",
    "full_token_conditional_history_linear_ridge",
)


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


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _autocast(device: torch.device):
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def _to_device(
    batch: dict[str, Tensor | list[str]], device: torch.device
) -> dict[str, Tensor | list[str]]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, Tensor) else value
        for key, value in batch.items()
    }


def _interval_action(
    batch: dict[str, Tensor | list[str]], pair: tuple[int, int]
) -> Tensor:
    key = {(0, 1): "action_01", (0, 2): "action_03", (1, 2): "action_13"}[pair]
    value = batch[key]
    assert isinstance(value, Tensor)
    return value


@torch.no_grad()
def _predict(
    model: SourceAnchoredTokenDynamicsJEPA,
    records: Sequence[PatientStates],
    *,
    batch_size: int,
    device: torch.device,
) -> Tensor:
    model.eval()
    rows: list[Tensor] = []
    for start in range(0, len(records), batch_size):
        selected = records[start : start + batch_size]
        source = torch.stack([item.features[1] for item in selected]).to(device)
        source_time = torch.stack([item.times_days[1] for item in selected]).to(device)
        target_time = torch.stack([item.times_days[2] for item in selected]).to(device)
        action = torch.stack([item.action_13 for item in selected]).to(device)
        with _autocast(device):
            rows.append(
                model.step(source, source_time, target_time, action).float().cpu()
            )
    return torch.cat(rows)


@torch.no_grad()
def _validation_primary_mae(
    model: SourceAnchoredTokenDynamicsJEPA,
    records: Sequence[PatientStates],
    weights: Tensor,
    config: V41TrainingConfig,
    device: torch.device,
) -> float:
    prediction = _predict(
        model, records, batch_size=config.batch_size, device=device
    )
    target = _record_tensor(records, "features")[:, 2]
    disease = REGIONS.index("t0_disease_bed")
    values = patient_region_mae(prediction, target, weights[:, disease])
    return float(torch.nanmean(values))


def _train(
    model: SourceAnchoredTokenDynamicsJEPA,
    train: list[PatientStates],
    val: list[PatientStates],
    val_weights: Tensor,
    config: V41TrainingConfig,
    device: torch.device,
) -> tuple[dict[str, Tensor], list[dict[str, Any]], int]:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    generator = torch.Generator().manual_seed(config.seed)
    pairs = ((0, 1), (0, 2), (1, 2))
    best_state: dict[str, Tensor] | None = None
    best_value = math.inf
    best_epoch = -1
    patience = 0
    history: list[dict[str, Any]] = []
    for epoch in range(config.epochs):
        model.train()
        order = torch.randperm(len(train), generator=generator)
        totals: list[float] = []
        fraction_counts = {str(value): 0 for value in config.mask_fractions}
        for start in range(0, len(train), config.batch_size):
            batch_indices = order[start : start + config.batch_size]
            batch = _to_device(stack_batch(train, batch_indices), device)
            features, times = batch["features"], batch["times"]
            assert isinstance(features, Tensor) and isinstance(times, Tensor)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device):
                latent_terms: list[Tensor] = []
                for pair_index, (source_index, target_index) in enumerate(pairs):
                    mask_seed = (
                        config.seed
                        + epoch * 100_003
                        + start * 11
                        + pair_index
                    )
                    context_mask, supervision_mask, fraction = scheduled_masks(
                        features.shape[0],
                        seed=mask_seed,
                        fractions=config.mask_fractions,
                    )
                    context_mask = context_mask.to(device)
                    supervision_mask = supervision_mask.to(device)
                    fraction_counts[str(fraction)] += 1
                    prediction = model.step(
                        features[:, source_index],
                        times[:, source_index],
                        times[:, target_index],
                        _interval_action(batch, (source_index, target_index)),
                        context_mask,
                    )
                    latent_terms.append(
                        masked_latent_loss(
                            prediction.float(),
                            features[:, target_index].float(),
                            supervision_mask,
                        )
                    )
                direct, composed = model.compose(
                    features[:, 0],
                    times,
                    _interval_action(batch, (0, 1)),
                    _interval_action(batch, (1, 2)),
                    _interval_action(batch, (0, 2)),
                )
                consistency_mask = torch.ones(
                    features.shape[0], 1, 7, 7, 7, dtype=torch.bool, device=device
                )
                composition = semigroup_loss(
                    direct.float(), composed.float(), consistency_mask
                )
                total = (
                    config.latent_weight * torch.stack(latent_terms).mean()
                    + config.semigroup_weight * composition
                )
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            totals.append(float(total.detach()))
        validation = _validation_primary_mae(
            model, val, val_weights, config, device
        )
        row: dict[str, Any] = {
            "epoch": epoch,
            "train_total_loss": sum(totals) / len(totals),
            "val_t0_disease_bed_full_source_token_mae": validation,
            "mask_fraction_batch_counts": fraction_counts,
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        if validation < best_value - config.early_stopping_min_delta:
            best_value = validation
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            patience = 0
        else:
            patience += 1
            if patience >= config.early_stopping_patience:
                break
    if best_state is None:
        raise RuntimeError("v4.1 training did not produce a finite checkpoint")
    return best_state, history, best_epoch


def _load_full_token_baselines(
    report_path: Path,
    private_path: Path,
    expected_ids: Sequence[str],
) -> tuple[dict[str, Tensor], dict[str, Any]]:
    report = _json(report_path)
    evidence = report.get("evidence", {})
    if (
        report.get("schema_version") != FULL_TOKEN_SCHEMA
        or report.get("status") != "completed_development_full_token_linear_baselines"
        or report.get("test_split_loaded") is not False
        or evidence.get("private_predictions_sha256") != _sha(private_path)
    ):
        raise ValueError("full-token baseline release is not approved")
    private = torch.load(private_path, map_location="cpu", weights_only=True)
    if (
        private.get("schema_version") != FULL_TOKEN_PRIVATE_SCHEMA
        or private.get("patient_ids") != list(expected_ids)
        or not isinstance(private.get("predictions"), dict)
    ):
        raise ValueError("private full-token baseline predictions do not align")
    missing = set(BASELINE_NAMES) - set(private["predictions"])
    if missing:
        raise ValueError("full-token baseline release lacks required models")
    return {
        name: private["predictions"][name].float() for name in BASELINE_NAMES
    }, report


def _evaluate(
    model: SourceAnchoredTokenDynamicsJEPA,
    train: list[PatientStates],
    val: list[PatientStates],
    train_weights: Tensor,
    val_weights: Tensor,
    config: V41TrainingConfig,
    baseline_report_path: Path,
    baseline_private_path: Path,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    predicted_train = _predict(
        model, train, batch_size=config.batch_size, device=device
    )
    predicted_val = _predict(
        model, val, batch_size=config.batch_size, device=device
    )
    train_features = _record_tensor(train, "features")
    val_features = _record_tensor(val, "features")
    source, target = val_features[:, 1], val_features[:, 2]
    ids = [item.patient_id for item in val]
    baselines, baseline_report = _load_full_token_baselines(
        baseline_report_path, baseline_private_path, ids
    )
    predictions = {
        "source_anchored_med_jepa_v4p1": predicted_val,
        "latent_persistence": source,
        **baselines,
    }
    disease = REGIONS.index("t0_disease_bed")
    full = REGIONS.index("full_shared_valid_grid")
    errors: dict[str, dict[str, Tensor]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    for name, prediction in predictions.items():
        errors[name] = {}
        summaries[name] = {}
        for region_index, region in ((disease, "t0_disease_bed"), (full, "full_shared_valid_grid")):
            value = patient_region_mae(prediction, target, val_weights[:, region_index])
            errors[name][region] = value
            summaries[name][region] = _summary(value)
    comparisons: dict[str, Any] = {}
    candidate = errors["source_anchored_med_jepa_v4p1"]["t0_disease_bed"]
    for counter, reference in enumerate(("latent_persistence", *BASELINE_NAMES)):
        comparisons[f"v4p1_vs_{reference}"] = _bootstrap_from_errors(
            candidate,
            errors[reference]["t0_disease_bed"],
            ids,
            seed=config.bootstrap_seed + counter,
            replicates=config.bootstrap_replicates,
        )
    passed = all(float(value["ci_low"]) > 0 for value in comparisons.values())
    train_target_pooled = _weighted_pool(
        train_features[:, 2], train_weights[:, disease]
    )
    train_eligible = torch.isfinite(train_target_pooled).all(1)
    val_eligible = val_weights[:, disease].flatten(1).sum(1) > 0
    scaler = TrainStandardizer().fit(train_target_pooled[train_eligible])
    retrieval: dict[str, Any] = {}
    ranks: dict[str, Any] = {}
    delta: dict[str, Any] = {}
    private_retrieval: dict[str, Tensor] = {}
    for name, prediction in predictions.items():
        retrieval[name], private_retrieval[name] = _retrieval(
            prediction, target, val_weights[:, disease], scaler, val_eligible
        )
        ranks[name] = {
            "predicted_t3": representation_rank_diagnostics(prediction, device),
            "predicted_delta": representation_rank_diagnostics(
                prediction - source, device
            ),
        }
        delta[name], _ = _delta_diagnostics(
            prediction, source, target, val_weights[:, disease]
        )
    public = {
        "development_gate_passed": passed,
        "gate_policy": {
            "all_four_patient_bootstrap_ci_lower_bounds_positive": True,
            "fixed_effective_rank_threshold": None,
            "best_seed_selection_permitted": False,
        },
        "validation_region_mae": summaries,
        "disease_bed_paired_bootstrap": comparisons,
        "patient_retrieval": retrieval,
        "representation_rank": ranks,
        "delta_diagnostics": delta,
        "full_token_baseline_evidence": {
            "report_schema_version": baseline_report["schema_version"],
            "primary_estimand": baseline_report["primary_estimand"],
        },
    }
    private = {
        "schema_version": PRIVATE_SCHEMA,
        "patient_ids": ids,
        "absolute_errors": errors,
        "retrieval_ranks": private_retrieval,
        "prediction": predicted_val.half(),
    }
    return public, private


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("v4.1 training requires exactly one CUDA GPU")
    config_path = Path(args.config).expanduser().resolve()
    design_path = Path(args.frozen_design).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest_report_path = Path(args.manifest_build_report).expanduser().resolve()
    calibration_path = Path(args.anchor_calibration_audit).expanduser().resolve()
    baseline_report_path = Path(args.full_token_baseline_report).expanduser().resolve()
    baseline_private_path = Path(args.full_token_baseline_private).expanduser().resolve()
    noise_path = Path(args.noise_floor_report).expanduser().resolve()
    mask_root = Path(args.context_mask_release).expanduser().resolve()
    mask_report_path = Path(args.context_mask_report).expanduser().resolve()
    mask_index_path = Path(args.context_mask_index).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    private_output = Path(args.private_output_dir).expanduser().resolve()
    if output.exists() or private_output.exists():
        raise FileExistsError("immutable v4.1 training output exists")
    config = V41TrainingConfig.load(config_path)
    design = _json(design_path)
    manifest_report = _json(manifest_report_path)
    calibration = _json(calibration_path)
    baseline_release = _json(baseline_report_path)
    noise = _json(noise_path)
    design_evidence = design.get("evidence", {})
    calibration_evidence = calibration.get("evidence", {})
    baseline_evidence = baseline_release.get("evidence", {})
    noise_evidence = noise.get("evidence", {})
    if (
        design.get("schema_version") != DESIGN_SCHEMA
        or design.get("status") != "frozen_not_trained"
        or design.get("test_split_loaded") is not False
        or manifest_report.get("status") != "ok"
        or manifest_report.get("manifest_sha256") != _sha(manifest_path)
        or calibration.get("schema_version") != CALIBRATION_SCHEMA
        or calibration.get("status") != "completed_read_only_v4_anchor_calibration_audit"
        or calibration.get("test_split_loaded") is not False
        or calibration_evidence.get("training_report_sha256")
        != design_evidence.get("v4_training_report_sha256")
        or calibration_evidence.get("checkpoint_sha256")
        != design_evidence.get("v4_checkpoint_sha256")
        or calibration_evidence.get("manifest_sha256") != _sha(manifest_path)
        or calibration_evidence.get("context_mask_report_sha256")
        != _sha(mask_report_path)
        or calibration_evidence.get("context_mask_index_sha256")
        != _sha(mask_index_path)
        or calibration_evidence.get("full_token_baseline_report_sha256")
        != _sha(baseline_report_path)
        or calibration_evidence.get("full_token_baseline_private_sha256")
        != _sha(baseline_private_path)
        or baseline_release.get("schema_version") != FULL_TOKEN_SCHEMA
        or baseline_release.get("status")
        != "completed_development_full_token_linear_baselines"
        or baseline_release.get("test_split_loaded") is not False
        or baseline_evidence.get("manifest_sha256") != _sha(manifest_path)
        or baseline_evidence.get("context_mask_report_sha256")
        != _sha(mask_report_path)
        or baseline_evidence.get("context_mask_index_sha256")
        != _sha(mask_index_path)
        or baseline_evidence.get("private_predictions_sha256")
        != _sha(baseline_private_path)
        or noise.get("schema_version") != NOISE_SCHEMA
        or noise.get("status") != "completed_training_only_frozen_3dino_noise_floor"
        or noise.get("test_split_loaded") is not False
        or noise_evidence.get("manifest_sha256") != _sha(manifest_path)
    ):
        raise ValueError("v4.1 governed prerequisite chain is not approved")
    train = load_development_states(manifest_path, "train")
    val = load_development_states(manifest_path, "val")
    development_ids = {item.patient_id for item in (*train, *val)}
    mask_records = _validate_mask_release(
        mask_root, mask_report_path, mask_index_path, development_ids
    )
    train_weights = _load_region_weights(train, mask_records)
    val_weights = _load_region_weights(val, mask_records)
    output.mkdir(parents=True, mode=0o700)
    private_output.mkdir(parents=True, mode=0o700)
    _set_seed(config.seed)
    device = torch.device("cuda")
    model = SourceAnchoredTokenDynamicsJEPA(
        config.hidden_channels, config.blocks
    ).to(device)
    best_state, history, best_epoch = _train(
        model, train, val, val_weights, config, device
    )
    model.load_state_dict(best_state, strict=True)
    evaluation, private = _evaluate(
        model,
        train,
        val,
        train_weights,
        val_weights,
        config,
        baseline_report_path,
        baseline_private_path,
        device,
    )
    checkpoint_path = output / "checkpoint_best.pt"
    torch.save(
        {
            "schema_version": SCHEMA,
            "model": best_state,
            "config": asdict(config),
            "best_epoch": best_epoch,
        },
        checkpoint_path,
    )
    history_path = output / "metrics.jsonl"
    with history_path.open("x", encoding="utf-8") as handle:
        for row in history:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    private_path = private_output / "validation_measurements.pt"
    torch.save(private, private_path)
    report = {
        "schema_version": SCHEMA,
        "status": "completed_development_training",
        "seed": config.seed,
        "patient_counts": {"train": len(train), "val": len(val), "test": 0},
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "selection_metric": "val_t0_disease_bed_full_source_token_mae",
        "model_contract": design["v4p1_contract"],
        "validation": evaluation,
        "noise_floor_triage": noise["triage"],
        "test_split_loaded": False,
        "future_pixel_reconstruction": False,
        "patient_identifiers_emitted": False,
        "confirmatory_claim_permitted": False,
        "evidence": {
            "config_sha256": _sha(config_path),
            "frozen_design_sha256": _sha(design_path),
            "manifest_sha256": _sha(manifest_path),
            "manifest_build_report_sha256": _sha(manifest_report_path),
            "anchor_calibration_audit_sha256": _sha(calibration_path),
            "full_token_baseline_report_sha256": _sha(baseline_report_path),
            "full_token_baseline_private_sha256": _sha(baseline_private_path),
            "noise_floor_report_sha256": _sha(noise_path),
            "context_mask_report_sha256": _sha(mask_report_path),
            "context_mask_index_sha256": _sha(mask_index_path),
            "checkpoint_sha256": _sha(checkpoint_path),
            "private_measurements_sha256": _sha(private_path),
            "model_source_sha256": _sha(Path(__file__).with_name("model.py")),
            "training_source_sha256": _sha(Path(__file__)),
        },
    }
    report_path = output / "aggregate_report.json"
    with report_path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    for path in (checkpoint_path, history_path, private_path, report_path):
        os.chmod(path, 0o600)
    print(
        json.dumps(
            {
                "status": report["status"],
                "development_gate_passed": evaluation["development_gate_passed"],
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
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manifest-build-report", required=True)
    parser.add_argument("--anchor-calibration-audit", required=True)
    parser.add_argument("--full-token-baseline-report", required=True)
    parser.add_argument("--full-token-baseline-private", required=True)
    parser.add_argument("--noise-floor-report", required=True)
    parser.add_argument("--context-mask-release", required=True)
    parser.add_argument("--context-mask-report", required=True)
    parser.add_argument("--context-mask-index", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--private-output-dir", required=True)
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["run"]
