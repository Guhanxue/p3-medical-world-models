"""One-seed development training and formal validation gate for Med-JEPA v4."""

from __future__ import annotations

import argparse
import copy
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
import torch.nn.functional as F

from med_jepa.frozen_audit.features import feature_views
from med_jepa.frozen_audit.run import (
    _bootstrap_error_comparison,
    _fit_ridge_probe,
    _fit_rrr_temporal,
)
from med_jepa.roi_v4.config import V4TrainingConfig
from med_jepa.roi_v4.data import PatientStates, load_development_states, stack_batch
from med_jepa.roi_v4.gate import Comparator, evaluate_v4_gate
from med_jepa.roi_v4.losses import masked_latent_loss, semigroup_loss, token_variance_loss
from med_jepa.roi_v4.masking import block_mask
from med_jepa.roi_v4.model import TokenDynamicsJEPA


SCHEMA = "med-jepa-roi-v4-development-training-v1"
PRIVATE_SCHEMA = "med-jepa-roi-v4-private-validation-errors-v1"


def _sha(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("release report must be a JSON object")
    return value


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _autocast(device: torch.device):
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else torch.autocast(device_type="cpu", enabled=False)


def _to_device(batch: dict[str, Tensor | list[str]], device: torch.device) -> dict[str, Tensor | list[str]]:
    return {key: value.to(device, non_blocking=True) if isinstance(value, Tensor) else value for key, value in batch.items()}


def _interval_action(batch: dict[str, Tensor | list[str]], pair: tuple[int, int]) -> Tensor:
    lookup = {(0, 1): "action_01", (0, 2): "action_03", (1, 2): "action_13"}
    value = batch[lookup[pair]]
    assert isinstance(value, Tensor)
    return value


def _masked_validation_loss(
    model: TokenDynamicsJEPA,
    records: list[PatientStates],
    config: V4TrainingConfig,
    device: torch.device,
) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for start in range(0, len(records), config.batch_size):
            indices = torch.arange(start, min(len(records), start + config.batch_size))
            batch = _to_device(stack_batch(records, indices), device)
            features = batch["features"]
            times = batch["times"]
            action = batch["action_13"]
            assert isinstance(features, Tensor) and isinstance(times, Tensor) and isinstance(action, Tensor)
            mask = block_mask(features.shape[0], seed=config.seed + 900_000 + start, masked_fraction=config.masked_fraction).to(device)
            with _autocast(device):
                prediction = model.step(features[:, 1], times[:, 1], times[:, 2], action, mask)
                loss = masked_latent_loss(prediction.float(), features[:, 2].float(), mask)
            losses.append(float(loss))
    return sum(losses) / len(losses)


def _train(
    model: TokenDynamicsJEPA,
    train: list[PatientStates],
    val: list[PatientStates],
    config: V4TrainingConfig,
    device: torch.device,
) -> tuple[dict[str, Tensor], list[dict[str, float]], int]:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    generator = torch.Generator().manual_seed(config.seed)
    best_state: dict[str, Tensor] | None = None
    best_value = math.inf
    best_epoch = -1
    patience = 0
    history: list[dict[str, float]] = []
    pairs = ((0, 1), (0, 2), (1, 2))
    for epoch in range(config.epochs):
        model.train()
        order = torch.randperm(len(train), generator=generator)
        totals: list[float] = []
        for start in range(0, len(train), config.batch_size):
            batch_indices = order[start : start + config.batch_size]
            batch = _to_device(stack_batch(train, batch_indices), device)
            features = batch["features"]
            times = batch["times"]
            assert isinstance(features, Tensor) and isinstance(times, Tensor)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device):
                latent_terms: list[Tensor] = []
                variance_terms: list[Tensor] = []
                for pair_index, (source, target) in enumerate(pairs):
                    mask_seed = config.seed + epoch * 100_003 + start * 11 + pair_index
                    mask = block_mask(features.shape[0], seed=mask_seed, masked_fraction=config.masked_fraction).to(device)
                    prediction = model.step(
                        features[:, source], times[:, source], times[:, target],
                        _interval_action(batch, (source, target)), mask,
                    )
                    latent_terms.append(masked_latent_loss(prediction.float(), features[:, target].float(), mask))
                    variance_terms.append(token_variance_loss(prediction.float(), config.variance_margin))
                direct = model.step(features[:, 0], times[:, 0], times[:, 2], _interval_action(batch, (0, 2)))
                middle = model.step(features[:, 0], times[:, 0], times[:, 1], _interval_action(batch, (0, 1)))
                composed = model.step(middle, times[:, 1], times[:, 2], _interval_action(batch, (1, 2)))
                consistency_mask = block_mask(features.shape[0], seed=config.seed + epoch * 200_003 + start, masked_fraction=config.masked_fraction).to(device)
                composition = semigroup_loss(direct.float(), composed.float(), consistency_mask)
                total = (
                    config.latent_weight * torch.stack(latent_terms).mean()
                    + config.semigroup_weight * composition
                    + config.variance_weight * torch.stack(variance_terms).mean()
                )
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            totals.append(float(total.detach()))
        val_loss = _masked_validation_loss(model, val, config, device)
        row = {"epoch": float(epoch), "train_total_loss": sum(totals) / len(totals), "val_masked_t1_t3_smooth_l1": val_loss}
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        if val_loss < best_value - config.early_stopping_min_delta:
            best_value = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= config.early_stopping_patience:
                break
    if best_state is None:
        raise RuntimeError("v4 training did not produce a finite checkpoint")
    return best_state, history, best_epoch


@torch.no_grad()
def _predict_t3(
    model: TokenDynamicsJEPA,
    records: list[PatientStates],
    batch_size: int,
    device: torch.device,
) -> Tensor:
    model.eval()
    predictions: list[Tensor] = []
    for start in range(0, len(records), batch_size):
        indices = torch.arange(start, min(len(records), start + batch_size))
        batch = _to_device(stack_batch(records, indices), device)
        features, times, action = batch["features"], batch["times"], batch["action_13"]
        assert isinstance(features, Tensor) and isinstance(times, Tensor) and isinstance(action, Tensor)
        with _autocast(device):
            prediction = model.step(features[:, 1], times[:, 1], times[:, 2], action)
        predictions.append(prediction.float().cpu())
    return torch.cat(predictions)


def _record_tensor(records: Sequence[PatientStates], field: str) -> Tensor:
    return torch.stack([getattr(record, field) for record in records])


def _views(features: Tensor) -> tuple[Tensor, Tensor]:
    global_values = features.mean(dim=(3, 4, 5))
    spatial = torch.stack(
        [torch.stack([feature_views(visit)["spatial_pyramid_3x3x3"].flatten() for visit in patient]) for patient in features]
    )
    return global_values.float(), spatial.float()


def _prediction_views(prediction: Tensor) -> tuple[Tensor, Tensor]:
    global_values = prediction.mean(dim=(2, 3, 4))
    spatial = torch.stack([feature_views(item)["spatial_pyramid_3x3x3"].flatten() for item in prediction])
    return global_values.float(), spatial.float()


def _entropy_effective_rank_centered_tokens(predictions: Tensor, device: torch.device) -> float:
    values = predictions.to(device)
    tokens = values.permute(0, 2, 3, 4, 1)
    tokens = tokens - tokens.mean(dim=(1, 2, 3), keepdim=True)
    matrix = tokens.reshape(-1, tokens.shape[-1]).float()
    matrix = matrix - matrix.mean(dim=0, keepdim=True)
    covariance = matrix.T @ matrix / max(1, matrix.shape[0] - 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
    probabilities = eigenvalues / eigenvalues.sum().clamp_min(torch.finfo(eigenvalues.dtype).eps)
    probabilities = probabilities[probabilities > 0]
    return float(torch.exp(-(probabilities * probabilities.log()).sum()).cpu())


def _formal_evaluation(
    model: TokenDynamicsJEPA,
    train: list[PatientStates],
    val: list[PatientStates],
    config: V4TrainingConfig,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    predicted_train = _predict_t3(model, train, config.batch_size, device)
    predicted_val = _predict_t3(model, val, config.batch_size, device)
    train_features = _record_tensor(train, "features")
    val_features = _record_tensor(val, "features")
    global_train, spatial_train = _views(train_features)
    global_val, spatial_val = _views(val_features)
    predicted_global_train, _ = _prediction_views(predicted_train)
    predicted_global_val, predicted_spatial_val = _prediction_views(predicted_val)
    train_ids = [record.patient_id for record in train]
    val_ids = [record.patient_id for record in val]

    global_context_train = torch.cat((global_train[:, 0], global_train[:, 1], global_train[:, 1] - global_train[:, 0]), dim=1)
    global_context_val = torch.cat((global_val[:, 0], global_val[:, 1], global_val[:, 1] - global_val[:, 0]), dim=1)
    global_target = global_val[:, 2] - global_val[:, 1]
    global_rrr_report, global_rrr = _fit_rrr_temporal(
        global_context_train, global_train[:, 2] - global_train[:, 1],
        global_context_val, global_target, train_ids, seed=config.bootstrap_seed + 10,
    )

    spatial_context_train = torch.cat((spatial_train[:, 0], spatial_train[:, 1], spatial_train[:, 1] - spatial_train[:, 0]), dim=1)
    spatial_context_val = torch.cat((spatial_val[:, 0], spatial_val[:, 1], spatial_val[:, 1] - spatial_val[:, 0]), dim=1)
    spatial_target = spatial_val[:, 2] - spatial_val[:, 1]
    spatial_rrr_report, spatial_rrr = _fit_rrr_temporal(
        spatial_context_train, spatial_train[:, 2] - spatial_train[:, 1],
        spatial_context_val, spatial_target, train_ids, seed=config.bootstrap_seed + 20,
    )

    candidate_global = predicted_global_val - global_val[:, 1]
    candidate_spatial = predicted_spatial_val - spatial_val[:, 1]
    full_candidate = predicted_val.flatten(1)
    full_reference = val_features[:, 1].flatten(1)
    full_target = val_features[:, 2].flatten(1)
    persistence = _bootstrap_error_comparison(
        full_candidate, full_reference, full_target, val_ids,
        seed=config.bootstrap_seed + 1, n_bootstrap=config.bootstrap_replicates,
    )
    global_comparison = _bootstrap_error_comparison(
        candidate_global, global_rrr, global_target, val_ids,
        seed=config.bootstrap_seed + 2, n_bootstrap=config.bootstrap_replicates,
    )
    spatial_comparison = _bootstrap_error_comparison(
        candidate_spatial, spatial_rrr, spatial_target, val_ids,
        seed=config.bootstrap_seed + 3, n_bootstrap=config.bootstrap_replicates,
    )
    rank = _entropy_effective_rank_centered_tokens(predicted_val, device)
    gate = evaluate_v4_gate(
        [
            Comparator("persistence", persistence["candidate_mean"], persistence["reference_mean"], persistence["ci_low"]),
            Comparator("global_rrr", global_comparison["candidate_mean"], global_comparison["reference_mean"], global_comparison["ci_low"]),
            Comparator("coarse_spatial_rrr", spatial_comparison["candidate_mean"], spatial_comparison["reference_mean"], spatial_comparison["ci_low"]),
        ],
        effective_rank=rank,
    )

    train_ftv = _record_tensor(train, "ftv")
    val_ftv = _record_tensor(val, "ftv")
    finite_train = torch.isfinite(train_ftv[:, 2])
    finite_val = torch.isfinite(val_ftv[:, 2]) & torch.isfinite(val_ftv[:, 1])
    ftv_report, ftv_prediction = _fit_ridge_probe(
        predicted_global_train[finite_train], torch.log1p(train_ftv[finite_train, 2]),
        predicted_global_val[finite_val], torch.log1p(val_ftv[finite_val, 2]),
        [patient for patient, keep in zip(train_ids, finite_train.tolist()) if keep],
        seed=config.bootstrap_seed + 30,
    )
    ftv_report["vs_t1_no_change"] = _bootstrap_error_comparison(
        ftv_prediction, torch.log1p(val_ftv[finite_val, 1]), torch.log1p(val_ftv[finite_val, 2]),
        [patient for patient, keep in zip(val_ids, finite_val.tolist()) if keep],
        seed=config.bootstrap_seed + 31, n_bootstrap=config.bootstrap_replicates,
    )
    public = {
        "gate": gate,
        "comparisons": {
            "v4_vs_persistence_full_grid": persistence,
            "v4_vs_global_rrr": global_comparison,
            "v4_vs_coarse_spatial_rrr": spatial_comparison,
        },
        "rrr": {"global": global_rrr_report, "coarse_spatial": spatial_rrr_report},
        "detached_ftv_probe": ftv_report,
        "centered_token_effective_rank": rank,
    }
    private = {
        "schema_version": PRIVATE_SCHEMA,
        "patient_ids": val_ids,
        "absolute_errors": {
            "v4_full_grid": (full_candidate - full_target).abs().mean(1),
            "persistence_full_grid": (full_reference - full_target).abs().mean(1),
            "v4_global": (candidate_global - global_target).abs().mean(1),
            "global_rrr": (global_rrr - global_target).abs().mean(1),
            "v4_coarse_spatial": (candidate_spatial - spatial_target).abs().mean(1),
            "coarse_spatial_rrr": (spatial_rrr - spatial_target).abs().mean(1),
        },
    }
    return public, private


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("v4 training requires CUDA")
    config = V4TrainingConfig.load(args.config)
    design = _json(args.frozen_design)
    manifest_report = _json(args.manifest_build_report)
    frozen_audit = _json(args.frozen_audit)
    manifest_sha = _sha(args.manifest)
    if (
        design.get("status") != "frozen_not_trained"
        or design.get("future_pixel_reconstruction") is not False
        or design.get("test_split_loaded") is not False
        or manifest_report.get("status") != "ok"
        or manifest_report.get("manifest_sha256") != manifest_sha
        or manifest_report.get("records") != 587
        or frozen_audit.get("status") != "completed_development_frozen_feature_audit"
        or frozen_audit.get("test_split_loaded") is not False
    ):
        raise ValueError("v4 governed input release is not approved")
    output = Path(args.output_dir).expanduser().resolve()
    private_output = Path(args.private_output_dir).expanduser().resolve()
    if output.exists() or private_output.exists():
        raise FileExistsError("immutable v4 output already exists")
    output.mkdir(parents=True, mode=0o700)
    private_output.mkdir(parents=True, mode=0o700)
    _set_seed(config.seed)
    train = load_development_states(args.manifest, "train")
    val = load_development_states(args.manifest, "val")
    device = torch.device("cuda")
    model = TokenDynamicsJEPA(config.hidden_channels, config.blocks).to(device)
    best_state, history, best_epoch = _train(model, train, val, config, device)
    model.load_state_dict(best_state, strict=True)
    evaluation, private = _formal_evaluation(model, train, val, config, device)

    checkpoint = output / "checkpoint_best.pt"
    torch.save({"schema_version": SCHEMA, "model": best_state, "config": asdict(config), "best_epoch": best_epoch}, checkpoint)
    history_path = output / "metrics.jsonl"
    with history_path.open("x", encoding="utf-8") as handle:
        for row in history:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    private_path = private_output / "validation_errors.pt"
    torch.save(private, private_path)
    evidence = {
        "training_config_sha256": _sha(args.config),
        "frozen_design_sha256": _sha(args.frozen_design),
        "manifest_sha256": manifest_sha,
        "manifest_build_report_sha256": _sha(args.manifest_build_report),
        "frozen_audit_sha256": _sha(args.frozen_audit),
        "checkpoint_sha256": _sha(checkpoint),
        "private_validation_errors_sha256": _sha(private_path),
    }
    report = {
        "schema_version": SCHEMA,
        "status": "completed_development_training",
        "seed": config.seed,
        "patient_counts": {"train": len(train), "val": len(val), "test": 0},
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "selection_metric": "val_masked_t1_t3_smooth_l1",
        "model_contract": {
            "future_pixel_reconstruction": False,
            "image_decoder_present": False,
            "target": "future_frozen_3dino_tokens",
            "ordered_pairs_per_patient": 3,
            "composition_triplets_per_patient": 1,
            "ftv_encoder_gradient": False,
        },
        "validation": evaluation,
        "evidence": evidence,
        "test_split_loaded": False,
        "patient_identifiers_emitted": False,
        "confirmatory_claim_permitted": False,
    }
    report_path = output / "aggregate_report.json"
    with report_path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    for path in (checkpoint, history_path, report_path, private_path):
        os.chmod(path, 0o600)
    print(json.dumps({"status": report["status"], "development_gate_passed": evaluation["gate"]["development_gate_passed"], "test_split_loaded": False}, sort_keys=True))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--frozen-design", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manifest-build-report", required=True)
    parser.add_argument("--frozen-audit", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--private-output-dir", required=True)
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
