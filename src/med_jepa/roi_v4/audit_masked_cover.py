"""Read-only masked-cover audit of a trained Med-JEPA v4 checkpoint.

The first v4 development run was supervised only at masked token locations but
was formally evaluated with an unmasked forward pass.  This audit evaluates the
checkpoint under the condition it was trained for: many deterministic block
masks are drawn, only predictions at masked locations are retained, and those
predictions are averaged into one complete 7x7x7 future-token grid.

No parameter is updated, no held-out test record is loaded, and the original
training report is never changed.  Patient identifiers and patient-level errors
are written only to the governed private output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor, nn

from med_jepa.frozen_audit.run import (
    _bootstrap_error_comparison,
    _fit_rrr_temporal,
)
from med_jepa.roi_v4.data import PatientStates, load_development_states
from med_jepa.roi_v4.gate import Comparator, evaluate_v4_gate
from med_jepa.roi_v4.masking import block_mask
from med_jepa.roi_v4.model import TokenDynamicsJEPA
from med_jepa.roi_v4.train import (
    _prediction_views,
    _record_tensor,
    _views,
)


SCHEMA = "med-jepa-roi-v4-masked-cover-audit-v1"
PRIVATE_SCHEMA = "med-jepa-roi-v4-private-masked-cover-errors-v1"
TRAINING_SCHEMA = "med-jepa-roi-v4-development-training-v1"
DEFAULT_MINIMUM_DRAWS = 32
DEFAULT_MAXIMUM_DRAWS = 128
REPRODUCTION_TOLERANCE = 1e-3


def _sha(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    return value


def _autocast(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return torch.autocast(device_type="cpu", enabled=False)


def _entropy_effective_rank(matrix: Tensor) -> tuple[float, int]:
    """Entropy effective rank using the smaller centered Gram matrix."""

    if matrix.ndim != 2 or not matrix.is_floating_point():
        raise ValueError("effective-rank input must be a floating matrix")
    if matrix.shape[0] < 2 or matrix.shape[1] < 1:
        raise ValueError("effective-rank input is too small")
    values = matrix.float()
    values = values - values.mean(dim=0, keepdim=True)
    maximum = min(values.shape[0] - 1, values.shape[1])
    if values.shape[0] <= values.shape[1]:
        spectrum = torch.linalg.eigvalsh(values @ values.T).clamp_min(0)
    else:
        spectrum = torch.linalg.eigvalsh(values.T @ values).clamp_min(0)
    total = spectrum.sum()
    if float(total) <= torch.finfo(spectrum.dtype).eps:
        return 0.0, maximum
    probabilities = spectrum / total
    probabilities = probabilities[probabilities > 0]
    rank = torch.exp(-(probabilities * probabilities.log()).sum())
    return float(rank.cpu()), maximum


@torch.no_grad()
def representation_rank_diagnostics(values: Tensor, device: torch.device) -> dict[str, float | int]:
    """Report patient-pooled and spatially centered channel effective rank."""

    if values.ndim != 5 or tuple(values.shape[1:]) != (1024, 7, 7, 7):
        raise ValueError("rank values must have shape [N,1024,7,7,7]")
    working = values.to(device=device, dtype=torch.float32)
    pooled = working.mean(dim=(2, 3, 4))
    pooled_rank, pooled_maximum = _entropy_effective_rank(pooled)
    tokens = working.permute(0, 2, 3, 4, 1)
    tokens = tokens - tokens.mean(dim=(1, 2, 3), keepdim=True)
    token_matrix = tokens.reshape(-1, tokens.shape[-1])
    centered_rank, centered_maximum = _entropy_effective_rank(token_matrix)
    channel_variance = token_matrix.var(dim=0, unbiased=False)
    return {
        "patient_pooled_effective_rank": pooled_rank,
        "patient_pooled_maximum_rank": pooled_maximum,
        "patient_pooled_effective_rank_ratio": pooled_rank / pooled_maximum,
        "centered_token_effective_rank": centered_rank,
        "centered_token_maximum_rank": centered_maximum,
        "centered_token_effective_rank_ratio": centered_rank / centered_maximum,
        "centered_token_mean_channel_variance": float(channel_variance.mean().cpu()),
        "tensor_rms": float(working.square().mean().sqrt().cpu()),
    }


@torch.no_grad()
def masked_cover_predict_batch(
    model: nn.Module,
    source: Tensor,
    source_time: Tensor,
    target_time: Tensor,
    action: Tensor,
    *,
    masked_fraction: float,
    seed: int,
    minimum_draws: int = DEFAULT_MINIMUM_DRAWS,
    maximum_draws: int = DEFAULT_MAXIMUM_DRAWS,
) -> tuple[Tensor, Tensor, int]:
    """Average only masked-location predictions until every token is covered."""

    if source.ndim != 5 or tuple(source.shape[1:]) != (1024, 7, 7, 7):
        raise ValueError("source violates the v4 feature-grid contract")
    if minimum_draws < 1 or maximum_draws < minimum_draws:
        raise ValueError("invalid masked-cover draw bounds")
    accumulated = torch.zeros_like(source, dtype=torch.float32)
    coverage = torch.zeros(
        source.shape[0], 1, 7, 7, 7, device=source.device, dtype=torch.int32
    )
    draws_used = 0
    for draw in range(maximum_draws):
        mask = block_mask(
            source.shape[0],
            seed=seed + draw * 1_000_003,
            masked_fraction=masked_fraction,
        ).to(source.device)
        with _autocast(source.device):
            prediction = model.step(source, source_time, target_time, action, mask)
        accumulated.add_(prediction.float() * mask)
        coverage.add_(mask)
        draws_used = draw + 1
        if draws_used >= minimum_draws and bool((coverage > 0).all()):
            break
    if not bool((coverage > 0).all()):
        raise RuntimeError("masked-cover draws did not cover every feature token")
    stitched = accumulated / coverage.to(accumulated.dtype)
    return stitched, coverage[:, 0], draws_used


@torch.no_grad()
def _predict_masked_cover(
    model: TokenDynamicsJEPA,
    records: Sequence[PatientStates],
    *,
    batch_size: int,
    masked_fraction: float,
    seed: int,
    minimum_draws: int,
    maximum_draws: int,
    device: torch.device,
) -> tuple[Tensor, Tensor, list[int]]:
    predictions: list[Tensor] = []
    coverages: list[Tensor] = []
    draws: list[int] = []
    for start in range(0, len(records), batch_size):
        selected = records[start : start + batch_size]
        source = torch.stack([item.features[1] for item in selected]).to(device)
        source_time = torch.stack([item.times_days[1] for item in selected]).to(device)
        target_time = torch.stack([item.times_days[2] for item in selected]).to(device)
        action = torch.stack([item.action_13 for item in selected]).to(device)
        prediction, coverage, used = masked_cover_predict_batch(
            model,
            source,
            source_time,
            target_time,
            action,
            masked_fraction=masked_fraction,
            seed=seed + start * 10_000_019,
            minimum_draws=minimum_draws,
            maximum_draws=maximum_draws,
        )
        predictions.append(prediction.cpu())
        coverages.append(coverage.cpu())
        draws.append(used)
    return torch.cat(predictions), torch.cat(coverages), draws


@torch.no_grad()
def _predict_unmasked(
    model: TokenDynamicsJEPA,
    records: Sequence[PatientStates],
    *,
    batch_size: int,
    device: torch.device,
) -> Tensor:
    predictions: list[Tensor] = []
    for start in range(0, len(records), batch_size):
        selected = records[start : start + batch_size]
        source = torch.stack([item.features[1] for item in selected]).to(device)
        source_time = torch.stack([item.times_days[1] for item in selected]).to(device)
        target_time = torch.stack([item.times_days[2] for item in selected]).to(device)
        action = torch.stack([item.action_13 for item in selected]).to(device)
        with _autocast(device):
            predictions.append(
                model.step(source, source_time, target_time, action).float().cpu()
            )
    return torch.cat(predictions)


def _fit_rrr_baselines(
    train: list[PatientStates],
    val: list[PatientStates],
    *,
    bootstrap_seed: int,
) -> dict[str, Any]:
    train_features = _record_tensor(train, "features")
    val_features = _record_tensor(val, "features")
    global_train, spatial_train = _views(train_features)
    global_val, spatial_val = _views(val_features)
    train_ids = [item.patient_id for item in train]

    global_context_train = torch.cat(
        (global_train[:, 0], global_train[:, 1], global_train[:, 1] - global_train[:, 0]),
        dim=1,
    )
    global_context_val = torch.cat(
        (global_val[:, 0], global_val[:, 1], global_val[:, 1] - global_val[:, 0]),
        dim=1,
    )
    global_target = global_val[:, 2] - global_val[:, 1]
    global_report, global_prediction = _fit_rrr_temporal(
        global_context_train,
        global_train[:, 2] - global_train[:, 1],
        global_context_val,
        global_target,
        train_ids,
        seed=bootstrap_seed + 10,
    )

    spatial_context_train = torch.cat(
        (
            spatial_train[:, 0],
            spatial_train[:, 1],
            spatial_train[:, 1] - spatial_train[:, 0],
        ),
        dim=1,
    )
    spatial_context_val = torch.cat(
        (
            spatial_val[:, 0],
            spatial_val[:, 1],
            spatial_val[:, 1] - spatial_val[:, 0],
        ),
        dim=1,
    )
    spatial_target = spatial_val[:, 2] - spatial_val[:, 1]
    spatial_report, spatial_prediction = _fit_rrr_temporal(
        spatial_context_train,
        spatial_train[:, 2] - spatial_train[:, 1],
        spatial_context_val,
        spatial_target,
        train_ids,
        seed=bootstrap_seed + 20,
    )
    return {
        "train_features": train_features,
        "val_features": val_features,
        "global_val": global_val,
        "spatial_val": spatial_val,
        "global_target": global_target,
        "spatial_target": spatial_target,
        "global_prediction": global_prediction,
        "spatial_prediction": spatial_prediction,
        "reports": {"global": global_report, "coarse_spatial": spatial_report},
    }


def _comparison_bundle(
    prediction: Tensor,
    baselines: dict[str, Any],
    val_ids: Sequence[str],
    *,
    bootstrap_seed: int,
    bootstrap_replicates: int,
) -> tuple[dict[str, Any], dict[str, Tensor]]:
    global_prediction, spatial_prediction = _prediction_views(prediction)
    val_features = baselines["val_features"]
    global_delta = global_prediction - baselines["global_val"][:, 1]
    spatial_delta = spatial_prediction - baselines["spatial_val"][:, 1]
    full_prediction = prediction.flatten(1)
    full_reference = val_features[:, 1].flatten(1)
    full_target = val_features[:, 2].flatten(1)
    persistence = _bootstrap_error_comparison(
        full_prediction,
        full_reference,
        full_target,
        val_ids,
        seed=bootstrap_seed + 1,
        n_bootstrap=bootstrap_replicates,
    )
    global_rrr = _bootstrap_error_comparison(
        global_delta,
        baselines["global_prediction"],
        baselines["global_target"],
        val_ids,
        seed=bootstrap_seed + 2,
        n_bootstrap=bootstrap_replicates,
    )
    spatial_rrr = _bootstrap_error_comparison(
        spatial_delta,
        baselines["spatial_prediction"],
        baselines["spatial_target"],
        val_ids,
        seed=bootstrap_seed + 3,
        n_bootstrap=bootstrap_replicates,
    )
    comparisons = {
        "vs_persistence_full_grid": persistence,
        "vs_global_rrr": global_rrr,
        "vs_coarse_spatial_rrr": spatial_rrr,
    }
    errors = {
        "candidate_full_grid": (full_prediction - full_target).abs().mean(1),
        "persistence_full_grid": (full_reference - full_target).abs().mean(1),
        "candidate_global": (global_delta - baselines["global_target"]).abs().mean(1),
        "global_rrr": (
            baselines["global_prediction"] - baselines["global_target"]
        ).abs().mean(1),
        "candidate_coarse_spatial": (
            spatial_delta - baselines["spatial_target"]
        ).abs().mean(1),
        "coarse_spatial_rrr": (
            baselines["spatial_prediction"] - baselines["spatial_target"]
        ).abs().mean(1),
    }
    return comparisons, errors


def masked_cover_decision(persistence: dict[str, Any]) -> str:
    if float(persistence["ci_low"]) > 0:
        return "masked_cover_beats_persistence_retrain_v4p1"
    if float(persistence["ci_high"]) < 0:
        return "masked_cover_loses_to_persistence_rethink_v4"
    return "masked_cover_vs_persistence_inconclusive"


def _reproduction_check(
    observed: dict[str, Any], training_report: dict[str, Any]
) -> dict[str, Any]:
    original = training_report["validation"]["comparisons"]
    mapping = {
        "vs_persistence_full_grid": "v4_vs_persistence_full_grid",
        "vs_global_rrr": "v4_vs_global_rrr",
        "vs_coarse_spatial_rrr": "v4_vs_coarse_spatial_rrr",
    }
    differences = {
        key: abs(
            float(observed[key]["candidate_mean"])
            - float(original[source]["candidate_mean"])
        )
        for key, source in mapping.items()
    }
    passed = all(value <= REPRODUCTION_TOLERANCE for value in differences.values())
    return {
        "passed": passed,
        "absolute_candidate_mae_differences": differences,
        "tolerance": REPRODUCTION_TOLERANCE,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("masked-cover audit requires CUDA")
    training_report_path = Path(args.training_report).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    private_output = Path(args.private_output_dir).expanduser().resolve()
    if output.exists() or private_output.exists():
        raise FileExistsError("immutable masked-cover audit output already exists")
    training_report = _read_json(training_report_path)
    if (
        training_report.get("schema_version") != TRAINING_SCHEMA
        or training_report.get("status") != "completed_development_training"
        or training_report.get("test_split_loaded") is not False
        or training_report.get("patient_counts")
        != {"train": 409, "val": 88, "test": 0}
    ):
        raise ValueError("training report is not an approved development-only v4 release")
    evidence = training_report.get("evidence", {})
    if evidence.get("checkpoint_sha256") != _sha(checkpoint_path):
        raise ValueError("checkpoint does not match the training report")
    if evidence.get("manifest_sha256") != _sha(manifest_path):
        raise ValueError("manifest does not match the training report")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("schema_version") != TRAINING_SCHEMA
        or not isinstance(checkpoint.get("config"), dict)
        or not isinstance(checkpoint.get("model"), dict)
    ):
        raise ValueError("checkpoint schema is invalid")
    config = checkpoint["config"]
    if config.get("seed") != training_report.get("seed"):
        raise ValueError("checkpoint seed does not match the training report")

    output.mkdir(parents=True, mode=0o700)
    private_output.mkdir(parents=True, mode=0o700)
    device = torch.device("cuda")
    model = TokenDynamicsJEPA(
        hidden_channels=int(config["hidden_channels"]),
        blocks=int(config["blocks"]),
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    train = load_development_states(manifest_path, "train")
    val = load_development_states(manifest_path, "val")
    val_ids = [item.patient_id for item in val]
    batch_size = int(config["batch_size"])
    bootstrap_seed = int(config["bootstrap_seed"])
    bootstrap_replicates = int(config["bootstrap_replicates"])

    masked_prediction, coverage, draws = _predict_masked_cover(
        model,
        val,
        batch_size=batch_size,
        masked_fraction=float(config["masked_fraction"]),
        seed=int(config["seed"]) + 4_000_000,
        minimum_draws=int(args.minimum_draws),
        maximum_draws=int(args.maximum_draws),
        device=device,
    )
    unmasked_prediction = _predict_unmasked(
        model, val, batch_size=batch_size, device=device
    )
    baselines = _fit_rrr_baselines(train, val, bootstrap_seed=bootstrap_seed)
    masked_comparisons, masked_errors = _comparison_bundle(
        masked_prediction,
        baselines,
        val_ids,
        bootstrap_seed=bootstrap_seed + 1_000,
        bootstrap_replicates=bootstrap_replicates,
    )
    unmasked_comparisons, unmasked_errors = _comparison_bundle(
        unmasked_prediction,
        baselines,
        val_ids,
        bootstrap_seed=bootstrap_seed,
        bootstrap_replicates=bootstrap_replicates,
    )
    reproduction = _reproduction_check(unmasked_comparisons, training_report)
    if not reproduction["passed"]:
        raise RuntimeError("checkpoint audit could not reproduce the original unmasked metrics")

    val_features = baselines["val_features"]
    source = val_features[:, 1]
    target = val_features[:, 2]
    representations = {
        "source_t1": source,
        "target_t3": target,
        "masked_cover_prediction_t3": masked_prediction,
        "true_delta_t3_minus_t1": target - source,
        "predicted_delta_t3_minus_t1": masked_prediction - source,
    }
    rank_diagnostics = {
        name: representation_rank_diagnostics(values, device)
        for name, values in representations.items()
    }
    target_rank = float(
        rank_diagnostics["target_t3"]["centered_token_effective_rank"]
    )
    prediction_rank = float(
        rank_diagnostics["masked_cover_prediction_t3"][
            "centered_token_effective_rank"
        ]
    )
    rank_diagnostics["masked_cover_prediction_t3"][
        "effective_rank_relative_to_target"
    ] = prediction_rank / max(target_rank, torch.finfo(torch.float32).eps)

    persistence = masked_comparisons["vs_persistence_full_grid"]
    decision = masked_cover_decision(persistence)
    counterfactual_gate = evaluate_v4_gate(
        [
            Comparator(
                "persistence",
                float(persistence["candidate_mean"]),
                float(persistence["reference_mean"]),
                float(persistence["ci_low"]),
            ),
            Comparator(
                "global_rrr",
                float(masked_comparisons["vs_global_rrr"]["candidate_mean"]),
                float(masked_comparisons["vs_global_rrr"]["reference_mean"]),
                float(masked_comparisons["vs_global_rrr"]["ci_low"]),
            ),
            Comparator(
                "coarse_spatial_rrr",
                float(
                    masked_comparisons["vs_coarse_spatial_rrr"]["candidate_mean"]
                ),
                float(
                    masked_comparisons["vs_coarse_spatial_rrr"]["reference_mean"]
                ),
                float(masked_comparisons["vs_coarse_spatial_rrr"]["ci_low"]),
            ),
        ],
        effective_rank=prediction_rank,
    )

    coverage_float = coverage.float()
    private = {
        "schema_version": PRIVATE_SCHEMA,
        "patient_ids": val_ids,
        "masked_cover_absolute_errors": masked_errors,
        "unmasked_reproduction_absolute_errors": unmasked_errors,
        "coverage_counts": coverage,
        "draws_used_by_batch": draws,
    }
    private_path = private_output / "validation_masked_cover_errors.pt"
    torch.save(private, private_path)
    report = {
        "schema_version": SCHEMA,
        "status": "completed_read_only_masked_cover_audit",
        "scope": "development_only_checkpoint_audit",
        "seed": int(config["seed"]),
        "patient_counts": {"train": len(train), "val": len(val), "test": 0},
        "masked_cover_contract": {
            "mask_generator": "training_block_mask_v1",
            "masked_fraction": float(config["masked_fraction"]),
            "minimum_draws": int(args.minimum_draws),
            "maximum_draws": int(args.maximum_draws),
            "draws_used_min": min(draws),
            "draws_used_max": max(draws),
            "coverage_count_min": int(coverage.min()),
            "coverage_count_max": int(coverage.max()),
            "coverage_count_mean": float(coverage_float.mean()),
            "only_masked_location_predictions_retained": True,
            "repeated_coverage_predictions_averaged": True,
            "all_343_tokens_covered_per_patient": bool((coverage > 0).all()),
        },
        "masked_cover_comparisons": masked_comparisons,
        "unmasked_reproduction": {
            "check": reproduction,
            "comparisons": unmasked_comparisons,
        },
        "rrr": baselines["reports"],
        "effective_rank_diagnostics": rank_diagnostics,
        "decision": decision,
        "counterfactual_masked_cover_gate": counterfactual_gate,
        "interpretation_contract": {
            "if_persistence_ci_low_positive": (
                "masked_objective_signal_present_and_v4p1_retraining_is_justified"
            ),
            "if_persistence_ci_high_negative": (
                "masked_objective_failed_and_v4_requires_redesign"
            ),
            "otherwise": "masked_objective_result_is_inconclusive",
            "original_formal_gate_changed": False,
        },
        "retraining_performed": False,
        "optimizer_created": False,
        "checkpoint_parameters_modified": False,
        "future_pixel_reconstruction": False,
        "test_split_loaded": False,
        "patient_identifiers_emitted": False,
        "confirmatory_claim_permitted": False,
        "evidence": {
            "training_report_sha256": _sha(training_report_path),
            "checkpoint_sha256": _sha(checkpoint_path),
            "manifest_sha256": _sha(manifest_path),
            "audit_source_sha256": _sha(Path(__file__)),
            "private_validation_errors_sha256": _sha(private_path),
        },
    }
    report_path = output / "aggregate_report.json"
    with report_path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    for path in (report_path, private_path):
        os.chmod(path, 0o600)
    print(
        json.dumps(
            {
                "status": report["status"],
                "decision": decision,
                "masked_cover_improvement_over_persistence": persistence[
                    "improvement"
                ],
                "masked_cover_improvement_95ci": [
                    persistence["ci_low"],
                    persistence["ci_high"],
                ],
                "test_split_loaded": False,
            },
            sort_keys=True,
        )
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-report", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--private-output-dir", required=True)
    parser.add_argument("--minimum-draws", type=int, default=DEFAULT_MINIMUM_DRAWS)
    parser.add_argument("--maximum-draws", type=int, default=DEFAULT_MAXIMUM_DRAWS)
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "masked_cover_decision",
    "masked_cover_predict_batch",
    "representation_rank_diagnostics",
    "run",
]
