"""Train and evaluate shrinkage-anchored Med-JEPA v4.2 without test access."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import statistics
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

from med_jepa.roi_v4.audit_masked_cover import representation_rank_diagnostics
from med_jepa.roi_v4.audit_mechanistic import (
    REGIONS,
    _summary,
    patient_region_mae,
)
from med_jepa.roi_v4.data import PatientStates, stack_batch
from med_jepa.roi_v4p2.config import V42TrainingConfig
from med_jepa.roi_v4p2.data import (
    load_selected_train_release,
    materialize_selected_validation_release,
)
from med_jepa.roi_v4p2.feature_triage import (
    ALPHAS,
    SCHEMA as TRIAGE_SCHEMA,
    _folds,
)
from med_jepa.roi_v4p2.gate import evaluate_v4p2_gate
from med_jepa.roi_v4p2.lock import (
    SCHEMA as LOCK_SCHEMA,
    SELECTED_VARIANT,
)
from med_jepa.roi_v4p2.losses import occupancy_weighted_masked_latent_loss
from med_jepa.roi_v4p2.masking import scheduled_uniform_masks, uniform_block_mask
from med_jepa.roi_v4p2.model import ShrinkageAnchoredTokenDynamicsJEPA


SCHEMA = "med-jepa-roi-v4p2-development-training-v1"
PRIVATE_SCHEMA = "med-jepa-roi-v4p2-private-validation-measurements-v1"
SELECTION_SCHEMA = "med-jepa-roi-v4p2-train-only-selection-v1"
PAIR_KEYS = {(0, 1): "T0_to_T1", (0, 2): "T0_to_T3", (1, 2): "T1_to_T3"}
PAIRS = tuple(PAIR_KEYS)


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


def _autocast():
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def _to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, Tensor) else value
        for key, value in batch.items()
    }


def _interval_action(batch: Mapping[str, Any], pair: tuple[int, int]) -> Tensor:
    key = {(0, 1): "action_01", (0, 2): "action_03", (1, 2): "action_13"}[pair]
    value = batch[key]
    if not isinstance(value, Tensor):
        raise ValueError("interval action is not a tensor")
    return value


def _record_features(records: Sequence[PatientStates]) -> Tensor:
    return torch.stack([item.features for item in records])


def _region_errors(prediction: Tensor, target: Tensor, weights: Tensor) -> Tensor:
    return torch.stack(
        [
            patient_region_mae(prediction, target, weights[:, index])
            for index in range(len(REGIONS))
        ],
        dim=1,
    )


def _mean_visit(features: Tensor, indices: Tensor, visit: int) -> Tensor:
    total = torch.zeros_like(features[0, visit], dtype=torch.float32)
    for index in indices.tolist():
        total.add_(features[index, visit].float())
    return total.div(float(len(indices))).unsqueeze(0)


def _select_pair_alpha(
    features: Tensor,
    weights: Tensor,
    indices: Tensor,
    pair: tuple[int, int],
    *,
    device: torch.device,
    batch_size: int = 2,
) -> tuple[float, dict[str, float]]:
    """Leave-one-patient-out alpha selection inside one training fold."""

    source_visit, target_visit = pair
    disease = REGIONS.index("t0_disease_bed")
    target_mean = _mean_visit(features, indices, target_visit)[0]
    target_sum = target_mean * float(len(indices))
    score_sum = torch.zeros(len(ALPHAS), dtype=torch.float64)
    score_count = torch.zeros(len(ALPHAS), dtype=torch.long)
    alpha_grid = torch.tensor(
        ALPHAS, device=device, dtype=torch.float32
    )[:, None, None, None, None, None]
    for start in range(0, len(indices), batch_size):
        selected = indices[start : start + batch_size]
        source = features[selected, source_visit].float().to(device)
        target = features[selected, target_visit].float().to(device)
        leave_one_out = (
            target_sum[None].to(device) - target
        ) / float(len(indices) - 1)
        prediction = alpha_grid * source[None] + (1.0 - alpha_grid) * leave_one_out[None]
        token_error = (prediction - target[None]).abs().mean(2)
        region = weights[selected, disease].float().to(device)
        denominator = region.flatten(1).sum(1)
        eligible = denominator > 0
        values = (
            (token_error[:, eligible] * region[eligible][None]).flatten(2).sum(2)
            / denominator[eligible][None]
        )
        score_sum.add_(values.double().sum(1).cpu())
        score_count.add_(int(eligible.sum()))
    scores = score_sum / score_count.clamp_min(1)
    selected_index = min(
        range(len(ALPHAS)), key=lambda index: (float(scores[index]), ALPHAS[index])
    )
    return ALPHAS[selected_index], {
        f"{alpha:.4f}": float(scores[index])
        for index, alpha in enumerate(ALPHAS)
    }


def _fit_anchors(
    features: Tensor,
    weights: Tensor,
    indices: Tensor,
    *,
    device: torch.device,
) -> tuple[dict[str, float], dict[str, dict[str, float]], dict[int, Tensor]]:
    alphas: dict[str, float] = {}
    scores: dict[str, dict[str, float]] = {}
    for pair in PAIRS:
        key = PAIR_KEYS[pair]
        alphas[key], scores[key] = _select_pair_alpha(
            features, weights, indices, pair, device=device
        )
    means = {
        1: _mean_visit(features, indices, 1),
        2: _mean_visit(features, indices, 2),
    }
    return alphas, scores, means


def _anchor_for_pair(
    pair: tuple[int, int],
    alphas: Mapping[str, float],
    means: Mapping[int, Tensor],
    device: torch.device,
) -> tuple[float, Tensor]:
    return (
        float(alphas[PAIR_KEYS[pair]]),
        means[pair[1]].to(device, non_blocking=True),
    )


def _train_epoch(
    model: ShrinkageAnchoredTokenDynamicsJEPA,
    optimizer: torch.optim.Optimizer,
    records: list[PatientStates],
    weights: Tensor,
    indices: Tensor,
    alphas: Mapping[str, float],
    means: Mapping[int, Tensor],
    config: V42TrainingConfig,
    occupancy_lambda: float,
    *,
    epoch: int,
    device: torch.device,
) -> float:
    model.train()
    generator = torch.Generator().manual_seed(config.seed + epoch * 100_003)
    order = indices[torch.randperm(len(indices), generator=generator)]
    totals: list[float] = []
    disease = REGIONS.index("t0_disease_bed")
    for start in range(0, len(order), config.batch_size):
        selected = order[start : start + config.batch_size]
        batch = _to_device(stack_batch(records, selected), device)
        features = batch["features"]
        times = batch["times"]
        if not isinstance(features, Tensor) or not isinstance(times, Tensor):
            raise ValueError("training batch tensors are unavailable")
        occupancy = weights[selected, disease, None].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with _autocast():
            latent_terms: list[Tensor] = []
            for pair_index, pair in enumerate(PAIRS):
                source_visit, target_visit = pair
                context_mask, supervision_mask, _ = scheduled_uniform_masks(
                    len(selected),
                    tuple(int(value) for value in features.shape[-3:]),
                    seed=(
                        config.seed
                        + epoch * 1_000_003
                        + start * 31
                        + pair_index * 101
                    ),
                    fractions=config.mask_fractions,
                )
                alpha, mean = _anchor_for_pair(pair, alphas, means, device)
                prediction = model.step(
                    features[:, source_visit],
                    mean,
                    alpha,
                    times[:, source_visit],
                    times[:, target_visit],
                    _interval_action(batch, pair),
                    context_mask.to(device),
                )
                latent_terms.append(
                    occupancy_weighted_masked_latent_loss(
                        prediction,
                        features[:, target_visit],
                        supervision_mask.to(device),
                        occupancy,
                        uniform_floor_lambda=occupancy_lambda,
                    )
                )
            direct, composed = model.compose(
                features[:, 0],
                means[1].to(device),
                means[2].to(device),
                alphas[PAIR_KEYS[(0, 1)]],
                alphas[PAIR_KEYS[(1, 2)]],
                alphas[PAIR_KEYS[(0, 2)]],
                times,
                _interval_action(batch, (0, 1)),
                _interval_action(batch, (1, 2)),
                _interval_action(batch, (0, 2)),
            )
            full = torch.ones(
                len(selected),
                1,
                *features.shape[-3:],
                dtype=torch.bool,
                device=device,
            )
            composition = occupancy_weighted_masked_latent_loss(
                direct,
                composed,
                full,
                occupancy,
                uniform_floor_lambda=occupancy_lambda,
            )
            total = (
                config.latent_weight * torch.stack(latent_terms).mean()
                + config.semigroup_weight * composition
            )
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        totals.append(float(total.detach()))
    return sum(totals) / len(totals)


@torch.no_grad()
def _predict_mask_ensemble(
    model: ShrinkageAnchoredTokenDynamicsJEPA,
    records: Sequence[PatientStates],
    indices: Tensor,
    alpha: float,
    mean_target: Tensor,
    config: V42TrainingConfig,
    *,
    seed: int,
    device: torch.device,
) -> Tensor:
    model.eval()
    rows: list[Tensor] = []
    for start in range(0, len(indices), config.batch_size):
        selected_indices = indices[start : start + config.batch_size]
        selected = [records[int(index)] for index in selected_indices.tolist()]
        source = torch.stack([item.features[1] for item in selected]).to(device)
        source_time = torch.stack([item.times_days[1] for item in selected]).to(device)
        target_time = torch.stack([item.times_days[2] for item in selected]).to(device)
        action = torch.stack([item.action_13 for item in selected]).to(device)
        predictions: list[Tensor] = []
        for fraction_index, fraction in enumerate(config.mask_fractions):
            for draw in range(config.evaluation_mask_draws_per_fraction):
                mask = uniform_block_mask(
                    len(selected),
                    (7, 7, 7),
                    seed=(
                        seed
                        + start * 10_007
                        + fraction_index * 1_009
                        + draw * 97
                    ),
                    masked_fraction=fraction,
                ).to(device)
                with _autocast():
                    predictions.append(
                        model.step(
                            source,
                            mean_target.to(device),
                            alpha,
                            source_time,
                            target_time,
                            action,
                            mask,
                        ).float()
                    )
        rows.append(torch.stack(predictions).mean(0).cpu())
    return torch.cat(rows)


def _fit_fold(
    records: list[PatientStates],
    features: Tensor,
    weights: Tensor,
    train_indices: Tensor,
    held_indices: Tensor,
    alphas: Mapping[str, float],
    means: Mapping[int, Tensor],
    config: V42TrainingConfig,
    occupancy_lambda: float,
    *,
    fold: int,
    device: torch.device,
) -> tuple[dict[str, Tensor], int, float]:
    _set_seed(config.seed + fold * 10_000)
    model = ShrinkageAnchoredTokenDynamicsJEPA(
        config.hidden_channels, config.blocks
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    disease = REGIONS.index("t0_disease_bed")
    best_state: dict[str, Tensor] | None = None
    best_epoch = -1
    best_mae = math.inf
    patience = 0
    for epoch in range(config.maximum_cv_epochs):
        train_loss = _train_epoch(
            model,
            optimizer,
            records,
            weights,
            train_indices,
            alphas,
            means,
            config,
            occupancy_lambda,
            epoch=epoch,
            device=device,
        )
        prediction = _predict_mask_ensemble(
            model,
            records,
            held_indices,
            alphas[PAIR_KEYS[(1, 2)]],
            means[2],
            config,
            # Keep the held-fold mask ensemble fixed across epochs so early
            # stopping measures model change rather than mask-sampling noise.
            seed=config.seed + fold * 100_000 + 313,
            device=device,
        )
        target = features[held_indices, 2]
        error = patient_region_mae(
            prediction, target, weights[held_indices, disease]
        )
        value = float(torch.nanmean(error))
        print(
            json.dumps(
                {
                    "stage": "train_only_cv",
                    "fold": fold,
                    "occupancy_lambda": occupancy_lambda,
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "held_disease_bed_MAE": value,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if value < best_mae - config.early_stopping_min_delta:
            best_mae = value
            best_epoch = epoch
            best_state = {
                key: tensor.detach().cpu().clone()
                for key, tensor in model.state_dict().items()
            }
            patience = 0
        else:
            patience += 1
            if patience >= config.early_stopping_patience:
                break
    if best_state is None:
        raise RuntimeError("v4.2 fold training produced no checkpoint")
    del model, optimizer
    torch.cuda.empty_cache()
    return best_state, best_epoch + 1, best_mae


def choose_occupancy_lambda(
    summaries: Mapping[float, Mapping[str, float]],
) -> float:
    """Apply the frozen mean-metric and tie-break ordering."""

    if not summaries:
        raise ValueError("occupancy lambda summaries are empty")
    return min(
        summaries,
        key=lambda value: (
            summaries[value]["disease_candidate_to_shrinkage_MAE_ratio"],
            -summaries[value]["disease_minus_background_relative_improvement"],
            summaries[value]["full_grid_candidate_MAE"],
            -float(value),
        ),
    )


def _cross_validate(
    records: list[PatientStates],
    weights: Tensor,
    config: V42TrainingConfig,
    device: torch.device,
    *,
    resume_dir: Path,
    resume_contract: Mapping[str, Any],
) -> tuple[dict[str, Any], float, int, dict[str, float], dict[int, Tensor]]:
    features = _record_features(records)
    folds = _folds(409, seed=config.seed, count=config.grouped_cv_folds)
    candidate_errors = {
        value: torch.full((409, len(REGIONS)), float("nan"))
        for value in config.occupancy_lambda_candidates
    }
    shrinkage_errors = {
        value: torch.full((409, len(REGIONS)), float("nan"))
        for value in config.occupancy_lambda_candidates
    }
    best_epochs = {value: [] for value in config.occupancy_lambda_candidates}
    fold_anchor_evidence: list[dict[str, Any]] = []
    resume_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    for fold, (train_indices, held_indices) in enumerate(folds):
        alphas, alpha_scores, means = _fit_anchors(
            features, weights, train_indices, device=device
        )
        fold_anchor_evidence.append(
            {"fold": fold, "alphas": alphas, "leave_one_out_scores": alpha_scores}
        )
        source = features[held_indices, 1]
        target = features[held_indices, 2]
        shrinkage = (
            alphas[PAIR_KEYS[(1, 2)]] * source
            + (1.0 - alphas[PAIR_KEYS[(1, 2)]]) * means[2]
        )
        fold_shrinkage_errors = _region_errors(
            shrinkage, target, weights[held_indices]
        )
        for occupancy_lambda in config.occupancy_lambda_candidates:
            cache_path = resume_dir / (
                f"fold-{fold:02d}-lambda-{occupancy_lambda:.4f}.pt"
            )
            if cache_path.is_file():
                cache = torch.load(cache_path, map_location="cpu", weights_only=True)
                if (
                    not isinstance(cache, Mapping)
                    or cache.get("schema_version")
                    != "med-jepa-roi-v4p2-private-cv-fold-v1"
                    or cache.get("resume_contract") != dict(resume_contract)
                    or cache.get("fold") != fold
                    or cache.get("occupancy_lambda") != occupancy_lambda
                    or not torch.equal(cache.get("held_indices"), held_indices)
                ):
                    raise ValueError("resumable v4.2 CV fold does not match")
                fold_candidate_errors = cache.get("candidate_errors")
                cached_shrinkage_errors = cache.get("shrinkage_errors")
                epoch_count = int(cache.get("best_epoch_count", 0))
                if (
                    not isinstance(fold_candidate_errors, Tensor)
                    or fold_candidate_errors.shape != (len(held_indices), len(REGIONS))
                    or not isinstance(cached_shrinkage_errors, Tensor)
                    or not torch.equal(
                        cached_shrinkage_errors, fold_shrinkage_errors
                    )
                    or epoch_count < 1
                ):
                    raise ValueError("resumable v4.2 CV fold payload is malformed")
                print(
                    json.dumps(
                        {
                            "stage": "train_only_cv_resume",
                            "fold": fold,
                            "occupancy_lambda": occupancy_lambda,
                            "best_epoch_count": epoch_count,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            else:
                state, epoch_count, _ = _fit_fold(
                    records,
                    features,
                    weights,
                    train_indices,
                    held_indices,
                    alphas,
                    means,
                    config,
                    occupancy_lambda,
                    fold=fold,
                    device=device,
                )
                model = ShrinkageAnchoredTokenDynamicsJEPA(
                    config.hidden_channels, config.blocks
                ).to(device)
                model.load_state_dict(state, strict=True)
                prediction = _predict_mask_ensemble(
                    model,
                    records,
                    held_indices,
                    alphas[PAIR_KEYS[(1, 2)]],
                    means[2],
                    config,
                    seed=config.seed + fold * 1_000_000 + 777,
                    device=device,
                )
                fold_candidate_errors = _region_errors(
                    prediction, target, weights[held_indices]
                )
                temporary = cache_path.with_suffix(".partial")
                torch.save(
                    {
                        "schema_version": "med-jepa-roi-v4p2-private-cv-fold-v1",
                        "resume_contract": dict(resume_contract),
                        "fold": fold,
                        "occupancy_lambda": occupancy_lambda,
                        "held_indices": held_indices,
                        "candidate_errors": fold_candidate_errors,
                        "shrinkage_errors": fold_shrinkage_errors,
                        "best_epoch_count": epoch_count,
                    },
                    temporary,
                )
                os.chmod(temporary, 0o600)
                os.replace(temporary, cache_path)
                del model, state, prediction
                torch.cuda.empty_cache()
            candidate_errors[occupancy_lambda][held_indices] = fold_candidate_errors
            shrinkage_errors[occupancy_lambda][held_indices] = fold_shrinkage_errors
            best_epochs[occupancy_lambda].append(epoch_count)
    disease = REGIONS.index("t0_disease_bed")
    background = REGIONS.index("valid_non_disease_bed")
    full = REGIONS.index("full_shared_valid_grid")
    summaries: dict[float, dict[str, float]] = {}
    gates: dict[str, Any] = {}
    patient_ids = [item.patient_id for item in records]
    for counter, occupancy_lambda in enumerate(config.occupancy_lambda_candidates):
        candidate = candidate_errors[occupancy_lambda]
        shrinkage = shrinkage_errors[occupancy_lambda]
        if not torch.isfinite(candidate[:, 2:]).all():
            raise RuntimeError("v4.2 cross-validation left missing predictions")
        disease_candidate = float(candidate[:, disease].mean())
        disease_reference = float(shrinkage[:, disease].mean())
        background_gain = 1.0 - float(
            candidate[:, background].mean() / shrinkage[:, background].mean()
        )
        disease_gain = 1.0 - disease_candidate / disease_reference
        summaries[occupancy_lambda] = {
            "disease_candidate_MAE": disease_candidate,
            "disease_shrinkage_MAE": disease_reference,
            "disease_candidate_to_shrinkage_MAE_ratio": (
                disease_candidate / disease_reference
            ),
            "disease_relative_improvement": disease_gain,
            "background_relative_improvement": background_gain,
            "disease_minus_background_relative_improvement": (
                disease_gain - background_gain
            ),
            "full_grid_candidate_MAE": float(candidate[:, full].mean()),
        }
        gates[f"{occupancy_lambda:.4f}"] = evaluate_v4p2_gate(
            candidate,
            shrinkage,
            patient_ids,
            seed=config.bootstrap_seed + counter * 100,
            replicates=config.bootstrap_replicates,
        )
    selected = choose_occupancy_lambda(summaries)
    final_epochs = max(
        1, int(round(statistics.median(best_epochs[selected])))
    )
    all_indices = torch.arange(409)
    final_alphas, final_alpha_scores, final_means = _fit_anchors(
        features, weights, all_indices, device=device
    )
    public = {
        "grouped_folds": config.grouped_cv_folds,
        "candidate_summaries": {
            f"{key:.4f}": value for key, value in summaries.items()
        },
        "candidate_oof_gates": gates,
        "best_epoch_count_by_lambda": {
            f"{key:.4f}": value for key, value in best_epochs.items()
        },
        "fold_anchor_evidence": fold_anchor_evidence,
        "selected_occupancy_lambda": selected,
        "selected_final_epoch_count": final_epochs,
        "final_pair_specific_alphas": final_alphas,
        "final_pair_specific_leave_one_out_scores": final_alpha_scores,
    }
    return public, selected, final_epochs, final_alphas, final_means


def _train_final(
    records: list[PatientStates],
    weights: Tensor,
    config: V42TrainingConfig,
    occupancy_lambda: float,
    epochs: int,
    alphas: Mapping[str, float],
    means: Mapping[int, Tensor],
    device: torch.device,
) -> tuple[dict[str, Tensor], list[dict[str, float]]]:
    _set_seed(config.seed)
    model = ShrinkageAnchoredTokenDynamicsJEPA(
        config.hidden_channels, config.blocks
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    indices = torch.arange(409)
    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        loss = _train_epoch(
            model,
            optimizer,
            records,
            weights,
            indices,
            alphas,
            means,
            config,
            occupancy_lambda,
            epoch=epoch,
            device=device,
        )
        row = {"epoch": float(epoch), "train_total_loss": loss}
        history.append(row)
        print(json.dumps({"stage": "final_train", **row}, sort_keys=True), flush=True)
    state = {
        key: tensor.detach().cpu().clone()
        for key, tensor in model.state_dict().items()
    }
    del model, optimizer
    torch.cuda.empty_cache()
    return state, history


def _validation_evaluation(
    model: ShrinkageAnchoredTokenDynamicsJEPA,
    records: list[PatientStates],
    weights: Tensor,
    config: V42TrainingConfig,
    alphas: Mapping[str, float],
    means: Mapping[int, Tensor],
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    indices = torch.arange(88)
    prediction = _predict_mask_ensemble(
        model,
        records,
        indices,
        alphas[PAIR_KEYS[(1, 2)]],
        means[2],
        config,
        seed=config.bootstrap_seed + 500_000,
        device=device,
    )
    features = _record_features(records)
    source, target = features[:, 1], features[:, 2]
    shrinkage = (
        alphas[PAIR_KEYS[(1, 2)]] * source
        + (1.0 - alphas[PAIR_KEYS[(1, 2)]]) * means[2]
    )
    predictions = {
        "shrinkage_anchored_med_jepa_v4p2": prediction,
        "train_only_shrinkage_to_mean_t3": shrinkage,
        "latent_persistence": source,
    }
    errors = {
        name: _region_errors(value, target, weights)
        for name, value in predictions.items()
    }
    summaries = {
        name: {
            region: _summary(value[:, index])
            for index, region in enumerate(REGIONS)
        }
        for name, value in errors.items()
    }
    patient_ids = [item.patient_id for item in records]
    gate = evaluate_v4p2_gate(
        errors["shrinkage_anchored_med_jepa_v4p2"],
        errors["train_only_shrinkage_to_mean_t3"],
        patient_ids,
        seed=config.bootstrap_seed,
        replicates=config.bootstrap_replicates,
    )
    ranks = {
        "source_T1": representation_rank_diagnostics(source, device),
        "target_T3": representation_rank_diagnostics(target, device),
        "prediction_T3": representation_rank_diagnostics(prediction, device),
        "shrinkage_T3": representation_rank_diagnostics(shrinkage, device),
        "true_delta": representation_rank_diagnostics(target - source, device),
        "predicted_delta": representation_rank_diagnostics(
            prediction - source, device
        ),
    }
    public = {
        "estimand": "locked_local64_resampled_7cube_mask_ensemble_T1_to_T3",
        "region_MAE": summaries,
        "development_gate": gate,
        "representation_rank": ranks,
        "validation_used_for_checkpoint_selection": False,
        "validation_used_for_hyperparameter_selection": False,
    }
    private = {
        "schema_version": PRIVATE_SCHEMA,
        "patient_ids": patient_ids,
        "patient_region_errors": errors,
        "prediction": prediction.half(),
        "shrinkage_prediction": shrinkage.half(),
    }
    return public, private


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("v4.2 training requires exactly one CUDA GPU")
    if args.license_acknowledgement != (
        "I_ACCEPT_CC_BY_NC_ND_4_0_FOR_AUTHORIZED_NONCOMMERCIAL_RESEARCH"
    ):
        raise ValueError("explicit 3DINO license acknowledgement is required")
    paths = {
        name: Path(value).expanduser().resolve()
        for name, value in {
            "config": args.config,
            "protocol_lock": args.protocol_lock,
            "frozen_design": args.frozen_design,
            "triage_report": args.feature_triage_report,
            "triage_private_root": args.feature_triage_private_root,
            "triage_private_index": args.feature_triage_private_index,
            "manifest": args.manifest,
            "manifest_report": args.manifest_build_report,
            "roi_root": args.roi_root,
            "roi_index": args.roi_index,
            "mask_root": args.context_mask_release,
            "mask_report": args.context_mask_report,
            "mask_index": args.context_mask_index,
            "three_dino_repo": args.three_dino_repo,
            "checkpoint": args.checkpoint,
            "output": args.output_dir,
            "private_output": args.private_output_dir,
        }.items()
    }
    final_output = paths["output"]
    final_private_output = paths["private_output"]
    if final_output.exists() or final_private_output.exists():
        raise FileExistsError("immutable v4.2 training output exists")
    paths["output"] = final_output.parent / f".{final_output.name}.staging"
    paths["private_output"] = (
        final_private_output.parent / f".{final_private_output.name}.staging"
    )
    config = V42TrainingConfig.load(paths["config"])
    lock = _json(paths["protocol_lock"])
    triage = _json(paths["triage_report"])
    manifest_report = _json(paths["manifest_report"])
    lock_evidence = lock.get("evidence", {})
    triage_evidence = triage.get("evidence", {})
    if (
        lock.get("schema_version") != LOCK_SCHEMA
        or lock.get("status") != "locked_train_only_cv_required_before_validation"
        or lock.get("selected_feature_variant") != SELECTED_VARIANT
        or lock.get("validation_88_loaded") is not False
        or lock.get("test_split_loaded") is not False
        or lock_evidence.get("config_sha256") != _sha(paths["config"])
        or lock_evidence.get("frozen_design_sha256") != _sha(paths["frozen_design"])
        or lock_evidence.get("feature_triage_report_sha256")
        != _sha(paths["triage_report"])
        or lock_evidence.get("feature_triage_private_index_sha256")
        != _sha(paths["triage_private_index"])
        or triage.get("schema_version") != TRIAGE_SCHEMA
        or triage.get("selection", {}).get("selected_variant") != SELECTED_VARIANT
        or triage_evidence.get("private_index_sha256")
        != _sha(paths["triage_private_index"])
        or manifest_report.get("status") != "ok"
        or manifest_report.get("manifest_sha256") != _sha(paths["manifest"])
        or _sha(paths["checkpoint"]) != args.expected_checkpoint_sha256
    ):
        raise ValueError("v4.2 training prerequisite chain is not approved")
    paths["output"].mkdir(parents=True, exist_ok=True, mode=0o700)
    paths["private_output"].mkdir(parents=True, exist_ok=True, mode=0o700)
    device = torch.device("cuda")
    train, train_weights, train_payload_merkle = load_selected_train_release(
        paths["manifest"],
        paths["triage_private_root"],
        paths["triage_private_index"],
        expected_index_sha256=_sha(paths["triage_private_index"]),
    )
    resume_contract = {
        "config_sha256": _sha(paths["config"]),
        "protocol_lock_sha256": _sha(paths["protocol_lock"]),
        "train_feature_payload_merkle_sha256": train_payload_merkle,
        "training_source_sha256": _sha(Path(__file__)),
    }
    cv, occupancy_lambda, final_epochs, alphas, means = _cross_validate(
        train,
        train_weights,
        config,
        device,
        resume_dir=paths["private_output"] / "cv_folds",
        resume_contract=resume_contract,
    )
    selection_report = {
        "schema_version": SELECTION_SCHEMA,
        "status": "locked_train_only_hyperparameters_before_validation",
        "patients": {"train": 409, "validation_loaded": 0, "test_loaded": 0},
        "selected_feature_variant": SELECTED_VARIANT,
        "train_only_cross_validation": cv,
        "selected_occupancy_lambda": occupancy_lambda,
        "selected_final_epoch_count": final_epochs,
        "selected_pair_specific_shrinkage_alphas": alphas,
        "validation_88_loaded": False,
        "test_split_loaded": False,
        "evidence": {
            "protocol_lock_sha256": _sha(paths["protocol_lock"]),
            "train_feature_payload_merkle_sha256": train_payload_merkle,
            "training_source_sha256": _sha(Path(__file__)),
        },
    }
    selection_path = paths["output"] / "train_only_selection.json"
    if selection_path.is_file():
        if _json(selection_path) != selection_report:
            raise ValueError("resumable train-only selection release changed")
    else:
        with selection_path.open("x", encoding="utf-8") as handle:
            json.dump(selection_report, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.chmod(selection_path, 0o600)
    final_fit_cache = paths["private_output"] / "final_fit_state.pt"
    final_fit_contract = {
        **resume_contract,
        "train_only_selection_sha256": _sha(selection_path),
    }
    if final_fit_cache.is_file():
        fit = torch.load(final_fit_cache, map_location="cpu", weights_only=True)
        if (
            not isinstance(fit, Mapping)
            or fit.get("schema_version")
            != "med-jepa-roi-v4p2-private-final-fit-v1"
            or fit.get("resume_contract") != final_fit_contract
            or not isinstance(fit.get("model"), Mapping)
            or not isinstance(fit.get("history"), list)
        ):
            raise ValueError("resumable v4.2 final fit does not match")
        state = dict(fit["model"])
        history = list(fit["history"])
        print(json.dumps({"stage": "final_train_resume"}, sort_keys=True), flush=True)
    else:
        state, history = _train_final(
            train,
            train_weights,
            config,
            occupancy_lambda,
            final_epochs,
            alphas,
            means,
            device,
        )
        temporary = final_fit_cache.with_suffix(".partial")
        torch.save(
            {
                "schema_version": "med-jepa-roi-v4p2-private-final-fit-v1",
                "resume_contract": final_fit_contract,
                "model": state,
                "history": history,
            },
            temporary,
        )
        os.chmod(temporary, 0o600)
        os.replace(temporary, final_fit_cache)
    # The validation split is first opened after train-only selection has been
    # made immutable and the final fit has completed without validation input.
    extraction_contract = {
        "selected_feature_variant": SELECTED_VARIANT,
        "train_only_selection_sha256": _sha(selection_path),
        "manifest_sha256": _sha(paths["manifest"]),
        "roi_index_sha256": _sha(paths["roi_index"]),
        "context_mask_index_sha256": _sha(paths["mask_index"]),
        "three_dino_checkpoint_sha256": _sha(paths["checkpoint"]),
        "feature_triage_preprocessing_source_sha256": triage_evidence.get(
            "preprocessing_source_sha256"
        ),
    }
    validation_root = paths["private_output"] / "validation_features"
    val, val_weights, val_payload_merkle, val_index_path = (
        materialize_selected_validation_release(
            manifest_path=paths["manifest"],
            roi_root=paths["roi_root"],
            roi_index_path=paths["roi_index"],
            mask_root=paths["mask_root"],
            mask_report_path=paths["mask_report"],
            mask_index_path=paths["mask_index"],
            three_dino_repo=paths["three_dino_repo"],
            checkpoint_path=paths["checkpoint"],
            expected_checkpoint_sha256=args.expected_checkpoint_sha256,
            output_root=validation_root,
            device=device,
            autocast_dtype=torch.bfloat16,
            extraction_contract=extraction_contract,
        )
    )
    gc.collect()
    torch.cuda.empty_cache()
    model = ShrinkageAnchoredTokenDynamicsJEPA(
        config.hidden_channels, config.blocks
    ).to(device)
    model.load_state_dict(state, strict=True)
    evaluation, private = _validation_evaluation(
        model, val, val_weights, config, alphas, means, device
    )
    checkpoint_path = paths["output"] / "checkpoint_best.pt"
    torch.save(
        {
            "schema_version": SCHEMA,
            "model": state,
            "config": asdict(config),
            "selected_feature_variant": SELECTED_VARIANT,
            "occupancy_lambda": occupancy_lambda,
            "epochs": final_epochs,
            "pair_specific_shrinkage_alphas": alphas,
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
    passed = bool(evaluation["development_gate"]["development_gate_passed"])
    report = {
        "schema_version": SCHEMA,
        "status": (
            "completed_development_training_gate_pass"
            if passed
            else "completed_development_training_gate_fail"
        ),
        "seed": config.seed,
        "patient_counts": {"train": 409, "val": 88, "test": 0},
        "selected_feature_variant": SELECTED_VARIANT,
        "selected_occupancy_lambda": occupancy_lambda,
        "final_epoch_count": final_epochs,
        "pair_specific_shrinkage_alphas": alphas,
        "training_selection": {
            "selection_report_sha256": _sha(selection_path),
            "validation_used": False,
            "test_used": False,
        },
        "validation": evaluation,
        "future_pixel_reconstruction": False,
        "validation_88_loaded_once_after_lock": True,
        "test_split_loaded": False,
        "patient_identifiers_emitted": False,
        "confirmatory_claim_permitted": False,
        "evidence": {
            "config_sha256": _sha(paths["config"]),
            "protocol_lock_sha256": _sha(paths["protocol_lock"]),
            "frozen_design_sha256": _sha(paths["frozen_design"]),
            "feature_triage_report_sha256": _sha(paths["triage_report"]),
            "feature_triage_private_index_sha256": _sha(
                paths["triage_private_index"]
            ),
            "train_feature_payload_merkle_sha256": train_payload_merkle,
            "validation_feature_index_sha256": _sha(val_index_path),
            "validation_feature_payload_merkle_sha256": val_payload_merkle,
            "manifest_sha256": _sha(paths["manifest"]),
            "manifest_build_report_sha256": _sha(paths["manifest_report"]),
            "context_mask_report_sha256": _sha(paths["mask_report"]),
            "context_mask_index_sha256": _sha(paths["mask_index"]),
            "roi_index_sha256": _sha(paths["roi_index"]),
            "three_dino_checkpoint_sha256": _sha(paths["checkpoint"]),
            "checkpoint_sha256": _sha(checkpoint_path),
            "private_validation_measurements_sha256": _sha(private_path),
            "model_source_sha256": _sha(Path(__file__).with_name("model.py")),
            "data_source_sha256": _sha(Path(__file__).with_name("data.py")),
            "training_source_sha256": _sha(Path(__file__)),
        },
    }
    report_path = paths["output"] / "aggregate_report.json"
    with report_path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    for path in (
        checkpoint_path,
        history_path,
        private_path,
        report_path,
        selection_path,
    ):
        os.chmod(path, 0o600)
    os.replace(paths["private_output"], final_private_output)
    os.replace(paths["output"], final_output)
    print(
        json.dumps(
            {
                "status": report["status"],
                "development_gate_passed": passed,
                "selected_feature_variant": SELECTED_VARIANT,
                "selected_occupancy_lambda": occupancy_lambda,
                "validation_88_loaded_once_after_lock": True,
                "test_split_loaded": False,
            },
            sort_keys=True,
        )
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--protocol-lock", required=True)
    parser.add_argument("--frozen-design", required=True)
    parser.add_argument("--feature-triage-report", required=True)
    parser.add_argument("--feature-triage-private-root", required=True)
    parser.add_argument("--feature-triage-private-index", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manifest-build-report", required=True)
    parser.add_argument("--roi-root", required=True)
    parser.add_argument("--roi-index", required=True)
    parser.add_argument("--context-mask-release", required=True)
    parser.add_argument("--context-mask-report", required=True)
    parser.add_argument("--context-mask-index", required=True)
    parser.add_argument("--three-dino-repo", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--license-acknowledgement", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--private-output-dir", required=True)
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["SCHEMA", "choose_occupancy_lambda", "run"]
