"""Train409-only patient-grouped OOF development pipeline for Med-JEPA v4.3.

This entry point intentionally has no validation- or test-split arguments.  It
fits six frozen candidates, selects every information-matched reference inside
the corresponding outer training fold, and opens an outer held patient only
for its one out-of-fold score.
"""

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

from med_jepa.roi_v4.audit_mechanistic import REGIONS, _summary, patient_region_mae
from med_jepa.roi_v4.data import PatientStates, stack_batch
from med_jepa.roi_v4p2.feature_triage import _folds
from med_jepa.roi_v4p2.masking import scheduled_uniform_masks, uniform_block_mask
from med_jepa.roi_v4p2.train import (
    PAIR_KEYS,
    PAIRS,
    _anchor_for_pair,
    _fit_anchors,
    _interval_action,
    _record_features,
    _region_errors,
    _to_device,
)
from med_jepa.roi_v4p3.baselines import select_and_predict_strongest_baseline
from med_jepa.roi_v4p3.config import V43TrainingConfig
from med_jepa.roi_v4p3.data import load_v4p3_train409_release
from med_jepa.roi_v4p3.freeze_design import SCHEMA as DESIGN_SCHEMA
from med_jepa.roi_v4p3.gate import evaluate_v4p3_train_oof_gate
from med_jepa.roi_v4p3.losses import dual_path_objective, semigroup_consistency_loss
from med_jepa.roi_v4p3.model import DualPathShrinkageTokenDynamicsJEPA


SCHEMA = "med-jepa-roi-v4p3-train409-oof-training-v1"
PRIVATE_SCHEMA = "med-jepa-roi-v4p3-private-train409-oof-v1"
FOLD_SCHEMA = "med-jepa-roi-v4p3-private-outer-fold-candidate-v1"
BASELINE_FOLD_SCHEMA = "med-jepa-roi-v4p3-private-outer-fold-baseline-v1"


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
    if path.is_file():
        if path.read_text(encoding="utf-8") != rendered:
            raise ValueError("resumable public v4.3 artifact changed")
        return
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(rendered, encoding="utf-8")
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))


def _autocast():
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def _candidate_key(beta: float, outside: float) -> str:
    return f"beta-{beta:.2f}_outside-{outside:.2f}"


def _derangement(indices: Tensor, *, seed: int) -> Tensor:
    if indices.ndim != 1 or len(indices) < 2:
        raise ValueError("occupancy shuffle requires at least two patients")
    generator = torch.Generator().manual_seed(int(seed))
    for _ in range(100):
        candidate = indices[torch.randperm(len(indices), generator=generator)]
        if not bool(torch.any(candidate == indices)):
            return candidate
    # Deterministic guaranteed fallback; patient order has no statistical role.
    return torch.roll(indices, shifts=1)


def _inner_split(indices: Tensor, *, seed: int) -> tuple[Tensor, Tensor]:
    if indices.ndim != 1 or len(indices) < 20:
        raise ValueError("outer training fold is too small for inner calibration")
    generator = torch.Generator().manual_seed(int(seed))
    shuffled = indices[torch.randperm(len(indices), generator=generator)]
    count = max(24, len(indices) // 5)
    calibration = shuffled[:count].sort().values
    fit = shuffled[count:].sort().values
    return fit, calibration


def _core_ring(input_occupancy: Tensor, indices: Tensor, device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
    core = input_occupancy[indices, 0, None].to(
        device, non_blocking=True
    )
    ring = input_occupancy[indices, 1, None].to(
        device, non_blocking=True
    )
    valid = input_occupancy[indices, 2, None].to(
        device, non_blocking=True
    )
    return core, ring, valid


def _train_epoch(
    model: DualPathShrinkageTokenDynamicsJEPA,
    optimizer: torch.optim.Optimizer,
    records: list[PatientStates],
    evaluation_weights: Tensor,
    input_occupancy: Tensor,
    indices: Tensor,
    alphas: Mapping[str, float],
    means: Mapping[int, Tensor],
    config: V43TrainingConfig,
    *,
    ring_gate_beta: float,
    outside_energy_weight: float,
    epoch: int,
    seed: int,
    device: torch.device,
) -> float:
    model.train()
    generator = torch.Generator().manual_seed(int(seed) + epoch * 100_003)
    order = indices[torch.randperm(len(indices), generator=generator)]
    totals: list[float] = []
    for start in range(0, len(order), config.batch_size):
        selected = order[start : start + config.batch_size]
        batch = _to_device(stack_batch(records, selected), device)
        features = batch["features"]
        times = batch["times"]
        if not isinstance(features, Tensor) or not isinstance(times, Tensor):
            raise ValueError("v4.3 training tensors are unavailable")
        core, ring, valid = _core_ring(input_occupancy, selected, device)
        optimizer.zero_grad(set_to_none=True)
        with _autocast():
            pair_losses: list[Tensor] = []
            outputs: dict[tuple[int, int], Any] = {}
            supervision: dict[tuple[int, int], Tensor] = {}
            contexts: dict[tuple[int, int], Tensor] = {}
            for pair_index, pair in enumerate(PAIRS):
                source_visit, target_visit = pair
                context_mask, supervised_mask, _ = scheduled_uniform_masks(
                    len(selected),
                    (7, 7, 7),
                    seed=(
                        int(seed)
                        + epoch * 1_000_003
                        + start * 37
                        + pair_index * 10_007
                    ),
                    fractions=config.mask_fractions,
                )
                alpha, mean = _anchor_for_pair(pair, alphas, means, device)
                history = (
                    features[:, 1] - features[:, 0]
                    if pair == (1, 2)
                    else None
                )
                output = model.step(
                    features[:, source_visit],
                    mean,
                    alpha,
                    times[:, source_visit],
                    times[:, target_visit],
                    _interval_action(batch, pair),
                    core,
                    ring,
                    ring_gate_beta=ring_gate_beta,
                    valid_occupancy=valid,
                    history_delta=history,
                    context_mask=context_mask.to(device),
                )
                outputs[pair] = output
                supervision[pair] = supervised_mask.to(device)
                contexts[pair] = context_mask.to(device)
                terms = dual_path_objective(
                    output,
                    features[:, target_visit],
                    supervision[pair],
                    core,
                    ring,
                    ring_gate_beta=ring_gate_beta,
                    outside_energy_weight=outside_energy_weight,
                    global_background_weight=config.global_background_weight,
                    disease_weight=config.disease_weight,
                    full_grid_safety_weight=config.full_grid_safety_weight,
                    decorrelation_weight=config.decorrelation_weight,
                )
                pair_losses.append(terms["total"])

            first = outputs[(0, 1)].predicted
            composed = model.step(
                first,
                means[2].to(device),
                alphas[PAIR_KEYS[(1, 2)]],
                times[:, 1],
                times[:, 2],
                _interval_action(batch, (1, 2)),
                core,
                ring,
                ring_gate_beta=ring_gate_beta,
                valid_occupancy=valid,
                history_delta=first - features[:, 0],
                context_mask=contexts[(1, 2)],
            )
            semigroup = semigroup_consistency_loss(
                outputs[(0, 2)].predicted,
                composed.predicted,
                supervision[(0, 2)],
            )
            total = torch.stack(pair_losses).mean() + config.semigroup_weight * semigroup
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        totals.append(float(total.detach()))
    if not totals:
        raise RuntimeError("v4.3 training epoch had no batches")
    return sum(totals) / len(totals)


@torch.no_grad()
def _calibration_mae(
    model: DualPathShrinkageTokenDynamicsJEPA,
    records: Sequence[PatientStates],
    features: Tensor,
    evaluation_weights: Tensor,
    input_occupancy: Tensor,
    indices: Tensor,
    alpha: float,
    mean_target: Tensor,
    *,
    ring_gate_beta: float,
    seed: int,
    batch_size: int,
    device: torch.device,
) -> float:
    model.eval()
    rows: list[Tensor] = []
    for start in range(0, len(indices), batch_size):
        chosen = indices[start : start + batch_size]
        selected = [records[int(index)] for index in chosen.tolist()]
        source = torch.stack([item.features[1] for item in selected]).to(device)
        history = source - torch.stack([item.features[0] for item in selected]).to(device)
        source_time = torch.stack([item.times_days[1] for item in selected]).to(device)
        target_time = torch.stack([item.times_days[2] for item in selected]).to(device)
        action = torch.stack([item.action_13 for item in selected]).to(device)
        core, ring, valid = _core_ring(input_occupancy, chosen, device)
        mask = uniform_block_mask(
            len(chosen), (7, 7, 7), seed=seed + start * 101, masked_fraction=0.75
        ).to(device)
        with _autocast():
            output = model.step(
                source,
                mean_target.to(device),
                alpha,
                source_time,
                target_time,
                action,
                core,
                ring,
                ring_gate_beta=ring_gate_beta,
                valid_occupancy=valid,
                history_delta=history,
                context_mask=mask,
            )
        rows.append(output.predicted.float().cpu())
    prediction = torch.cat(rows)
    error = patient_region_mae(
        prediction,
        features[indices, 2],
        evaluation_weights[indices, REGIONS.index("t0_disease_bed")],
    )
    return float(torch.nanmean(error))


def _select_epoch_count(
    records: list[PatientStates],
    features: Tensor,
    evaluation_weights: Tensor,
    input_occupancy: Tensor,
    outer_train_indices: Tensor,
    alphas: Mapping[str, float],
    means: Mapping[int, Tensor],
    config: V43TrainingConfig,
    *,
    ring_gate_beta: float,
    outside_energy_weight: float,
    fold: int,
    device: torch.device,
) -> tuple[int, list[dict[str, float]]]:
    fit_indices, calibration_indices = _inner_split(
        outer_train_indices, seed=config.seed + fold * 70_001
    )
    seed = (
        config.seed
        + fold * 1_000_003
        + int(round(ring_gate_beta * 100)) * 1_009
        + int(round(outside_energy_weight * 100)) * 101
    )
    _set_seed(seed)
    model = DualPathShrinkageTokenDynamicsJEPA(
        config.hidden_channels,
        config.conditioned_blocks,
        config.global_coarse_grid,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    best = math.inf
    best_epoch_count = 0
    patience = 0
    history: list[dict[str, float]] = []
    for epoch in range(config.maximum_cv_epochs):
        train_loss = _train_epoch(
            model,
            optimizer,
            records,
            evaluation_weights,
            input_occupancy,
            fit_indices,
            alphas,
            means,
            config,
            ring_gate_beta=ring_gate_beta,
            outside_energy_weight=outside_energy_weight,
            epoch=epoch,
            seed=seed,
            device=device,
        )
        value = _calibration_mae(
            model,
            records,
            features,
            evaluation_weights,
            input_occupancy,
            calibration_indices,
            alphas[PAIR_KEYS[(1, 2)]],
            means[2],
            ring_gate_beta=ring_gate_beta,
            seed=seed + 500_009,
            batch_size=config.batch_size,
            device=device,
        )
        row = {
            "epoch": float(epoch),
            "fit_loss": train_loss,
            "inner_calibration_disease_bed_MAE": value,
        }
        history.append(row)
        print(
            json.dumps(
                {
                    "stage": "v4p3_inner_epoch_selection",
                    "fold": fold,
                    "ring_gate_beta": ring_gate_beta,
                    "outside_energy_weight": outside_energy_weight,
                    **row,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if value < best - config.early_stopping_min_delta:
            best = value
            best_epoch_count = epoch + 1
            patience = 0
        else:
            patience += 1
            if patience >= config.early_stopping_patience:
                break
    if best_epoch_count < 1:
        raise RuntimeError("v4.3 inner calibration selected no epoch")
    del model, optimizer
    torch.cuda.empty_cache()
    return best_epoch_count, history


def _fit_for_epochs(
    records: list[PatientStates],
    evaluation_weights: Tensor,
    input_occupancy: Tensor,
    indices: Tensor,
    alphas: Mapping[str, float],
    means: Mapping[int, Tensor],
    config: V43TrainingConfig,
    *,
    ring_gate_beta: float,
    outside_energy_weight: float,
    epochs: int,
    seed: int,
    device: torch.device,
) -> tuple[dict[str, Tensor], list[dict[str, float]]]:
    _set_seed(seed)
    model = DualPathShrinkageTokenDynamicsJEPA(
        config.hidden_channels,
        config.conditioned_blocks,
        config.global_coarse_grid,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        loss = _train_epoch(
            model,
            optimizer,
            records,
            evaluation_weights,
            input_occupancy,
            indices,
            alphas,
            means,
            config,
            ring_gate_beta=ring_gate_beta,
            outside_energy_weight=outside_energy_weight,
            epoch=epoch,
            seed=seed,
            device=device,
        )
        history.append({"epoch": float(epoch), "train_total_loss": loss})
        print(
            json.dumps(
                {
                    "stage": "v4p3_outer_refit",
                    "epoch": epoch,
                    "epochs": epochs,
                    "loss": loss,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    state = {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }
    del model, optimizer
    torch.cuda.empty_cache()
    return state, history


@torch.no_grad()
def _predict_ensemble(
    model: DualPathShrinkageTokenDynamicsJEPA,
    records: Sequence[PatientStates],
    input_occupancy: Tensor,
    indices: Tensor,
    alpha: float,
    mean_target: Tensor,
    config: V43TrainingConfig,
    *,
    ring_gate_beta: float,
    seed: int,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, Tensor, float]:
    model.eval()
    permutation = _derangement(indices, seed=int(seed) + 55_003)
    candidate_rows: list[Tensor] = []
    global_rows: list[Tensor] = []
    shuffled_rows: list[Tensor] = []
    anchor_rows: list[Tensor] = []
    outside_max = 0.0
    draw_count = len(config.mask_fractions) * config.evaluation_mask_draws_per_fraction
    for start in range(0, len(indices), config.batch_size):
        chosen = indices[start : start + config.batch_size]
        shuffled = permutation[start : start + config.batch_size]
        selected = [records[int(index)] for index in chosen.tolist()]
        source = torch.stack([item.features[1] for item in selected]).to(device)
        t0 = torch.stack([item.features[0] for item in selected]).to(device)
        source_time = torch.stack([item.times_days[1] for item in selected]).to(device)
        target_time = torch.stack([item.times_days[2] for item in selected]).to(device)
        action = torch.stack([item.action_13 for item in selected]).to(device)
        core, ring, valid = _core_ring(input_occupancy, chosen, device)
        shuffled_core, shuffled_ring, shuffled_valid = _core_ring(
            input_occupancy, shuffled, device
        )
        candidate_sum = torch.zeros_like(source, dtype=torch.float32)
        global_sum = torch.zeros_like(source, dtype=torch.float32)
        shuffled_sum = torch.zeros_like(source, dtype=torch.float32)
        anchor: Tensor | None = None
        for fraction_index, fraction in enumerate(config.mask_fractions):
            for draw in range(config.evaluation_mask_draws_per_fraction):
                mask = uniform_block_mask(
                    len(chosen),
                    (7, 7, 7),
                    seed=(
                        int(seed)
                        + start * 10_007
                        + fraction_index * 1_009
                        + draw * 97
                    ),
                    masked_fraction=fraction,
                ).to(device)
                with _autocast():
                    output = model.step(
                        source,
                        mean_target.to(device),
                        alpha,
                        source_time,
                        target_time,
                        action,
                        core,
                        ring,
                        ring_gate_beta=ring_gate_beta,
                        valid_occupancy=valid,
                        history_delta=source - t0,
                        context_mask=mask,
                    )
                    shuffled_output = model.step(
                        source,
                        mean_target.to(device),
                        alpha,
                        source_time,
                        target_time,
                        action,
                        shuffled_core,
                        shuffled_ring,
                        ring_gate_beta=ring_gate_beta,
                        valid_occupancy=shuffled_valid,
                        history_delta=source - t0,
                        context_mask=mask,
                    )
                candidate_sum.add_(output.predicted.float())
                global_sum.add_(
                    output.shrinkage_anchor.float() + output.global_residual.float()
                )
                shuffled_sum.add_(shuffled_output.predicted.float())
                anchor = output.shrinkage_anchor.float()
                outside = output.disease_gate == 0
                if bool(outside.any()):
                    value = float(
                        output.gated_disease_residual[
                            outside.expand_as(output.gated_disease_residual)
                        ]
                        .abs()
                        .max()
                    )
                    outside_max = max(outside_max, value)
        if anchor is None:
            raise RuntimeError("v4.3 evaluation produced no mask draws")
        candidate_rows.append((candidate_sum / draw_count).cpu())
        global_rows.append((global_sum / draw_count).cpu())
        shuffled_rows.append((shuffled_sum / draw_count).cpu())
        anchor_rows.append(anchor.cpu())
    return (
        torch.cat(candidate_rows),
        torch.cat(global_rows),
        torch.cat(anchor_rows),
        torch.cat(shuffled_rows),
        outside_max,
    )


def _fold_cache_path(root: Path, fold: int, beta: float, outside: float) -> Path:
    return root / f"fold-{fold:02d}_{_candidate_key(beta, outside)}.pt"


def _cross_validate(
    records: list[PatientStates],
    evaluation_weights: Tensor,
    input_occupancy: Tensor,
    config: V43TrainingConfig,
    *,
    resume_dir: Path,
    resume_contract: Mapping[str, Any],
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    features = _record_features(records)
    folds = _folds(409, seed=config.seed, count=config.grouped_cv_folds)
    candidates = [
        (beta, outside)
        for beta in config.ring_gate_beta_candidates
        for outside in config.outside_energy_weight_candidates
    ]
    names = [_candidate_key(*value) for value in candidates]
    matrices: dict[str, dict[str, Tensor]] = {
        name: {
            key: torch.full((409, len(REGIONS)), float("nan"))
            for key in (
                "candidate",
                "strongest_baseline",
                "global_only",
                "anchor",
                "occupancy_shuffled",
            )
        }
        for name in names
    }
    best_epochs: dict[str, list[int]] = {name: [] for name in names}
    maximum_outside: dict[str, float] = {name: 0.0 for name in names}
    fold_evidence: dict[str, list[dict[str, Any]]] = {name: [] for name in names}
    resume_dir.mkdir(parents=True, exist_ok=True, mode=0o700)

    for fold, (outer_train, held) in enumerate(folds):
        alphas, alpha_scores, means = _fit_anchors(
            features, evaluation_weights, outer_train, device=device
        )
        source = features[held, 1]
        target = features[held, 2]
        anchor = (
            alphas[PAIR_KEYS[(1, 2)]] * source
            + (1.0 - alphas[PAIR_KEYS[(1, 2)]]) * means[2]
        )
        anchor_errors = _region_errors(anchor, target, evaluation_weights[held])

        baseline_by_beta: dict[float, tuple[Tensor, dict[str, Any]]] = {}
        for beta in config.ring_gate_beta_candidates:
            baseline_path = resume_dir / f"fold-{fold:02d}_baseline_beta-{beta:.2f}.pt"
            if baseline_path.is_file():
                cached = torch.load(baseline_path, map_location="cpu", weights_only=True)
                if (
                    not isinstance(cached, Mapping)
                    or cached.get("schema_version") != BASELINE_FOLD_SCHEMA
                    or cached.get("resume_contract") != dict(resume_contract)
                    or cached.get("fold") != fold
                    or cached.get("ring_gate_beta") != beta
                    or not torch.equal(cached.get("held_indices"), held)
                    or not isinstance(cached.get("errors"), Tensor)
                ):
                    raise ValueError("resumable v4.3 baseline fold changed")
                baseline_errors = cached["errors"]
                baseline_public = dict(cached["public_evidence"])
            else:
                result = select_and_predict_strongest_baseline(
                    records,
                    features,
                    evaluation_weights,
                    input_occupancy,
                    outer_train,
                    held,
                    alpha=alphas[PAIR_KEYS[(1, 2)]],
                    population_target_mean=means[2],
                    ring_gate_beta=beta,
                    seed=config.seed + fold * 100_003 + int(beta * 100),
                )
                baseline_errors = _region_errors(
                    result.prediction, target, evaluation_weights[held]
                )
                baseline_public = {
                    "selected_family": result.selected_family,
                    "inside_fold_calibration_disease_MAE": dict(
                        result.selection_disease_mae
                    ),
                    "selected_fit_report": dict(result.selected_fit_report),
                    "maximum_outside_gated_local_absolute": (
                        result.outside_gated_local_maximum_absolute
                    ),
                }
                temporary = baseline_path.with_suffix(".partial")
                torch.save(
                    {
                        "schema_version": BASELINE_FOLD_SCHEMA,
                        "resume_contract": dict(resume_contract),
                        "fold": fold,
                        "ring_gate_beta": beta,
                        "held_indices": held,
                        "errors": baseline_errors,
                        "public_evidence": baseline_public,
                    },
                    temporary,
                )
                os.chmod(temporary, 0o600)
                os.replace(temporary, baseline_path)
            baseline_by_beta[beta] = (baseline_errors, baseline_public)

        for beta, outside in candidates:
            name = _candidate_key(beta, outside)
            cache_path = _fold_cache_path(resume_dir, fold, beta, outside)
            if cache_path.is_file():
                cached = torch.load(cache_path, map_location="cpu", weights_only=True)
                if (
                    not isinstance(cached, Mapping)
                    or cached.get("schema_version") != FOLD_SCHEMA
                    or cached.get("resume_contract") != dict(resume_contract)
                    or cached.get("fold") != fold
                    or cached.get("ring_gate_beta") != beta
                    or cached.get("outside_energy_weight") != outside
                    or not torch.equal(cached.get("held_indices"), held)
                ):
                    raise ValueError("resumable v4.3 candidate fold changed")
                payload = dict(cached)
                print(
                    json.dumps(
                        {"stage": "v4p3_outer_fold_resume", "fold": fold, "candidate": name},
                        sort_keys=True,
                    ),
                    flush=True,
                )
            else:
                epoch_count, inner_history = _select_epoch_count(
                    records,
                    features,
                    evaluation_weights,
                    input_occupancy,
                    outer_train,
                    alphas,
                    means,
                    config,
                    ring_gate_beta=beta,
                    outside_energy_weight=outside,
                    fold=fold,
                    device=device,
                )
                state, outer_history = _fit_for_epochs(
                    records,
                    evaluation_weights,
                    input_occupancy,
                    outer_train,
                    alphas,
                    means,
                    config,
                    ring_gate_beta=beta,
                    outside_energy_weight=outside,
                    epochs=epoch_count,
                    seed=config.seed + fold * 10_000_019 + int(beta * 1000) + int(outside * 100),
                    device=device,
                )
                model = DualPathShrinkageTokenDynamicsJEPA(
                    config.hidden_channels,
                    config.conditioned_blocks,
                    config.global_coarse_grid,
                ).to(device)
                model.load_state_dict(state, strict=True)
                candidate, global_only, observed_anchor, shuffled, outside_max = _predict_ensemble(
                    model,
                    records,
                    input_occupancy,
                    held,
                    alphas[PAIR_KEYS[(1, 2)]],
                    means[2],
                    config,
                    ring_gate_beta=beta,
                    seed=config.bootstrap_seed + fold * 1_000_003,
                    device=device,
                )
                if not torch.allclose(
                    observed_anchor, anchor.float(), rtol=0.0, atol=1e-6
                ):
                    raise RuntimeError("v4.3 neural anchor diverged from fold shrinkage")
                payload = {
                    "schema_version": FOLD_SCHEMA,
                    "resume_contract": dict(resume_contract),
                    "fold": fold,
                    "ring_gate_beta": beta,
                    "outside_energy_weight": outside,
                    "held_indices": held,
                    "best_epoch_count": epoch_count,
                    "inner_history": inner_history,
                    "outer_refit_history": outer_history,
                    "candidate_errors": _region_errors(candidate, target, evaluation_weights[held]),
                    "global_only_errors": _region_errors(global_only, target, evaluation_weights[held]),
                    "anchor_errors": anchor_errors,
                    "occupancy_shuffled_errors": _region_errors(shuffled, target, evaluation_weights[held]),
                    "maximum_outside_gated_disease_absolute": outside_max,
                }
                temporary = cache_path.with_suffix(".partial")
                torch.save(payload, temporary)
                os.chmod(temporary, 0o600)
                os.replace(temporary, cache_path)
                del model, state, candidate, global_only, shuffled
                torch.cuda.empty_cache()

            baseline_errors, baseline_public = baseline_by_beta[beta]
            entries = {
                "candidate": payload["candidate_errors"],
                "strongest_baseline": baseline_errors,
                "global_only": payload["global_only_errors"],
                "anchor": payload["anchor_errors"],
                "occupancy_shuffled": payload["occupancy_shuffled_errors"],
            }
            for key, value in entries.items():
                if not isinstance(value, Tensor) or value.shape != (len(held), len(REGIONS)):
                    raise ValueError("v4.3 fold error payload is malformed")
                matrices[name][key][held] = value
            epoch_count = int(payload["best_epoch_count"])
            best_epochs[name].append(epoch_count)
            maximum_outside[name] = max(
                maximum_outside[name],
                float(payload["maximum_outside_gated_disease_absolute"]),
            )
            fold_evidence[name].append(
                {
                    "fold": fold,
                    "held_patient_count": len(held),
                    "best_epoch_count": epoch_count,
                    "pair_specific_shrinkage_alphas": alphas,
                    "pair_specific_leave_one_out_scores": alpha_scores,
                    "strongest_baseline": baseline_public,
                }
            )
        gc.collect()
        torch.cuda.empty_cache()

    ids = [item.patient_id for item in records]
    disease_index = REGIONS.index("t0_disease_bed")
    full_index = REGIONS.index("full_shared_valid_grid")
    gates: dict[str, Any] = {}
    summaries: dict[str, Any] = {}
    for counter, name in enumerate(names):
        value = matrices[name]
        if any(torch.isnan(matrix[:, disease_index:]).any() for matrix in value.values()):
            raise RuntimeError("v4.3 OOF matrix contains missing required regions")
        gate = evaluate_v4p3_train_oof_gate(
            value["candidate"],
            value["strongest_baseline"],
            value["global_only"],
            value["anchor"],
            value["occupancy_shuffled"],
            ids,
            maximum_outside_gated_disease_absolute=maximum_outside[name],
            seed=config.bootstrap_seed + counter * 1_009,
            replicates=config.bootstrap_replicates,
        )
        gates[name] = gate
        summaries[name] = {
            "development_gate_passed": gate["development_gate_passed"],
            "candidate_region_MAE": {
                region: _summary(value["candidate"][:, index])
                for index, region in enumerate(REGIONS)
            },
            "strongest_baseline_region_MAE": {
                region: _summary(value["strongest_baseline"][:, index])
                for index, region in enumerate(REGIONS)
            },
            "mean_disease_bed_MAE": float(value["candidate"][:, disease_index].mean()),
            "mean_full_grid_MAE": float(value["candidate"][:, full_index].mean()),
            "best_epoch_counts": best_epochs[name],
            "fold_evidence": fold_evidence[name],
        }
    passed = [name for name in names if gates[name]["development_gate_passed"]]
    selected = (
        min(
            passed,
            key=lambda name: (
                summaries[name]["mean_disease_bed_MAE"],
                summaries[name]["mean_full_grid_MAE"],
                name,
            ),
        )
        if passed
        else None
    )
    public = {
        "candidate_summaries": summaries,
        "candidate_oof_gates": gates,
        "passing_candidates": passed,
        "selected_candidate": selected,
        "selection_rule": (
            "minimum_mean_train409_OOF_T0_disease_bed_MAE_among_candidates_"
            "passing_every_frozen_gate"
        ),
        "outer_folds": config.grouped_cv_folds,
        "outer_held_fold_used_for_early_stopping": False,
        "baseline_selected_inside_outer_training_fold": True,
    }
    private = {
        "schema_version": PRIVATE_SCHEMA,
        "patient_ids": ids,
        "candidate_error_matrices": matrices,
        "selected_candidate": selected,
    }
    return public, private


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("v4.3 train409 OOF requires exactly one CUDA GPU")
    paths = {
        name: Path(value).expanduser().resolve()
        for name, value in {
            "config": args.config,
            "frozen_design": args.frozen_design,
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
        "config",
        "frozen_design",
        "manifest",
        "manifest_report",
        "feature_index",
        "mask_report",
        "mask_index",
    ):
        if not paths[name].is_file():
            raise FileNotFoundError(f"required v4.3 input is unavailable: {name}")
    if not paths["feature_root"].is_dir():
        raise FileNotFoundError("v4.3 train409 feature root is unavailable")
    if not paths["mask_root"].is_dir():
        raise FileNotFoundError("v4.3 T0-only context-mask release is unavailable")
    if paths["output"].exists() or paths["private_output"].exists():
        raise FileExistsError("immutable v4.3 train409 output already exists")
    output_staging = paths["output"].parent / f".{paths['output'].name}.staging"
    private_staging = paths["private_output"].parent / f".{paths['private_output'].name}.staging"
    output_staging.mkdir(parents=True, exist_ok=True, mode=0o700)
    private_staging.mkdir(parents=True, exist_ok=True, mode=0o700)

    config = V43TrainingConfig.load(paths["config"])
    design = _json(paths["frozen_design"])
    manifest_report = _json(paths["manifest_report"])
    design_evidence = design.get("evidence", {})
    if (
        design.get("schema_version") != DESIGN_SCHEMA
        or design.get("status") != "frozen_train409_oof_required_training_not_started"
        or design.get("training_started") is not False
        or design.get("validation_88_loaded") is not False
        or design.get("test_split_loaded") is not False
        or design_evidence.get("config_sha256") != _sha(paths["config"])
        or design_evidence.get("manifest_sha256") != _sha(paths["manifest"])
        or design_evidence.get("train_feature_index_sha256") != _sha(paths["feature_index"])
        or manifest_report.get("status") != "ok"
        or manifest_report.get("manifest_sha256") != _sha(paths["manifest"])
    ):
        raise ValueError("v4.3 frozen prerequisite chain is invalid")

    # Freeze the concrete implementation before the first train409 payload is
    # opened.  This closes the gap between an abstract frozen design and the
    # later-added executable OOF pipeline.
    implementation_lock = {
        "schema_version": "med-jepa-roi-v4p3-implementation-lock-v1",
        "status": "locked_before_train409_payload_loading",
        "frozen_design_sha256": _sha(paths["frozen_design"]),
        "config_sha256": _sha(paths["config"]),
        "source_sha256": {
            name: _sha(Path(__file__).with_name(name))
            for name in (
                "model.py",
                "losses.py",
                "baselines.py",
                "data.py",
                "gate.py",
                "train.py",
            )
        },
        "T0_only_occupancy_input": True,
        "shared_T0_T1_T3_valid_support_used_only_for_evaluation": True,
        "validation_88_loaded": False,
        "test_split_loaded": False,
    }
    implementation_lock_path = output_staging / "implementation_lock.json"
    _write_json(implementation_lock_path, implementation_lock)

    (
        records,
        evaluation_weights,
        input_occupancy,
        feature_merkle,
        input_occupancy_digest,
    ) = load_v4p3_train409_release(
        paths["manifest"],
        paths["feature_root"],
        paths["feature_index"],
        expected_feature_index_sha256=_sha(paths["feature_index"]),
        context_mask_root=paths["mask_root"],
        context_mask_report=paths["mask_report"],
        context_mask_index=paths["mask_index"],
    )
    if (
        len(records) != 409
        or evaluation_weights.shape != (409, len(REGIONS), 7, 7, 7)
        or input_occupancy.shape != (409, 3, 7, 7, 7)
    ):
        raise ValueError("v4.3 train409 release cardinality changed")
    resume_contract = {
        "config_sha256": _sha(paths["config"]),
        "frozen_design_sha256": _sha(paths["frozen_design"]),
        "implementation_lock_sha256": _sha(implementation_lock_path),
        "manifest_sha256": _sha(paths["manifest"]),
        "feature_index_sha256": _sha(paths["feature_index"]),
        "feature_payload_merkle_sha256": feature_merkle,
        "context_mask_report_sha256": _sha(paths["mask_report"]),
        "context_mask_index_sha256": _sha(paths["mask_index"]),
        "T0_only_input_occupancy_digest": input_occupancy_digest,
        "training_source_sha256": _sha(Path(__file__)),
        "model_source_sha256": _sha(Path(__file__).with_name("model.py")),
        "baseline_source_sha256": _sha(Path(__file__).with_name("baselines.py")),
    }
    device = torch.device("cuda")
    cv, private = _cross_validate(
        records,
        evaluation_weights,
        input_occupancy,
        config,
        resume_dir=private_staging / "outer_fold_caches",
        resume_contract=resume_contract,
        device=device,
    )
    selected = cv["selected_candidate"]
    checkpoint_path: Path | None = None
    final_history: list[dict[str, float]] = []
    final_epoch_count: int | None = None
    final_alphas: Mapping[str, float] | None = None
    if selected is not None:
        selected_summary = cv["candidate_summaries"][selected]
        final_epoch_count = max(
            1, int(round(statistics.median(selected_summary["best_epoch_counts"])))
        )
        beta_text, outside_text = selected.split("_")
        beta = float(beta_text.removeprefix("beta-"))
        outside = float(outside_text.removeprefix("outside-"))
        all_indices = torch.arange(409)
        features = _record_features(records)
        final_alphas, _, final_means = _fit_anchors(
            features, evaluation_weights, all_indices, device=device
        )
        final_cache = private_staging / "final_train409_fit.pt"
        final_contract = {
            **resume_contract,
            "selected_candidate": selected,
            "final_epoch_count": final_epoch_count,
            "pair_specific_shrinkage_alphas": dict(final_alphas),
        }
        if final_cache.is_file():
            cached_fit = torch.load(
                final_cache, map_location="cpu", weights_only=True
            )
            if (
                not isinstance(cached_fit, Mapping)
                or cached_fit.get("schema_version")
                != "med-jepa-roi-v4p3-private-final-train409-fit-v1"
                or cached_fit.get("resume_contract") != final_contract
                or not isinstance(cached_fit.get("model"), Mapping)
                or not isinstance(cached_fit.get("history"), list)
            ):
                raise ValueError("resumable v4.3 final train409 fit changed")
            state = dict(cached_fit["model"])
            final_history = list(cached_fit["history"])
            print(json.dumps({"stage": "v4p3_final_fit_resume"}), flush=True)
        else:
            state, final_history = _fit_for_epochs(
                records,
                evaluation_weights,
                input_occupancy,
                all_indices,
                final_alphas,
                final_means,
                config,
                ring_gate_beta=beta,
                outside_energy_weight=outside,
                epochs=final_epoch_count,
                seed=config.seed,
                device=device,
            )
            temporary = final_cache.with_suffix(".partial")
            torch.save(
                {
                    "schema_version": (
                        "med-jepa-roi-v4p3-private-final-train409-fit-v1"
                    ),
                    "resume_contract": final_contract,
                    "model": state,
                    "history": final_history,
                },
                temporary,
            )
            os.chmod(temporary, 0o600)
            os.replace(temporary, final_cache)
        checkpoint_path = output_staging / "checkpoint_train409.pt"
        torch.save(
            {
                "schema_version": SCHEMA,
                "model": state,
                "config": asdict(config),
                "selected_candidate": selected,
                "ring_gate_beta": beta,
                "outside_energy_weight": outside,
                "final_epoch_count": final_epoch_count,
                "pair_specific_shrinkage_alphas": dict(final_alphas),
                "population_target_means": {
                    str(key): value.half() for key, value in final_means.items()
                },
                "validation_88_loaded": False,
                "test_split_loaded": False,
            },
            checkpoint_path,
        )
        os.chmod(checkpoint_path, 0o600)
        del state, features
        torch.cuda.empty_cache()

    history_path = output_staging / "final_training_metrics.jsonl"
    if not history_path.exists():
        with history_path.open("x", encoding="utf-8") as handle:
            for row in final_history:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        os.chmod(history_path, 0o600)
    private_path = private_staging / "train409_oof_measurements.pt"
    if not private_path.exists():
        torch.save(private, private_path)
        os.chmod(private_path, 0o600)
    status = (
        "completed_train409_oof_gate_pass_final_model_fitted"
        if selected is not None
        else "completed_train409_oof_gate_fail_no_final_model"
    )
    report = {
        "schema_version": SCHEMA,
        "status": status,
        "scope": "train409_patient_grouped_five_fold_OOF_only",
        "seed": config.seed,
        "patient_counts": {"train_loaded": 409, "validation_loaded": 0, "test_loaded": 0},
        "selected_feature_variant": design["feature_contract"]["variant"],
        "train409_cross_validation": cv,
        "selected_candidate": selected,
        "final_epoch_count": final_epoch_count,
        "pair_specific_shrinkage_alphas": final_alphas,
        "final_checkpoint_created": checkpoint_path is not None,
        "future_pixel_reconstruction": False,
        "validation_88_loaded": False,
        "test_split_loaded": False,
        "patient_identifiers_emitted": False,
        "confirmatory_claim_permitted": False,
        "evidence": {
            **resume_contract,
            "manifest_build_report_sha256": _sha(paths["manifest_report"]),
            "private_train409_oof_measurements_sha256": _sha(private_path),
            "checkpoint_sha256": _sha(checkpoint_path) if checkpoint_path else None,
        },
    }
    _write_json(output_staging / "aggregate_report.json", report)
    os.replace(output_staging, paths["output"])
    os.replace(private_staging, paths["private_output"])
    print(
        json.dumps(
            {
                "status": status,
                "selected_candidate": selected,
                "train_patients": 409,
                "validation_88_loaded": False,
                "test_split_loaded": False,
            },
            sort_keys=True,
        )
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--frozen-design", required=True)
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


__all__ = ["SCHEMA", "run"]
