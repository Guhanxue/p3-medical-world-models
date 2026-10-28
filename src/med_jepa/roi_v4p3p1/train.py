"""Train409-only nested OOF pipeline for Med-JEPA v4.3.1.

This executable has no validation- or test-split arguments.  In every outer
fold it fits an occupancy-conditioned coarse RRR on the outer-training
patients, then trains one neural core residual against that fixed anchor.
Epoch selection uses a further inner split; the outer-held patients are scored
once and never influence fitting or stopping.
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
from med_jepa.roi_v4p2.train import _record_features, _region_errors, _to_device
from med_jepa.roi_v4p3.data import load_v4p3_train409_release
from med_jepa.roi_v4p3p1.anchor import anchor_bank, fit_predict_coarse_rrr_anchor
from med_jepa.roi_v4p3p1.config import V431TrainingConfig
from med_jepa.roi_v4p3p1.freeze_design import SCHEMA as DESIGN_SCHEMA
from med_jepa.roi_v4p3p1.gate import evaluate_v431_train_oof_gate
from med_jepa.roi_v4p3p1.losses import masked_fractional_core_objective
from med_jepa.roi_v4p3p1.model import CoarseRRRAnchoredCoreResidualJEPA


SCHEMA = "med-jepa-roi-v4p3p1-train409-oof-training-v1"
PRIVATE_SCHEMA = "med-jepa-roi-v4p3p1-private-train409-oof-v1"
FOLD_SCHEMA = "med-jepa-roi-v4p3p1-private-outer-fold-v1"


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
            raise ValueError("resumable public v4.3.1 artifact changed")
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


def _derangement(indices: Tensor, *, seed: int) -> Tensor:
    if indices.ndim != 1 or len(indices) < 2:
        raise ValueError("occupancy shuffle requires at least two patients")
    generator = torch.Generator().manual_seed(int(seed))
    for _ in range(100):
        candidate = indices[torch.randperm(len(indices), generator=generator)]
        if not bool(torch.any(candidate == indices)):
            return candidate
    return torch.roll(indices, shifts=1)


def _inner_split(indices: Tensor, *, seed: int) -> tuple[Tensor, Tensor]:
    if indices.ndim != 1 or len(indices) < 20:
        raise ValueError("outer training fold is too small for inner calibration")
    generator = torch.Generator().manual_seed(int(seed))
    shuffled = indices[torch.randperm(len(indices), generator=generator)]
    count = max(24, len(indices) // 5)
    return shuffled[count:].sort().values, shuffled[:count].sort().values


def _core_ring(
    input_occupancy: Tensor,
    indices: Tensor,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    return (
        input_occupancy[indices, 0, None].to(device, non_blocking=True),
        input_occupancy[indices, 1, None].to(device, non_blocking=True),
    )


def _interval_action(batch: Mapping[str, object]) -> Tensor:
    value = batch["action_13"]
    if not isinstance(value, Tensor):
        raise ValueError("T1-to-T3 action is unavailable")
    return value


def _fit_anchor_bank(
    records: Sequence[PatientStates],
    features: Tensor,
    evaluation_weights: Tensor,
    input_occupancy: Tensor,
    fit_indices: Tensor,
    predict_indices: Tensor,
    *,
    seed: int,
    device: torch.device,
) -> tuple[Tensor, dict[str, Any]]:
    fit = fit_predict_coarse_rrr_anchor(
        records,
        features,
        evaluation_weights,
        input_occupancy,
        fit_indices,
        predict_indices,
        seed=seed,
        device=device,
    )
    bank = anchor_bank(len(records), predict_indices, fit.predictions)
    report = {
        "model": "fold_local_occupancy_conditioned_coarse_RRR",
        "fit_patient_count": fit.fit_patient_count,
        "prediction_patient_count": fit.prediction_patient_count,
        "shrinkage_alpha": fit.shrinkage_alpha,
        "shrinkage_leave_one_out_scores": dict(
            fit.shrinkage_leave_one_out_scores
        ),
        "coarse_rrr": dict(fit.coarse_rrr_report),
        "prediction_outcomes_used_for_model_or_hyperparameter_fitting": False,
    }
    return bank, report


def _train_epoch(
    model: CoarseRRRAnchoredCoreResidualJEPA,
    optimizer: torch.optim.Optimizer,
    records: list[PatientStates],
    input_occupancy: Tensor,
    anchors: Tensor,
    indices: Tensor,
    config: V431TrainingConfig,
    *,
    epoch: int,
    seed: int,
    device: torch.device,
) -> float:
    model.train()
    generator = torch.Generator().manual_seed(int(seed) + epoch * 100_003)
    order = indices[torch.randperm(len(indices), generator=generator)]
    totals: list[float] = []
    for start in range(0, len(order), config.batch_size):
        chosen = order[start : start + config.batch_size]
        batch = _to_device(stack_batch(records, chosen), device)
        features = batch["features"]
        times = batch["times"]
        if not isinstance(features, Tensor) or not isinstance(times, Tensor):
            raise ValueError("v4.3.1 training tensors are unavailable")
        core, ring = _core_ring(input_occupancy, chosen, device)
        context_mask, supervised_mask, _ = scheduled_uniform_masks(
            len(chosen),
            (7, 7, 7),
            seed=int(seed) + epoch * 1_000_003 + start * 37,
            fractions=config.mask_fractions,
        )
        anchor = anchors[chosen].to(device, non_blocking=True)
        if not torch.isfinite(anchor).all():
            raise ValueError("training anchor bank is incomplete")
        optimizer.zero_grad(set_to_none=True)
        with _autocast():
            output = model.step(
                features[:, 1],
                anchor,
                times[:, 1],
                times[:, 2],
                _interval_action(batch),
                core,
                ring,
                history_delta=features[:, 1] - features[:, 0],
                context_mask=context_mask.to(device),
            )
            terms = masked_fractional_core_objective(
                output,
                features[:, 2],
                supervised_mask.to(device),
                core,
            )
        terms["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        totals.append(float(terms["total"].detach()))
    if not totals:
        raise RuntimeError("v4.3.1 epoch had no batches")
    return sum(totals) / len(totals)


@torch.no_grad()
def _calibration_core_mae(
    model: CoarseRRRAnchoredCoreResidualJEPA,
    records: Sequence[PatientStates],
    features: Tensor,
    evaluation_weights: Tensor,
    input_occupancy: Tensor,
    anchors: Tensor,
    indices: Tensor,
    config: V431TrainingConfig,
    *,
    seed: int,
    device: torch.device,
) -> float:
    model.eval()
    rows: list[Tensor] = []
    for start in range(0, len(indices), config.batch_size):
        chosen = indices[start : start + config.batch_size]
        selected = [records[int(index)] for index in chosen.tolist()]
        source = torch.stack([item.features[1] for item in selected]).to(device)
        t0 = torch.stack([item.features[0] for item in selected]).to(device)
        times = torch.stack([item.times_days for item in selected]).to(device)
        action = torch.stack([item.action_13 for item in selected]).to(device)
        core, ring = _core_ring(input_occupancy, chosen, device)
        mask = uniform_block_mask(
            len(chosen),
            (7, 7, 7),
            seed=int(seed) + start * 101,
            masked_fraction=0.75,
        ).to(device)
        anchor = anchors[chosen].to(device, non_blocking=True)
        with _autocast():
            output = model.step(
                source,
                anchor,
                times[:, 1],
                times[:, 2],
                action,
                core,
                ring,
                history_delta=source - t0,
                context_mask=mask,
            )
        rows.append(output.predicted.float().cpu())
    prediction = torch.cat(rows)
    error = patient_region_mae(
        prediction,
        features[indices, 2],
        evaluation_weights[indices, REGIONS.index("t0_core")],
    )
    return float(torch.nanmean(error))


def _select_epoch_count(
    records: list[PatientStates],
    features: Tensor,
    evaluation_weights: Tensor,
    input_occupancy: Tensor,
    outer_train: Tensor,
    config: V431TrainingConfig,
    *,
    fold: int,
    device: torch.device,
) -> tuple[int, list[dict[str, float]], dict[str, Any]]:
    fit_indices, calibration_indices = _inner_split(
        outer_train, seed=config.seed + fold * 70_001
    )
    combined = torch.cat((fit_indices, calibration_indices))
    anchors, anchor_report = _fit_anchor_bank(
        records,
        features,
        evaluation_weights,
        input_occupancy,
        fit_indices,
        combined,
        seed=config.seed + fold * 1_000_003 + 41,
        device=device,
    )
    seed = config.seed + fold * 10_000_019
    _set_seed(seed)
    model = CoarseRRRAnchoredCoreResidualJEPA(
        config.hidden_channels, config.conditioned_blocks
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    best = math.inf
    best_epoch_count = 0
    patience = 0
    history: list[dict[str, float]] = []
    for epoch in range(config.maximum_cv_epochs):
        fit_loss = _train_epoch(
            model,
            optimizer,
            records,
            input_occupancy,
            anchors,
            fit_indices,
            config,
            epoch=epoch,
            seed=seed,
            device=device,
        )
        value = _calibration_core_mae(
            model,
            records,
            features,
            evaluation_weights,
            input_occupancy,
            anchors,
            calibration_indices,
            config,
            seed=seed + 500_009,
            device=device,
        )
        row = {
            "epoch": float(epoch),
            "fit_loss": fit_loss,
            "inner_calibration_core_MAE": value,
        }
        history.append(row)
        print(
            json.dumps(
                {"stage": "v4p3p1_inner_epoch_selection", "fold": fold, **row},
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
        raise RuntimeError("v4.3.1 inner calibration selected no epoch")
    del model, optimizer, anchors
    torch.cuda.empty_cache()
    return best_epoch_count, history, anchor_report


def _fit_for_epochs(
    records: list[PatientStates],
    input_occupancy: Tensor,
    anchors: Tensor,
    indices: Tensor,
    config: V431TrainingConfig,
    *,
    epochs: int,
    seed: int,
    device: torch.device,
) -> tuple[dict[str, Tensor], list[dict[str, float]]]:
    _set_seed(seed)
    model = CoarseRRRAnchoredCoreResidualJEPA(
        config.hidden_channels, config.conditioned_blocks
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
            input_occupancy,
            anchors,
            indices,
            config,
            epoch=epoch,
            seed=seed,
            device=device,
        )
        history.append({"epoch": float(epoch), "train_total_loss": loss})
        print(
            json.dumps(
                {
                    "stage": "v4p3p1_outer_refit",
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
    model: CoarseRRRAnchoredCoreResidualJEPA,
    records: Sequence[PatientStates],
    input_occupancy: Tensor,
    anchors: Tensor,
    indices: Tensor,
    config: V431TrainingConfig,
    *,
    seed: int,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, float]:
    model.eval()
    permutation = _derangement(indices, seed=int(seed) + 55_003)
    candidate_rows: list[Tensor] = []
    anchor_rows: list[Tensor] = []
    shuffled_rows: list[Tensor] = []
    maximum_outside = 0.0
    draws = len(config.mask_fractions) * config.evaluation_mask_draws_per_fraction
    for start in range(0, len(indices), config.batch_size):
        chosen = indices[start : start + config.batch_size]
        shuffled = permutation[start : start + config.batch_size]
        selected = [records[int(index)] for index in chosen.tolist()]
        source = torch.stack([item.features[1] for item in selected]).to(device)
        t0 = torch.stack([item.features[0] for item in selected]).to(device)
        times = torch.stack([item.times_days for item in selected]).to(device)
        action = torch.stack([item.action_13 for item in selected]).to(device)
        core, ring = _core_ring(input_occupancy, chosen, device)
        shuffled_core, shuffled_ring = _core_ring(
            input_occupancy, shuffled, device
        )
        anchor = anchors[chosen].to(device, non_blocking=True)
        candidate_sum = torch.zeros_like(source, dtype=torch.float32)
        shuffled_sum = torch.zeros_like(source, dtype=torch.float32)
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
                        anchor,
                        times[:, 1],
                        times[:, 2],
                        action,
                        core,
                        ring,
                        history_delta=source - t0,
                        context_mask=mask,
                    )
                    shuffled_output = model.step(
                        source,
                        anchor,
                        times[:, 1],
                        times[:, 2],
                        action,
                        shuffled_core,
                        shuffled_ring,
                        history_delta=source - t0,
                        context_mask=mask,
                    )
                candidate_sum.add_(output.predicted.float())
                shuffled_sum.add_(shuffled_output.predicted.float())
                outside = output.core_gate == 0
                if bool(outside.any()):
                    maximum_outside = max(
                        maximum_outside,
                        float(
                            output.gated_core_residual[
                                outside.expand_as(output.gated_core_residual)
                            ]
                            .abs()
                            .max()
                        ),
                    )
        candidate_rows.append((candidate_sum / draws).cpu())
        shuffled_rows.append((shuffled_sum / draws).cpu())
        anchor_rows.append(anchor.float().cpu())
    return (
        torch.cat(candidate_rows),
        torch.cat(anchor_rows),
        torch.cat(shuffled_rows),
        maximum_outside,
    )


def _cross_validate(
    records: list[PatientStates],
    evaluation_weights: Tensor,
    input_occupancy: Tensor,
    config: V431TrainingConfig,
    *,
    resume_dir: Path,
    resume_contract: Mapping[str, Any],
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    features = _record_features(records)
    folds = _folds(409, seed=config.seed, count=config.grouped_cv_folds)
    matrices = {
        key: torch.full((409, len(REGIONS)), float("nan"))
        for key in ("candidate", "coarse_rrr", "occupancy_shuffled")
    }
    best_epochs: list[int] = []
    fold_evidence: list[dict[str, Any]] = []
    maximum_outside = 0.0
    resume_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    for fold, (outer_train, held) in enumerate(folds):
        cache_path = resume_dir / f"fold-{fold:02d}.pt"
        if cache_path.is_file():
            payload = torch.load(cache_path, map_location="cpu", weights_only=True)
            if (
                not isinstance(payload, Mapping)
                or payload.get("schema_version") != FOLD_SCHEMA
                or payload.get("resume_contract") != dict(resume_contract)
                or payload.get("fold") != fold
                or not torch.equal(payload.get("held_indices"), held)
            ):
                raise ValueError("resumable v4.3.1 fold changed")
            print(
                json.dumps(
                    {"stage": "v4p3p1_outer_fold_resume", "fold": fold},
                    sort_keys=True,
                ),
                flush=True,
            )
        else:
            epochs, inner_history, inner_anchor_report = _select_epoch_count(
                records,
                features,
                evaluation_weights,
                input_occupancy,
                outer_train,
                config,
                fold=fold,
                device=device,
            )
            combined = torch.cat((outer_train, held))
            anchors, outer_anchor_report = _fit_anchor_bank(
                records,
                features,
                evaluation_weights,
                input_occupancy,
                outer_train,
                combined,
                seed=config.seed + fold * 1_000_003 + 91,
                device=device,
            )
            state, outer_history = _fit_for_epochs(
                records,
                input_occupancy,
                anchors,
                outer_train,
                config,
                epochs=epochs,
                seed=config.seed + fold * 10_000_019 + 7,
                device=device,
            )
            model = CoarseRRRAnchoredCoreResidualJEPA(
                config.hidden_channels, config.conditioned_blocks
            ).to(device)
            model.load_state_dict(state, strict=True)
            candidate, anchor, shuffled, outside = _predict_ensemble(
                model,
                records,
                input_occupancy,
                anchors,
                held,
                config,
                seed=config.bootstrap_seed + fold * 1_000_003,
                device=device,
            )
            target = features[held, 2]
            payload = {
                "schema_version": FOLD_SCHEMA,
                "resume_contract": dict(resume_contract),
                "fold": fold,
                "held_indices": held,
                "best_epoch_count": epochs,
                "inner_history": inner_history,
                "outer_refit_history": outer_history,
                "inner_anchor_evidence": inner_anchor_report,
                "outer_anchor_evidence": outer_anchor_report,
                "candidate_errors": _region_errors(
                    candidate, target, evaluation_weights[held]
                ),
                "coarse_rrr_errors": _region_errors(
                    anchor, target, evaluation_weights[held]
                ),
                "occupancy_shuffled_errors": _region_errors(
                    shuffled, target, evaluation_weights[held]
                ),
                "maximum_outside_core_support_absolute": outside,
            }
            temporary = cache_path.with_suffix(".partial")
            torch.save(payload, temporary)
            os.chmod(temporary, 0o600)
            os.replace(temporary, cache_path)
            del model, state, anchors, candidate, anchor, shuffled
            torch.cuda.empty_cache()
        for key in matrices:
            value = payload[f"{key}_errors"]
            if not isinstance(value, Tensor) or value.shape != (
                len(held), len(REGIONS)
            ):
                raise ValueError("v4.3.1 fold error payload is malformed")
            matrices[key][held] = value
        best_epochs.append(int(payload["best_epoch_count"]))
        maximum_outside = max(
            maximum_outside,
            float(payload["maximum_outside_core_support_absolute"]),
        )
        fold_evidence.append(
            {
                "fold": fold,
                "held_patient_count": len(held),
                "best_epoch_count": int(payload["best_epoch_count"]),
                "inner_anchor_evidence": payload["inner_anchor_evidence"],
                "outer_anchor_evidence": payload["outer_anchor_evidence"],
            }
        )
        gc.collect()
        torch.cuda.empty_cache()
    if any(torch.isnan(value).any() for value in matrices.values()):
        raise RuntimeError("v4.3.1 OOF matrix is incomplete")
    ids = [item.patient_id for item in records]
    gate = evaluate_v431_train_oof_gate(
        matrices["candidate"],
        matrices["coarse_rrr"],
        matrices["occupancy_shuffled"],
        ids,
        maximum_outside_core_support_absolute=maximum_outside,
        seed=config.bootstrap_seed,
        replicates=config.bootstrap_replicates,
    )
    summary = {
        "development_gate_passed": gate["development_gate_passed"],
        "candidate_region_MAE": {
            region: _summary(matrices["candidate"][:, index])
            for index, region in enumerate(REGIONS)
        },
        "coarse_RRR_region_MAE": {
            region: _summary(matrices["coarse_rrr"][:, index])
            for index, region in enumerate(REGIONS)
        },
        "best_epoch_counts": best_epochs,
        "fold_evidence": fold_evidence,
    }
    public = {
        "candidate": "coarse_RRR_anchored_fractional_T0_core_residual",
        "candidate_count": 1,
        "candidate_summary": summary,
        "oof_gate": gate,
        "outer_folds": config.grouped_cv_folds,
        "outer_held_fold_used_for_early_stopping": False,
        "coarse_RRR_selected_and_fitted_inside_training_folds": True,
        "selected_candidate": (
            "coarse_RRR_anchored_fractional_T0_core_residual"
            if gate["development_gate_passed"]
            else None
        ),
    }
    private = {
        "schema_version": PRIVATE_SCHEMA,
        "patient_ids": ids,
        "error_matrices": matrices,
        "selected_candidate": public["selected_candidate"],
    }
    return public, private


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("v4.3.1 train409 OOF requires exactly one CUDA GPU")
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
            raise FileNotFoundError(f"required v4.3.1 input unavailable: {name}")
    if not paths["feature_root"].is_dir() or not paths["mask_root"].is_dir():
        raise FileNotFoundError("required v4.3.1 governed release is unavailable")
    if paths["output"].exists() or paths["private_output"].exists():
        raise FileExistsError("immutable v4.3.1 output exists")
    output_staging = paths["output"].parent / f".{paths['output'].name}.staging"
    private_staging = (
        paths["private_output"].parent / f".{paths['private_output'].name}.staging"
    )
    output_staging.mkdir(parents=True, exist_ok=True, mode=0o700)
    private_staging.mkdir(parents=True, exist_ok=True, mode=0o700)

    config = V431TrainingConfig.load(paths["config"])
    design = _json(paths["frozen_design"])
    manifest_report = _json(paths["manifest_report"])
    evidence = design.get("evidence", {})
    if (
        design.get("schema_version") != DESIGN_SCHEMA
        or design.get("status")
        != "frozen_train409_core_residual_oof_required_training_not_started"
        or design.get("training_started") is not False
        or design.get("validation_88_loaded") is not False
        or design.get("test_split_loaded") is not False
        or evidence.get("config_sha256") != _sha(paths["config"])
        or evidence.get("manifest_sha256") != _sha(paths["manifest"])
        or evidence.get("train_feature_index_sha256") != _sha(paths["feature_index"])
        or manifest_report.get("status") != "ok"
        or manifest_report.get("manifest_sha256") != _sha(paths["manifest"])
    ):
        raise ValueError("v4.3.1 frozen prerequisite chain is invalid")

    implementation_lock = {
        "schema_version": "med-jepa-roi-v4p3p1-implementation-lock-v1",
        "status": "locked_before_train409_payload_loading",
        "frozen_design_sha256": _sha(paths["frozen_design"]),
        "config_sha256": _sha(paths["config"]),
        "source_sha256": {
            name: _sha(Path(__file__).with_name(name))
            for name in (
                "anchor.py",
                "config.py",
                "freeze_design.py",
                "gate.py",
                "losses.py",
                "model.py",
                "train.py",
            )
        },
        "fixed_reference": "occupancy_conditioned_coarse_RRR",
        "neural_global_branch": False,
        "neural_ring_residual_branch": False,
        "T0_only_occupancy_input": True,
        "validation_88_loaded": False,
        "test_split_loaded": False,
    }
    lock_path = output_staging / "implementation_lock.json"
    _write_json(lock_path, implementation_lock)

    (
        records,
        evaluation_weights,
        input_occupancy,
        feature_merkle,
        occupancy_digest,
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
        raise ValueError("v4.3.1 train409 release cardinality changed")
    resume_contract = {
        "config_sha256": _sha(paths["config"]),
        "frozen_design_sha256": _sha(paths["frozen_design"]),
        "implementation_lock_sha256": _sha(lock_path),
        "manifest_sha256": _sha(paths["manifest"]),
        "feature_index_sha256": _sha(paths["feature_index"]),
        "feature_payload_merkle_sha256": feature_merkle,
        "context_mask_report_sha256": _sha(paths["mask_report"]),
        "context_mask_index_sha256": _sha(paths["mask_index"]),
        "T0_only_input_occupancy_digest": occupancy_digest,
        "training_source_sha256": _sha(Path(__file__)),
        "anchor_source_sha256": _sha(Path(__file__).with_name("anchor.py")),
        "model_source_sha256": _sha(Path(__file__).with_name("model.py")),
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
    final_anchor_report: Mapping[str, Any] | None = None
    if selected is not None:
        final_epoch_count = max(
            1,
            int(
                round(
                    statistics.median(
                        cv["candidate_summary"]["best_epoch_counts"]
                    )
                )
            ),
        )
        all_indices = torch.arange(409)
        features = _record_features(records)
        anchors, final_anchor_report = _fit_anchor_bank(
            records,
            features,
            evaluation_weights,
            input_occupancy,
            all_indices,
            all_indices,
            seed=config.seed + 900_001,
            device=device,
        )
        state, final_history = _fit_for_epochs(
            records,
            input_occupancy,
            anchors,
            all_indices,
            config,
            epochs=final_epoch_count,
            seed=config.seed,
            device=device,
        )
        checkpoint_path = output_staging / "checkpoint_train409.pt"
        torch.save(
            {
                "schema_version": SCHEMA,
                "model": state,
                "config": asdict(config),
                "selected_candidate": selected,
                "final_epoch_count": final_epoch_count,
                "coarse_RRR_train409_fit_report": final_anchor_report,
                "coarse_RRR_refit_required_from_bound_train409_release": True,
                "validation_88_loaded": False,
                "test_split_loaded": False,
            },
            checkpoint_path,
        )
        os.chmod(checkpoint_path, 0o600)
        del state, anchors, features
        torch.cuda.empty_cache()

    history_path = output_staging / "final_training_metrics.jsonl"
    with history_path.open("x", encoding="utf-8") as handle:
        for row in final_history:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    os.chmod(history_path, 0o600)
    private_path = private_staging / "train409_oof_measurements.pt"
    torch.save(private, private_path)
    os.chmod(private_path, 0o600)
    status = (
        "completed_train409_core_residual_gate_pass_final_model_fitted"
        if selected is not None
        else "completed_train409_core_residual_gate_fail_no_final_model"
    )
    report = {
        "schema_version": SCHEMA,
        "status": status,
        "scope": "adaptive_train409_patient_grouped_five_fold_OOF_only",
        "seed": config.seed,
        "patient_counts": {
            "train_loaded": 409,
            "validation_loaded": 0,
            "test_loaded": 0,
        },
        "selected_feature_variant": design["feature_contract"]["variant"],
        "train409_cross_validation": cv,
        "selected_candidate": selected,
        "final_epoch_count": final_epoch_count,
        "final_coarse_RRR_fit_report": final_anchor_report,
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
