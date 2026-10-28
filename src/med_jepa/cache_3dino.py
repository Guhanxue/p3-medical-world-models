"""Fail-closed 3DINO feature caching for governed, preprocessed MRI ROIs.

This module deliberately does not download, vendor, or modify 3DINO.  It is intended
to run on the authorized cluster against an independently checked-out upstream
repository and a locally approved checkpoint.  Raw DICOM conversion is a separate
stage because its geometry must be audited against the actual ISPY2 series layout.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from torch import Tensor, nn

from med_jepa.ispy2_manifest_builder import (
    ManifestBuildError,
    validate_spatial_provenance,
)


THREE_DINO_REVISION = "85bd4435c1b2ada41cd34cd15cad17c4d3c88d89"
THREE_DINO_HF_REVISION = "8a00a2bb14becb3fbe955064837322a3217f7ac0"
THREE_DINO_LICENSE = "CC BY-NC-ND 4.0"
# The pinned public checkpoint stores the ViT backbone alongside both
# self-supervised training heads under ``teacher``.  The heads are not part of the
# inference model, but the complete paired namespace layout is verified before they
# are discarded.  The independently pinned checkpoint digest verifies their exact
# key counts and tensor contents.
THREE_DINO_TEACHER_AUXILIARY_PREFIXES = (
    "dino_head.",
    "ibot_head.",
)
DEFAULT_INPUT_SHAPE = (112, 112, 112)
DEFAULT_FEATURE_DIM = 1024
DEFAULT_PATCH_SIZE = 16
ROI_SCHEMA_VERSION = "med-jepa-ispy2-roi-v2"
SER_NORMALIZATION = "fixed_ser_ratio_clip_0_3_to_minus1_plus1_v1"
SER_INPUT_RANGE = (0.0, 3.0)
SER_RANGE_TOLERANCE = 1e-4
REQUIRED_INDEX_COLUMNS = (
    "PatientID",
    "tp_label",
    "scan_id",
    "roi_path",
    "source_content_sha256",
)
OUTPUT_INDEX_COLUMNS = (
    "PatientID",
    "tp_label",
    "scan_id",
    "feature_path",
    "content_sha256",
    "source_content_sha256",
)


@dataclass(frozen=True)
class ROIRecord:
    patient_id: str
    timepoint: str
    scan_id: str
    roi_path: Path
    source_content_sha256: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _valid_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def read_roi_index(
    path: str | Path,
    *,
    allowed_input_root: str | Path,
    expected_timepoints: Sequence[str] = ("T0", "T1", "T3"),
    expected_patients: int | None = None,
) -> list[ROIRecord]:
    """Read and globally validate the governed ROI index without loading tensors."""

    index_path = Path(path).expanduser().resolve()
    allowed_root = Path(allowed_input_root).expanduser().resolve()
    if not allowed_root.is_dir():
        raise ValueError("allowed input root is not an existing directory")
    with index_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("ROI index has no header")
        missing = sorted(set(REQUIRED_INDEX_COLUMNS) - set(reader.fieldnames))
        if missing:
            raise ValueError(f"ROI index is missing required columns: {missing}")
        raw_rows = list(reader)
    if not raw_rows:
        raise ValueError("ROI index is empty")

    allowed_timepoints = tuple(expected_timepoints)
    if len(set(allowed_timepoints)) != len(allowed_timepoints) or not allowed_timepoints:
        raise ValueError("expected timepoints must be non-empty and unique")
    records: list[ROIRecord] = []
    patient_visits: dict[str, set[str]] = {}
    scan_ids: set[str] = set()
    source_hashes: set[str] = set()
    for row_number, row in enumerate(raw_rows, start=2):
        patient_id = str(row["PatientID"]).strip()
        timepoint = str(row["tp_label"]).strip().upper()
        scan_id = str(row["scan_id"]).strip()
        digest = str(row["source_content_sha256"]).strip().lower()
        if not patient_id or not scan_id:
            raise ValueError(f"ROI index row {row_number} has a blank identifier")
        if timepoint not in allowed_timepoints:
            raise ValueError(f"ROI index row {row_number} has an unexpected timepoint")
        if not _valid_sha256(digest):
            raise ValueError(f"ROI index row {row_number} has an invalid SHA-256")
        roi_path = Path(str(row["roi_path"])).expanduser()
        if not roi_path.is_absolute():
            roi_path = (index_path.parent / roi_path).resolve()
        else:
            roi_path = roi_path.resolve()
        if not _within(roi_path, allowed_root):
            raise ValueError(f"ROI index row {row_number} escapes allowed input root")
        visits = patient_visits.setdefault(patient_id, set())
        if timepoint in visits:
            raise ValueError(f"ROI index row {row_number} duplicates a patient visit")
        visits.add(timepoint)
        if scan_id in scan_ids:
            raise ValueError(f"ROI index row {row_number} reuses a scan_id")
        scan_ids.add(scan_id)
        if digest in source_hashes:
            raise ValueError(f"ROI index row {row_number} reuses tensor content")
        source_hashes.add(digest)
        records.append(
            ROIRecord(
                patient_id=patient_id,
                timepoint=timepoint,
                scan_id=scan_id,
                roi_path=roi_path,
                source_content_sha256=digest,
            )
        )

    expected_visit_set = set(allowed_timepoints)
    if any(visits != expected_visit_set for visits in patient_visits.values()):
        raise ValueError("every patient must have exactly the expected ROI visits")
    if expected_patients is not None and len(patient_visits) != expected_patients:
        raise ValueError(
            f"ROI index patient count mismatch: expected {expected_patients}, "
            f"observed {len(patient_visits)}"
        )
    return sorted(records, key=lambda item: (item.patient_id, item.timepoint))


def load_roi_payload(
    path: Path, expected_shape: Sequence[int]
) -> tuple[Tensor, dict[str, Any], dict[str, Any]]:
    """Load a formal SER ROI and its context-only spatial provenance."""

    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict):
        raise TypeError("formal ROI cache must contain a dictionary payload")
    if value.get("schema_version") != ROI_SCHEMA_VERSION:
        raise ValueError("ROI payload schema_version does not match the formal contract")
    if str(value.get("modality", "")).upper() != "SER":
        raise ValueError("ROI payload modality must be SER")
    if value.get("tensor_kind") != "volume":
        raise ValueError("ROI payload tensor_kind must be volume")
    volume = value.get("volume")
    if not isinstance(volume, Tensor):
        raise TypeError("ROI payload volume is not a tensor")
    if not volume.is_floating_point():
        raise TypeError("ROI payload volume must already be floating point")
    shape = tuple(int(item) for item in expected_shape)
    if volume.shape != (1, *shape):
        raise ValueError(f"ROI tensor must have shape [1,{','.join(map(str, shape))}]")
    volume = volume.float()
    if not torch.isfinite(volume).all():
        raise ValueError("ROI tensor contains non-finite values")
    try:
        spatial = validate_spatial_provenance(value.get("spatial_provenance"))
    except (ManifestBuildError, TypeError) as error:
        raise ValueError("ROI payload has invalid spatial_provenance") from error
    if tuple(spatial["roi_voxel_shape"]) != shape:
        raise ValueError("ROI tensor shape disagrees with spatial_provenance")
    intensity = value.get("intensity_provenance")
    if not isinstance(intensity, dict):
        raise ValueError("ROI payload must contain intensity_provenance")
    required_intensity = {
        "quantity",
        "units",
        "dicom_rescale_applied",
        "dicom_rescale_slope",
        "dicom_rescale_intercept",
        "value_state",
        "prior_intensity_transform",
        "converter_version",
        "converter_source_sha256",
        "source_dicom_content_sha256",
    }
    if not required_intensity.issubset(intensity):
        raise ValueError("ROI intensity_provenance lacks required fields")
    if intensity["quantity"] != "signal_enhancement_ratio":
        raise ValueError("ROI intensity quantity must be signal_enhancement_ratio")
    if intensity["units"] != "unitless_ratio":
        raise ValueError("ROI intensity units must be unitless_ratio")
    if intensity["dicom_rescale_applied"] is not True:
        raise ValueError("ROI intensity must have the DICOM rescale applied")
    if intensity["value_state"] != "dicom_rescaled_ser":
        raise ValueError("ROI intensity value_state must be dicom_rescaled_ser")
    if intensity["prior_intensity_transform"] != "none":
        raise ValueError("ROI intensity must not have prior percentile or min-max scaling")
    if intensity["converter_version"] != "med-jepa-ispy2-roi-converter-v2":
        raise ValueError("ROI intensity converter_version is not approved")
    for field in ("converter_source_sha256", "source_dicom_content_sha256"):
        digest = str(intensity[field]).strip().lower()
        if not _valid_sha256(digest):
            raise ValueError(f"ROI intensity {field} must be a valid SHA-256")
    try:
        slope = float(intensity["dicom_rescale_slope"])
        intercept = float(intensity["dicom_rescale_intercept"])
    except (TypeError, ValueError) as error:
        raise ValueError("ROI DICOM rescale provenance must be numeric") from error
    if not torch.isfinite(torch.tensor([slope, intercept])).all() or slope == 0.0:
        raise ValueError("ROI DICOM rescale provenance is invalid")
    normalized_intensity = deepcopy(intensity)
    normalized_intensity["dicom_rescale_slope"] = slope
    normalized_intensity["dicom_rescale_intercept"] = intercept
    normalized_intensity["converter_source_sha256"] = str(
        intensity["converter_source_sha256"]
    ).strip().lower()
    normalized_intensity["source_dicom_content_sha256"] = str(
        intensity["source_dicom_content_sha256"]
    ).strip().lower()
    return volume, spatial, normalized_intensity


def fixed_ser_normalize(
    volume: Tensor,
    *,
    input_minimum: float = SER_INPUT_RANGE[0],
    input_maximum: float = SER_INPUT_RANGE[1],
    tolerance: float = SER_RANGE_TOLERANCE,
) -> Tensor:
    """Preserve absolute SER while mapping the fixed physical range to ``[-1, 1]``."""

    if (
        not input_minimum < input_maximum
        or not torch.isfinite(torch.tensor([input_minimum, input_maximum])).all()
        or not 0 <= tolerance < (input_maximum - input_minimum)
    ):
        raise ValueError("fixed SER normalization range is invalid")
    if not torch.isfinite(volume).all():
        raise ValueError("cannot normalize non-finite input")
    observed_minimum = float(volume.min())
    observed_maximum = float(volume.max())
    if observed_minimum < input_minimum - tolerance or observed_maximum > input_maximum + tolerance:
        raise ValueError(
            "ROI is outside the fixed rescaled SER range; DICOM rescale may be missing"
        )
    if observed_maximum - observed_minimum <= tolerance:
        raise ValueError("ROI is constant and fails intensity QC")
    clipped = torch.clamp(volume.float(), input_minimum, input_maximum)
    normalized = (
        (clipped - input_minimum) / (input_maximum - input_minimum) * 2.0 - 1.0
    )
    if not torch.isfinite(normalized).all():
        raise ValueError("normalization produced non-finite values")
    return normalized


def extract_spatial_features(
    model: nn.Module,
    volume: Tensor,
    *,
    device: torch.device,
    expected_feature_dim: int = DEFAULT_FEATURE_DIM,
    patch_size: int = DEFAULT_PATCH_SIZE,
    autocast_dtype: torch.dtype | None = None,
) -> Tensor:
    """Return the final normalized patch map as ``[F,Df,Hf,Wf]`` on CPU."""

    if volume.ndim != 4 or volume.shape[0] != 1:
        raise ValueError("normalized ROI must have shape [1,D,H,W]")
    spatial = tuple(int(value) for value in volume.shape[-3:])
    if any(value % patch_size for value in spatial):
        raise ValueError("all ROI dimensions must be divisible by the patch size")
    batch = volume.unsqueeze(0).to(device)
    enabled = autocast_dtype is not None and device.type == "cuda"
    with torch.inference_mode(), torch.autocast(
        device_type=device.type,
        dtype=autocast_dtype or torch.float32,
        enabled=enabled,
    ):
        outputs = model.get_intermediate_layers(batch, n=1, reshape=True)
    if not isinstance(outputs, (list, tuple)) or len(outputs) != 1:
        raise RuntimeError("3DINO returned an unexpected intermediate-layer structure")
    features = outputs[0]
    expected_grid = tuple(value // patch_size for value in spatial)
    if features.shape != (1, expected_feature_dim, *expected_grid):
        raise RuntimeError(
            "3DINO feature shape mismatch: expected "
            f"{(1, expected_feature_dim, *expected_grid)}, observed {tuple(features.shape)}"
        )
    if not torch.isfinite(features).all():
        raise RuntimeError("3DINO produced non-finite features")
    return features[0].detach().to(device="cpu", dtype=torch.float16).contiguous()


def verify_three_dino_checkout(repo: str | Path, expected_revision: str) -> Path:
    root = Path(repo).expanduser().resolve()
    if not (root / "dinov2" / "models" / "vision_transformer.py").is_file():
        raise ValueError("3DINO repository does not contain the expected source tree")
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()
    if revision != expected_revision:
        raise RuntimeError(
            f"3DINO revision mismatch: expected {expected_revision}, observed {revision}"
        )
    dirty = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()
    if dirty:
        raise RuntimeError("3DINO checkout has modified or untracked files")
    return root


def _strip_upstream_prefixes(
    state: dict[str, Tensor],
) -> dict[str, Tensor]:
    normalized_state: dict[str, Tensor] = {}
    module_prefix_flags: list[bool] = []
    for key, value in state.items():
        module_prefixed = key.startswith("module.")
        module_prefix_flags.append(module_prefixed)
        normalized = key
        if module_prefixed:
            normalized = normalized[len("module.") :]
        if normalized.startswith("module."):
            raise RuntimeError("checkpoint contains a repeated module prefix")
        if normalized in normalized_state:
            raise RuntimeError("checkpoint keys collide after prefix normalization")
        normalized_state[normalized] = value
    if any(module_prefix_flags) and not all(module_prefix_flags):
        raise RuntimeError("checkpoint mixes module-prefixed and unprefixed keys")

    backbone: dict[str, Tensor] = {}
    direct: dict[str, Tensor] = {}
    auxiliary_counts = {
        prefix: 0 for prefix in THREE_DINO_TEACHER_AUXILIARY_PREFIXES
    }
    for key, value in normalized_state.items():
        if key.startswith("backbone."):
            stripped = key[len("backbone.") :]
            if stripped in backbone:
                raise RuntimeError("checkpoint backbone keys collide after normalization")
            backbone[stripped] = value
            continue
        matched_prefix = next(
            (
                prefix
                for prefix in THREE_DINO_TEACHER_AUXILIARY_PREFIXES
                if key.startswith(prefix)
            ),
            None,
        )
        if matched_prefix is not None:
            auxiliary_counts[matched_prefix] += 1
        else:
            direct[key] = value

    observed_auxiliary = {
        prefix for prefix, count in auxiliary_counts.items() if count > 0
    }
    if backbone:
        if direct:
            raise RuntimeError(
                "checkpoint mixes backbone.* with an unapproved sibling namespace"
            )
        if observed_auxiliary and observed_auxiliary != set(
            THREE_DINO_TEACHER_AUXILIARY_PREFIXES
        ):
            raise RuntimeError(
                "checkpoint contains an incomplete auxiliary-head namespace pair"
            )
        return backbone
    if observed_auxiliary:
        raise RuntimeError("checkpoint auxiliary heads are present without backbone.*")
    return direct


def load_checkpoint_fail_closed(
    model: nn.Module,
    checkpoint: str | Path,
    *,
    expected_sha256: str,
    checkpoint_key: str | None = "teacher",
    allowed_missing_keys: Iterable[str] = (),
    allowed_unexpected_keys: Iterable[str] = (),
) -> str:
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    expected_digest = expected_sha256.lower()
    if not _valid_sha256(expected_digest):
        raise ValueError("expected checkpoint SHA-256 is invalid")
    if not checkpoint_path.is_file():
        raise FileNotFoundError("3DINO checkpoint does not exist")
    observed_digest = sha256_file(checkpoint_path)
    if observed_digest != expected_digest:
        raise RuntimeError("3DINO checkpoint SHA-256 mismatch")
    value = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict):
        raise RuntimeError("checkpoint root is not a dictionary")
    if checkpoint_key is None:
        state = value
    else:
        if checkpoint_key not in value:
            raise RuntimeError(
                f"checkpoint lacks required {checkpoint_key!r} state dictionary"
            )
        state = value[checkpoint_key]
    if not isinstance(state, dict) or not state or any(
        not isinstance(key, str) or not isinstance(tensor, Tensor)
        for key, tensor in state.items()
    ):
        raise RuntimeError("checkpoint teacher entry is not a tensor state dictionary")
    model_state = _strip_upstream_prefixes(state)
    result = model.load_state_dict(model_state, strict=False)
    observed_missing = set(result.missing_keys)
    observed_unexpected = set(result.unexpected_keys)
    expected_missing = set(allowed_missing_keys)
    expected_unexpected = set(allowed_unexpected_keys)
    if observed_missing != expected_missing or observed_unexpected != expected_unexpected:
        raise RuntimeError(
            "checkpoint/model key mismatch; observed missing/unexpected keys do not "
            "exactly match the preregistered allowlists"
        )
    return observed_digest


def build_three_dino_model(
    repo: Path,
    checkpoint: Path,
    *,
    expected_checkpoint_sha256: str,
    checkpoint_key: str | None = "teacher",
    allowed_missing_keys: Iterable[str] = (),
    allowed_unexpected_keys: Iterable[str] = (),
) -> nn.Module:
    """Dynamically import an unmodified checkout and strictly load its teacher."""

    if os.environ.get("XFORMERS_DISABLED") != "1":
        raise RuntimeError(
            "formal 3DINO extraction requires XFORMERS_DISABLED=1"
        )
    if any(name == "dinov2" or name.startswith("dinov2.") for name in sys.modules):
        raise RuntimeError("dinov2 was imported before the audited checkout was selected")
    sys.path.insert(0, str(repo))
    try:
        from dinov2.configs import load_and_merge_config_3d
        from dinov2.models import build_model_from_cfg

        config = load_and_merge_config_3d("train/vit3d_highres")
        model, feature_dim = build_model_from_cfg(config, only_teacher=True)
    finally:
        if sys.path[0] == str(repo):
            sys.path.pop(0)
    if int(feature_dim) != DEFAULT_FEATURE_DIM:
        raise RuntimeError("audited 3DINO configuration no longer has feature_dim=1024")
    load_checkpoint_fail_closed(
        model,
        checkpoint,
        expected_sha256=expected_checkpoint_sha256,
        checkpoint_key=checkpoint_key,
        allowed_missing_keys=allowed_missing_keys,
        allowed_unexpected_keys=allowed_unexpected_keys,
    )
    return model.eval()


def _opaque_name(record: ROIRecord) -> str:
    digest = hashlib.sha256(
        "\0".join((record.patient_id, record.timepoint, record.scan_id)).encode("utf-8")
    ).hexdigest()
    return f"{digest[:32]}.pt"


def _atomic_torch_save(value: Any, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError("refusing to overwrite an existing feature tensor")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
        torch.save(value, temporary)
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def cache_shard(
    records: Sequence[ROIRecord],
    *,
    model: nn.Module,
    output_dir: str | Path,
    shard_index: int,
    num_shards: int,
    input_shape: Sequence[int] = DEFAULT_INPUT_SHAPE,
    checkpoint_sha256: str,
    device: torch.device,
    autocast_dtype: torch.dtype | None = torch.float16,
    expected_feature_dim: int = DEFAULT_FEATURE_DIM,
    patch_size: int = DEFAULT_PATCH_SIZE,
    checkpoint_key: str = "teacher",
    source_revision: str = THREE_DINO_REVISION,
    roi_index_sha256: str | None = None,
) -> dict[str, Any]:
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must satisfy 0 <= shard_index < num_shards")
    selected = [record for index, record in enumerate(records) if index % num_shards == shard_index]
    if not selected:
        raise ValueError("cache shard would be empty; reduce num_shards")
    shard_dir = Path(output_dir).expanduser().resolve() / (
        f"shard-{shard_index:05d}-of-{num_shards:05d}"
    )
    if shard_dir.exists():
        raise FileExistsError("refusing to overwrite an existing cache shard")
    rows: list[dict[str, str]] = []
    shape = tuple(int(value) for value in input_shape)
    canonical_spatial: dict[str, Any] | None = None
    if len(source_revision) != 40 or any(
        character not in "0123456789abcdefABCDEF" for character in source_revision
    ):
        raise ValueError("source_revision must be a full 40-character git revision")
    if roi_index_sha256 is not None and not _valid_sha256(roi_index_sha256):
        raise ValueError("roi_index_sha256 is invalid")
    autocast_name = (
        "float32" if autocast_dtype is None else str(autocast_dtype).removeprefix("torch.")
    )
    feature_provenance = {
        "extractor": "AICONSlab/3DINO-ViT",
        "source_revision": source_revision.lower(),
        "hf_revision": THREE_DINO_HF_REVISION,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_key": checkpoint_key,
        "layer": "last_normalized_patch_tokens",
        "feature_dim": expected_feature_dim,
        "feature_grid_shape": [value // patch_size for value in shape],
        "input_shape": list(shape),
        "normalization": SER_NORMALIZATION,
        "input_quantity": "signal_enhancement_ratio",
        "input_units": "unitless_ratio",
        "input_range": list(SER_INPUT_RANGE),
        "per_roi_intensity_rescaling": False,
        "attention_backend": "pytorch_eager_xformers_disabled_v1",
        "torch_version": str(torch.__version__),
        "torch_cuda_version": (
            None if torch.version.cuda is None else str(torch.version.cuda)
        ),
        "compute_device": device.type,
        "autocast_dtype": autocast_name,
        "feature_storage_dtype": "float16",
        "license": THREE_DINO_LICENSE,
    }
    if roi_index_sha256 is not None:
        feature_provenance["roi_index_sha256"] = roi_index_sha256.lower()
    if device.type == "cuda":
        feature_provenance["cuda_device_type"] = torch.cuda.get_device_name(device)
        feature_provenance["cuda_device_capability"] = list(
            torch.cuda.get_device_capability(device)
        )
    try:
        for record in selected:
            observed_source_hash = sha256_file(record.roi_path)
            if observed_source_hash != record.source_content_sha256:
                raise RuntimeError("source ROI hash mismatch")
            roi, spatial_provenance, intensity_provenance = load_roi_payload(
                record.roi_path, shape
            )
            if canonical_spatial is None:
                canonical_spatial = spatial_provenance
            elif spatial_provenance != canonical_spatial:
                raise RuntimeError("ROI spatial provenance is inconsistent within the shard")
            normalized = fixed_ser_normalize(roi)
            feature = extract_spatial_features(
                model,
                normalized,
                device=device,
                expected_feature_dim=expected_feature_dim,
                patch_size=patch_size,
                autocast_dtype=autocast_dtype,
            )
            # Delay materializing the immutable shard until one real model forward
            # succeeds. A generic GPU/runtime incompatibility therefore does not
            # consume the run ID by leaving an empty cache directory.
            if not shard_dir.exists():
                shard_dir.mkdir(parents=True, exist_ok=False)
            destination = shard_dir / _opaque_name(record)
            payload = {
                "volume": feature,
                "feature_provenance": deepcopy(feature_provenance),
                "source_content_sha256": record.source_content_sha256,
                "spatial_provenance": spatial_provenance,
                "source_intensity_provenance": intensity_provenance,
            }
            _atomic_torch_save(payload, destination)
            rows.append(
                {
                    "PatientID": record.patient_id,
                    "tp_label": record.timepoint,
                    "scan_id": record.scan_id,
                    "feature_path": str(destination),
                    "content_sha256": sha256_file(destination),
                    "source_content_sha256": record.source_content_sha256,
                }
            )
        index_path = shard_dir / "feature_index.csv"
        with index_path.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=OUTPUT_INDEX_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        assert canonical_spatial is not None
        spatial_json = json.dumps(
            canonical_spatial, allow_nan=False, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
        report = {
            "schema_version": "med-jepa-3dino-cache-v2",
            "shard_index": shard_index,
            "num_shards": num_shards,
            "cached_records": len(rows),
            "input_shape": list(shape),
            "feature_grid_shape": [value // patch_size for value in shape],
            "feature_dim": expected_feature_dim,
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_key": checkpoint_key,
            "source_revision": source_revision.lower(),
            "hf_revision": THREE_DINO_HF_REVISION,
            "feature_provenance": feature_provenance,
            "spatial_provenance": canonical_spatial,
            "spatial_provenance_sha256": hashlib.sha256(spatial_json).hexdigest(),
        }
        with (shard_dir / "report.json").open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
        return report
    except Exception:
        # Keep the partial shard for forensic inspection; a rerun must use a fresh
        # output directory instead of silently replacing potentially mixed artifacts.
        raise


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True, help="Governed preprocessed ROI CSV")
    parser.add_argument("--allowed-input-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--three-dino-repo", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument(
        "--checkpoint-key",
        default="teacher",
        help="Checkpoint state-dict key, or NONE for a root state dict",
    )
    parser.add_argument("--expected-revision", default=THREE_DINO_REVISION)
    parser.add_argument("--expected-patients", type=int, required=True)
    parser.add_argument("--input-shape", type=int, nargs=3, default=DEFAULT_INPUT_SHAPE)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--allow-missing-key", action="append", default=[])
    parser.add_argument("--allow-unexpected-key", action="append", default=[])
    parser.add_argument(
        "--autocast-dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float16",
    )
    parser.add_argument(
        "--acknowledge-cc-by-nc-nd-4.0",
        action="store_true",
        dest="license_acknowledged",
        help="Confirm this run is authorized under the upstream noncommercial license",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.license_acknowledged:
        raise SystemExit(
            "Refusing to load 3DINO without --acknowledge-cc-by-nc-nd-4.0"
        )
    if not torch.cuda.is_available():
        raise SystemExit("3DINO feature caching requires a CUDA compute node")
    repo = verify_three_dino_checkout(args.three_dino_repo, args.expected_revision)
    records = read_roi_index(
        args.index,
        allowed_input_root=args.allowed_input_root,
        expected_patients=args.expected_patients,
    )
    roi_index_sha256 = sha256_file(Path(args.index).expanduser().resolve())
    checkpoint_key = None if args.checkpoint_key.upper() == "NONE" else args.checkpoint_key
    model = build_three_dino_model(
        repo,
        Path(args.checkpoint),
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
        checkpoint_key=checkpoint_key,
        allowed_missing_keys=args.allow_missing_key,
        allowed_unexpected_keys=args.allow_unexpected_key,
    ).cuda()
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": None,
    }[args.autocast_dtype]
    report = cache_shard(
        records,
        model=model,
        output_dir=args.output_dir,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        input_shape=args.input_shape,
        checkpoint_sha256=args.expected_checkpoint_sha256.lower(),
        device=torch.device("cuda"),
        autocast_dtype=dtype,
        checkpoint_key="ROOT" if checkpoint_key is None else checkpoint_key,
        source_revision=args.expected_revision,
        roi_index_sha256=roi_index_sha256,
    )
    # Aggregate-only output: no patient or scan identifiers are printed.
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
