"""Adapter for preprocessed local trajectory tensors described by JSON Lines.

Raw DICOM conversion is intentionally outside this package: it should run where the
ISPY2 data are mounted. This loader only reads de-identified, spatially standardized
``.pt`` tensors produced by that preprocessing step.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from med_jepa.data.contracts import TrajectorySample


_CACHED_TENSOR_KINDS = {"cached_features"}
_SUPPORTED_TENSOR_KINDS = {"volume", *_CACHED_TENSOR_KINDS}
_FEATURE_PROVENANCE_FIELDS = (
    "extractor",
    "source_revision",
    "checkpoint_sha256",
    "layer",
    "feature_dim",
    "feature_grid_shape",
    "input_shape",
    "normalization",
)


def _as_float(value: Any) -> float:
    return float("nan") if value is None else float(value)


def _validate_feature_names(value: Any, *, field: str, owner: str) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(name, str) or not name.strip() for name in value
    ):
        raise ValueError(f"{owner} {field} must be a list of non-empty strings")
    if len(set(value)) != len(value):
        raise ValueError(f"{owner} {field} contains duplicate names")
    return list(value)


def _nonempty_text(value: Any, *, field: str, owner: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{owner} {field} must be a non-empty string")
    return value.strip()


def _sha256(value: Any, *, field: str, owner: str) -> str:
    digest = _nonempty_text(value, field=field, owner=owner).lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{owner} {field} must be a 64-character hex digest")
    return digest


def _source_revision(value: Any, *, owner: str) -> str:
    revision = _nonempty_text(value, field="source_revision", owner=owner).lower()
    if len(revision) not in {40, 64} or any(
        character not in "0123456789abcdef" for character in revision
    ):
        raise ValueError(
            f"{owner} source_revision must be a full 40- or 64-character hex revision"
        )
    return revision


def _positive_int_triplet(value: Any, *, field: str, owner: str) -> list[int]:
    if (
        not isinstance(value, list)
        or len(value) != 3
        or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in value)
    ):
        raise ValueError(f"{owner} {field} must contain three positive integers")
    return [int(item) for item in value]


def _top_level_contract(
    records: list[dict[str, Any]],
    *,
    expected_schema_version: str | None,
    expected_modality: str | None,
    expected_tensor_kind: str | None,
    formal_provenance_required: bool,
) -> dict[str, str] | None:
    fields = ("schema_version", "modality", "tensor_kind")
    expected = {
        "schema_version": expected_schema_version,
        "modality": expected_modality,
        "tensor_kind": expected_tensor_kind,
    }
    expected = {
        field: (
            _nonempty_text(value, field=field, owner="configured")
            if value is not None
            else None
        )
        for field, value in expected.items()
    }
    active = (
        formal_provenance_required
        or any(value is not None for value in expected.values())
        or any(
            any(record.get(field) is not None for field in fields)
            or record.get("feature_provenance") is not None
            for record in records
        )
    )
    if not active:
        return None

    canonical: dict[str, str] | None = None
    for index, record in enumerate(records):
        observed = {
            field: _nonempty_text(record.get(field), field=field, owner=f"record {index}")
            for field in fields
        }
        if observed["tensor_kind"] not in _SUPPORTED_TENSOR_KINDS:
            raise ValueError(
                f"record {index} tensor_kind {observed['tensor_kind']!r} is unsupported"
            )
        if canonical is None:
            canonical = observed
        elif observed != canonical:
            raise ValueError(
                f"record {index} top-level data contract does not match the global contract: "
                f"expected {canonical}, observed {observed}"
            )
    assert canonical is not None
    for field, expected_value in expected.items():
        if expected_value is not None and canonical[field] != expected_value:
            raise ValueError(
                f"manifest {field} does not match configured expectation: "
                f"expected {expected_value!r}, observed {canonical[field]!r}"
            )
    return canonical


def _validate_feature_provenance(value: Any, *, owner: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{owner} must be a JSON object")
    missing = [field for field in _FEATURE_PROVENANCE_FIELDS if field not in value]
    if missing:
        raise ValueError(f"{owner} lacks required fields: {missing}")

    result = dict(value)
    result["extractor"] = _nonempty_text(
        value["extractor"], field="extractor", owner=owner
    )
    result["source_revision"] = _source_revision(value["source_revision"], owner=owner)
    result["checkpoint_sha256"] = _sha256(
        value["checkpoint_sha256"], field="checkpoint_sha256", owner=owner
    )
    layer = value["layer"]
    if isinstance(layer, bool) or not isinstance(layer, (str, int)):
        raise ValueError(f"{owner} layer must be a non-empty string or an integer")
    if isinstance(layer, str) and not layer.strip():
        raise ValueError(f"{owner} layer must be a non-empty string or an integer")
    result["layer"] = layer.strip() if isinstance(layer, str) else layer
    feature_dim = value["feature_dim"]
    if isinstance(feature_dim, bool) or not isinstance(feature_dim, int) or feature_dim <= 0:
        raise ValueError(f"{owner} feature_dim must be a positive integer")
    result["feature_dim"] = int(feature_dim)
    result["feature_grid_shape"] = _positive_int_triplet(
        value["feature_grid_shape"], field="feature_grid_shape", owner=owner
    )
    result["input_shape"] = _positive_int_triplet(
        value["input_shape"], field="input_shape", owner=owner
    )
    normalization = value["normalization"]
    if isinstance(normalization, str):
        if not normalization.strip():
            raise ValueError(f"{owner} normalization must be non-empty")
        result["normalization"] = normalization.strip()
    elif isinstance(normalization, dict) and normalization:
        try:
            json.dumps(normalization, allow_nan=False, sort_keys=True)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{owner} normalization must be finite JSON") from error
    else:
        raise ValueError(f"{owner} normalization must be a non-empty string or object")
    return result


def _feature_provenance_contract(
    records: list[dict[str, Any]],
    *,
    required: bool,
    expected_extractor: str | None,
    expected_source_revision: str | None,
    expected_checkpoint_sha256: str | None,
    expected_checkpoint_key: str | None,
) -> dict[str, Any] | None:
    active = (
        required
        or expected_extractor is not None
        or expected_source_revision is not None
        or expected_checkpoint_sha256 is not None
        or expected_checkpoint_key is not None
        or any(record.get("feature_provenance") is not None for record in records)
    )
    if not active:
        return None
    canonical: dict[str, Any] | None = None
    for index, record in enumerate(records):
        observed = _validate_feature_provenance(
            record.get("feature_provenance"), owner=f"record {index} feature_provenance"
        )
        if canonical is None:
            canonical = observed
        elif observed != canonical:
            raise ValueError(
                f"record {index} feature_provenance does not match the global contract"
            )
    assert canonical is not None
    normalized_extractor = (
        _nonempty_text(expected_extractor, field="expected_extractor", owner="configured")
        if expected_extractor is not None
        else None
    )
    if normalized_extractor is not None and canonical["extractor"] != normalized_extractor:
        raise ValueError(
            "manifest extractor does not match configured expectation: "
            f"expected {normalized_extractor!r}, observed {canonical['extractor']!r}"
        )
    if expected_source_revision is not None:
        expected_revision = _source_revision(expected_source_revision, owner="configured")
        if canonical["source_revision"] != expected_revision:
            raise ValueError(
                "manifest source_revision does not match configured expectation"
            )
    if expected_checkpoint_sha256 is not None:
        expected_digest = _sha256(
            expected_checkpoint_sha256,
            field="expected_checkpoint_sha256",
            owner="configured",
        )
        if canonical["checkpoint_sha256"] != expected_digest:
            raise ValueError(
                "manifest checkpoint_sha256 does not match configured expectation"
            )
    if expected_checkpoint_key is not None:
        expected_key = _nonempty_text(
            expected_checkpoint_key, field="expected_checkpoint_key", owner="configured"
        )
        if canonical.get("checkpoint_key") != expected_key:
            raise ValueError(
                "manifest checkpoint_key does not match configured expectation"
            )
    return canonical


def _validate_spatial_provenance(
    value: Any,
    *,
    context_visit_ids: set[str],
    owner: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{owner} must be a JSON object")
    if "crop_shape" in value:
        raise ValueError(
            f"{owner} crop_shape is ambiguous; use roi_voxel_shape for raw ROI geometry"
        )
    if "feature_grid_shape" in value:
        raise ValueError(
            f"{owner} feature_grid_shape belongs in feature_provenance, not spatial_provenance"
        )
    reference = _nonempty_text(
        value.get("reference_visit"), field="reference_visit", owner=owner
    )
    sources = value.get("roi_source_visits")
    if (
        not isinstance(sources, list)
        or not sources
        or any(not isinstance(item, str) or not item.strip() for item in sources)
    ):
        raise ValueError(f"{owner} ROI sources must be non-empty context visits")
    normalized_sources = [item.strip() for item in sources]
    if len(set(normalized_sources)) != len(normalized_sources):
        raise ValueError(f"{owner} ROI sources must not contain duplicates")
    if reference not in context_visit_ids:
        raise ValueError(f"{owner} spatial reference must be a context visit")
    if not set(normalized_sources).issubset(context_visit_ids):
        raise ValueError(f"{owner} ROI sources must be non-empty context visits")
    spacing = value.get("spacing_mm")
    if (
        not isinstance(spacing, list)
        or len(spacing) != 3
        or any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            or float(item) <= 0.0
            for item in spacing
        )
    ):
        raise ValueError(f"{owner} has invalid spacing_mm")
    roi_voxel_shape = _positive_int_triplet(
        value.get("roi_voxel_shape"), field="roi_voxel_shape", owner=owner
    )
    orientation = _nonempty_text(value.get("orientation"), field="orientation", owner=owner)
    transform_id = _nonempty_text(value.get("transform_id"), field="transform_id", owner=owner)
    registration_type = _nonempty_text(
        value.get("registration_type"), field="registration_type", owner=owner
    ).casefold()
    if registration_type not in {"identity", "rigid", "affine", "rigid_then_affine"}:
        raise ValueError(f"{owner} has unsupported registration_type")
    if value.get("deformable_registration") is not False:
        raise ValueError(f"{owner} deformable_registration must be false")
    result = dict(value)
    result.update(
        {
            "reference_visit": reference,
            "roi_source_visits": normalized_sources,
            "spacing_mm": [float(item) for item in spacing],
            "roi_voxel_shape": roi_voxel_shape,
            "orientation": orientation,
            "transform_id": transform_id,
            "registration_type": registration_type,
            "deformable_registration": False,
        }
    )
    return result


def _feature_contract(
    records: list[dict[str, Any]],
    *,
    record_field: str,
    expected: list[str] | None,
) -> list[str] | None:
    expected_names = (
        _validate_feature_names(expected, field=record_field, owner="expected")
        if expected is not None
        else None
    )
    declarations = [record.get(record_field) for record in records]
    if expected_names is None and all(value is None for value in declarations):
        # Legacy manifests had anonymous vectors. They remain readable when no named
        # contract is requested.
        return None

    canonical = expected_names
    for index, declaration in enumerate(declarations):
        if declaration is None:
            raise ValueError(
                f"record {index} must declare {record_field} because a named feature "
                "contract is active"
            )
        names = _validate_feature_names(
            declaration, field=record_field, owner=f"record {index}"
        )
        if canonical is None:
            canonical = names
        elif names != canonical:
            raise ValueError(
                f"record {index} {record_field} does not match the global feature contract: "
                f"expected {canonical}, observed {names}"
            )
    return canonical


def _load_volume(path: Path) -> torch.Tensor:
    if path.suffix != ".pt":
        raise ValueError(f"Only safe preprocessed .pt tensors are supported, got: {path}")
    value = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(value, dict):
        value = value.get("volume")
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Expected a tensor or {{'volume': tensor}} in {path}")
    if value.ndim != 4:
        raise ValueError(f"Expected [C,D,H,W] tensor in {path}, got {tuple(value.shape)}")
    return value.float()


class ManifestTrajectoryDataset(Dataset[TrajectorySample]):
    """Read one future-target training example per JSONL record.

    Required record shape is documented in ``docs/ispy2_manifest_contract.md``.
    Patient-level split leakage is checked across the complete manifest at startup.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        expected_patient_counts: dict[str, int] | None = None,
        expected_history_length: int | None = None,
        expected_context_visits: list[str] | None = None,
        expected_target_visit: str | None = None,
        expected_clinical_features: list[str] | None = None,
        expected_action_features: list[str] | None = None,
        expected_schema_version: str | None = None,
        expected_modality: str | None = None,
        expected_tensor_kind: str | None = None,
        expected_extractor: str | None = None,
        expected_source_revision: str | None = None,
        expected_checkpoint_sha256: str | None = None,
        expected_checkpoint_key: str | None = None,
        expected_volume_shape: list[int] | tuple[int, int, int] | None = None,
        require_spatial_provenance: bool = False,
        require_feature_provenance: bool = False,
        require_content_hashes: bool = False,
    ) -> None:
        self.path = Path(manifest_path).expanduser().resolve()
        self.root = self.path.parent
        self._verified_hashes: set[Path] = set()
        self.expected_volume_shape = (
            tuple(int(value) for value in expected_volume_shape)
            if expected_volume_shape is not None
            else None
        )
        if self.expected_volume_shape is not None and (
            len(self.expected_volume_shape) != 3 or min(self.expected_volume_shape) < 1
        ):
            raise ValueError("expected_volume_shape must contain three positive integers")
        with self.path.open("r", encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle if line.strip()]
        if not records:
            raise ValueError(f"manifest is empty: {self.path}")

        formal_provenance_required = (
            require_spatial_provenance
            or require_feature_provenance
            or expected_extractor is not None
            or expected_source_revision is not None
            or expected_checkpoint_sha256 is not None
            or expected_checkpoint_key is not None
        )
        self.data_contract = _top_level_contract(
            records,
            expected_schema_version=expected_schema_version,
            expected_modality=expected_modality,
            expected_tensor_kind=expected_tensor_kind,
            formal_provenance_required=formal_provenance_required,
        )
        self.schema_version = (
            self.data_contract["schema_version"] if self.data_contract is not None else None
        )
        self.modality = self.data_contract["modality"] if self.data_contract is not None else None
        self.tensor_kind = (
            self.data_contract["tensor_kind"] if self.data_contract is not None else None
        )
        cached_features = self.tensor_kind in _CACHED_TENSOR_KINDS
        self.feature_provenance = _feature_provenance_contract(
            records,
            required=require_feature_provenance or cached_features,
            expected_extractor=expected_extractor,
            expected_source_revision=expected_source_revision,
            expected_checkpoint_sha256=expected_checkpoint_sha256,
            expected_checkpoint_key=expected_checkpoint_key,
        )

        self.clinical_features = _feature_contract(
            records,
            record_field="clinical_features",
            expected=expected_clinical_features,
        )
        self.action_features = _feature_contract(
            records,
            record_field="action_features",
            expected=expected_action_features,
        )

        patient_splits: dict[str, str] = {}
        split_patients: dict[str, set[str]] = {}
        global_identifiers: dict[tuple[str, str], tuple[str, str]] = {}
        for record in records:
            patient_id = str(record["patient_id"])
            record_split = str(record["split"])
            split_patients.setdefault(record_split, set()).add(patient_id)
            previous = patient_splits.setdefault(patient_id, record_split)
            if previous != record_split:
                raise ValueError(
                    f"patient {patient_id} occurs in both {previous!r} and {record_split!r}"
                )
            visits = [*record.get("context", []), record.get("target", {})]
            for visit in visits:
                scan_id = visit.get("scan_id")
                if scan_id is None:
                    if require_content_hashes:
                        raise ValueError("every formal manifest visit must declare scan_id")
                    identifiers: list[tuple[str, str]] = []
                else:
                    identifiers = [("scan_id", str(scan_id))]
                content_hash = visit.get("content_sha256")
                if require_content_hashes and content_hash is None:
                    raise ValueError(
                        "every manifest visit must declare content_sha256 when hashes are required"
                    )
                if content_hash is not None:
                    digest = str(content_hash).lower()
                    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
                        raise ValueError("content_sha256 must be a 64-character hex digest")
                    identifiers.append(("content_sha256", digest))
                volume = visit.get("volume")
                if volume is not None:
                    identifiers.append(("volume", str(self._resolve(str(volume)))))
                for identifier in identifiers:
                    owner = global_identifiers.setdefault(
                        identifier, (patient_id, record_split)
                    )
                    if owner != (patient_id, record_split):
                        raise ValueError(
                            f"{identifier[0]} {identifier[1]!r} is shared by "
                            f"{owner} and {(patient_id, record_split)}"
                        )
        if expected_patient_counts is not None:
            observed = {name: len(patients) for name, patients in split_patients.items()}
            if observed != expected_patient_counts:
                raise ValueError(
                    "patient split counts do not match the preregistered manifest: "
                    f"expected {expected_patient_counts}, observed {observed}"
                )
        for index, record in enumerate(records):
            context = record.get("context", [])
            if self.clinical_features is not None:
                clinical = record.get("clinical", [])
                if not isinstance(clinical, list) or len(clinical) != len(
                    self.clinical_features
                ):
                    raise ValueError(
                        f"record {index} clinical vector does not match clinical_features"
                    )
            if self.action_features is not None:
                next_action = record.get("next_action", [])
                if not isinstance(next_action, list) or len(next_action) != len(
                    self.action_features
                ):
                    raise ValueError(
                        f"record {index} next_action vector does not match action_features"
                    )
                for visit_index, visit in enumerate(context):
                    exposure = visit.get("state_action", [0.0] * len(self.action_features))
                    if not isinstance(exposure, list) or len(exposure) != len(
                        self.action_features
                    ):
                        raise ValueError(
                            f"record {index} context visit {visit_index} state_action "
                            "does not match action_features"
                        )
            if expected_history_length is not None and len(context) != expected_history_length:
                raise ValueError(
                    f"record {index} has {len(context)} context visits; "
                    f"expected {expected_history_length}"
                )
            if expected_context_visits is not None:
                observed_visits = [str(visit.get("visit_id")) for visit in context]
                if observed_visits != expected_context_visits:
                    raise ValueError(
                        f"record {index} context visits are {observed_visits}; "
                        f"expected {expected_context_visits}"
                    )
            if expected_target_visit is not None:
                observed_target = str(record.get("target", {}).get("visit_id"))
                if observed_target != expected_target_visit:
                    raise ValueError(
                        f"record {index} target visit is {observed_target!r}; "
                        f"expected {expected_target_visit!r}"
                    )
            provenance = record.get("spatial_provenance")
            spatial_provenance_required = (
                require_spatial_provenance or self.tensor_kind == "volume"
            )
            if spatial_provenance_required and provenance is None:
                raise ValueError(f"record {index} lacks spatial_provenance")
            if provenance is not None and self.data_contract is not None:
                record["spatial_provenance"] = _validate_spatial_provenance(
                    provenance,
                    context_visit_ids={str(visit.get("visit_id")) for visit in context},
                    owner=f"record {index} spatial_provenance",
                )
        self.records = [record for record in records if str(record["split"]) == split]
        if not self.records:
            raise ValueError(f"no records for split {split!r} in {self.path}")

    def __len__(self) -> int:
        return len(self.records)

    def _resolve(self, value: str) -> Path:
        path = Path(value).expanduser()
        return path if path.is_absolute() else (self.root / path).resolve()

    def _verify_content_hash(self, path: Path, expected: str | None) -> None:
        if expected is None or path in self._verified_hashes:
            return
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        observed = digest.hexdigest()
        if observed != str(expected).lower():
            raise ValueError(
                f"content hash mismatch for {path}: expected {expected}, observed {observed}"
            )
        self._verified_hashes.add(path)

    def _validate_tensor_contract(
        self,
        tensor: torch.Tensor,
        *,
        record: dict[str, Any],
        index: int,
        visit_id: Any,
    ) -> None:
        spatial_shape = tuple(int(value) for value in tensor.shape[-3:])
        label = str(visit_id)
        if self.tensor_kind in _CACHED_TENSOR_KINDS:
            assert self.feature_provenance is not None
            expected_grid = tuple(self.feature_provenance["feature_grid_shape"])
            expected_channels = int(self.feature_provenance["feature_dim"])
            if spatial_shape != expected_grid:
                raise ValueError(
                    f"record {index} visit {label} tensor grid {spatial_shape} does not "
                    f"match feature_provenance.feature_grid_shape {expected_grid}"
                )
            if int(tensor.shape[0]) != expected_channels:
                raise ValueError(
                    f"record {index} visit {label} tensor channels {int(tensor.shape[0])} "
                    f"do not match feature_provenance.feature_dim {expected_channels}"
                )
        elif self.tensor_kind == "volume":
            provenance = record.get("spatial_provenance")
            if isinstance(provenance, dict):
                expected_roi_shape = tuple(int(value) for value in provenance["roi_voxel_shape"])
                if spatial_shape != expected_roi_shape:
                    raise ValueError(
                        f"record {index} visit {label} volume shape {spatial_shape} does not "
                        f"match spatial_provenance.roi_voxel_shape {expected_roi_shape}"
                    )
        elif self.data_contract is None:
            # Legacy manifests had no tensor-kind declaration and called the serialized
            # tensor shape ``crop_shape``. Preserve that behavior only outside the formal
            # contract; formal records must separate raw ROI and cached-feature geometry.
            provenance = record.get("spatial_provenance")
            if isinstance(provenance, dict) and "crop_shape" in provenance:
                expected_legacy_shape = tuple(int(value) for value in provenance["crop_shape"])
                if spatial_shape != expected_legacy_shape:
                    raise ValueError(
                        f"record {index} visit {label} tensor shape does not match legacy "
                        "spatial provenance"
                    )
        if self.expected_volume_shape is not None and spatial_shape != self.expected_volume_shape:
            raise ValueError(
                f"record {index} visit {label} tensor shape {spatial_shape} does not match "
                f"configured volume_shape {self.expected_volume_shape}"
            )

    def __getitem__(self, index: int) -> TrajectorySample:
        record = self.records[index]
        context = record["context"]
        if not context:
            raise ValueError(f"record {index} has no context visits")
        next_action = torch.tensor(record.get("next_action", []), dtype=torch.float32)
        action_dim = next_action.numel()
        volumes = []
        times = []
        ftv = []
        state_actions = []
        context_paths = [self._resolve(visit["volume"]) for visit in context]
        target = record["target"]
        target_path = self._resolve(target["volume"])
        all_paths = [str(path) for path in context_paths] + [str(target_path)]
        if len(set(all_paths)) != len(all_paths):
            raise ValueError(f"record {index} reuses a volume across context and target")
        scan_ids = [visit.get("scan_id") for visit in context] + [target.get("scan_id")]
        if any(value is None for value in scan_ids):
            raise ValueError(f"record {index} must declare a scan_id for every visit")
        if len(set(str(value) for value in scan_ids)) != len(scan_ids):
            raise ValueError(f"record {index} reuses a scan_id across visits")
        content_hashes = [
            str(value).lower()
            for value in [
                *(visit.get("content_sha256") for visit in context),
                target.get("content_sha256"),
            ]
            if value is not None
        ]
        if len(set(content_hashes)) != len(content_hashes):
            raise ValueError(f"record {index} reuses tensor content across visits")

        for visit, volume_path in zip(context, context_paths):
            self._verify_content_hash(volume_path, visit.get("content_sha256"))
        self._verify_content_hash(target_path, target.get("content_sha256"))

        for visit, volume_path in zip(context, context_paths):
            volume = _load_volume(volume_path)
            self._validate_tensor_contract(
                volume,
                record=record,
                index=index,
                visit_id=visit.get("visit_id"),
            )
            volumes.append(volume)
            times.append(float(visit["time_days"]))
            ftv.append(_as_float(visit.get("ftv")))
            exposure = visit.get("state_action", [0.0] * action_dim)
            if len(exposure) != action_dim:
                raise ValueError(f"record {index} has inconsistent action dimensions")
            state_actions.append(exposure)
        target_volume = _load_volume(target_path)
        self._validate_tensor_contract(
            target_volume,
            record=record,
            index=index,
            visit_id=target.get("visit_id"),
        )
        sample = TrajectorySample(
            patient_id=str(record["patient_id"]),
            context_volumes=torch.stack(volumes),
            context_times_days=torch.tensor(times, dtype=torch.float32),
            context_valid=torch.ones(len(context), dtype=torch.bool),
            state_actions=torch.tensor(state_actions, dtype=torch.float32).reshape(
                len(context), action_dim
            ),
            clinical=torch.tensor(record.get("clinical", []), dtype=torch.float32),
            context_ftv=torch.tensor(ftv, dtype=torch.float32),
            target_volume=target_volume,
            target_time_days=torch.tensor(float(target["time_days"])),
            target_ftv=torch.tensor(_as_float(target.get("ftv"))),
            next_action=next_action,
            target_pcr=torch.tensor(_as_float(target.get("pcr"))),
        )
        sample.validate()
        return sample
