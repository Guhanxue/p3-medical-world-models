from __future__ import annotations

import csv
import hashlib
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from med_jepa.cache_3dino import (
    ROI_SCHEMA_VERSION,
    ROIRecord,
    SER_NORMALIZATION,
    THREE_DINO_TEACHER_AUXILIARY_PREFIXES,
    cache_shard,
    extract_spatial_features,
    fixed_ser_normalize,
    load_checkpoint_fail_closed,
    load_roi_payload,
    read_roi_index,
    sha256_file,
)


SPATIAL_PROVENANCE = {
    "reference_visit": "T0",
    "roi_source_visits": ["T0"],
    "spacing_mm": [1.0, 1.0, 1.0],
    "orientation": "RAS",
    "transform_id": "context-only-t0-anchor-v1",
    "roi_voxel_shape": [4, 4, 4],
    "registration_type": "rigid",
    "deformable_registration": False,
}
INTENSITY_PROVENANCE = {
    "quantity": "signal_enhancement_ratio",
    "units": "unitless_ratio",
    "dicom_rescale_applied": True,
    "dicom_rescale_slope": 0.001,
    "dicom_rescale_intercept": 0.0,
    "value_state": "dicom_rescaled_ser",
    "prior_intensity_transform": "none",
    "converter_version": "med-jepa-ispy2-roi-converter-v2",
    "converter_source_sha256": "a" * 64,
    "source_dicom_content_sha256": "b" * 64,
}


class _Fake3DINO(nn.Module):
    def get_intermediate_layers(self, value, n=1, reshape=True):
        del n, reshape
        pooled = torch.nn.functional.avg_pool3d(value, kernel_size=2, stride=2)
        return (pooled.repeat(1, 4, 1, 1, 1),)


class _Recording3DINO(_Fake3DINO):
    def __init__(self) -> None:
        super().__init__()
        self.seen: torch.Tensor | None = None

    def get_intermediate_layers(self, value, n=1, reshape=True):
        self.seen = value.detach().cpu().clone()
        return super().get_intermediate_layers(value, n=n, reshape=reshape)


class Cache3DINOTests(unittest.TestCase):
    def test_fixed_ser_normalization_preserves_absolute_scale(self) -> None:
        anchors = torch.tensor([0.0, 0.75, 1.5, 2.25, 3.0]).reshape(1, 1, 1, 5)
        normalized = fixed_ser_normalize(anchors)
        torch.testing.assert_close(
            normalized.flatten(), torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])
        )
        half_scale = fixed_ser_normalize(anchors * 0.5)
        self.assertFalse(torch.equal(normalized, half_scale))

    def test_fixed_ser_normalization_fails_on_units_and_corruption(self) -> None:
        with self.assertRaisesRegex(ValueError, "DICOM rescale"):
            fixed_ser_normalize(torch.tensor([0.0, 3000.0]).reshape(1, 1, 1, 2))
        with self.assertRaisesRegex(ValueError, "fixed rescaled SER range"):
            fixed_ser_normalize(torch.tensor([-0.1, 1.0]).reshape(1, 1, 1, 2))
        with self.assertRaisesRegex(ValueError, "fixed rescaled SER range"):
            fixed_ser_normalize(torch.tensor([0.0, 3.1]).reshape(1, 1, 1, 2))
        with self.assertRaisesRegex(ValueError, "non-finite"):
            fixed_ser_normalize(torch.tensor([0.0, float("nan")]).reshape(1, 1, 1, 2))
        with self.assertRaisesRegex(ValueError, "constant"):
            fixed_ser_normalize(torch.ones(1, 2, 2, 2))

    def test_spatial_extraction_contract(self) -> None:
        volume = torch.linspace(-1, 1, 64).reshape(1, 4, 4, 4)
        features = extract_spatial_features(
            _Fake3DINO(),
            volume,
            device=torch.device("cpu"),
            expected_feature_dim=4,
            patch_size=2,
            autocast_dtype=None,
        )
        self.assertEqual(features.shape, (4, 2, 2, 2))
        self.assertEqual(features.dtype, torch.float16)

    def test_formal_roi_requires_valid_intensity_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "roi.pt"
            base = {
                "schema_version": ROI_SCHEMA_VERSION,
                "modality": "SER",
                "tensor_kind": "volume",
                "volume": torch.linspace(0.0, 3.0, 64).reshape(1, 4, 4, 4),
                "spatial_provenance": SPATIAL_PROVENANCE,
            }
            torch.save(base, path)
            with self.assertRaisesRegex(ValueError, "intensity_provenance"):
                load_roi_payload(path, (4, 4, 4))

            invalid = dict(base)
            invalid["intensity_provenance"] = {
                **INTENSITY_PROVENANCE,
                "dicom_rescale_applied": False,
            }
            torch.save(invalid, path)
            with self.assertRaisesRegex(ValueError, "DICOM rescale"):
                load_roi_payload(path, (4, 4, 4))

            integer_volume = {
                **base,
                "volume": torch.ones(1, 4, 4, 4, dtype=torch.int16),
                "intensity_provenance": INTENSITY_PROVENANCE,
            }
            torch.save(integer_volume, path)
            with self.assertRaisesRegex(TypeError, "floating point"):
                load_roi_payload(path, (4, 4, 4))

    def test_roi_index_rejects_paths_outside_allowed_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index = root / "index.csv"
            with index.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=(
                        "PatientID",
                        "tp_label",
                        "scan_id",
                        "roi_path",
                        "source_content_sha256",
                    ),
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "PatientID": "P1",
                        "tp_label": "T0",
                        "scan_id": "S0",
                        "roi_path": "/outside/value.pt",
                        "source_content_sha256": "a" * 64,
                    }
                )
            with self.assertRaisesRegex(ValueError, "escapes"):
                read_roi_index(
                    index,
                    allowed_input_root=root,
                    expected_timepoints=("T0",),
                    expected_patients=1,
                )

    def test_checkpoint_hash_and_key_mismatch_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.pt"
            source = nn.Linear(2, 2)
            torch.save({"teacher": source.state_dict()}, path)
            destination = nn.Linear(2, 2)
            digest = sha256_file(path)
            self.assertEqual(
                load_checkpoint_fail_closed(
                    destination, path, expected_sha256=digest
                ),
                digest,
            )
            torch.save(source.state_dict(), path)
            self.assertEqual(
                load_checkpoint_fail_closed(
                    destination,
                    path,
                    expected_sha256=sha256_file(path),
                    checkpoint_key=None,
                ),
                sha256_file(path),
            )
            torch.save({"teacher": source.state_dict()}, path)
            with self.assertRaisesRegex(RuntimeError, "SHA-256"):
                load_checkpoint_fail_closed(
                    destination, path, expected_sha256="0" * 64
                )
            torch.save({"teacher": {"unexpected": torch.ones(1)}}, path)
            with self.assertRaisesRegex(RuntimeError, "key mismatch"):
                load_checkpoint_fail_closed(
                    destination, path, expected_sha256=sha256_file(path)
                )

    def test_official_teacher_layout_selects_only_verified_backbone(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.pt"
            source = nn.Linear(2, 2)
            state = {
                "module.backbone.weight": source.weight.detach().clone(),
                "module.backbone.bias": source.bias.detach().clone(),
            }
            for prefix in THREE_DINO_TEACHER_AUXILIARY_PREFIXES:
                for index in range(8):
                    state[f"module.{prefix}tensor_{index}"] = torch.ones(1)
            torch.save({"teacher": state}, path)

            destination = nn.Linear(2, 2)
            load_checkpoint_fail_closed(
                destination,
                path,
                expected_sha256=sha256_file(path),
            )
            torch.testing.assert_close(destination.weight, source.weight)
            torch.testing.assert_close(destination.bias, source.bias)

            for index in range(8):
                state.pop(f"module.ibot_head.tensor_{index}")
            torch.save({"teacher": state}, path)
            with self.assertRaisesRegex(RuntimeError, "incomplete auxiliary-head"):
                load_checkpoint_fail_closed(
                    destination,
                    path,
                    expected_sha256=sha256_file(path),
                )

            for index in range(8):
                state[f"module.ibot_head.tensor_{index}"] = torch.ones(1)
            state["module.predictor.tensor"] = torch.ones(1)
            torch.save({"teacher": state}, path)
            with self.assertRaisesRegex(RuntimeError, "unapproved sibling"):
                load_checkpoint_fail_closed(
                    destination,
                    path,
                    expected_sha256=sha256_file(path),
                )

    def test_prefixed_backbone_only_layout_and_mixed_layout_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.pt"
            source = nn.Linear(2, 2)
            prefixed = {
                f"backbone.{key}": value.detach().clone()
                for key, value in source.state_dict().items()
            }
            torch.save({"teacher": prefixed}, path)
            destination = nn.Linear(2, 2)
            load_checkpoint_fail_closed(
                destination,
                path,
                expected_sha256=sha256_file(path),
            )
            torch.testing.assert_close(destination.weight, source.weight)
            torch.testing.assert_close(destination.bias, source.bias)

            prefixed["weight"] = source.weight.detach().clone()
            torch.save({"teacher": prefixed}, path)
            with self.assertRaisesRegex(RuntimeError, "unapproved sibling"):
                load_checkpoint_fail_closed(
                    destination,
                    path,
                    expected_sha256=sha256_file(path),
                )

    def test_cache_shard_writes_opaque_feature_and_index(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            roi_path = root / "source.pt"
            torch.save(
                {
                    "schema_version": ROI_SCHEMA_VERSION,
                    "modality": "SER",
                    "tensor_kind": "volume",
                    "volume": torch.linspace(0.0, 3.0, 64).reshape(1, 4, 4, 4),
                    "spatial_provenance": SPATIAL_PROVENANCE,
                    "intensity_provenance": INTENSITY_PROVENANCE,
                },
                roi_path,
            )
            source_hash = sha256_file(roi_path)
            records = [
                ROIRecord("P1", "T0", "S0", roi_path, source_hash),
            ]
            model = _Recording3DINO()
            report = cache_shard(
                records,
                model=model,
                output_dir=root / "features",
                shard_index=0,
                num_shards=1,
                input_shape=(4, 4, 4),
                checkpoint_sha256="f" * 64,
                device=torch.device("cpu"),
                autocast_dtype=None,
                expected_feature_dim=4,
                patch_size=2,
                source_revision="e" * 40,
                roi_index_sha256="d" * 64,
            )
            shard = root / "features" / "shard-00000-of-00001"
            self.assertEqual(report["cached_records"], 1)
            self.assertIsNotNone(model.seen)
            torch.testing.assert_close(
                model.seen,
                fixed_ser_normalize(
                    torch.linspace(0.0, 3.0, 64).reshape(1, 4, 4, 4)
                ).unsqueeze(0),
            )
            with (shard / "feature_index.csv").open(newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertNotIn("P1", Path(row["feature_path"]).name)
            self.assertEqual(
                hashlib.sha256(Path(row["feature_path"]).read_bytes()).hexdigest(),
                row["content_sha256"],
            )
            payload = torch.load(
                Path(row["feature_path"]), map_location="cpu", weights_only=True
            )
            self.assertEqual(payload["spatial_provenance"], SPATIAL_PROVENANCE)
            self.assertEqual(
                payload["source_intensity_provenance"], INTENSITY_PROVENANCE
            )
            self.assertEqual(
                payload["feature_provenance"]["normalization"], SER_NORMALIZATION
            )
            self.assertEqual(
                payload["feature_provenance"]["compute_device"], "cpu"
            )
            self.assertEqual(
                payload["feature_provenance"]["autocast_dtype"], "float32"
            )
            self.assertEqual(
                payload["feature_provenance"]["feature_storage_dtype"], "float16"
            )
            self.assertIn("torch_version", payload["feature_provenance"])
            self.assertEqual(
                payload["feature_provenance"]["source_revision"], "e" * 40
            )
            self.assertEqual(
                payload["feature_provenance"]["roi_index_sha256"], "d" * 64
            )
            self.assertEqual(report["source_revision"], "e" * 40)
            with self.assertRaises(FileExistsError):
                cache_shard(
                    records,
                    model=_Fake3DINO(),
                    output_dir=root / "features",
                    shard_index=0,
                    num_shards=1,
                    input_shape=(4, 4, 4),
                    checkpoint_sha256="f" * 64,
                    device=torch.device("cpu"),
                    autocast_dtype=None,
                    expected_feature_dim=4,
                    patch_size=2,
                )


if __name__ == "__main__":
    unittest.main()
