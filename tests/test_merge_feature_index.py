from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch

from med_jepa.cache_3dino import OUTPUT_INDEX_COLUMNS, sha256_file
from med_jepa.merge_feature_index import merge_feature_indices


class MergeFeatureIndexTests(unittest.TestCase):
    def test_validates_and_merges_complete_shards(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "cache"
            feature_provenance = {
                "extractor": "fake",
                "source_revision": "a" * 40,
                "checkpoint_sha256": "c" * 64,
                "roi_index_sha256": "d" * 64,
                "checkpoint_key": "teacher",
                "layer": "last",
                "feature_dim": 2,
                "feature_grid_shape": [2, 2, 2],
                "input_shape": [32, 32, 32],
                "normalization": "test",
            }
            spatial_provenance = {
                "reference_visit": "T0",
                "roi_source_visits": ["T0"],
                "spacing_mm": [1.0, 1.0, 1.0],
                "orientation": "RAS",
                "transform_id": "test",
                "roi_voxel_shape": [32, 32, 32],
                "registration_type": "rigid",
                "deformable_registration": False,
            }
            spatial_digest = hashlib.sha256(
                json.dumps(
                    spatial_provenance,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode()
            ).hexdigest()
            for shard_index, timepoint in enumerate(("T0", "T1", "T3")):
                shard = root / f"shard-{shard_index:05d}-of-00003"
                shard.mkdir(parents=True)
                feature = shard / f"feature-{timepoint}.pt"
                torch.save({"volume": torch.ones(2, 2, 2, 2) * shard_index}, feature)
                report = {
                    "schema_version": "med-jepa-3dino-cache-v2",
                    "shard_index": shard_index,
                    "num_shards": 3,
                    "cached_records": 1,
                    "input_shape": [32, 32, 32],
                    "feature_grid_shape": [2, 2, 2],
                    "feature_dim": 2,
                    "checkpoint_sha256": "c" * 64,
                    "checkpoint_key": "teacher",
                    "source_revision": "a" * 40,
                    "hf_revision": "hf-revision",
                    "feature_provenance": feature_provenance,
                    "spatial_provenance": spatial_provenance,
                    "spatial_provenance_sha256": spatial_digest,
                }
                (shard / "report.json").write_text(json.dumps(report), encoding="utf-8")
                with (shard / "feature_index.csv").open(
                    "w", encoding="utf-8", newline=""
                ) as handle:
                    writer = csv.DictWriter(handle, fieldnames=OUTPUT_INDEX_COLUMNS)
                    writer.writeheader()
                    writer.writerow(
                        {
                            "PatientID": "P1",
                            "tp_label": timepoint,
                            "scan_id": f"S-{timepoint}",
                            "feature_path": str(feature),
                            "content_sha256": sha256_file(feature),
                            "source_content_sha256": f"{shard_index + 1:064x}",
                        }
                    )
            output = Path(directory) / "merged.csv"
            result = merge_feature_indices(
                root,
                output,
                expected_num_shards=3,
                expected_patients=1,
            )
            self.assertEqual(result["patients"], 1)
            self.assertEqual(result["records"], 3)
            with output.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["tp_label"] for row in rows], ["T0", "T1", "T3"])
            self.assertTrue(
                (Path(directory) / "merged.feature_provenance.json").is_file()
            )
            self.assertTrue(
                (Path(directory) / "merged.spatial_provenance.json").is_file()
            )
            with self.assertRaises(FileExistsError):
                merge_feature_indices(
                    root,
                    output,
                    expected_num_shards=3,
                    expected_patients=1,
                )


if __name__ == "__main__":
    unittest.main()
