# Data and encoder preparation

The study uses I-SPY2 longitudinal breast DCE-MRI with SER volumes and lesion
annotations. Obtain the collection from The Cancer Imaging Archive under its
data-use conditions. The imaging resource is cited in the paper as **I-SPY 2
Breast Dynamic Contrast Enhanced MRI Trial (ISPY2)**, Wen Li and colleagues,
2022. Patient data and patient-level derived artifacts are not included here.

## Study representation

1. Use T0 as the fixed spatial reference. Derive crop placement only from the
   T0 lesion annotation; rigidly register T1 and T3 to that reference.
2. The canonical ROI contract in
   `configs/ispy2_ser_roi_112_2p1mm_v1.json` uses a 112³ RAS grid at 2.1 mm.
   Future masks check coverage and never determine crop placement.
3. The paper's selected local representation uses a T0-anchored 64³ field from
   this ROI, resized to 112³ before frozen 3DINO encoding. It yields a
   `1024 × 7 × 7 × 7` feature grid per visit.
4. Project fractional T0 lesion-core and 20-mm-ring occupancy to the same local
   grid. The imaging inputs are T0/T1; T3 supplies the prediction target.
5. Keep the patient's visits in one partition. The study has 409 training,
   88 validation, and 90 test patients after eight coverage exclusions.

The full canonical grid and the local64 view are distinct steps. Feeding the
canonical full-field features directly into the paper model changes the
representation used in the study.

## 3DINO

Download the encoder from [AICONSlab/3DINO](https://github.com/AICONSlab/3DINO)
and its linked model release. This repository does not bundle third-party
source or weights. The implementation pins:

- Source revision: `85bd4435c1b2ada41cd34cd15cad17c4d3c88d89`.
- Model release revision: `8a00a2bb14becb3fbe955064837322a3217f7ac0`.
- Checkpoint SHA-256: `b68dde2e435648bdf72edae70f970b07e4a903a8122c5be9e360bdb076cf695c`.
- Checkpoint key: `teacher`, with the backbone selected by the loader.

The feature loader verifies revision, checkpoint, shape, and spatial provenance.
Read the upstream license before using or redistributing the encoder.

## Local inputs

The original study pipeline expects a trajectory manifest, patient split,
ROI/feature indices, context-mask sidecars, and their aggregate provenance
reports. Paths are supplied through command-line arguments. The public default
dataset directory is `data/ispy2`; use an explicit `--root` or corresponding
CLI path argument for your own installation.

Inspect the available preprocessing interfaces with:

```bash
python -m med_jepa.ispy2_ser_roi_converter --help
python -m med_jepa.cache_3dino --help
python -m med_jepa.merge_feature_index --help
python -m med_jepa.ispy2_manifest_builder --help
python -m med_jepa.roi_v4p2.feature_triage --help
```

The manifest builder retains legacy pre-QC default split counts. Supply the
released split counts explicitly with
`--expected-split-counts-json '{"train":409,"val":88,"test":90}'`, together
with the required QC evidence. Read `--help` for
the complete arguments. The Cancer JEPA loaders enforce the study's training
and validation counts and do not evaluate the test split.

The unit-test identifiers and measurements are programmatically created
fixtures. They are not a study roster, imaging examples, or experimental results.
