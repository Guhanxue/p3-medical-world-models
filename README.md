# Patient, Place, Prior (P³)

## What Counts as Personalization in Medical World Models?

**Xingrui Gu, Hanxue Gu, Yuxiang Zhang, Yang Yang**  
NeurIPS 2026 Workshop on Physical Understanding for Decision-Making

[Paper](https://arxiv.org/abs/2610.09194) · [Reproduction guide](docs/reproduction.md) · [Data preparation](docs/data.md) · [Code map](docs/code-map.md)

An accurate medical forecast does not by itself establish that the patient's
history adds predictive value beyond a population-level pattern. This repository
contains **P³** and **Cancer JEPA**, the model proposed in our longitudinal breast
MRI case study.

| P³ comparison | Scientific question |
| --- | --- |
| **Patient** | Does the patient's own longitudinal history improve the forecast relative to another patient's history? |
| **Place** | Does the forecast benefit from patient-matched, externally supplied spatial support? |
| **Prior** | Does the patient's own history improve on a population-average prediction under matched support and observed context? |

Cancer JEPA forecasts the frozen 3DINO representation of preoperative T3 SER MRI
from T0/T1 history. It combines a patient-conditioned reduced-rank regression
(RRR) anchor with a lesion-constrained neural correction trained using an
occlusion-based latent prediction objective. Core occupancy gates the correction;
ring occupancy provides context.

## Install and check

Use Python 3.10 or later. For CUDA training, install the PyTorch build appropriate
for your GPU before installing this package.

```bash
git clone https://github.com/Guhanxue/p3-medical-world-models.git
cd p3-medical-world-models
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,imaging]'
python -m pytest
python examples/model_smoke.py
```

The tests and smoke example use generated tensors and temporary fixtures to check
software behavior. They do not reproduce the paper's experimental results or
download patient data. The smoke example exercises the released neural model,
its occlusion loss, a backward pass, and its exact spatial gate on CPU.

## Main entry points

```bash
cancer-jepa-finalize --help
cancer-jepa-validate --help
p3-factorial-audit --help
p3-prior-audit --help
```

The full study pipeline requires locally prepared I-SPY2 data, frozen 3DINO
features, and the provenance artifacts described in the
[reproduction guide](docs/reproduction.md). Images, patient manifests, feature
caches, patient-level measurements, and trained checkpoints are not distributed
in this repository. Obtain I-SPY2 and the 3DINO encoder from their original
providers under their respective terms.

## Published case study

The paper reports a descriptive analysis of 88 validation patients, with 87
eligible for the primary lesion-core endpoint, after training on 409 patients.
Patient and Place have positive mean contrasts; the Prior interval includes
zero. The current audit attributes the **neural correction**, with the
patient-conditioned RRR anchor held fixed. It does not establish personalization
of the complete model or clinical benefit. See the paper for effect sizes,
intervals, and limitations.

The paper's Prior result comes from `audit_matched_prior.py`. The older `POP` arm
inside `audit_factorial.py` is a different, unmatched population-template control
and must not be substituted for the reported Prior comparison.

## Repository scope

This is the public code release for the P³ paper. The `med_jepa` Python namespace
and versioned module paths are retained for compatibility with the study code.
`roi_v4p3p1` contains the Cancer JEPA model, loss, RRR anchor, and training code;
`roi_v4p3p2` contains the fixed final fit, validation, and P³ audits. Earlier
modules are retained only where these components import them.

See [release provenance](docs/release.md) for the included source inventory and
packaging changes. No repository-wide software license is specified in this
release; contact the authors about reuse terms. Third-party software and data
retain their own licenses.

## Citation

```bibtex
@misc{gu2026patientplaceprior,
  title={Patient, Place, Prior (P$^3$): What Counts as Personalization in Medical World Models?},
  author={Xingrui Gu and Hanxue Gu and Yuxiang Zhang and Yang Yang},
  year={2026},
  eprint={2610.09194},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  doi={10.48550/arXiv.2610.09194}
}
```
