# Public release provenance

This repository contains the code used by **Patient, Place, Prior (P³): What
Counts as Personalization in Medical World Models?** and its Cancer JEPA case
study. The source snapshot is `cc23ad1bbeaa1f8175220a98062c3c6e3f8c3f1c`.

The public repository starts with a new Git history. The original development
history, unrelated projects, internal notes, compute-cluster scripts, data,
checkpoints, and patient-level outputs are excluded.

## Packaging changes

- Added a paper-specific README, citation metadata, code map, reproduction
  guide, CPU smoke example, and installation metadata.
- Exposed the existing model as `med_jepa.CancerJEPA`; retained its original
  implementation and versioned module path.
- Removed the top-level import of an unrelated early prototype and an unused
  synthetic-dataset export from the package initializers.
- Replaced two institution-specific default dataset paths with `data/ispy2`,
  with the corresponding test expectation updated.
- Retained model, loss, anchor, training, and P³ audit implementations without
  scientific changes. Required earlier modules remain as internal dependencies.

`source_manifest.json` lists the included upstream source, tests, and
configurations, with source and released SHA-256 digests. Newly written public
documentation and packaging are identified separately there.

## Materials and terms

This is a source-code release. I-SPY2 images, patient rosters, masks, feature
caches, per-patient measurements, and trained checkpoints are not distributed.
Tests generate their own small tensor/metadata fixtures. Third-party encoder
source and weights must be obtained from the original provider.

The source snapshot has no repository-wide software license. This release does
not add one; authors can specify reuse terms separately. See
[3DINO's upstream license](https://github.com/AICONSlab/3DINO/blob/main/LICENSE)
for that dependency's terms.

## Release checks

On Python 3.12.14 with the dependencies in `requirements-test.txt`, all 86 unit
tests and 21 subtests passed. The CPU smoke example passed a forward/backward
step and verified that predictions outside the core remain exactly equal to
the RRR anchor. All four installed study commands and five preprocessing
module entry points passed `--help` checks. No GPU study run or new experiment
was performed as part of this release.
