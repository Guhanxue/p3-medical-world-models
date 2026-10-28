# Reproduction guide

## What this release provides

The release contains the study model, RRR fitting, occlusion objective, fixed
final-fit configuration, validation, Patient/Place factorial audit, matched
Prior audit, and their transitive Python dependencies. These scientific
implementations are preserved from the study source.

The included unit tests and CPU smoke example run without study data. Repeating
the paper's numbers additionally requires the original data partitions,
preprocessing/feature releases, trained checkpoint, and provenance reports.
These patient-bearing and model artifacts are not part of this public release.
The original command-line programs check their provenance chain and reject
missing or mismatched artifacts; the public packaging does not bypass those
checks.

## Fixed model settings

| Setting | Value |
| --- | --- |
| Input/target feature grid | `1024 × 7 × 7 × 7` |
| Neural input channels | 2,053: latest state, history change, core/ring occupancy, coordinates |
| Hidden channels / conditioned blocks | 128 / 4 |
| Final-fit epochs / batch size / seed | 18 / 4 / 2026 |
| Optimizer | AdamW, learning rate `1e-4`, weight decay `1e-4` |
| Gradient clipping | 1.0 |
| Mask fractions | 0.50, 0.625, 0.75, 0.875, 1.00 |
| Evaluation masks | Two fixed draws per fraction |
| Patient/Place donor assignments | 32 unique whole-cohort derangements; no self-match |
| Prior reference | All 409 training histories |
| Bootstrap | 10,000 paired patient resamples |

The configuration in `ispy2_roi_v4p3p1_seed2026.dev.json` includes the earlier
cross-validation epoch ceiling of 100. The final fit uses the **18 epochs** in
`ispy2_roi_v4p3p2_seed2026.recovery.json`. Do not treat the CV ceiling as the
paper's final training duration.

## Execution order

For a fresh adaptation, begin with the [data guide](data.md), then inspect the
model, anchor, and objective APIs in the [code map](code-map.md). The historical
study CLI sequence is:

1. Build and verify the ROI, feature, and context-mask releases.
2. Fit the RRR anchor and neural model using training patients only. The
   historical grouped-CV driver is `med_jepa.roi_v4p3p1.train`; the fixed final
   fit is `med_jepa.roi_v4p3p2.finalize`.
3. Evaluate the frozen final fit with `med_jepa.roi_v4p3p2.validate`.
4. Run `med_jepa.roi_v4p3p2.audit_factorial` for Patient and Place.
5. Run `med_jepa.roi_v4p3p2.audit_matched_prior` for Prior, supplying the
   factorial report from step 4.

The training and study audit CLIs require one CUDA GPU. Use `--help` to inspect
all required input paths and reports before allocating a GPU. Their study-size
and provenance checks are intentional, so applying the framework to another
cohort requires an explicit protocol adaptation.

## Example audit commands for prepared study artifacts

`P3_LOCAL` below is a local directory populated with your authorized study
artifacts. The names illustrate the required roles; this repository does not
provide those files.

```bash
export P3_LOCAL=/path/to/local/study-artifacts

p3-factorial-audit \
  --recovery-config configs/ispy2_roi_v4p3p2_seed2026.recovery.json \
  --v4p3p1-config configs/ispy2_roi_v4p3p1_seed2026.dev.json \
  --final-fit-report "$P3_LOCAL/final_fit/aggregate_report.json" \
  --checkpoint "$P3_LOCAL/final_fit/checkpoint_train409.pt" \
  --validation-report "$P3_LOCAL/validation/aggregate_report.json" \
  --validation-private-measurements "$P3_LOCAL/validation_private/validation88_measurements.pt" \
  --manifest "$P3_LOCAL/manifest.jsonl" \
  --train-feature-root "$P3_LOCAL/train_features" \
  --train-feature-index "$P3_LOCAL/train_features/feature_variant_index.pt" \
  --validation-feature-root "$P3_LOCAL/validation_features" \
  --context-mask-release "$P3_LOCAL/context_masks" \
  --context-mask-report "$P3_LOCAL/context_masks/aggregate_report.json" \
  --context-mask-index "$P3_LOCAL/context_masks/index.csv" \
  --output-dir "$P3_LOCAL/factorial_public" \
  --private-output-dir "$P3_LOCAL/factorial_private"

p3-prior-audit \
  --recovery-config configs/ispy2_roi_v4p3p2_seed2026.recovery.json \
  --v4p3p1-config configs/ispy2_roi_v4p3p1_seed2026.dev.json \
  --final-fit-report "$P3_LOCAL/final_fit/aggregate_report.json" \
  --checkpoint "$P3_LOCAL/final_fit/checkpoint_train409.pt" \
  --validation-report "$P3_LOCAL/validation/aggregate_report.json" \
  --validation-private-measurements "$P3_LOCAL/validation_private/validation88_measurements.pt" \
  --factorial-report "$P3_LOCAL/factorial_public/aggregate_report.json" \
  --manifest "$P3_LOCAL/manifest.jsonl" \
  --train-feature-root "$P3_LOCAL/train_features" \
  --train-feature-index "$P3_LOCAL/train_features/feature_variant_index.pt" \
  --validation-feature-root "$P3_LOCAL/validation_features" \
  --context-mask-release "$P3_LOCAL/context_masks" \
  --context-mask-report "$P3_LOCAL/context_masks/aggregate_report.json" \
  --context-mask-index "$P3_LOCAL/context_masks/index.csv" \
  --output-dir "$P3_LOCAL/prior_public" \
  --private-output-dir "$P3_LOCAL/prior_private"
```

Output directories are immutable. Choose new output paths for a new run.

## Interpretation

The paper's current P³ audit keeps the patient-conditioned RRR anchor fixed.
Patient and Place substitutions use unmatched donors, so incompatibility
between history, support, and context can contribute to their effects. The
Prior reference averages neural corrections under patient-matched support and
context; it is not a fully population-average forecast of the complete model.

The validation cohort was previously used during development. Intervals are
descriptive and unadjusted for multiplicity. Small changes in latent MAE have
no established mapping to clinical outcomes. This release adds no experiments
or new test-set results.
