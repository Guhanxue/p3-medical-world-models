# Paper-to-code map

Paths below are relative to `src/med_jepa/`. Versioned names are retained from
the study so that model, configuration, and report schemas remain compatible.

| Paper component | Implementation |
| --- | --- |
| Cancer JEPA neural correction and multiplicative core gate | `roi_v4p3p1/model.py` |
| Occlusion-based latent prediction objective | `roi_v4p3p1/losses.py` |
| Occlusion schedule | `roi_v4p2/masking.py` |
| Fold-local RRR anchor | `roi_v4p3p1/anchor.py`, `roi_v4p3/baselines.py` |
| Training and fixed 18-epoch final fit | `roi_v4p3p1/train.py`, `roi_v4p3p2/finalize.py` |
| Validation and comparison with the RRR anchor | `roi_v4p3p2/validate.py` |
| Patient, Place, joint, and interaction contrasts | `roi_v4p3p2/audit_factorial.py` |
| Prior with patient-matched support and context | `roi_v4p3p2/audit_matched_prior.py` |
| Paired patient bootstrap | `frozen_audit/statistics.py` |
| T0-anchored SER ROI conversion and coverage checks | `ispy2_ser_roi_converter.py` |
| Frozen 3DINO feature extraction | `cache_3dino.py` |
| Local64 feature preparation and occupancy | `roi_v4p2/preprocessing.py`, `roi_v4p2/feature_triage.py` |
| Training/validation data loading | `roi_v4p3/data.py`, `roi_v4p3p2/data.py` |

## Audit labels

The code uses `CC`, `SC`, `CS`, and `SS` for the paper's matched, history-swapped,
support-swapped, and both-swapped conditions, respectively. Here `C` in the
first position means correct imaging content; `C` in the second means correct
spatial support. The strings are preserved for report compatibility.

- Patient: `SC - CC` loss.
- Place: `CS - CC` loss.
- Joint: `SS - CC` loss.
- Interaction: `(CS - CC) - (SS - SC)` loss.
- Prior: `MATCHED_PRIOR_train409_content_marginalized - CC_frozen_recipient_content` loss.

Positive values favor patient-matched inputs. The reported Prior comparison
averages neural corrections from all 409 training histories under each evaluated
patient's support, timing, treatment-phase context, and masks. It then applies
that patient's core gate and adds the fixed RRR anchor.

`audit_factorial.py` also retains an earlier `POP` control that averages raw
training-patient corrections without the same patient-context matching. Use
`audit_matched_prior.py` for the paper's Prior result.
