# Review of uploaded run fc83aef

Evidence: saved outputs in AssemLens_HP_Overnight.ipynb, not direct access to the HP.

| Observed result | Value |
|---|---:|
| Completed run directory | assembly101_v1/overnight |
| Optimizer steps / epochs | 76 / 2 |
| Training runtime | 277.0846 seconds |
| Train loss | 0.5787483006715775 |
| Visible late validation loss | 0.3454417586326599 |
| Adapted action exact match | 0/40 (0%) |
| Adapted verb exact match | 5/40 (12.5%) |
| Adapted object exact match | 3/40 (7.5%) |
| JSON validity | 40/40 (100%) |
| Trainable parameters | 11,796,480 (0.2651%) |

The pipeline completes and emits valid JSON, but semantic accuracy is weak. There is no full-run baseline visible in the committed outputs; the pilot's two-example baseline cannot substitute for it. No claim of improvement or regression is justified. Loss is dominated partly by predictable JSON tokens and is not a substitute for prediction accuracy. The uploaded final report cell appears stale: it says no report, whereas the later-visible worker status is finished. Inspect the on-device report.

The top config selects 24 train / 4 validation recordings with 2,472 / 589 segments and 22.59 GiB videos. Later cells reset ROOT to v1. Preparation output does not show a completed v2 extraction, and image_side is missing from that edited config. The current script says eight epochs and 300-step checkpoints; completed output says two epochs and contains a step-75 evaluation. A rerun of these mixed cells is not reproducible as one experiment.

Next: AssemLens_Scale_Run.ipynb and train_assemlens_v2.py. New immutable root, full config/code fingerprints, 32/8 recording selection, 180 segments max per recording, 3 epochs, LR 5e-5, checkpoints/evaluation every 50 steps, early stopping, best-loss selection, representative per-recording generation sample, label coverage and macro F1. Start from the base model for a clean paired baseline, not the weak first adapter. Reuse cached HF videos. No GPU result is claimed for the new code until the HP pilot and run complete.
