# Verifier v2: annotation-assisted mistake classification

Run from the HP repository root using the existing project .venv. No package installation or new video downloads are required. Official mistake annotations must already exist in runs/verification_100/mistake_annotations. Video files are resolved from the pinned heavy-v4 HF cache. The code fails on missing cached videos instead of initiating a large download.

## Prepare

    python -u scripts/verifier_v2.py prepare

Uses all valid attach/detach rows from cached heavy-v4 training recordings. Reserves the old heavy-v4 validation recordings entirely. Splits eligible recordings deterministically into approximately 80% training and 20% validation. Requires at least twelve recordings and class support in both partitions; never falls back to a random frame split. This is a custom development split, not a published official benchmark score. No final test evaluation is performed here.

Inspect runs/verifier_v2/prepared.json and inspect_correct.jpg, inspect_mistake.jpg, inspect_correction.jpg. Four frames are sampled inside each action interval, using the existing project's 30-fps annotation interpretation. Check several examples against full videos and their timestamps. Contact sheets alone cannot establish all ordering errors. If alignment is wrong, stop and correct preparation before training. Invalid intervals and non-attach/detach annotation verbs are excluded and recorded. The builder uses natural class frequencies, not oversampling.

## Train in tmux on the HP

    tmux new-session -A -s assemlens
    cd /home/hp15/git_happens/assemlens
    source /home/hp15/miniforge3/etc/profile.d/conda.sh
    conda activate zgx
    source .venv/bin/activate
    set -o pipefail
    python -u scripts/verifier_v2.py train --confirm-alignment 2>&1 | tee -a runs/verifier_v2/train.log

The confirmation flag records your decision to proceed after inspecting alignment; it is not an automatic alignment test. Ctrl+B, then D detaches. Or detach from another HP terminal using tmux detach-client -s assemlens. Closing the Mac afterward does not stop the HP process. HP reboot, shutdown, sleep, or administrative termination can stop it.

Reattach with tmux attach -t assemlens. Alternatively inspect runs/verifier_v2/training/status.json and tail runs/verifier_v2/train.log. Do not run two training sessions on the GPU. A file lock prevents duplicate v2 workers only; it does not detect unrelated GPU jobs.

## Experiment

Fresh Qwen3-VL-4B-Instruct BF16, pinned revision from heavy-v4 config, frozen base/vision weights, language-attention LoRA rank16/alpha32, four images at up to 448 pixels, learning rate 2e-5, microbatch1, accumulation8, up to six epochs. Eight-hour training-loop budget per invocation, not including preparation or generation evaluation. Early stopping after four non-improving validation-loss checks may finish earlier. No promise of five to eight hours: use measured progress. Do not increase epochs solely to occupy the GPU.

Checkpoints every fifty optimizer steps; best validation-loss checkpoint is restored. Selection by loss is not guaranteed to maximize mistake recall. Short JSON output has just assessment. Report class precision/recall/F1, macro-F1, accuracy, majority baseline, invalid JSON, and abstention rate. The dataset has no uncertain ground-truth class; uncertain is an abstention, not a learned supervised class. Base and adapted models use identical validation examples and prompts. Natural imbalance may bias the model toward correct; inspect class metrics.

If interrupted, rerun the same train command after confirming the previous process stopped. Resume uses the last checkpoint (up to fifty steps may be lost); code/config/manifests/image hashes must match. The time budget restarts on resume. Completed runs refuse to relaunch. Do not edit the worker or data mid-run.

Inputs include current and preceding annotation-derived action descriptions, but no assessments, remarks, fabricated verified state, or fabricated product instructions. These are oracle action inputs and may leak useful semantics: the experiment is not end-to-end camera verification. Evaluate with model-predicted actions and image/text ablations before claiming visual reasoning gains. No supervision is provided for corrective instructions, defect localization, calibrated confidence, or physical step completion. Do not claim these abilities from this run.

Results: runs/verifier_v2/training/comparison.json, baseline.json, adapted.json, adapter/, checkpoints/, trainer_state.json. Back up weights separately before the lab node is wiped. Data license/source: https://github.com/assembly-101/assembly101-mistake-detection (CC BY-NC 4.0). Preserve attribution. CPU syntax and helper checks were performed; the new worker has not been run on the HP GPU by the assistant.
