# Jeep demo: next experiment

The Assembly101 verifier is an annotation-assisted action classifier. Its adapted
mistake recall was 2/18, versus 9/18 before adaptation. Higher overall accuracy
does not establish reliable assembly verification. Preserve those results; do
not lengthen the same training job solely to occupy the GPU.

## Immediate goal

Test one visible state on the actual FYD jeep, with one human-approved reference
photo and six development photos: two matching, two visibly wrong, two obscured.
Use the real kit instructions to choose a state. Do not infer screw tightness,
internal engagement, or safety from photos. Hold the camera reasonably close and
move hands away for completion photos. An incomplete step while someone is still
working is not automatically an error; this pilot checks a requested checkpoint.

Run from the repository root with the existing HP environment:

```bash
python scripts/jeep_state_eval.py init
```

Edit `runs/jeep_demo_v1/task.json`: replace EDIT instructions and visible conditions.
Save the correct reference at `references/step_01.jpg` within that folder.
Save observations in `captures/`. Add one line per observation to `examples.jsonl`:

```json
{"id":"trial_01","session":"dev_01","split":"dev","step_id":"step_01","image":"captures/trial_01.jpg","label":"mismatch"}
```

Labels are `matches`, `mismatch`, `uncertain`. These describe visible state,
not the old action labels. Goal text is permitted model input; actual action,
ground-truth label, filenames, and session IDs are not passed to the model.
Keep entire recording attempts in one split. Use independent held-out test
sessions after development. A six-photo pilot is a debugging check, not an
accuracy claim. Reference/test exact duplicates and cross-split exact duplicates
are rejected, but humans must also prevent near-duplicate frame leakage.

```bash
python scripts/jeep_state_eval.py evaluate --check-only
python -u scripts/jeep_state_eval.py evaluate --name base_dev_01
```

The model/revision comes from the existing `runs/verifier_v2/config.json`.
No new packages or vendor PyTorch changes should be necessary.
Outputs go into `runs/jeep_demo_v1/results/base_dev_01/` and include raw responses,
input image hashes, timing, confusion matrix, mismatch recall and false approvals.
Existing output directories cannot be overwritten. Use a new name to rerun.

Optional comparison, only after the base run works:

```bash
python -u scripts/jeep_state_eval.py evaluate --name verifier_v2_dev_01 --adapter runs/verifier_v2/training/adapter
```

The old adapter learned a different task/output schema and may perform worse.
Never choose it simply because it is fine-tuned. Invalid JSON is counted separately
and never interpreted as a successful verification. The outputs are experimental
judgments, not authoritative completion signals for the frontend.

## Next training decision

First inspect actual toy failure examples. Collect separate training recordings
across all selected steps, including clear mismatches and occlusions; an initial
target is 60–120 varied labeled checkpoints, with separate development and test
sessions. This target is a starting point, not a guarantee of adequate coverage.
Review labels with a second teammate. Reference photos must be independent of
the reserved test sessions.

Only then build toy-specific supervised training with the same reference/current
input and output schema. Start with a short fit test. Extend training if held-out
mistake detection improves without increasing false approvals. Compare the base
model and adapted model on identical examples, and run a shuffled-current-image
control to test dependence on visual evidence. Stop when development performance
plateaus. A 5–8-hour budget is a ceiling, not a useful objective on its own.

## Demo milestone

Reference + deliberately incorrect state -> visible mismatch -> user fixes it ->
new observation matches. Treat correction as the verified transition across
observations, not as a label inferred from a detach action. Add automatic progress
tracking only after this manual checkpoint test works on the real toy.

The script is syntax/CPU-logic tested. GPU execution and accuracy on the physical
toy must be tested on HP. It does not train, detect steps automatically, or verify
mechanical assembly strength.
