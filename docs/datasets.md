# Dataset access and provenance

## Assembly101: tonight's action-recognition warm-up

- Official dataset: https://huggingface.co/datasets/cvml-nus/assembly101
- Official annotation schema: https://github.com/assembly-101/assembly101-annotations/tree/main/fine-grained-annotations
- Separate mistake task: https://github.com/assembly-101/assembly101-mistake-detection
- Model: https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct

Assembly101 contains take-apart toy assembly recordings. It is CC BY-NC 4.0 and gated by access conditions. Accept those conditions in your own Hugging Face account, then use a read token. Do not commit tokens. This is not an unrestricted commercial dataset, a furniture dataset, or a PC-building dataset.

The notebook contains working `hf_hub_download` calls. A plain public wget URL does not bypass the gate. Downloads use your cached authentication and a recorded commit revision. No raw videos or weights belong in GitHub.

```python
from huggingface_hub import HfApi, hf_hub_download
repo = 'cvml-nus/assembly101'
revision = HfApi().dataset_info(repo).sha
annotations = hf_hub_download(
    repo_id=repo,
    filename='annotations/fine-grained-annotations/actions.csv',
    repo_type='dataset', revision=revision,
)
# For videos, use an exact path from the notebook's size-checked video_plan.json.
# hf_hub_download(repo, selected_path, repo_type='dataset', revision=revision)
```

Default subset: 4 train recordings, 2 validation recordings, one RGB camera, up to 80 segments per recording, two chronological frames per segment. Video budget is 8 GiB; the preparation stage stops before video downloads if exceeded. Never download the whole multi-terabyte repository.

Annotation frame indices are at 30 fps; decoding uses seconds, not the raw video's frame index. Full recording identities are disjoint across train and validation. The official test set stays untouched. Attribution: Sener et al., Assembly101, CVPR 2022. Processing changes: subset selection, frame extraction and resizing.

## Target-product data: still to collect

For furniture or an HP/other manufacturer PC workflow, collect your own authorized footage and product reference states. Label complete/incomplete/incorrect/uncertain, visible error, proposed correction and step ID. Split by full recording session before extracting frames. Have a second annotator review labels. Keep one final test session untouched.

Do not relabel every assembly action as correct. Detecting an action does not prove the resulting assembly state. Dataset action recognition and product-state verification are separate tasks.
