"""Reference-conditioned jeep state pilot. No action annotations are model inputs.

Run init, edit task.json, add reference/test images and examples.jsonl, then evaluate.
This is an experimental visible-state comparator, not a mechanical safety check.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LABELS = ('matches', 'mismatch', 'uncertain')


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2) + '\n')


def initialize(root):
    root.mkdir(parents=True, exist_ok=True)
    for folder in ('references', 'captures', 'results'):
        (root / folder).mkdir(exist_ok=True)
    task = root / 'task.json'
    if not task.exists():
        write(task, {'product': 'FYD take-apart jeep', 'steps': [{
            'id': 'step_01', 'instruction': 'EDIT: describe one real assembly step',
            'visible_completion': 'EDIT: list the visible conditions to check',
            'reference': 'references/step_01.jpg'}]})
    manifest = root / 'examples.jsonl'
    if not manifest.exists():
        manifest.write_text('')
    print('Created:', root)
    print('Edit task.json; replace EDIT text using your actual toy.')
    print('Save a correct reference and test photos, then add JSON lines such as:')
    print(json.dumps({'id': 'trial_01', 'session': 'dev_01', 'split': 'dev',
                      'step_id': 'step_01', 'image': 'captures/trial_01.jpg',
                      'label': 'mismatch'}))


def load_inputs(root):
    task = json.loads((root / 'task.json').read_text())
    steps = {s['id']: s for s in task['steps']}
    assert len(steps) == len(task['steps']) and steps, 'Duplicate or missing steps'
    for step in steps.values():
        for key in ('instruction', 'visible_completion'):
            assert step[key].strip() and 'EDIT' not in step[key], 'Finish task.json first'
        assert (root / step['reference']).is_file(), step['reference']
    rows = [json.loads(s) for s in (root / 'examples.jsonl').read_text().splitlines() if s.strip()]
    assert rows, 'Add labeled test photos to examples.jsonl first'
    ids, sessions, hashes = set(), {}, {}
    for row in rows:
        assert row['id'] not in ids, 'Duplicate example ID'
        ids.add(row['id'])
        assert row['step_id'] in steps and row['label'] in LABELS
        assert row['split'] in ('train', 'dev', 'test') and row['session']
        previous = sessions.setdefault(row['session'], row['split'])
        assert previous == row['split'], 'Same recording session in different splits'
        path = root / row['image']
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        previous = hashes.setdefault(digest, row['split'])
        assert previous == row['split'], 'Identical image crosses splits'
        reference = root / steps[row['step_id']]['reference']
        assert digest != hashlib.sha256(reference.read_bytes()).hexdigest(), 'Test photo equals reference'
    return steps, rows


def prompt(step):
    return ('Image 1 is a human-approved reference for the required assembly state. '
            'Image 2 is the current observation. Compare visible parts and their relationships, '
            'not background, camera position, or hands. The instruction is a goal, not an '
            'observation. Never assume it has been completed. If relevant parts or connections '
            'are hidden or too small, answer uncertain. Do not infer tightness or internal '
            'connections. matches means ALL specified visible conditions can be confirmed; '
            'mismatch means at least one visible condition is contradicted; otherwise uncertain. '
            'Return only JSON: {"assessment":"matches|mismatch|uncertain",'
            '"evidence":"one short sentence", "next_action":"one short instruction"}. '
            'Keep each sentence under 20 words. Goal: ' + step['instruction'] +
            '. Visible conditions: ' + step['visible_completion'])


def parse(raw):
    text = raw.strip()
    if text.startswith('```') and text.endswith('```'):
        text = '\n'.join(text.splitlines()[1:-1])
    try:
        obj = json.loads(text)
        assert obj['assessment'] in LABELS
        assert all(isinstance(obj[k], str) for k in ('evidence', 'next_action'))
        return obj
    except (ValueError, KeyError, TypeError, AssertionError):
        return {'assessment': 'invalid_json', 'evidence': '', 'next_action': ''}


def metrics(rows):
    matrix = {k: {p: 0 for p in (*LABELS, 'invalid_json')} for k in LABELS}
    for row in rows:
        matrix[row['label']][row['prediction']['assessment']] += 1
    mismatch_n = sum(matrix['mismatch'].values())
    n = len(rows)
    return {'n': n, 'accuracy': sum(matrix[k][k] for k in LABELS) / n,
            'mismatch_recall': matrix['mismatch']['mismatch'] / mismatch_n if mismatch_n else None,
            'false_approvals': matrix['mismatch']['matches'],
            'unclear_views_approved': matrix['uncertain']['matches'],
            'invalid_json_count': sum(v['invalid_json'] for v in matrix.values()),
            'confusion_matrix': matrix}


def evaluate(args):
    steps, all_rows = load_inputs(args.root)
    rows = [r for r in all_rows if r['split'] == args.split]
    assert rows, 'No examples in selected split'
    if args.check_only:
        from PIL import Image
        for row in rows:
            for name in (row['image'], steps[row['step_id']]['reference']):
                with Image.open(args.root / name) as im:
                    im.verify()
        print(f'PASS: {len(rows)} {args.split} examples; task, files, sessions checked')
        return
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    assert torch.cuda.is_available(), 'Select the HP project Python environment'
    cfg = json.loads((REPO / 'runs/verifier_v2/config.json').read_text())
    out = args.root / 'results' / args.name
    out.mkdir(parents=True, exist_ok=False)
    write(out / 'run.json', {'model': cfg['model'], 'revision': cfg['model_revision'],
                            'adapter': str(args.adapter) if args.adapter else None,
                            'split': args.split, 'scope': 'reference-conditioned visible-state pilot',
                            'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                            'task': json.loads((args.root / 'task.json').read_text())})
    processor = AutoProcessor.from_pretrained(cfg['model'], revision=cfg['model_revision'],
                                              min_pixels=64*32*32, max_pixels=256*32*32)
    model = Qwen3VLForConditionalGeneration.from_pretrained(cfg['model'], revision=cfg['model_revision'],
                 dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda')
    if args.adapter:
        from peft import PeftModel
        import peft.tuners.lora.model as lm
        if hasattr(lm, 'dispatch_torchao'):
            lm.dispatch_torchao = lambda *a, **kw: None
        model = PeftModel.from_pretrained(model, str(args.adapter))
    model.eval()
    results = []
    with (out / 'predictions.jsonl').open('w') as stream:
        for row in rows:
            step = steps[row['step_id']]
            images = []
            image_hashes = []
            for name in (step['reference'], row['image']):
                path = args.root / name
                image_hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
                with Image.open(path) as im:
                    images.append(im.convert('RGB').copy())
            messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'image'},
                         {'type': 'text', 'text': prompt(step)}]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            torch.cuda.synchronize()
            start = time.monotonic()
            batch = processor(text=[text], images=images, return_tensors='pt').to('cuda')
            with torch.inference_mode():
                tokens = model.generate(**batch, max_new_tokens=192, do_sample=False)
            torch.cuda.synchronize()
            elapsed = time.monotonic() - start
            raw = processor.batch_decode(tokens[:, batch['input_ids'].shape[1]:], skip_special_tokens=True)[0]
            result = {**row, 'prediction': parse(raw), 'raw': raw, 'seconds': elapsed,
                      'image_sha256': image_hashes}
            results.append(result)
            stream.write(json.dumps(result) + '\n'); stream.flush()
            print(row['id'], result['prediction'], flush=True)
    summary = metrics(results)
    write(out / 'metrics.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['init', 'evaluate'])
    parser.add_argument('--root', type=Path, default=REPO / 'runs/jeep_demo_v1')
    parser.add_argument('--split', choices=['dev', 'test'], default='dev')
    parser.add_argument('--name', default='base_dev_01')
    parser.add_argument('--adapter', type=Path)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if args.command == 'init': initialize(args.root)
    else: evaluate(args)
