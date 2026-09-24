"""Assembly101 mistake-label pilot. Run as a module from the repository root.

Uses cached videos and the already downloaded official mistake CSVs only.
This is annotation-assisted classification, not product-state verification.
"""
import argparse
import csv
import hashlib
import json
import random
import shutil
import subprocess
import sys
from dataclasses import asdict
from collections import Counter
from pathlib import Path

LABELS = ('correct', 'mistake', 'correction')
REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / 'runs/verifier_v2'


def read(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def save(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


def prompt(row):
    return ('Classify this assembly action using four chronological images sampled '
            'inside its annotated interval and the preceding action descriptions. '
            'correct means normal assembly progress; mistake means an assembly error; '
            'correction means an action addressing an earlier error. Detaching alone '
            'does not imply correction. If evidence is insufficient, use uncertain. '
            'Return only JSON with one key assessment. No explanation. '
            'Descriptions are annotation-assisted observations, not expected instructions. '
            + json.dumps({'observed_action': row['action'], 'history': row['history']}))


def metrics(rows):
    matrix = {a: {p: 0 for p in (*LABELS, 'uncertain', 'invalid_json')} for a in LABELS}
    for r in rows:
        matrix[r['label']][r['prediction']] += 1
    scores = {}
    for label in LABELS:
        tp = matrix[label][label]
        fp = sum(matrix[a][label] for a in LABELS if a != label)
        fn = sum(matrix[label].values()) - tp
        scores[label] = {'precision': tp / (tp + fp) if tp + fp else 0,
                         'recall': tp / (tp + fn) if tp + fn else 0,
                         'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0}
    n = len(rows)
    return {'n': n, 'accuracy': sum(r['label'] == r['prediction'] for r in rows) / n,
            'macro_f1': sum(s['f1'] for s in scores.values()) / 3,
            'valid_json_fraction': sum(r['prediction'] != 'invalid_json' for r in rows) / n,
            'abstention_fraction': sum(r['prediction'] == 'uncertain' for r in rows) / n,
            'majority_baseline': max(Counter(r['label'] for r in rows).values()) / n,
            'classes': scores, 'confusion_matrix': matrix}


def prepare():
    import PIL.Image
    if (ROOT / 'prepared.json').exists():
        print('Already prepared. Inspect prepared.json and contact sheets.'); return
    assert not (ROOT / 'training').exists(), 'Never rebuild an existing training run.'
    ROOT.mkdir(parents=True, exist_ok=True)
    old = REPO / 'runs/assembly101_heavy_v4'
    cfg = json.loads((old / 'config.json').read_text())
    annotations = REPO / 'runs/verification_100/mistake_annotations'
    assert annotations.is_dir(), f'Missing {annotations}'
    ffmpeg = shutil.which('ffmpeg') or str(Path.home() / 'miniforge3/envs/assemlens-media/bin/ffmpeg')
    assert Path(ffmpeg).is_file(), 'FFmpeg is missing'
    from huggingface_hub import hf_hub_download
    # Keep prior heavy-run validation recordings reserved: do not use their labels.
    eligible = sorted({r['sequence'] for r in read(old / 'train.jsonl')})
    reserved = {r['sequence'] for r in read(old / 'validation.jsonl')}
    eligible = [s for s in eligible if s not in reserved and (annotations / (s + '.csv')).exists()]
    random.Random(42).shuffle(eligible)
    assert len(eligible) >= 12, 'Need at least 12 cached annotated training recordings.'
    nval = max(4, round(len(eligible) * .2))
    validation = set(eligible[:nval])
    results = {'train': [], 'validation': []}
    skips = []
    annotation_hashes = {}
    for seq in eligible:
        ann = annotations / (seq + '.csv')
        annotation_hashes[seq] = hashlib.sha256(ann.read_bytes()).hexdigest()
        with ann.open() as f:
            raw = list(csv.reader(f))
        actions = []
        for idx, cells in enumerate(raw):
            if not cells or cells[0].strip().lower() == 'start': continue
            try:
                start, end = int(cells[0]), int(cells[1])
                verb, a, b, label = [x.strip() for x in cells[2:6]]
                assert end > start >= 0 and label in LABELS
                assert verb in ('attach', 'detach')
            except (ValueError, IndexError, AssertionError):
                skips.append({'sequence': seq, 'row': idx, 'reason': 'invalid_annotation'}); continue
            actions.append({'start': start, 'end': end, 'row': idx,
                            'action': f'{verb} {a} with {b}', 'label': label})
        actions.sort(key=lambda a: (a['start'], a['end']))
        video = hf_hub_download(cfg['dataset'], f'recordings/{seq}/{cfg["camera"]}',
                                repo_type='dataset', revision=cfg['dataset_revision'], local_files_only=True)
        for i, a in enumerate(actions):
            key = hashlib.sha256(f'{seq}:{a["row"]}'.encode()).hexdigest()[:20]
            folder = ROOT / 'frames' / key
            folder.mkdir(parents=True, exist_ok=True)
            paths = []
            try:
                for j, frac in enumerate((.05, .35, .65, .95)):
                    dest = folder / f'{j}.jpg'
                    # Matches the existing project's 30-fps annotation interpretation.
                    seconds = (a['start'] + frac * (a['end'] - a['start'])) / 30
                    subprocess.run([ffmpeg, '-v', 'error', '-y', '-ss', str(seconds), '-i', video,
                                    '-frames:v', '1', '-vf', 'scale=448:448:force_original_aspect_ratio=decrease',
                                    str(dest)], check=True, capture_output=True)
                    with PIL.Image.open(dest) as im: im.verify()
                    paths.append(str(dest))
            except (OSError, subprocess.CalledProcessError) as exc:
                skips.append({'sequence': seq, 'row': a['row'], 'reason': str(exc)}); continue
            split = 'validation' if seq in validation else 'train'
            results[split].append({'id': key, 'sequence': seq, 'images': paths,
                                   'action': a['action'], 'history': [x['action'] for x in actions[max(0, i-5):i]],
                                   'label': a['label'], 'start_frame': a['start'], 'end_frame': a['end']})
        print(seq, {s: len(v) for s, v in results.items()}, flush=True)
    for split, rows in results.items():
        assert len(rows) >= (100 if split == 'train' else 30), f'Too few {split} examples'
        assert all(Counter(r['label'] for r in rows)[l] >= 5 for l in LABELS), f'Insufficient class support: {split}'
        (ROOT / f'{split}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    from PIL import Image, ImageDraw, ImageOps
    for label in LABELS:
        samples = [r for r in results['train'] if r['label'] == label][:4]
        canvas = Image.new('RGB', (1000, len(samples) * 220), 'white')
        for i, row in enumerate(samples):
            for j, p in enumerate(row['images']):
                with Image.open(p) as im:
                    canvas.paste(ImageOps.contain(im.convert('RGB'), (245, 180)), (j*250, i*220))
            ImageDraw.Draw(canvas).text((5, i*220+182), row['action'] + ' | ' + row['id'], fill='black')
        canvas.save(ROOT / f'inspect_{label}.jpg')
    cfg.update(epochs=6, learning_rate=2e-5, training_hours=8)
    save(ROOT / 'config.json', cfg)
    save(ROOT / 'prepared.json', {'counts': {s: dict(Counter(r['label'] for r in v)) for s,v in results.items()},
                                  'recordings': {s: len({r['sequence'] for r in v}) for s,v in results.items()},
                                  'annotation_sha256': annotation_hashes, 'skips': skips,
                                  'scope': '3-class annotation-assisted benchmark; uncertain is abstention only',
                                  'fps_assumption': 30, 'reserved_prior_validation_recordings': sorted(reserved)})
    print('Prepared. Inspect the three inspect_*.jpg sheets and prepared.json before training.')


def train(confirmed):
    assert confirmed, 'Inspect contact sheets first, then pass --confirm-alignment.'
    import fcntl
    import time
    import traceback
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, Trainer, TrainingArguments, TrainerCallback, EarlyStoppingCallback, set_seed
    from transformers.trainer_utils import get_last_checkpoint
    from peft import LoraConfig, get_peft_model
    import peft.tuners.lora.model as lm
    if hasattr(lm, 'dispatch_torchao'): lm.dispatch_torchao = lambda *a, **kw: None
    out = ROOT / 'training'
    out.mkdir(exist_ok=True)
    lock = (out / 'lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (out / 'comparison.json').exists(), 'Run is finished; do not relaunch.'
    try:
        cfg = json.loads((ROOT / 'config.json').read_text())
        sets = {s: read(ROOT / f'{s}.jsonl') for s in ('train', 'validation')}
        assert not ({r['sequence'] for r in sets['train']} & {r['sequence'] for r in sets['validation']})
        fingerprint = hashlib.sha256(Path(__file__).read_bytes() + (ROOT/'config.json').read_bytes())
        for s in sets:
            fingerprint.update((ROOT/f'{s}.jsonl').read_bytes())
            for r in sets[s]:
                for p in r['images']: fingerprint.update(Path(p).read_bytes())
        signature = fingerprint.hexdigest()
        sig = out / 'signature.json'
        if sig.exists(): assert json.loads(sig.read_text()) == signature, 'Run inputs changed; do not resume.'
        else: save(sig, signature)
        save(out/'status.json', {'state': 'loading'})
        set_seed(42)
        assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        processor = AutoProcessor.from_pretrained(cfg['model'], revision=cfg['model_revision'], min_pixels=64*32*32, max_pixels=196*32*32)
        model = Qwen3VLForConditionalGeneration.from_pretrained(cfg['model'], revision=cfg['model_revision'], dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda')
        targets = [n for n,m in model.named_modules() if 'language_model' in n and '.self_attn.' in n and n.rsplit('.',1)[-1] in ('q_proj','k_proj','v_proj','o_proj')]
        assert targets
        model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=.05, bias='none', task_type='CAUSAL_LM', target_modules=targets))
        model.enable_input_require_grads()
        model.config.use_cache = False
        model.print_trainable_parameters()

        def encode(row, training=False):
            images = []
            for p in row['images']:
                with Image.open(p) as im: images.append(im.convert('RGB').copy())
            messages = [{'role':'user', 'content':[{'type':'image'} for _ in images]+[{'type':'text','text':prompt(row)}]}]
            prefix = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            batch = processor(text=[prefix], images=images, return_tensors='pt')
            if not training: return batch
            messages.append({'role':'assistant','content':[{'type':'text','text':json.dumps({'assessment':row['label']})}]})
            full = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            encoded = processor(text=[full], images=images, return_tensors='pt')
            n = batch['input_ids'].shape[1]
            assert torch.equal(encoded['input_ids'][0,:n], batch['input_ids'][0])
            assert encoded['input_ids'].shape[1] <= 4096
            encoded['labels'] = encoded['input_ids'].clone()
            encoded['labels'][:,:n] = -100
            return encoded

        def evaluate_generation(filename):
            predictions = []
            model.eval()
            for row in sets['validation']:
                with torch.inference_mode():
                    batch = encode(row).to('cuda')
                    tokens = model.generate(**batch, max_new_tokens=40, do_sample=False, use_cache=True)
                raw = processor.batch_decode(tokens[:,batch['input_ids'].shape[1]:], skip_special_tokens=True)[0]
                try:
                    pred = json.loads(raw)['assessment']
                    assert pred in (*LABELS, 'uncertain')
                except (ValueError, KeyError, TypeError, AssertionError): pred = 'invalid_json'
                predictions.append({'id':row['id'], 'sequence':row['sequence'], 'label':row['label'], 'prediction':pred, 'raw':raw})
            result = {'metrics':metrics(predictions), 'predictions':predictions}
            save(out/filename, result)
            return result['metrics']

        basefile = out/'baseline.json'
        base = json.loads(basefile.read_text())['metrics'] if basefile.exists() else evaluate_generation('baseline.json')
        class Dataset(torch.utils.data.Dataset):
            def __init__(self, rows): self.rows=rows
            def __len__(self): return len(self.rows)
            def __getitem__(self, i): return self.rows[i]
        class Budget(TrainerCallback):
            def on_train_begin(self, args, state, control, **kw): self.start=time.monotonic()
            def on_step_end(self, args, state, control, **kw):
                if time.monotonic()-self.start > cfg['training_hours']*3600:
                    control.should_training_stop=True; control.should_save=True
                return control
        args = TrainingArguments(output_dir=str(out/'checkpoints'), num_train_epochs=cfg['epochs'],
            per_device_train_batch_size=1, per_device_eval_batch_size=1, gradient_accumulation_steps=8,
            learning_rate=cfg['learning_rate'], bf16=True, gradient_checkpointing=True,
            gradient_checkpointing_kwargs={'use_reentrant':False}, eval_strategy='steps', eval_steps=50,
            save_steps=50, save_total_limit=2, load_best_model_at_end=True, metric_for_best_model='eval_loss',
            greater_is_better=False, logging_steps=5, warmup_ratio=.05, weight_decay=.01,
            report_to='none', remove_unused_columns=False, label_names=['labels'], optim='adamw_torch', seed=42)
        trainer = Trainer(model=model, args=args, train_dataset=Dataset(sets['train']), eval_dataset=Dataset(sets['validation']),
                          data_collator=lambda rows: encode(rows[0], True),
                          callbacks=[Budget(), EarlyStoppingCallback(early_stopping_patience=4)])
        save(out/'training_arguments.json', args.to_dict())
        checkpoint = get_last_checkpoint(str(out/'checkpoints')) if (out/'checkpoints').exists() else None
        save(out/'status.json', {'state':'training', 'resume_from':checkpoint})
        trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model(str(out/'adapter')); processor.save_pretrained(out/'adapter')
        save(out/'trainer_state.json', asdict(trainer.state))
        adapted = evaluate_generation('adapted.json')
        save(out/'comparison.json', {'baseline':base, 'adapted':adapted, 'best_checkpoint':trainer.state.best_model_checkpoint,
                                    'steps':trainer.state.global_step, 'scope':'annotation-assisted validation; no final test evaluation'})
        save(out/'status.json', {'state':'finished'})
        print((out/'comparison.json').read_text(), flush=True)
    except BaseException:
        save(out/'status.json', {'state':'failed','error':traceback.format_exc()}); raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare','train'])
    parser.add_argument('--confirm-alignment', action='store_true')
    options = parser.parse_args()
    if options.command == 'prepare': prepare()
    else: train(options.confirm_alignment)
