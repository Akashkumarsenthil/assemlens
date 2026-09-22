import argparse, contextlib, json, time, traceback, os, fcntl
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, Trainer, TrainingArguments, TrainerCallback, set_seed
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig, get_peft_model
import peft.tuners.lora.model as peft_lora_model
peft_lora_model.dispatch_torchao = lambda *args, **kwargs: None

PROMPT = ('These two images are chronological frames from one assembly action. '
          'Identify the action. Return only JSON with string keys action, verb, object. '
          'Use concise assembly-action names. Do not judge whether the assembly is correct.')

def read_rows(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]

def encode(processor, row, training=True):
    images = []
    for path in row['images']:
        with Image.open(path) as im:
            images.append(im.convert('RGB').copy())
    messages = [{'role':'user','content':[{'type':'image'} for _ in images]+[{'type':'text','text':PROMPT}]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    batch = processor(text=[prompt], images=images, return_tensors='pt')
    if not training:
        return batch
    target = json.dumps(row['target'], ensure_ascii=False)
    full = processor.apply_chat_template(messages+[{'role':'assistant','content':[{'type':'text','text':target}]}], tokenize=False, add_generation_prompt=False)
    encoded = processor(text=[full], images=images, return_tensors='pt')
    n = batch['input_ids'].shape[1]
    assert torch.equal(encoded['input_ids'][0,:n], batch['input_ids'][0]), 'Chat-template prefix mismatch; stop rather than train wrong labels.'
    assert encoded['input_ids'].shape[1] <= 4096, 'Example too long; reduce image resolution, never truncate image tokens.'
    labels = encoded['input_ids'].clone()
    labels[:,:n] = -100
    assert (labels != -100).any()
    encoded['labels'] = labels
    return encoded

class Rows(torch.utils.data.Dataset):
    def __init__(self, rows): self.rows = rows
    def __len__(self): return len(self.rows)
    def __getitem__(self, i): return self.rows[i]

class Budget(TrainerCallback):
    def __init__(self, hours): self.hours=hours; self.started=None
    def on_train_begin(self,args,state,control,**kw): self.started=time.monotonic()
    def on_step_end(self,args,state,control,**kw):
        if time.monotonic()-self.started > self.hours*3600:
            control.should_save=True; control.should_training_stop=True
        return control

def score(model, processor, rows, output, base=False):
    predictions=[]; model.eval()
    context = model.disable_adapter() if base else contextlib.nullcontext()
    with context, torch.inference_mode():
        for row in rows:
            batch=encode(processor,row,False).to('cuda')
            started=time.monotonic()
            tokens=model.generate(**batch,max_new_tokens=96,do_sample=False,use_cache=True)
            torch.cuda.synchronize()
            text=processor.batch_decode(tokens[:,batch['input_ids'].shape[1]:],skip_special_tokens=True)[0]
            try:
                pred=json.loads(text.strip()); assert isinstance(pred,dict)
            except Exception: pred={}
            predictions.append({'id':row['id'],'target':row['target'],'prediction':pred,'raw':text,'seconds':time.monotonic()-started})
    def norm(s): return str(s).strip().lower().replace('_',' ')
    metrics={k+'_exact_match':sum(norm(p['prediction'].get(k,''))==norm(p['target'][k]) for p in predictions)/len(predictions) for k in ['action','verb','object']}
    metrics['valid_json_fraction']=sum(bool(p['prediction']) for p in predictions)/len(predictions)
    metrics['n']=len(predictions)
    Path(output).write_text(json.dumps({'metrics':metrics,'predictions':predictions},indent=2))
    print(output,metrics,flush=True)
    return metrics

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); ap.add_argument('--pilot',action='store_true'); args=ap.parse_args()
    cfg=json.loads(Path(args.config).read_text()); root=Path(cfg['root']); out=root/('pilot' if args.pilot else 'overnight'); out.mkdir(exist_ok=True)
    lock=(out/'worker.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status=out/'status.json'
    def write_status(**v): status.write_text(json.dumps(v,indent=2))
    write_status(state='starting',pid=os.getpid())
    try:
        set_seed(42)
        assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(), 'Need working HP CUDA/BF16 environment.'
        processor=AutoProcessor.from_pretrained(cfg['model'],revision=cfg['model_revision'],min_pixels=64*32*32,max_pixels=196*32*32)
        model=Qwen3VLForConditionalGeneration.from_pretrained(cfg['model'],revision=cfg['model_revision'],dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda')
        targets=[name for name,module in model.named_modules() if 'language_model' in name and '.self_attn.' in name and name.rsplit('.',1)[-1] in {'q_proj','k_proj','v_proj','o_proj'}]
        assert targets, 'No language attention targets found; model layout changed.'
        model=get_peft_model(model,LoraConfig(r=16,lora_alpha=32,lora_dropout=0.05,bias='none',task_type='CAUSAL_LM',target_modules=targets))
        model.print_trainable_parameters(); model.enable_input_require_grads(); model.config.use_cache=False
        train=read_rows(root/'train.jsonl'); val=read_rows(root/'validation.jsonl')
        if args.pilot: train=train[:8]; val=val[:2]
        measured=val[:2 if args.pilot else 40]
        signature=json.dumps({'config':cfg,'train_records':train,'val_records':val},sort_keys=True)
        sigfile=out/'signature.json'
        if sigfile.exists(): assert sigfile.read_text()==signature, 'Data/config changed: use a new ROOT/run directory.'
        else: sigfile.write_text(signature)
        baseline=score(model,processor,measured,out/'baseline.json',base=True)
        def collate(rows):
            assert len(rows)==1
            return encode(processor,rows[0])
        training_args=TrainingArguments(output_dir=str(out/'checkpoints'),per_device_train_batch_size=1,per_device_eval_batch_size=1,gradient_accumulation_steps=2 if args.pilot else 8,num_train_epochs=8,max_steps=2 if args.pilot else -1,learning_rate=1e-4,warmup_ratio=0.05,weight_decay=0.01,bf16=True,gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},logging_steps=1 if args.pilot else 5,save_steps=300,save_total_limit=2,eval_strategy='steps',eval_steps=300,report_to='none',remove_unused_columns=False,label_names=['labels'],dataloader_num_workers=0,optim='adamw_torch',seed=42)
        trainer=Trainer(model=model,args=training_args,train_dataset=Rows(train),eval_dataset=Rows(val),data_collator=collate,callbacks=[Budget(cfg['training_hours'])])
        checkpoint=get_last_checkpoint(str(out/'checkpoints')) if (out/'checkpoints').exists() else None
        write_status(state='training',pid=os.getpid(),resume_from=checkpoint)
        trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model(str(out/'adapter')); processor.save_pretrained(out/'adapter')
        trainer.save_state()
        loss=trainer.evaluate(); after=score(model,processor,measured,out/'adapted.json')
        (out/'comparison.json').write_text(json.dumps({'baseline':baseline,'adapted':after,'validation_loss':loss,'scope':'action recognition on held-out recordings; not furniture defect detection'},indent=2))
        write_status(state='finished',pid=os.getpid(),adapter=str(out/'adapter'))
    except BaseException:
        write_status(state='failed',pid=os.getpid(),error=traceback.format_exc()); raise

if __name__=='__main__': main()
