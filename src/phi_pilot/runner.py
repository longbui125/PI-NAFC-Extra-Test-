"""Single-process stages, supervised externally by guard.py in the notebook."""
import argparse
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import statistics as stats
import sys

from hooks import read, save, latest_checkpoint

CODE = Path(__file__).resolve().parent
SHARED = Path(os.environ.get('PILOT_OUTPUT', '/content/drive/MyDrive/NAFC_Phi_Olympic_3Seeds_v2'))
OUT = SHARED
MODEL = Path('/content/phi_same_source_model')
MERGED = Path('/content/phi_same_source_merged')
PLAN = read(CODE/'plan.json')
CAP = PLAN['max_tokens']
ARMS = tuple(PLAN['arms'])
SEED = 1
PREPARATION = ('anchors.json', 'buckets_train.json', 'buckets_validation.json',
               'buckets_test.json', 'context.json', 'readiness.json',
               'results/base_test.json', 'results/base_validation.json')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def verify_sources():
    for name, expected in read(CODE/'manifest.json').items():
        if file_hash(CODE/name) != expected:
            raise ValueError('Bundled source changed: '+name)


def configure_seed(seed):
    """Share only frozen preparation; never copy another seed's trained state."""
    global OUT, SEED
    verify_sources()
    shared_identity = read(SHARED/'identity.json')
    if shared_identity['protocol'] != digest(PLAN) or shared_identity['sources'] != read(CODE/'manifest.json'):
        raise ValueError('Shared preparation belongs to another experiment.')
    hashes = {name:file_hash(SHARED/name) for name in PREPARATION}
    folder = SHARED/f'seed_{seed}'
    identity = dict(shared=shared_identity, seed=seed, inputs=hashes)
    path = folder/'identity.json'
    if path.exists():
        if read(path) != identity:
            raise ValueError('Seed inputs changed; preserve its checkpoints and use a new experiment folder.')
    elif folder.exists() and any(folder.iterdir()):
        raise ValueError('Refusing to overwrite an unidentified seed folder.')
    save(path, identity)
    for name, expected in hashes.items():
        target = folder/name
        if target.exists():
            if file_hash(target) != expected:
                raise ValueError('Seed preparation changed: '+name)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix+'.tmp')
            shutil.copyfile(SHARED/name, temporary)
            if file_hash(temporary) != expected:
                raise ValueError('Incomplete preparation copy: '+name)
            temporary.replace(target)
    save(folder/'plan.json', dict(PLAN, seed=seed))
    OUT, SEED = folder, seed


def initialize():
    verify_sources()
    versions = {name: importlib.metadata.version(name) for name in PLAN['versions']}
    if versions != PLAN['versions']:
        raise ValueError(f'Wrong dependency versions: {versions}')
    identity = dict(protocol=digest(PLAN), sources=read(CODE/'manifest.json'), versions=versions)
    path = OUT/'identity.json'
    if path.exists() and read(path) != identity:
        raise ValueError('This Drive folder belongs to a different protocol. Preserve it and use a new folder.')
    if not path.exists() and OUT.exists() and any(p.name != 'logs' for p in OUT.iterdir()):
        raise ValueError('Refusing to overwrite an unidentified experiment folder.')
    OUT.mkdir(parents=True, exist_ok=True)
    save(path, identity)
    save(OUT/'plan.json', PLAN)
    shutil.copyfile(CODE/'questions.json', OUT/'questions.json')
    for name in (*read(CODE/'manifest.json'), 'manifest.json'):
        target = OUT/'source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(CODE/name, target)
    import torch
    if not torch.cuda.is_available() or 'A100' not in torch.cuda.get_device_name(0):
        raise RuntimeError('Select an A100 runtime for this notebook.')
    from huggingface_hub import snapshot_download
    snapshot_download(PLAN['model'], revision=PLAN['revision'], local_dir=MODEL,
                      allow_patterns=['*.json', '*.safetensors', '*.model', '*.tiktoken', '*.jinja'], max_workers=4)
    from transformers import AutoConfig, AutoTokenizer
    from common import chat_messages
    data = read(CODE/'questions.json')
    tok = AutoTokenizer.from_pretrained(MODEL)
    length = max(len(tok.apply_chat_template(chat_messages(row['question']), tokenize=True,
                                            add_generation_prompt=True, return_dict=False))
                 for split in ('train', 'anchor_pool', 'validation', 'test') for row in data[split])
    native = AutoConfig.from_pretrained(MODEL).max_position_embeddings
    context = PLAN['context_tokens']
    if not length < context <= native:
        raise ValueError('Frozen prompts do not fit the declared total context.')
    # Keep the actual seed-1 context explicitly rather than relying on len(BatchEncoding).
    save(OUT/'context.json', dict(max_prompt=length, context=context, completion_cap=CAP))
    print(f'[ready] A100; total context {context}; max prompt {length}; completion cap {CAP}; Drive: {OUT}', flush=True)


def preflight():
    """Exercise the longest real prompt plus full completion backward, without updating weights."""
    if (OUT/'backward_ok.json').exists():
        print('[reuse] successful full-length backward check', flush=True)
        return
    import torch
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    import torch.nn.functional as F
    length = read(OUT/'context.json')['context']
    model = get_peft_model(AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation='sdpa').cuda(),
        LoraConfig(r=32, lora_alpha=64, lora_dropout=0., target_modules='all-linear', task_type='CAUSAL_LM'))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.enable_input_require_grads()
    model.config.use_cache = False
    torch.cuda.reset_peak_memory_stats()
    x = torch.full((1, length), 42, dtype=torch.long, device='cuda')
    logits = model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False).logits
    loss = F.cross_entropy(logits[:, -CAP:-1].float().reshape(-1, logits.size(-1)), x[:, -CAP+1:].reshape(-1))
    if not torch.isfinite(loss):
        raise RuntimeError('Full-length backward preflight produced a non-finite loss.')
    loss.backward()
    peak = torch.cuda.max_memory_allocated()/1024**3
    save(OUT/'backward_ok.json', dict(context=length, peak_allocated_gib=peak, optimizer_updates=0))
    print(f'[backward OK] micro-batch 1; peak {peak:.2f} GiB; no optimizer update', flush=True)


class Sampler:
    def __init__(self, model=MODEL, stamp=None):
        self.model, self.stamp, self.llm = str(model), stamp or PLAN['revision'], None

    def sample(self, rows, n, tag, seed):
        from common import chat_messages, is_correct
        folder = OUT/'samples'/tag
        identity = dict(rows=digest(rows), model=self.stamp, n=n, seed=seed,
                        cap=CAP, batch=16, protocol=digest(PLAN))
        if (folder/'identity.json').exists() and read(folder/'identity.json') != identity:
            raise ValueError('Cached generation belongs to different inputs: '+tag)
        save(folder/'identity.json', identity)
        result = []
        for offset in range(0, len(rows), 16):
            batch = rows[offset:offset+16]
            path = folder/f'{offset:05d}.json.gz'
            if path.exists():
                saved = json.loads(gzip.decompress(path.read_bytes()))
                if ([r['question'] for r in saved] != [r['question'] for r in batch]
                    or any(len(r['samples']) != n for r in saved)):
                    raise ValueError('Incomplete or misaligned cached batch: '+str(path))
                result.extend(saved)
                print(f'[reuse {tag}] {min(offset+16,len(rows))}/{len(rows)}', flush=True)
                continue
            if self.llm is None:
                from transformers import AutoTokenizer
                from vllm import LLM
                self.tok = AutoTokenizer.from_pretrained(self.model)
                self.llm = LLM(self.model, dtype='bfloat16', gpu_memory_utilization=.80,
                               max_model_len=read(OUT/'context.json')['context'],
                               max_num_seqs=64, max_num_batched_tokens=4096, enforce_eager=True, seed=0)
            from vllm import SamplingParams
            prompts = [self.tok.apply_chat_template(chat_messages(r['question']), tokenize=False,
                                                   add_generation_prompt=True) for r in batch]
            outs = self.llm.generate(prompts, SamplingParams(n=n, temperature=1., top_p=1.,
                                     max_tokens=CAP, seed=seed+offset), use_tqdm=True)
            if len(outs) != len(batch):
                raise RuntimeError('Generation returned an incomplete batch.')
            saved = []
            for i, (row, output) in enumerate(zip(batch, outs)):
                if len(output.outputs) != n:
                    raise RuntimeError('Generation returned fewer draws than requested.')
                saved.append(dict(question=row['question'], answer=row['answer'], samples=[
                    dict(text=c.text, ids=list(c.token_ids), ok=bool(is_correct(c.text, row['answer'])))
                    for c in output.outputs]))
                if i % 4 == 0:
                    print(f'[grade {tag}] question {offset+i+1}/{len(rows)}', flush=True)
            temporary = path.with_suffix('.tmp')
            with temporary.open('wb') as stream:
                stream.write(gzip.compress(json.dumps(saved, ensure_ascii=False).encode()))
                stream.flush(); os.fsync(stream.fileno())
            temporary.replace(path)
            result.extend(saved)
            print(f'[saved {tag}] {len(result)}/{len(rows)} questions on Drive', flush=True)
        return result


def raw_items(samples):
    from preparation import bucket_of
    return [dict(question=r['question'], answer=r['answer'],
                 p=sum(s['ok'] for s in r['samples'])/len(r['samples']),
                 bucket=bucket_of(sum(s['ok'] for s in r['samples'])/len(r['samples']), [.75,.25]),
                 samples=[[s['ok'],len(s['ids'])] for s in r['samples']]) for r in samples]


def calibrate(samples, split, budget, reference=None):
    """The bucket statistics and calibration formulas from Ngoc's prepare_buckets.py."""
    from constraints import calibrate_lambda
    items = raw_items(samples)
    counts = [sum(r['bucket'] == k for r in items) for k in range(3)]
    weights = [n/len(items) for n in counts]
    natural = []
    for k in range(3):
        pairs = [s for r in items if r['bucket'] == k for s in r['samples']]
        natural.append(stats.median([L for ok,L in pairs if ok] or [L for _,L in pairs] or [budget]))
    scale = budget/sum(w*l for w,l in zip(weights,natural))
    budgets = [l*scale for l in natural]
    init = [calibrate_lambda([r['samples'] for r in items if r['bucket']==k], budgets[k],
             budgets[k]/max(CAP-budgets[k],1.), beta=.5) if counts[k] else 0. for k in range(3)]
    init_global = calibrate_lambda([r['samples'] for r in items],budget,budget/max(CAP-budget,1.),beta=.5)
    if reference:
        budgets,init,init_global = (reference[k] for k in ('budgets','init_lambdas','init_lambda_global'))
    meta = dict(model=str(MODEL), dataset='NuminaMath-1.5/Olympic', split=split, n=8,
                max_len=CAP,budget=budget,thresholds=[.75,.25], counts=counts, weights=weights,
                natural_len=natural,budgets=budgets,difficulty=[0.,.5,1.],init_lambdas=init,
                init_lambda_global=init_global,
                base=dict(acc=stats.mean(r['p'] for r in items),
                          mean_len=stats.mean(s[1] for r in items for s in r['samples'])))
    return dict(meta=meta, **meta, items=items)


def readiness():
    rows = read(CODE/'questions.json')['validation']
    samples = Sampler().sample(rows,8,'validation_ref',0)
    items = raw_items(samples)
    accuracy = stats.mean(r['p'] for r in items)
    capped = stats.mean(s[1]>=CAP for r in items for s in r['samples'])
    mixed = stats.mean(0<r['p']<1 for r in items)
    budget = PLAN['pi_budget_tokens']
    gate = PLAN['readiness']
    ready = (accuracy>=gate['min_validation_accuracy'] and mixed>=gate['min_mixed_group_fraction']
             and capped<=gate['max_capped_fraction'] and 0<budget<CAP)
    save(OUT/'readiness.json',dict(accuracy=accuracy,capped=capped,mixed_groups=mixed,
                                  pi_budget_tokens=budget,ready=ready,criteria=PLAN['readiness']))
    print(json.dumps(read(OUT/'readiness.json'),indent=2),flush=True)
    if not ready:
        raise RuntimeError('Capability/truncation readiness gate failed. Preserve diagnostics; no main training was run.')


def anchors():
    if (OUT/'anchors.json').exists():
        if len(read(OUT/'anchors.json')['items']) != 128:
            raise ValueError('Incomplete anchor file.')
        print('[reuse] 128 Phi anchors', flush=True); return
    from preparation import pick_anchors, noise_floors
    rows = read(CODE/'questions.json')['anchor_pool']
    sampler = Sampler()
    scored = []
    # All selection chunks have the same identities even if selection stops early.
    for offset in range(0,len(rows),128):
        samples = sampler.sample(rows[offset:offset+128],8,f'anchor_select_{offset:04d}',0)
        for r in samples:
            good = [s for s in r['samples'] if s['ok']]
            scored.append(dict(question=r['question'],answer=r['answer'],
                p_select=len(good)/8,base_first_correct=r['samples'][0]['ok'],
                base_response=good[0]['text'] if good else None,
                base_response_ids=good[0]['ids'] if good else None,
                base_response_len=len(good[0]['ids']) if good else None))
        selected = pick_anchors(scored,128,.75,0)
        save(OUT/'anchor_selection_status.json',dict(tested=len(scored),stable=len(selected),required=128))
        print(f'[anchors] {len(selected)}/128 stable after {len(scored)}/2048 candidates',flush=True)
        if len(scored)>=800 and len(selected)==128: break
    if len(selected)!=128:
        raise RuntimeError('The frozen 2048 candidates do not provide 128 stable anchors; no training is allowed.')
    check = sampler.sample(selected,4,'anchor_check',100000)
    for row,sample in zip(selected,check):
        row.update(p_check=sum(s['ok'] for s in sample['samples'])/4,
                   check_first_correct=sample['samples'][0]['ok'])
    meta = dict(model=str(MODEL),dataset='NuminaMath-1.5/Olympic',n_anchor=128,
                k_select=8,k_check=4,hi=.75,lo=.25,max_len=CAP,
                tested_candidates=len(scored),**noise_floors(selected,.25))
    save(OUT/'anchors.json',dict(meta=meta,items=selected))
    print('[anchors complete]',json.dumps(meta),flush=True)


def prepare(split):
    if not read(OUT/'readiness.json')['ready']:
        raise RuntimeError('Readiness gate has not passed.')
    path = OUT/f'buckets_{split}.json'
    if path.exists():
        print('[reuse]',path,flush=True); return
    rows = read(CODE/'questions.json')[split]
    samples = Sampler().sample(rows,8,f'{split}_ref',0)
    ref = read(OUT/'buckets_train.json')['meta'] if split!='train' else None
    save(path,calibrate(samples,split,read(OUT/'readiness.json')['pi_budget_tokens'],ref))
    print('[prepared]',split,flush=True)


def evaluate(model, stamp, split, name):
    from common import last_boxed
    target = OUT/'results'/f'{name}_{split}.json'
    if target.exists():
        print('[reuse]',target,flush=True); return
    buckets = read(OUT/f'buckets_{split}.json')
    rows = buckets['items']
    samples = Sampler(model,stamp).sample(rows,4,f'eval_{name}_{split}',1000000)
    per = []
    for i,(row,sample) in enumerate(zip(rows,samples)):
        draws = sample['samples']; lens=[len(s['ids']) for s in draws]
        per.append(dict(id=digest([row['question'],row['answer']]),bucket=row['bucket'],
            acc=sum(s['ok'] for s in draws)/4,len=stats.mean(lens),
            over=sum(L>buckets['budgets'][row['bucket']] for L in lens)/4,
            trunc=sum(L>=CAP for L in lens)/4,
            boxed=sum(last_boxed(s['text']) is not None for s in draws)/4))
    result = dict(model=str(model),eval_set='NuminaMath-1.5/Olympic/'+split,
        budget=buckets['budget'],decoding=dict(n=4,seed=1000000,temperature=1.,top_p=1.,batch_size=16),
        overall=dict(acc=100*stats.mean(r['acc'] for r in per),mean_len=stats.mean(r['len'] for r in per),
                     truncated_rate=100*stats.mean(r['trunc'] for r in per),
                     boxed_rate=100*stats.mean(r['boxed'] for r in per)),per_question=per)
    save(target,result)
    print('[evaluation complete]',name,split,result['overall'],flush=True)


def training_args(arm, output, steps=300, smoke=False):
    args = ['--method','laconic','--model',str(MODEL),'--buckets',str(OUT/'buckets_train.json'),
            '--output_dir',str(output),'--seed',str(SEED),'--max_steps',str(steps),
            '--max_completion_length',str(CAP),'--prompts_per_step','8','--num_generations','8',
            '--micro_batch','1','--lr','1e-5','--lora_r','32','--eta','.05','--kp','4',
            '--lambda_max','3','--two_sided','--adv_norm','decoupled','--scale_rewards','none',
            '--init','calibrated','--beta',str(PLAN['kl_beta']) if arm=='kl' else '0','--vllm_mem','.3',
            '--flip_mode',dict(mon='monitor',rep='nafc_replay',quota='replay_fixed',kl='monitor')[arm],
            '--anchors',str(OUT/'anchors.json'),'--anchor_split','.5','--split_seed','0',
            '--flip_every','25','--flip_k','4','--flip_eta','10','--flip_lam_max','1',
            '--flip_margin','.02','--replay_frac_max','.5',
            '--save_steps','1' if smoke else '25','--save_total_limit','3',
            '--keep_steps','' if smoke else '50,100,150,200,250,300']
    return args


def train(arm, smoke=False):
    if not (OUT/'anchors.json').exists() or not read(OUT/'readiness.json')['ready']:
        raise RuntimeError('Complete preparation before training.')
    folder = OUT/'runs'/arm if not smoke else OUT/'smoke'
    if smoke and (folder/'run_info.json').exists():
        print('[reuse] colocated training smoke check',flush=True);return
    checkpoint = latest_checkpoint(folder)
    if not smoke and checkpoint and int(checkpoint.name.split('-')[-1])==300:
        print('[reuse] trained arm',arm,flush=True);return
    if arm=='quota':
        parent = latest_checkpoint(OUT/'runs'/'rep')
        if parent is None or parent.name!='checkpoint-300':
            raise RuntimeError('The quota control requires the complete NAFC trace, not another notebook.')
    start = int(checkpoint.name.split('-')[-1]) if checkpoint else 0
    # Discard only uncommitted diagnostics beyond the resumed optimizer state.
    for path in (folder/'observations').glob('*.json'):
        if int(path.stem)>=start:
            path.unlink()
    log = folder/'flip_log.jsonl'
    if log.exists():
        history = read(checkpoint/'flip_state.json')['history'] if checkpoint else []
        log.write_text(''.join(json.dumps(r)+'\n' for r in history),encoding='utf8')
    if not checkpoint and folder.exists():
        print('[restart] no checkpoint was committed yet; starting from step 0.',flush=True)
    os.environ['PILOT_ARM'] = arm
    os.environ['PILOT_RESUME'] = str(checkpoint) if checkpoint else ''
    sys.argv = ['train.py',*training_args(arm,folder,2 if smoke else 300,smoke)]
    if checkpoint: sys.argv += ['--resume',str(checkpoint)]
    print('[training]',arm,'resume',checkpoint,'micro=1, accumulation=64',flush=True)
    import train as upstream
    upstream.main()


def merge(arm,step):
    split = 'test' if step==300 else 'validation'
    name = arm if step==300 else f'{arm}_step{step}'
    if (OUT/'results'/f'{name}_{split}.json').exists():
        print('[reuse] evaluation already completed',name,flush=True);return
    path = latest_checkpoint(OUT/'runs'/arm) if step==300 else OUT/'runs'/arm/'keep'/f'step-{step}'
    if path is None or not (path/'adapter_model.safetensors').exists():
        raise ValueError('Missing adapter for evaluation.')
    if step==300 and path.name!='checkpoint-300':
        raise ValueError('Final evaluation requires the step-300 checkpoint.')
    stamp = file_hash(path/'adapter_model.safetensors')
    if (MERGED/'pilot_stamp.json').exists() and read(MERGED/'pilot_stamp.json')==dict(stamp=stamp):
        print('[reuse] merged adapter',flush=True);return
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from peft import PeftModel
    if MERGED.exists(): shutil.rmtree(MERGED)
    base = AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.bfloat16,device_map='cpu')
    merged = PeftModel.from_pretrained(base,path).merge_and_unload()
    merged.save_pretrained(MERGED)
    AutoTokenizer.from_pretrained(MODEL).save_pretrained(MERGED)
    save(MERGED/'pilot_stamp.json',dict(stamp=stamp))
    print('[merged]',arm,step,flush=True)


def report():
    from metrics import summarize, write_report
    reference = [dict(id=digest([r['question'],r['answer']]),p=r['p'])
                 for r in read(OUT/'buckets_test.json')['items']]
    evaluations, traces = {}, {}
    for name in ARMS:
        path = OUT/'results'/f'{name}_test.json'
        if not path.exists():
            raise ValueError('Missing prespecified final result: '+name)
        obs = [read(p) for p in sorted((OUT/'runs'/name/'observations').glob('*.json'))]
        if len(obs)!=300 or [r['step'] for r in obs]!=list(range(300)):
            raise ValueError('Incomplete/extra training observation steps: '+name)
        evaluations[name], traces[name] = read(path), obs
    rows = summarize(reference, evaluations)
    for row in rows:
        obs = traces[row['arm']]
        row.update(replay_groups=sum(r['replay_groups'] for r in obs),
            training_response_tokens=sum(sum(g['lengths']) for r in obs for g in r['records']),
            task_groups=sum(r['groups']-r['replay_groups'] for r in obs))
    write_report(OUT, rows, f'Phi Olympic — seed {SEED}; PI {PLAN["pi_budget_tokens"]} tokens')
    save(OUT/'summary.json', dict(seed=SEED, rows=rows, protocol=digest(PLAN)))


def export():
    import zipfile
    target=Path('/content/nafc_phi_olympic_3seeds.zip')
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as bundle:
        for path in SHARED.rglob('*'):
            if path.is_file() and not path.name.endswith('.tmp'):
                bundle.write(path,path.relative_to(SHARED).as_posix())
    print('[ZIP ready]',target,flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['initialize','preflight','readiness','anchors','prepare','base','smoke','train','merge','evaluate','report','aggregate','export'])
    parser.add_argument('--split',choices=['train','validation','test'],default='test')
    parser.add_argument('--arm',choices=ARMS,default='mon')
    parser.add_argument('--step',type=int,choices=[50,100,150,200,250,300],default=300)
    parser.add_argument('--seed',type=int,choices=PLAN['seeds'])
    args=parser.parse_args()
    if args.command in ('train','merge','evaluate','report'):
        if args.seed is None: parser.error('This command requires --seed 1, 2 or 3.')
        configure_seed(args.seed)
    elif args.seed is not None:
        parser.error('Preparation is shared; --seed is only for training, evaluation and reporting.')
    if args.command=='initialize': initialize()
    elif args.command=='preflight': preflight()
    elif args.command=='readiness': readiness()
    elif args.command=='anchors': anchors()
    elif args.command=='prepare': prepare(args.split)
    elif args.command=='base': evaluate(MODEL,PLAN['revision'],args.split,'base')
    elif args.command=='smoke': train('mon',smoke=True)
    elif args.command=='train': train(args.arm)
    elif args.command=='merge': merge(args.arm,args.step)
    elif args.command=='evaluate':
        name=args.arm if args.step==300 else f'{args.arm}_step{args.step}'
        if (OUT/'results'/f'{name}_{args.split}.json').exists():
            print('[reuse] completed evaluation',name,args.split,flush=True)
        else:
            stamp=read(MERGED/'pilot_stamp.json')['stamp']
            evaluate(MERGED,stamp,args.split,name)
    elif args.command=='report': report()
    elif args.command=='aggregate':
        from metrics import aggregate
        aggregate(SHARED, PLAN['seeds'], ARMS, digest(PLAN))
    elif args.command=='export': export()



if __name__=='__main__':
    try:
        main()
    except BaseException:
        import traceback
        traceback.print_exc();sys.stdout.flush();sys.stderr.flush();os._exit(1)
    sys.stdout.flush();sys.stderr.flush();os._exit(0)
