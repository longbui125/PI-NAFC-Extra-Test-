"""Observation and a replay-quota control; the NAFC update itself stays upstream."""
import json
import os
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w', encoding='utf8') as stream:
        json.dump(value, stream, ensure_ascii=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


REQUIRED = ('adapter_model.safetensors', 'adapter_config.json', 'trainer_state.json',
            'optimizer.pt', 'scheduler.pt', 'rng_state.pth', 'lambda_state.json',
            'flip_state.json', 'pilot_state.json')


def latest_checkpoint(folder):
    paths = sorted(Path(folder).glob('checkpoint-*'),
                   key=lambda p: int(p.name.split('-')[-1]), reverse=True)
    for path in paths:
        if not (path/'COMPLETE.json').exists():
            continue
        marker = read(path/'COMPLETE.json')
        if all((path/f).is_file() and (path/f).stat().st_size == marker.get(f, -1) > 0
               for f in REQUIRED):
            step = int(path.name.split('-')[-1])
            if read(path/'trainer_state.json')['global_step'] == step:
                return path
    if paths:
        raise ValueError('No complete checkpoint. Preserve Drive output; do not silently restart training.')
    return None


def quota(path, step, groups):
    record = read(Path(path)/'observations'/f'{step:04d}.json')
    n = record['replay_groups']
    if record['step'] != step or record['groups'] != groups or not 0 <= n <= groups//2:
        raise ValueError('NAFC quota trace does not match this generation batch.')
    return n


TRAINER = None


def instrument(base):
    from flipcon import replace_groups

    class ObservedTrainer(base):
        def __init__(self, *args, **kwargs):
            global TRAINER
            super().__init__(*args, **kwargs)
            self._pilot_calls = 0
            checkpoint = os.environ.get('PILOT_RESUME')
            if checkpoint:
                state = read(Path(checkpoint)/'pilot_state.json')
                for name, value in state.items():
                    setattr(self, name, value)
            TRAINER = self

        def _replay(self, inputs):
            if os.environ['PILOT_ARM'] != 'quota':
                return super()._replay(inputs)
            key = lambda x: json.dumps(x['prompt'], sort_keys=True, ensure_ascii=False)
            n = quota(Path(self.args.output_dir).parent/'rep', self.state.global_step,
                      len({key(x) for x in inputs}))

            def make_item(item, anchor):
                return dict(item, prompt=self._anchor_msgs[anchor],
                            answer=self.anchors.answers[anchor], bucket=0)

            result, self._replay_ptr, used = replace_groups(
                inputs, n, self.flip_ctrl.repair_idx, self._replay_ptr, make_item, key=key)
            self._metrics['train']['flip/n_replay'].append(float(len(used)))
            return result

    return ObservedTrainer


def observe(reward, ctrl):
    from common import chat_messages, completion_text, is_correct

    def observed(completions, completion_ids, answer, bucket, **kwargs):
        trainer = TRAINER
        step = trainer.state.global_step
        # This pinned TRL generates once per optimizer step; fail loudly if that changes.
        if trainer._pilot_calls != step:
            raise RuntimeError('Generation/optimizer steps no longer align; quota matching would be invalid.')
        prompts = kwargs['prompts']
        keys = [json.dumps(p, sort_keys=True, ensure_ascii=False) for p in prompts]
        anchor_keys = {json.dumps(chat_messages(q), sort_keys=True, ensure_ascii=False): i
                       for i, q in enumerate(trainer.anchors.questions)}
        lams = list(ctrl.lambdas)
        values = reward(completions=completions, completion_ids=completion_ids,
                        answer=answer, bucket=bucket, **kwargs)
        groups = {}
        for key, completion, ids, gold, value in zip(keys, completions, completion_ids, answer, values):
            text = completion_text(completion)
            group = groups.setdefault(key, dict(anchor=anchor_keys.get(key), correct=[], lengths=[], advantages=[]))
            group['correct'].append(bool(is_correct(text, gold)))
            group['lengths'].append(len(ids))
            group['advantages'].append(float(value))
            if key in anchor_keys:
                group.setdefault('texts', []).append(text)
        records = list(groups.values())
        if len(records) != 8 or any(len(r['correct']) != 8 for r in records):
            raise RuntimeError('Expected exactly 8 distinct prompt groups x 8 responses.')
        save(Path(trainer.args.output_dir)/'observations'/f'{step:04d}.json',
             dict(step=step, groups=len(records), replay_groups=sum(r['anchor'] is not None for r in records),
                  pi_lambda_before=lams, pi_lambda_after=list(ctrl.lambdas), records=records))
        trainer._pilot_calls += 1
        return values

    observed.__name__ = reward.__name__
    return observed


def checkpoint_callback():
    from transformers import TrainerCallback

    class CommitCheckpoint(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            trainer = TRAINER
            path = Path(args.output_dir)/f'checkpoint-{state.global_step}'
            save(path/'pilot_state.json', {name: getattr(trainer, name) for name in
                 ('_pilot_calls', '_replay_ptr', '_flip_ptr', '_flip_last_check')})
            if trainer._pilot_calls != state.global_step:
                raise RuntimeError('Refusing a checkpoint with mismatched observation state.')
            sizes = {f: (path/f).stat().st_size for f in REQUIRED}
            if not all(sizes.values()):
                raise RuntimeError('Checkpoint contains an empty state file.')
            save(path/'COMPLETE.json', sizes)
            print(f'[checkpoint committed] {path}', flush=True)

    return CommitCheckpoint()
