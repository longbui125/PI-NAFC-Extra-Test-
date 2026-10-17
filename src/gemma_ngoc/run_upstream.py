"""Supply the previously frozen Olympic data to Ngoc's training and evaluation scripts."""
import json
from pathlib import Path
import runpy
import sys
import common

ROOT = Path(__file__).resolve().parent
original_load_split = common.load_split


def load_split(dataset, split, limit=None):
    if dataset == 'olympic':
        data = json.loads((ROOT/'frozen_questions.json').read_text(encoding='utf-8'))
        if split == 'train':
            rows = data['train'] if limit is not None else data['train']+data['anchor_pool']
        elif split == 'test':
            rows = data['test']
        else:
            raise ValueError('Only the frozen Olympic train and test splits are available.')
        return rows if limit is None else rows[:limit]
    return original_load_split(dataset, split, limit)


if __name__ == '__main__':
    script, *arguments = sys.argv[1:]
    if script not in ('prepare_buckets.py', 'build_anchors.py', 'train.py', 'evaluate.py'):
        raise ValueError('Unexpected upstream script.')
    common.load_split = load_split
    sys.argv = [str(ROOT/script), *arguments]
    runpy.run_path(str(ROOT/script), run_name='__main__')
