"""Rebuild embedded fixed questions and tokenizer QA; CPU only, no weight download."""
import os
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
from collections import Counter
from transformers import AutoTokenizer
from colab_pilot import build_data, validate_data
from io_utils import ROOT, read, write, digest
from model import prompt_ids


def main():
    c = read(ROOT/'config_colab.json')
    data = build_data(c)
    validate_data(data, c)
    tok = AutoTokenizer.from_pretrained(c['base_model'], revision=c['base_revision'],
                                      cache_dir=str(ROOT/'hf_cache'/'hub'), local_files_only=True)
    counts = {}
    for name, rows in data.items():
        if name == 'provenance': continue
        maximum = max(len(prompt_ids(tok, q['question'])) for q in rows)
        if maximum > c['max_prompt_tokens']:
            raise ValueError(f'{name}: prompt exceeds cap; no row removed.')
        counts[name] = {'count': len(rows), 'max_prompt_tokens': maximum,
                        'sources': dict(Counter(q['source'] for q in rows))}
    settings = {k: c[k] for k in ('split_seed', 'train_count', 'validation_count', 'anchor_pool',
        'test_count', 'numina_revision', 'omni_revision', 'numina_sources')}
    write(ROOT/'colab_splits.json', {'settings': settings, 'splits': data})
    write(ROOT/'COLAB_DATA_CHECK.json', {'mode': 'CPU-only; no model inference or training',
          'counts': counts, 'splits_sha256': digest(data)})
    print(counts, flush=True)


if __name__ == '__main__': main()
