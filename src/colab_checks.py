"""Offline CPU checks shipped inside the notebook; no model/dataset downloads."""
import copy
import torch
from model import token_logps
from transformers import Phi3Config, Phi3ForCausalLM, Gemma3TextConfig, Gemma3ForCausalLM
from peft import LoraConfig, get_peft_model
from core import LagrangianController, FlipController, robust_flags
from io_utils import ROOT, read, file_hash


def check():
    torch.set_num_threads(1)
    assert file_hash(ROOT/'core.py') == read(ROOT/'SOURCE.json')['core_sha256']
    pi = LagrangianController('laconic', budgets=[100], max_len=4096, eta=.05, kp=4, two_sided=True)
    pi.ceiling = lambda k: 3
    pi.dual_update([110, 110], [0, 0])
    assert abs(pi.lambdas[0]-.405) < 1e-8
    ctrl = FlipController('nafc_replay', eps=.02, eta=10, repair_idx=[0, 1], held_idx=[2, 3])
    ctrl.update([True, False, False, False], [True]*4)
    assert ctrl.lam == 0  # Repair flags alone cannot activate the held controller.
    ctrl.update([True, False, True, False], [True]*4)
    assert ctrl.lam > 0
    assert robust_flags([[1, 0, 0, 0], [1, 1, 0, 0]]) == [True, False]
    configs = [
        (Phi3ForCausalLM, Phi3Config(vocab_size=13, hidden_size=16, intermediate_size=32,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
            max_position_embeddings=128, original_max_position_embeddings=128,
            pad_token_id=0, bos_token_id=1, eos_token_id=12)),
        (Gemma3ForCausalLM, Gemma3TextConfig(vocab_size=13, hidden_size=16, intermediate_size=32,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
            max_position_embeddings=128, sliding_window=16, sliding_window_pattern=2,
            final_logit_softcapping=10, pad_token_id=0, bos_token_id=1, eos_token_id=12))]
    for model_class, config in configs:
        torch.manual_seed(1)
        a = get_peft_model(model_class(config), LoraConfig(r=2, lora_alpha=4, lora_dropout=0,
            target_modules='all-linear', task_type='CAUSAL_LM'))
        b = copy.deepcopy(a); b._pilot_chunked_head = True
        response = [i % 13 for i in range(65)]
        original, chunked = token_logps(a, [1, 2], response), token_logps(b, [1, 2], response)
        torch.testing.assert_close(original, chunked, rtol=1e-5, atol=1e-6)
        original.mean().backward(); chunked.mean().backward()
        for p, q in zip(a.parameters(), b.parameters()):
            if p.requires_grad:
                torch.testing.assert_close(p.grad, q.grad, rtol=1e-4, atol=1e-6)
    print('[CPU checks] unchanged PI/NAFC formulas and Phi/Gemma chunked-head values/gradients passed.', flush=True)


if __name__ == '__main__': check()
