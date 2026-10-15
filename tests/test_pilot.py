"""CPU checks only; no dataset downloads, checkpoint downloads or CUDA training."""
import ast
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]/'src'
sys.path.insert(0, str(ROOT))
import core
import data
import engine
import io_utils
import model
import report
import run

torch.set_num_threads(1)


def question(i, domain="math"):
    return {"id": f"{domain}-{i}", "question": f"{domain} question {i}", "answer": "4", "topic": "algebra"}


def records(questions, n, seed=0, bad=False):
    return [{"id": q["id"], "sample_index": j, "tokens": [2, 3] + ([4] if j % 2 else []),
             "text": "\\boxed{4}", "correct": (not bad and j % 2 == 0), "capped": False, "no_box": False}
            for q in questions for j in range(n)]


def config():
    return io_utils.read(ROOT / "config.json")


def test_core_hash():
    provenance = io_utils.read(ROOT / "SOURCE.json")
    assert io_utils.file_hash(ROOT / "core.py") == provenance["core_sha256"]


def test_upstream_symbols():
    provenance = io_utils.read(ROOT / "SOURCE.json")
    upstream = ROOT.parents[1] / "DATN-FPTU-rl-nafc" / "code" / "qwen-nafc"
    if not upstream.is_dir():
        pytest.skip("Optional upstream checkout is not available; bundled core hash is tested separately.")
    bundled = ast.parse((ROOT / "core.py").read_text(encoding="utf-8"))
    for name, info in provenance["files"].items():
        path = upstream / name
        assert io_utils.file_hash(path) == info["sha256"]
        original = ast.parse(path.read_text(encoding="utf-8"))
        def named(tree, symbol):
            return next(x for x in tree.body if getattr(x, "name", None) == symbol or
                        isinstance(x, ast.Assign) and any(isinstance(t, ast.Name) and t.id == symbol for t in x.targets))
        for symbol in info["symbols"]:
            assert ast.dump(named(original, symbol)) == ast.dump(named(bundled, symbol))


def test_pi_and_nafc_formula_and_replay():
    pi = core.LagrangianController("laconic", budgets=[100], max_len=1536, eta=.05, kp=4, two_sided=True)
    pi.ceiling = lambda k: 3
    pi.dual_update([110, 110], [0, 0])
    assert pi.lambdas[0] == pytest.approx(.405)
    anchors = [dict(question(i), base_first_correct=True) for i in range(4)]
    bundle = {"anchors": anchors, "repair_idx": [0, 1], "held_idx": [2, 3]}
    ctrl = core.FlipController("nafc_replay", eps=.1, eta=10, repair_idx=[0, 1], held_idx=[2, 3])
    ctrl.update([True, True, False, False], [True]*4)
    assert ctrl.lam == 0 and ctrl.selected == [0, 1]
    ctrl.update([True, False, True, False], [True]*4)
    assert ctrl.lam == 1 and ctrl.selected == [0]
    items = [question(i, "gsm8k") for i in range(8)]
    replaced, ptr, used = engine.replay(items, bundle, ctrl, config(), 0)
    assert used == [0] and ptr == 1 and replaced[0]["id"] == anchors[0]["id"]
    assert all(x["id"] not in {anchors[2]["id"], anchors[3]["id"]} for x in replaced)
    assert core.robust_flags([[1, 0, 0, 0], [1, 1, 0, 0]]) == [True, False]


def test_advantage_population_std_and_pi_sign():
    values = core.decoupled_advantages(["a"]*4, [1, 1, 0, 0], [-1, -1, 1, 1], [.5]*4)
    assert values == [1.5, 1.5, -1.5, -1.5]
    assert core.decoupled_advantages(["a"]*2, [1, 1], [0, 0], [3, 3]) == [0, 0]


@pytest.mark.parametrize("key,value", [("warmup_steps", 0), ("learning_rate", -1), ("train_count", 1001),
                                      ("anchor_count", 127), ("eval_draws", 1), ("noise_margin", -1)])
def test_invalid_config(key, value):
    c = config(); c[key] = value
    with pytest.raises(ValueError):
        io_utils.validate_config(c)


def test_schedule_and_gold():
    q = [question(i) for i in range(8)]
    batches = data.schedule(q, 8, 2, 4)
    assert batches == data.schedule(q, 8, 2, 4)
    assert {x["id"] for b in batches[:4] for x in b} == {x["id"] for x in q}
    assert all(len({x["id"] for x in b}) == 2 for b in batches)
    assert core.gold_answer("gsm8k", "steps #### 1,234") == "1234"
    assert core.last_boxed("\\boxed{2} then \\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"


class TinyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(9, 4)
        self.head = torch.nn.Linear(4, 9)
    def forward(self, input_ids, **kwargs):
        return SimpleNamespace(logits=self.head(self.embedding(input_ids)))
    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), Path(path) / "tiny.pt")


def test_token_logps_values_and_gradients():
    torch.manual_seed(0)
    a = TinyLM(); b = copy.deepcopy(a)
    actual = model.token_logps(a, [1, 2], [3, 4, 5])
    logits = b(torch.tensor([[1, 2, 3, 4, 5]])).logits[0, 1:4]
    expected = -F.cross_entropy(logits, torch.tensor([3, 4, 5]), reduction="none")
    torch.testing.assert_close(actual, expected)
    actual.sum().backward(); expected.sum().backward()
    for p, q in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(p.grad, q.grad)


def test_clipped_loss_clips_positive_and_negative():
    current = torch.tensor([1.0, -1.0], requires_grad=True)
    old = torch.zeros(2)
    assert model.clipped_loss(current, old, 1, 2).item() == pytest.approx(-(1.2 + torch.exp(torch.tensor(-1.)).item()) / 2)
    assert model.clipped_loss(current, old, -1, 2).item() == pytest.approx((torch.exp(torch.tensor(1.)).item() + .8) / 2)


def test_evaluation_resume_and_identity(tmp_path):
    q = [question(i) for i in range(35)]
    calls = []
    def sampler(items, n, seed):
        calls.append(seed)
        return records(items, n)
    path = tmp_path / "eval.json"
    saved = model.evaluate(sampler, q, 4, 100, path)
    assert len(saved) == 140 and len(calls) == 3
    assert calls == [100, 100 + 16*4*1009, 100 + 32*4*1009]
    model.evaluate(sampler, q, 4, 100, path)
    assert len(calls) == 3
    with pytest.raises(ValueError, match="identity"):
        model.evaluate(sampler, q, 4, 101, path)
    damaged = io_utils.read(path); damaged["records"][0]["sample_index"] = 1
    io_utils.write(path, damaged)
    with pytest.raises(ValueError, match="reordered"):
        model.evaluate(sampler, q, 4, 100, path)


class FakeTokenizer:
    def apply_chat_template(self, *args, **kwargs):
        return [1, 2]
    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "tiny_tokenizer.json").write_text("{}")


class FakeSampler:
    bad_anchors = False
    fail_step = None
    def __init__(self, m, tok, c):
        pass
    def __call__(self, questions, n, seed):
        if self.fail_step is not None and seed == 700_000_000 + self.fail_step*1_000_000 + 2:
            raise RuntimeError("Simulated interruption")
        auditing = 500_000_000 <= seed < 600_000_000
        return records(questions, n, seed, bad=self.bad_anchors and auditing)


def toy_run(c, data_rows, bundle, arm):
    def loader(*args, **kwargs):
        torch.manual_seed(123)
        return TinyLM(), FakeTokenizer()
    engine.train_arm(c, data_rows, bundle, arm, loader=loader, sampler_factory=FakeSampler)


def toy_inputs():
    c = config(); c.update(seed=2, steps=28, prompts_per_step=2, group_size=2, max_new_tokens=8,
                          warmup_steps=2, audit_every=5, validation_steps=[10, 20], precision="bf16")
    rows = {"task_train": [question(i, "gsm8k") for i in range(6)]}
    for domain in ("numina", "omni"):
        rows[f"{domain}_validation"] = [question(i+20, domain) for i in range(2)]
        rows[f"{domain}_test"] = [question(i+30, domain) for i in range(2)]
    rows["provenance"] = {"task_domain": "numina"}
    anchors = [dict(question(i+100), base_first_correct=True) for i in range(4)]
    bundle = {"anchors": anchors, "repair_idx": [0, 1], "held_idx": [2, 3], "eps": .02, "budget": 2}
    return c, rows, bundle


def test_inactive_nafc_matches_control_and_active_resume(tmp_path, monkeypatch):
    c, data_rows, bundle = toy_inputs()
    root = tmp_path / "full"; root.mkdir()
    monkeypatch.setattr(engine, "OUT", root)
    FakeSampler.bad_anchors = False; FakeSampler.fail_step = None
    toy_run(c, data_rows, bundle, "grpo_pi")
    toy_run(c, data_rows, bundle, "grpo_pi_nafc")
    a = torch.load(root / "grpo_pi" / "resume.pt", weights_only=True)
    b = torch.load(root / "grpo_pi_nafc" / "resume.pt", weights_only=True)
    for key in a["parameters"]:
        torch.testing.assert_close(a["parameters"][key], b["parameters"][key], rtol=0, atol=0)
    assert b["state"]["nafc"]["lam"] == 0
    active = tmp_path / "active"; active.mkdir(); monkeypatch.setattr(engine, "OUT", active)
    FakeSampler.bad_anchors = True
    toy_run(c, data_rows, bundle, "grpo_pi_nafc")
    complete = torch.load(active / "grpo_pi_nafc" / "resume.pt", weights_only=True)
    assert sum(m["anchor_prompts"] for m in complete["state"]["metrics"]) > 0
    resumed = tmp_path / "resume"; resumed.mkdir(); monkeypatch.setattr(engine, "OUT", resumed)
    FakeSampler.fail_step = 26
    with pytest.raises(RuntimeError, match="interruption"):
        toy_run(c, data_rows, bundle, "grpo_pi_nafc")
    FakeSampler.fail_step = None
    toy_run(c, data_rows, bundle, "grpo_pi_nafc")
    saved = torch.load(resumed / "grpo_pi_nafc" / "resume.pt", weights_only=True)
    assert saved["state"]["pi"] == complete["state"]["pi"]
    assert saved["state"]["nafc"] == complete["state"]["nafc"]
    assert saved["state"]["replay_ptr"] == complete["state"]["replay_ptr"]
    assert saved["state"]["last_audit_step"] == complete["state"]["last_audit_step"]
    for key in saved["parameters"]:
        torch.testing.assert_close(saved["parameters"][key], complete["parameters"][key], rtol=0, atol=0)


def test_tiny_phi_lora_forward_sampling_and_rng():
    from transformers import Phi3Config, Phi3ForCausalLM
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(12)
    m = get_peft_model(Phi3ForCausalLM(Phi3Config(vocab_size=9, hidden_size=16, intermediate_size=32,
             num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
             max_position_embeddings=32, original_max_position_embeddings=32, eos_token_id=[7, 8], pad_token_id=0)),
             LoraConfig(r=2, lora_alpha=4, lora_dropout=0, target_modules="all-linear", task_type="CAUSAL_LM"))
    m.train()
    class Tokenizer(FakeTokenizer):
        pad_token_id = 0
        eos_token_id = 8
        def decode(self, *args, **kwargs): return "\\boxed{4}"
    c = config(); c.update(generation_batch=2, max_new_tokens=3, max_prompt_tokens=10)
    sampler = model.Sampler(m, Tokenizer(), c, scorer=lambda text, gold: True)
    rng_before = torch.random.get_rng_state().clone()
    sampled = sampler([question(1), question(2)], 2, 9)
    assert torch.equal(rng_before, torch.random.get_rng_state()) and m.training
    assert sampled == sampler([question(1), question(2)], 2, 9)
    assert [(r["id"], r["sample_index"]) for r in sampled] == [(question(i)["id"], j) for i in (1, 2) for j in range(2)]
    assert all(1 <= len(r["tokens"]) <= 3 for r in sampled)
    logps = model.token_logps(m, [1, 2], sampled[0]["tokens"])
    loss = model.clipped_loss(logps, logps.detach(), 1, 3)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters() if p.requires_grad)
    assert all(p.grad is None for p in m.parameters() if not p.requires_grad)


def test_fixed_replay_never_uses_held_anchors():
    c, rows, bundle = toy_inputs()
    ctrl = core.FlipController("monitor", eps=.02, repair_idx=[0, 1], held_idx=[2, 3])
    batch = [question(i, "numina") for i in range(8)]
    new, ptr, used = engine.replay(batch, bundle, ctrl, c, 0, "grpo_pi_fixed_replay")
    assert used == [0, 1] and ptr == 2
    assert len(new) == 8 and len({q['id'] for q in new}) == 8
    assert not {bundle['anchors'][i]['id'] for i in bundle['held_idx']} & {q['id'] for q in new}


def test_olympic_partitions_block_all_benchmark_rows(monkeypatch):
    import datasets
    monkeypatch.setattr(data, "numeric_gold", lambda a: 4 if a == "4" else None)
    def item(i, **kwargs):
        return dict(problem=f"Olympic question {i}", answer="4", source="olympiads_ref",
                    question_type="math-word-problem", problem_is_valid="Yes",
                    solution_is_valid="Yes", problem_type="Algebra", **kwargs)
    source = [item(i) for i in range(25)]
    source += [dict(item(100), source="amc_aime"), dict(item(101), answer="proof"),
               dict(item(102), problem_is_valid="Incomplete")]
    external = [dict(item(0), answer="proof"), dict(item(200), domain=["Algebra"], difficulty=5)]
    monkeypatch.setattr(datasets, "load_dataset", lambda name, **kw: external if name == data.DATASETS['omni'] else source)
    c = config(); c.update(train_count=8, validation_count=4, test_count=4, anchor_pool=4)
    a = data.prepare_splits(c)
    assert a == data.prepare_splits(c)
    data.validate_splits(a, c)
    questions = {q['question'] for name, rows in a.items() if name not in ('provenance', 'omni_test') for q in rows}
    assert not questions & {'Olympic question 0', 'Olympic question 100', 'Olympic question 101', 'Olympic question 102'}
    assert a['provenance']['omni_total'] == 2 and a['provenance']['omni_scored'] == 1
    a['numina_test'][0] = a['task_train'][0]
    with pytest.raises(ValueError, match="leakage"):
        data.validate_splits(a, c)


def test_report_three_arms_caps_long_endpoint_and_aggregate(tmp_path, monkeypatch):
    shared = tmp_path/'outputs_phi_olympic'/'shared'; shared.mkdir(parents=True)
    q = [question(i, 'numina') for i in range(4)]
    o = [question(i, 'omni') for i in range(4)]
    io_utils.write(shared/'splits.json', {'numina_test': q, 'omni_test': o, 'provenance': {'task_domain': 'numina'}})
    io_utils.write(shared/'prepared.json', {'eps': .04, 'budget': 4})
    for prefix in ('', 'long_'):
        for domain, questions in (('numina', q), ('omni', o)):
            for suffix, n in (('test', 8), ('noise', 4)):
                rr = records(questions, n)
                for r in rr: r['correct'] = True
                io_utils.write(shared/f'reference_{prefix}{domain}_{suffix}.json', {'records': rr})
    c = config(); c['seeds'] = [42, 43]
    monkeypatch.setattr(report, 'ROOT', tmp_path)
    for seed in c['seeds']:
        out = shared.parent/f'seed_{seed}'; out.mkdir()
        monkeypatch.setattr(report, 'OUT', out)
        io_utils.write(out/'identity.json', {'settings': {'seed': seed}, 'splits_sha256': 'same', 'reference': 'same', 'code_sha256': 'same'})
        for arm in engine.ARMS:
            folder = out/arm; folder.mkdir()
            io_utils.write(folder/'complete.json', {'step': 2, 'optimizer_updates': 2, 'anchor_prompts': int(arm.endswith('nafc'))})
            for prefix in ('final', 'long'):
                for domain, questions in (('numina', q), ('omni', o)):
                    rr = records(questions, 4)
                    for r in rr: r.update(correct=arm.endswith('nafc'), capped=True)
                    io_utils.write(folder/f'{prefix}_{domain}_test.json', {'records': rr})
            io_utils.write(folder/'train_metrics.json', [{'step': 1, 'pi_lambda_used': .2}])
            io_utils.write(folder/'audit_metrics.json', [{'step': 0, 'F_used': .2, 'lam': .5}])
        result = report.make_report(shared)
        assert result['results']['grpo_pi_nafc']['primary_omni']['capped'] == 1
        assert result['results']['grpo_pi']['long_omni']['retention']['correct_to_low_success'] == 4
        assert result['differences']['primary_numina_vs_grpo_pi']['flip_rate'] == -1
        assert (out/'accuracy.png').exists()
    report.aggregate(c)
    summary = io_utils.read(shared.parent/'summary.json')
    assert summary['differences_nafc_minus_baseline']['primary_numina_vs_grpo_pi']['accuracy']['mean'] == 1
    with pytest.raises(ValueError, match='incomplete'):
        report.aggregate(dict(c, seeds=[42, 43, 44]))


def test_numeric_scoring_scope_and_equivalence():
    import scoring
    assert scoring.numeric_gold('proof') is None
    assert scoring.numeric_gold('x + 2') is None
    assert scoring.numeric_gold('2,3') is None
    assert scoring.numeric_gold('4 or 5') is None
    assert scoring.numeric_gold('4') is not None
    assert scoring.is_correct(r'Answer: \boxed{2/4}', r'\frac{1}{2}')
    assert not scoring.is_correct('4', '4')
    assert not scoring.is_correct(r'\boxed{4}', '5')


def test_output_isolation_and_valid_new_config(tmp_path, monkeypatch):
    io_utils.validate_config(config())
    monkeypatch.setattr(run, 'ROOT', tmp_path)
    monkeypatch.setattr(run, 'SHARED', tmp_path/'outputs_phi_olympic'/'shared')
    monkeypatch.setattr(run, 'OUT', run.OUT)
    for module in (io_utils, model, engine, report):
        monkeypatch.setattr(module, 'OUT', module.OUT)
    run.set_output(42)
    assert run.OUT == engine.OUT == model.OUT == report.OUT
    assert run.OUT != run.set_output(43)
    assert run.set_output() == run.SHARED
    assert not (ROOT/'nafc_continual_pi_kaggle.ipynb').exists()


def test_shared_preparation_and_unchanged_config(tmp_path, monkeypatch):
    monkeypatch.setattr(run, 'ROOT', tmp_path)
    shared = tmp_path/'outputs_phi_olympic'/'shared'
    monkeypatch.setattr(run, 'SHARED', shared)
    monkeypatch.setattr(run, 'OUT', run.OUT)
    for module in (io_utils, model, engine, report):
        monkeypatch.setattr(module, 'OUT', module.OUT)
    monkeypatch.setattr(run, 'check_gpu', lambda c: None)
    monkeypatch.setattr(run, 'base_path', lambda c: str(tmp_path/'reference'))
    monkeypatch.setattr(run, 'checkpoint_identity', lambda path: {'kind': 'merged', 'sha': 'same'})
    monkeypatch.setattr(run, 'identity', lambda c, parent, splits: {'settings': c, 'splits': io_utils.digest(splits)})
    monkeypatch.setattr(run, 'load_model', lambda *args: (TinyLM(), FakeTokenizer()))
    monkeypatch.setattr(run, 'clear_memory', lambda: None)
    c = config(); c.update(train_count=8, validation_count=4, test_count=6, anchor_pool=8, anchor_count=4)
    splits = {'task_train': [dict(question(i, 'numina'), source='olympiads') for i in range(8)],
        'numina_validation': [dict(question(i+20, 'numina'), source='olympiads') for i in range(4)],
        'numina_test': [dict(question(i+30, 'numina'), source='olympiads') for i in range(6)],
        'anchor_candidates': [dict(question(i+100, 'numina'), source='olympiads') for i in range(8)],
        'omni_test': [dict(question(i, 'omni'), source='HMMT') for i in range(6)],
        'provenance': {'task_domain': 'numina'}}
    monkeypatch.setattr(run, 'prepare_splits', lambda cfg: splits)
    class PreparationSampler:
        def __init__(self, m, t, cfg): self.c = cfg
        def __call__(self, questions, n, seed):
            rr = records(questions, n)
            for r in rr: r['correct'] = True
            return rr
    monkeypatch.setattr(run, 'Sampler', PreparationSampler)
    run.prepare(c)
    saved = io_utils.read(shared/'prepared.json')
    assert saved['budget'] == 2 and saved['eps'] == .02
    assert saved['reference_test_summary']['omni']['initially_stable'] == 6
    assert len(saved['repair_idx']) == len(saved['held_idx']) == 2
    assert (shared/'reference_long_omni_noise.json').exists()
    run.prepare(c)
    with pytest.raises(ValueError, match='Different frozen'):
        run.prepare(dict(c, learning_rate=1e-4))
    with pytest.raises(ValueError, match='Configuration changed'):
        run.train(dict(c, learning_rate=1e-4), 42, 'grpo_pi')


def test_scoring_worker_errors_do_not_assign_false_rewards(monkeypatch):
    import io
    import scoring
    scoring._stop_worker()
    if scoring.os.name != 'nt':
        return  # The direct Unix path uses Math-Verify's signal timeout.
    class Worker:
        stdin = io.StringIO()
        stdout = io.StringIO('{"error": "parser failed"}\n')
        killed = False
        def poll(self): return None
        def kill(self): self.killed = True
        def wait(self): return 0
    worker = Worker()
    monkeypatch.setattr(scoring.subprocess, 'Popen', lambda *args, **kwargs: worker)
    with pytest.raises(RuntimeError, match='no false reward'):
        scoring._bounded({'op': 'valid', 'gold': '4'})
    assert worker.killed and scoring._worker is None


def test_evaluation_rejects_changed_sampling_cap(tmp_path):
    class Sampler:
        c = {'max_new_tokens': 4}
        def __call__(self, questions, n, seed): return records(questions, n)
    sampler = Sampler()
    path = tmp_path/'eval.json'
    model.evaluate(sampler, [question(1)], 4, 100, path)
    sampler.c = {'max_new_tokens': 8}
    with pytest.raises(ValueError, match='identity'):
        model.evaluate(sampler, [question(1)], 4, 100, path)
