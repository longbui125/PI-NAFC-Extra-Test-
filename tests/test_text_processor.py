"""Replay the trainer-construction call without allocating model/GPU memory."""
import ast
from types import SimpleNamespace

from test_gemma_ngoc import SOURCE


def test_text_tokenizer_is_passed_explicitly_before_trainer_auto_processor_fallback():
    tree = ast.parse((SOURCE/'train.py').read_text(encoding='utf-8'))
    call = next(node for node in ast.walk(tree) if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name) and node.func.id == 'trainer_cls')
    tokenizer = object(); loads = []
    def load(model, **kwargs):
        loads.append((model, kwargs)); return tokenizer
    def trainer(**kwargs):
        if kwargs.get('processing_class') is None:
            raise OSError("Can't load image processor")
        assert kwargs['processing_class'] is tokenizer
        assert kwargs['peft_config']['r'] == 32
        return kwargs
    args = SimpleNamespace(model='/content/gemma_ngoc_model', method='laconic', lora_r=32,
                           adv_norm='decoupled', topq=0)
    scope = dict(args=args, trainer_cls=trainer, AutoTokenizer=SimpleNamespace(from_pretrained=load),
                 make_reward_fn=lambda *a: 'unchanged reward', ctrl=object(), num_buckets=3,
                 config=object(), ds=object(), LoraConfig=lambda **kw: kw, SaveLambda=lambda: object(),
                 extra={'flip_ctrl': 'unchanged controller'})
    result = eval(compile(ast.Expression(call), 'trainer_constructor', 'eval'), scope)
    assert loads == [('/content/gemma_ngoc_model', {'padding_side': 'left'})]
    assert result['flip_ctrl'] == 'unchanged controller'
