"""诊断哈希必须支持真实训练审计的 BF16 输出，且不修改输出张量。"""
import ast
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


def historical_digest(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize('tuple_output', [False, True])
def test_prediction_hash_preserves_output(dtype, tuple_output):
    source = Path(__file__).resolve().parents[1]/'tools/e1_cfg_validate.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'BranchAudit')
    namespace = dict(np=np, digest=historical_digest)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
    network = SimpleNamespace(adapter=SimpleNamespace(injection_counts=[3, 2, 2, 3]))
    audit = namespace['BranchAudit'](network)
    audit.current = 'positive'
    prediction = torch.linspace(-1, 1, 16).reshape(1, 4, 2, 2).to(dtype)
    original = prediction.clone()
    output = (prediction,) if tuple_output else SimpleNamespace(sample=prediction)
    audit.after(None, (), {}, output)
    event = audit.events[0]
    assert prediction.dtype == dtype and torch.equal(prediction, original)
    assert event['first_prediction_sha256'] == historical_digest(original.float())
    assert event['prediction_source_dtype'] == str(dtype)
