"""在冻结 E2 外检查权重完整性、观察梯度，不改写原网络或旧损失。"""
from contextlib import contextmanager

import torch


def bare_network(network):
    return getattr(network, 'module', network)


def load_trainable_strict(network, state):
    expected = {name: p for name, p in network.named_parameters() if p.requires_grad}
    missing = sorted(set(expected) - set(state))
    extra = sorted(set(state) - set(expected))
    wrong = [name for name in set(expected) & set(state)
             if expected[name].shape != state[name].shape]
    if missing or extra or wrong:
        raise ValueError('E2 checkpoint不完整: missing=%s extra=%s shape=%s' % (missing, extra, wrong))
    # 旧checkpoint只存可训练参数；其余冻结权重由原E1/SD初始化加载。
    network.load_trainable(state)
    return {'keys': len(expected), 'parameters': sum(p.numel() for p in expected.values())}


@contextmanager
def observe_prediction(pipe, network):
    """只观察旧调用的返回值；不重新解码、不detach、不消耗RNG。"""
    captured = {}
    original = pipe.vae.decode
    instance_decode = pipe.vae.__dict__.get('decode')
    had_instance_decode = 'decode' in pipe.vae.__dict__

    def decode(*args, **kwargs):
        captured['vae_input'] = args[0]
        output = original(*args, **kwargs)
        captured['x0'] = args[0] * pipe.vae.config.scaling_factor
        captured['decoded'] = output.sample
        return output

    def prediction(_module, args, output):
        captured['noisy'], captured['timesteps'] = args[0], args[1]
        captured['epsilon'] = output
        captured['features'] = list(bare_network(network).adapter.features)

    handle = network.register_forward_hook(prediction)
    pipe.vae.decode = decode
    try:
        yield captured
    finally:
        handle.remove()
        if had_instance_decode:
            pipe.vae.decode = instance_decode
        else:
            del pipe.vae.decode


def e2_objective(raw):
    # 与冻结核心的加法结合顺序一致，避免关闭开关时因重排浮点运算而改变结果。
    return raw['base'] + (.1 * raw['body'] + .03816244801608925 * raw['rest'])


def compute_losses(pipe, network, batch, rng, style, core, fft=None, fft_weight=0., audit_mode=False):
    with observe_prediction(pipe, network) as captured:
        raw, token = core.losses(pipe, network, batch, rng, style, audit_mode)
    old = e2_objective(raw)
    if fft is None or fft_weight == 0:
        return old, raw, token, captured, {}
    keep = not (.2 <= token['drop'] < .3)
    extra, stats = fft(captured['decoded'], batch['texture_image'], batch['garment_mask'],
                       captured['timesteps'], batch['dataset_index'], reference_keep=keep)
    components = stats.pop('_component_tensors', None)
    extra_raw = dict(auxiliary_total=extra, **components) if components is not None else dict(fft=extra)
    return old + fft_weight * extra, dict(raw, **extra_raw), token, captured, stats


def cpu_snapshot(raw, token, captured, total, rng):
    return dict(token=token, total=total.detach().float().cpu(),
                raw={k: v.detach().float().cpu() for k, v in raw.items()},
                tensors={k: captured[k].detach().float().cpu()
                         for k in ['epsilon', 'x0', 'decoded', 'noisy', 'timesteps']},
                features=[x.detach().float().cpu() for x in captured['features']],
                rng=rng.get_state().cpu())


def compare_snapshots(first, second, atol=1e-6, rtol=1e-5):
    assert first['token'] == second['token'], '旧噪声/t/dropout/数据索引改变'
    assert torch.equal(first['rng'], second['rng']), '旧训练RNG改变'
    assert len(first['features']) == len(second['features']), '尺度数量改变'
    errors = {}
    pairs = [('total', first['total'], second['total'])]
    for key in first['raw']:
        pairs.append(('loss/' + key, first['raw'][key], second['raw'][key]))
    for key in first['tensors']:
        pairs.append((key, first['tensors'][key], second['tensors'][key]))
    for index, (a, b) in enumerate(zip(first['features'], second['features'])):
        pairs.append(('feature/%d' % index, a, b))
    for name, a, b in pairs:
        torch.testing.assert_close(a, b, atol=atol, rtol=rtol)
        errors[name] = float((a - b).abs().max())
    return errors
