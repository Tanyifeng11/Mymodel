"""双rank CPU/Gloo探针：一个rank的FFT无有效参考，另一个正常反传。"""
import argparse
import json
import os
from pathlib import Path

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from models.e2_local_fft import LocalFFTLoss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(42)
    torch.distributed.init_process_group('gloo')
    rank = torch.distributed.get_rank()
    assert torch.distributed.get_world_size() == 2
    model = DistributedDataParallel(nn.Conv2d(3, 3, 3, padding=1))
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    coordinate = torch.arange(96).float()
    x = (.7 * torch.sin(coordinate * (2 * torch.pi / 8)))[None, None, None].expand(1, 3, 96, 96).clone()
    reference = x.transpose(-1, -2).contiguous() if rank else torch.zeros_like(x)
    mask = torch.zeros(1, 1, 96, 96); mask[:, :, 6:-6, 6:-6] = 1
    prediction = model(x)
    objective = LocalFFTLoss(seed=110642 + rank)
    extra, stats = objective(prediction, reference, mask, torch.tensor([100]), torch.tensor([rank]))
    total = prediction.square().mean() + .1 * extra
    total.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    vector = torch.cat([p.detach().flatten() for p in model.parameters()])
    gathered = [torch.empty_like(vector) for _ in range(2)]
    torch.distributed.all_gather(gathered, vector)
    assert torch.equal(gathered[0], gathered[1])
    record = dict(rank=rank, fft=float(extra.detach()), valid_n=stats['valid_n'], total=float(total.detach()))
    records = [None, None]
    torch.distributed.all_gather_object(records, record)
    if rank == 0:
        assert records[0]['valid_n'] == 0 and records[1]['valid_n'] == 1
        assert records[0]['fft'] == 0 and records[1]['fft'] > 0
        path = Path(args.out); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(passed=True, backend='gloo', world=2, ranks=records,
            parameters_identical=True), indent=2) + '\n', encoding='utf-8')
        print('DDP_PARTIAL_VALID_PASS', records, flush=True)
    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
