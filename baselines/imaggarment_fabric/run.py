import argparse
import datetime
import os
import traceback
import torch
import torch.distributed as dist
from .common import ROOT, write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['cache', 'train', 'smoke', 'generate', 'evaluate', 'package'])
    parser.add_argument('--stage', choices=['color', 'gam'])
    parser.add_argument('--short', action='store_true')
    args = parser.parse_args()
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    local = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if world > 1:
        dist.init_process_group('nccl', timeout=datetime.timedelta(minutes=30))
    try:
        if args.command == 'cache':
            from .cache import build
            build(device, rank, world, args.short)
        elif args.command == 'train':
            from .train import train
            assert args.stage
            train(args.stage, device, rank, world, args.short)
        elif args.command in ['generate', 'smoke']:
            from .generate import generate
            generate(device, rank, world, args.command == 'smoke')
        elif args.command == 'evaluate':
            from .evaluate import evaluate
            assert world == 1
            evaluate()
        elif args.command == 'package':
            from .evaluate import package
            assert world == 1
            package()
    except Exception:
        write(ROOT / 'audit' / ('failure_%s_%s_rank%d.json' % (args.command, args.stage, rank)),
              dict(command=vars(args), traceback=traceback.format_exc(), status='FAILED'))
        raise
    finally:
        if world > 1 and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
