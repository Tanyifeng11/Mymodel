import argparse
import datetime
import os
import traceback
import subprocess
import torch
import torch.distributed as dist
from .common import ROOT, HERE, CONFIG, read, write, sha


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
        frozen = read(ROOT / 'protocol.json')
        assert all(frozen[key] == value for key, value in CONFIG.items()), '已锁定实验配置发生变化'
        if rank == 0:
            commit = subprocess.check_output(['git', '-C', str(HERE), 'rev-parse', 'HEAD'], text=True).strip()
            write(ROOT / 'audit' / ('source_%s_%s_%s_%s.json' % (args.command, args.stage, args.short, commit[:8])),
                  dict(commit=commit, command=vars(args),
                       files={str(p.relative_to(HERE)): sha(p) for p in HERE.rglob('*.py') if '__pycache__' not in p.parts}))
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
