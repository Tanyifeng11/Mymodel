"""冻结B400与Lab约束，验证独立新身份和原始真实纹理。"""
import argparse
from pathlib import Path

import cv2
import torch

from tools import e2_state_b_generalization_data as inputs


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare',))
    parser.add_argument('--project',type=Path,default=Path('/share/home/u2515283058/Mymodel'))
    parser.add_argument('--data',type=Path,default=Path('/share/home/u2515283058/datasets/BF'))
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--review',type=Path)
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);cv2.setNumThreads(2)
    inputs.prepare(args)


if __name__=='__main__':main()
