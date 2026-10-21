"""E2A-SUP-COMP-01 独立根；旧 B/C 与原冻结源码只读复用。"""
import os
from pathlib import Path
from tools.e2a_attr_common import read, write, freeze, sha, csv_write, commit, PROJECT, SOURCE

OLD = PROJECT / 'output_eval/e2a_attr_01_20261019'
ROOT = PROJECT / 'output_eval/e2a_sup_comp_01_20261021'
PLAN_SHA = '2b35f00feef70abfcdd3166b3110801aad3e065c612c8a3eeba667d57205b0a6'
ARMS = ['T3', 'T1', 'T2']
EXPECTED = dict(E1='e51d71dbf17c424fb639293ab99cfa83b9658451e127e6a61e8263e15bd5e39a',
    B='1b398bad19d24cbf3bcb78dde0050f0adc61733ed6bda387513f2acc5ef9bb39',
    C='517be044acd7ca149c2b09c20d7b8971680e24516ad4f357927c6bb5d72ed7b6')
CONFIG = dict(protocol='E2A-SUP-COMP-01', plan_sha256=PLAN_SHA, seed=42, updates=28365,
    world_size=2, effective_batch=8, accumulation=4, pilot_updates=800,
    calibration_effective_batches=64, calibration_samples=512,
    eta=249, dino_model='DINO ViT-S/16', dino_key_block_zero_based=5,
    dino_size=224, dino_roi='sketch body+boundary bbox; square white pad; bilinear 224',
    canny=dict(low=100, high=200, aperture=3, L2gradient=False, target='GT uint8 gray'),
    auxiliary_rgb='T1/T2 clamp((decoded+1)/2,0,1); historical T3 helpers unchanged',
    hyperparameter_selection='TRAIN-CAL only; P2 never used to choose coefficients',
    P2a=32, P2b=96, LOCK=128, LOCK_opened=False,
    bootstrap_samples=10000, statistics_seed=221042,
    primary_family='C minus each T1/T2/T3 IoU; 3 two-sided tests; Holm',
    reference_swap_n=32, references_per_swap_identity=2,
    additional_seeds_authorized=False, automatic_factorial_authorized=False)


def folder(arm, phase='formal'):
    assert arm in ARMS and phase in ['calibrate','pilot','formal']
    return ROOT / 'training' / phase / arm


def model_path(arm):
    if arm == 'E1': return PROJECT/'output/phase1_e1_grouped_e5/checkpoint-final/joint_model.pt'
    if arm in ['B','C']:
        return OLD/'training/seed42'/('F000' if arm=='B' else 'F111')/'checkpoint-final/joint_model.pt'
    return folder(arm)/'checkpoint-final/joint_model.pt'


def status(**changes):
    path=ROOT/'status.json'
    value=read(path) if path.exists() else dict(preparation='NOT_RUN', complete=False,
        arms={a:'NOT_RUN' for a in ARMS}, engineering_failure=None, LOCK='NOT_OPENED')
    value.update(changes); write(path,value); return value
