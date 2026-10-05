"""逐seed重跑原真实/controlled分母，和正式RF2端点成对比较。"""
import numpy as np
import torch
from data.e33rf_real_rotation_dataset import reference_group
from models.e33tm_generation_wrapper import TriModalField
from models.apacc_features import load_dino
from tools.e33rf_common import build, WEIGHTS
from tools.e33r_evaluate import axial, evaluate_records
from tools.e33tm_protocol import *
from tools.e33tm_caption_audit import prepare

@torch.inference_mode()
def run():
    prepare()
    split = read(RF/'split_manifest.json')
    dino, _ = load_dino('cuda', WEIGHTS)
    reports = []
    for seed in SEEDS:
        dest = OUT/'field_precheck'/('seed%d'%seed)
        model, _ = build(seed, checkpoint=RF/('seed%d'%seed)/'RF2/checkpoint_final.pt')
        wrapper = TriModalField(model).eval()
        rows = []
        for row in split['dev']:
            case = reference_group(row)
            ref = torch.cat([case['reference'], torch.zeros_like(case['reference'][:1])]).cuda()
            structure = case['structure'][None].cuda().expand(4,-1,-1,-1)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                p = wrapper(ref, structure, text='fixed caption')
                original = model(ref, structure)
            assert all(torch.equal(p[k], original[k]) for k in p), 'wrapper altered RF2 output'
            ori = p['orientation'].float().cpu().numpy()
            support = case['support'].numpy()
            weight = case['gt'][3].numpy()[support]
            avg = lambda x: float(np.average(x[support], weights=weight)) if support.any() else None
            r90, r180 = avg(axial(ori[1], -ori[0])), avg(axial(ori[2], ori[0]))
            rows.append(dict(id=row['id'], rot90_response_error=r90, r180_response_error=r180,
                             rot90_response_success=None if r90 is None else r90<=15,
                             r180_identity_success=None if r180 is None else r180<=15))
            path = dest/'real/fields'/(row['id']+'.npz')
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, orientation=ori, q=p['q'].float().cpu().numpy(),
                                support=support, gt=case['gt'].numpy())
        write(dest/'real/rows.json', rows)
        rate = lambda key: float(np.mean([r[key] for r in rows if r[key] is not None]))
        real90, real180 = rate('rot90_response_success'), rate('r180_identity_success')
        _, cf = evaluate_records(wrapper, read(RF/'controlled_manifest.json')['dev'], dino,
                                 dest/'controlled', save_fields=False)
        old_real = read(RF/('seed%d'%seed)/'RF2/real/summary.json')
        old_cf = read(RF/('seed%d'%seed)/'RF2/controlled/summary.json')
        diffs = dict(real_r90=abs(real90-old_real['rot90_success']['mean']),
                     controlled_r90=abs(cf['clean_r90_success']['mean']-old_cf['clean_r90_success']['mean']),
                     real_r180=abs(real180-old_real['r180_identity']['mean']),
                     controlled_r180=abs(cf['r180_identity_success']['mean']-old_cf['r180_identity_success']['mean']))
        record = dict(seed=seed, differences=diffs, real_r90=real90, real_r180=real180,
                      real_denominator=256, controlled_denominator=128, direct_outputs_equal=True,
                      **{'pass':all(v<=.02+1e-12 for v in diffs.values())})
        write(dest/'summary.json', record)
        reports.append(record)
        print('[E33TM precheck]', record, flush=True)
        del wrapper, model
        torch.cuda.empty_cache()
    passed = all(r['pass'] for r in reports)
    write(OUT/'field_precheck/summary.json', dict(seeds=reports, **{'pass':passed}))
    decision(field_precheck_pass=passed, next_route='smoke_generation' if passed else 'trimodal_wrapper_integration_bug')
    freeze_check()
    return passed

if __name__ == '__main__':
    run()
