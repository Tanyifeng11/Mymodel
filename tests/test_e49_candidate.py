"""新96只能使用两个开发集合都通过门槛的冻结 step800 候选。"""
import tempfile
from pathlib import Path
from tools import e49_confirm as module
from tools import e49_protocol as protocol


def test_candidate_requires_both_sets_and_freezes_selection():
    previous,previous_protocol=module.OUT,protocol.OUT
    with tempfile.TemporaryDirectory(prefix='e49_candidate_') as directory:
        module.OUT=Path(directory)
        protocol.OUT=module.OUT
        try:
            module.write(module.OUT/'audit/complete.json',dict(passed=True))
            for stage in ['probe','dev32']:
                module.write(module.OUT/stage/'evaluation_complete.json',dict(complete=True))
            module.write(module.OUT/'probe/gates.json',dict(CLIP=dict(passed=True),SSVM=dict(passed=False)))
            module.write(module.OUT/'dev32/gates.json',dict(CLIP=dict(passed=False),SSVM=dict(passed=True)))
            assert module.candidate() is None
            assert module.read(module.OUT/'decision.json')['new96']=='not_run'
            for stage in ['probe','dev32']:
                module.write(module.OUT/stage/'metrics.json',[])
                module.write(module.OUT/stage/'gates.json',dict(CLIP=dict(passed=True),SSVM=dict(passed=True)))
            for arm in module.TRAIN_ARMS:
                folder=module.OUT/'training'/arm
                module.write(folder/'complete.json',dict(updates=800))
                (folder/'checkpoints').mkdir(parents=True)
                (folder/'checkpoints/step0800.pt').write_bytes(arm.encode())
            module.write(module.OUT/'dev32/means.json',dict(CLIP=dict(lpips_gt=.5),SSVM=dict(lpips_gt=.4)))
            assert module.candidate()['candidate']=='SSVM'
            module.write(module.OUT/'dev32/means.json',dict(CLIP=dict(lpips_gt=.3),SSVM=dict(lpips_gt=.4)))
            try: module.candidate()
            except AssertionError: pass
            else: raise AssertionError('冻结后不能改换候选')
        finally: module.OUT,protocol.OUT=previous,previous_protocol
