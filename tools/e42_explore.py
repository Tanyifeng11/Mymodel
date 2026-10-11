"""用户明确跳过G0后执行的探索性后续；不得把旧calibration改为通过。"""
import os
import shutil
import subprocess
from pathlib import Path
import cv2
from tools import e42_pipeline as p
from tools.e42_repeat import OUT as BASE,read,write,sha

OUT=BASE/'exploratory_override_20261011'


def run():
    cv2.setNumThreads(1)
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'override_protocol.json').exists(),'探索性批次已有记录，禁止覆盖或重复提交'
    names=['manifest.csv','manifest_records.json','geometry.json','input_hashes.json','calibration.json','detector_lock.json','audit.json']
    base_hashes={str(BASE/n):sha(BASE/n) for n in names}
    for n in names:shutil.copy2(BASE/n,OUT/n)
    cal=read(OUT/'calibration.json');rows=read(OUT/'manifest_records.json');geometry=read(OUT/'geometry.json')
    assert not cal['pass_calibration'],'本批次专用于已失败G0的用户覆盖'
    assert read(OUT/'audit.json')['protocol_verified']
    protocol=dict(authorized_by='用户：跳过G0测量，继续进行后续实验',created=p.now(),
        original_status='INCONCLUSIVE_MEASUREMENT / measurement_invalid',calibration_repeated=False,
        calibration_sha256=sha(BASE/'calibration.json'),detector_lock_sha256=sha(BASE/'detector_lock.json'),
        original_file_hashes=base_hashes,thresholds=cal['threshold'],compatibility_threshold=cal['compatibility_threshold'],
        GT_E5_geometry='exact frozen geometry.json; no mask or ROI reselection',
        reference_geometry='original patch minus5% each border; unchanged greedy NMS unless fewer than2; then largest fixed single scale fitting2 disjoint windows, content independent',
        support_rule='Report A and A+ even below30 groups; numerical thresholds are descriptive only; no preregistered GO allowed',
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        training_steps=0,new_generations=0,gpu_count=0)
    write(OUT/'override_protocol.json',protocol)
    p.OUT=OUT
    p.event('user_override_locked',protocol_sha256=sha(OUT/'override_protocol.json'),calibration_pass=False)
    assert all(sha(path)==digest for path,digest in read(OUT/'input_hashes.json').items())
    eligibility,details=p.observability(rows,geometry,cal,allow_uncalibrated=True)
    results=p.evaluate_e5(rows,geometry,cal,eligibility,details)
    stats=p.gate(rows,cal,eligibility,results,exploratory=True)
    p.contact_sheets(rows,eligibility,geometry)
    stats['override_protocol_sha256']=sha(OUT/'override_protocol.json')
    write(OUT/'paired_statistics.json',stats)
    audit=read(OUT/'audit.json');audit.update(original_job=audit['slurm'].get('SLURM_JOB_ID'),
        slurm={k:v for k,v in os.environ.items() if k.startswith('SLURM_')},started=protocol['created'],finished=p.now(),
        code_commit=protocol['code_commit'],status=stats['status'],stage_status=stats['stages'],
        G0_bypassed_by_user=True,calibration_pass=False,base_file_hashes_unchanged=all(sha(path)==h for path,h in base_hashes.items()),
        input_hashes_unchanged=all(sha(path)==h for path,h in read(OUT/'input_hashes.json').items()))
    assert audit['base_file_hashes_unchanged'] and audit['input_hashes_unchanged']
    write(OUT/'audit.json',audit)
    p.event('exploratory_complete',status=stats['status'],funnel=stats['funnel'])


if __name__=='__main__':run()
