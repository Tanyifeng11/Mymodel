"""把已校验的LOTS正式结果并入本地本轮归档；不改变原服务器包。"""
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps
from tools.t3_p0p1 import read, write, sha, csv_write


def rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def closeout(project, lots):
    root = project/'output_eval/t3_p0_p1_20261028'
    output = lots/'output/t3_p0_p1_20261028'
    server = output/'formal_server_results'
    integrity = read(output/'formal_download_integrity.json')
    assert integrity['all_member_sha256_and_sizes_match']
    assert sha(output/'formal_review.tar.gz') == integrity['archive_sha256']
    for item in read(server/'bundle_manifest.json'):
        path = server/item['path']
        assert sha(path) == item['sha256'] and path.stat().st_size == item['bytes']
    inputs = {r['id']: r for r in read(root/'protocol/input_manifest.json') if r['condition'] == 'original'}
    generated = {r['id']: r for r in read(server/'formal/manifest.json')}
    metrics = {r['id']: r for r in rows(server/'metrics/per_image.csv')}
    assert len(inputs) == len(generated) == len(metrics) == 96
    assert set(inputs) == set(generated) == set(metrics)
    assert read(server/'metrics/complete.json')['missing_generation'] == 0
    for identity, record in generated.items():
        native = server/'formal/native'/(identity+'.png')
        evaluation = server/'formal/eval384x512'/(identity+'.png')
        meta = read(evaluation.with_suffix('.json'))
        assert sha(native) == record['output_sha256']
        assert sha(evaluation) == meta['output_sha256'] == metrics[identity]['output_sha256']
        assert record['sketch_sha256'] == inputs[identity]['sketch_sha256']
        assert record['seed'] == inputs[identity]['seed'] and record['caption'] == inputs[identity]['caption']
        assert record['GT_used'] is False and record['reference_texture_used'] is False
        assert record['official_pipeline_sha256'] == 'a2df89021b94751b26c5897205247a3d23e9f8685a61bd1e326fd1a8d5edcfca'
        assert Image.open(native).size == (512, 512) and Image.open(evaluation).size == (384, 512)
        assert all(metrics[identity][key] == '' for key in ['tcf_lab', 'dists_margin', 'color_margin'])
    target = root/'lots'
    shutil.copytree(server, target, dirs_exist_ok=True)
    for name in ['formal_download_integrity.json', 'server_formal_jobs_and_bundle.png']:
        shutil.copy2(output/name, target/name)
    shutil.copytree(server/'metrics', root/'metrics/LOTS', dirs_exist_ok=True)
    shutil.copytree(server/'formal/eval384x512', root/'outputs/LOTS/original', dirs_exist_ok=True)
    fields = ['edge_f1', 'leak', 'iou', 'edge_l1', 'lpips']
    summary_rows = []
    means = {}
    for metric in fields:
        values = [float(r[metric]) for r in metrics.values() if r[metric] != '']
        assert len(values) == 96 and all(math.isfinite(v) for v in values)
        means[metric] = sum(values)/len(values)
        summary_rows.append(dict(model='LOTS', input_modalities='text+sketch; no reference texture',
            condition='original', metric=metric, mean=means[metric], n=len(values), planned_n=96,
            direction='up' if metric in ['edge_f1', 'iou'] else 'down',
            protocol='uniform CUDA float32 metrics, TF32 disabled; reference metrics N/A', output_native_size='512x512 -> 384x512'))
    csv_write(target/'metric_summary.csv', summary_rows)
    comparison = [r for r in rows(root/'external_comparison_table.csv') if r['model'] != 'LOTS']
    csv_write(root/'external_comparison_table.csv', comparison+summary_rows)
    registry = rows(root/'model_registry.csv')
    for row in registry:
        if row['model_id'] == 'LOTS':
            row.update(role='external_text_sketch_pretrained_zero_shot',
                checkpoint_sha256=next(iter(generated.values()))['adapter_sha256'],
                training_seed='upstream unknown; identity inference seeds recorded separately',
                training_steps='no current-dataset fine-tuning; no training this round',
                initialization_checkpoint='fixed official SDXL+LOTS+DINOv2 revisions; see assets provenance',
                dataset_split='P2b original96; smoke4 reused in96; no swap',
                inference_config='native512 square fp16 Euler30 CFG5 scale0.8; direct BILINEAR384x512 evaluation',
                metric_protocol_version='same uniform CUDA float32; TF32 disabled; structure4+LPIPS; reference metrics N/A',
                result_source=str(target/'metrics/per_image.csv'),
                status='96 generation and five-metric evaluation complete; no missing output',
                comparability_notes='minimum task gate passed; native backgrounds/quality failures retained; no reference texture; official pretrained zero-shot vs fine-tuned BF/CN; no Margin ranking')
    csv_write(root/'model_registry.csv', registry)
    external = [r for r in rows(root/'external_model_registry.csv') if r['model_id'] != 'LOTS']
    external.append(dict(model_id='LOTS',input_modalities='text+sketch; no reference texture',
        checkpoint_sha256=next(iter(generated.values()))['adapter_sha256'],training_seed='upstream unknown',
        training_steps='official pretrained; no current-dataset fine-tuning',dataset_split='P2b original96; no swap',
        inference_config='native512 fp16 Euler30 CFG5 scale0.8; direct BILINEAR384x512',
        code_version='upstream49e3955; generation a394b32/f6408a1; audit dac569c; metrics c9c89ea',
        status='96 generation and structure4+LPIPS complete',
        limitations='no reference texture; TCF/Margins N/A; backgrounds/humans/category failures retained; pretrained zero-shot versus other fine-tuned methods'))
    csv_write(root/'external_model_registry.csv',external)
    protocol=root/'external_protocol.md'
    text=protocol.read_text(encoding='utf-8-sig')
    text=text.replace('| LOTS 候选 | 文本、草图；无独立参考纹理 | 官方源码和权重；先完成4例冒烟和实际任务兼容检查 | 未通过前不扩展、不进入参考响应表 |',
        '| LOTS（补充） | 文本、草图；无独立参考纹理 | 固定官方预训练；fp16 Euler30、CFG5、scale0.8；4例图审后扩展96 | 原生512²保存，直接双线性384×512评价；不进入参考响应表 |')
    if 'LOTS正式96补充（2026-10-29）' not in text:
        text+='\nLOTS正式96补充（2026-10-29）：24项固定资产已本地下载、门户上传并通过服务器哈希校验。四例冒烟只放行最低任务可运行性，原样复用在96中，未改网络或再调参。全部输出保留，GT与独立参考纹理均不进入LOTS推理；仅计算结构四项和LPIPS，TCF-LAB及参考Margin不适用。完整批次含摄影背景、人物入镜、包袋类别失败，Leak接近1不能解释为纹理泄漏率；官方预训练零样本与已微调方法不同，结果只作有任务限制的文本+草图补充，不作同条件因果排名。逐图seed、实际latent SHA、输出SHA、版本、评价和失败图见lots/。\n'
    protocol.write_text(text,encoding='utf-8')
    # 原27作业包含早期LOTS下载尝试；按job_id去重，保留失败/取消历史。
    snapshot = root/'audit/resource_ledger_before_lots_followup.csv'
    if not snapshot.exists(): shutil.copy2(root/'resource_ledger.csv', snapshot)
    resources = {r['job_id']: r for r in rows(snapshot)}
    for row in rows(server/'resource_ledger.csv'):
        row['closure_source'] = 'lots/resource_ledger.csv'
        resources[row['job_id']] = row
    # 最后一个打包作业无法在自身包内捕获结束；补录已保存的最终sacct截图。
    closure = read(output/'final_job_closure.json')
    for row in closure['jobs']:
        resources[row['job_id']] = dict(resources.get(row['job_id'], {}), **row,
            closure_source='lots/server_formal_job_fields.png')
    resource_rows = sorted(resources.values(), key=lambda r: int(r['job_id']))
    assert all(r['state'].startswith(('COMPLETED', 'FAILED', 'CANCELLED')) for r in resource_rows)
    csv_write(root/'resource_ledger.csv', resource_rows)
    states = {}
    for row in resource_rows:
        state = row['state'].split()[0]; states[state] = states.get(state, 0)+1
    gpu = [r for r in resource_rows if int(r['allocated_gpus']) > 0]
    write(root/'audit/resource_summary_local.json', dict(total_jobs=len(resource_rows), job_states=states,
        actual_gpu_hours=sum(float(r['gpu_hours']) for r in gpu), gpu_jobs=len(gpu),
        all_gpu_jobs_completed=all(r['state'] == 'COMPLETED' for r in gpu), new_training=0,
        original_ledger_snapshot_preserved=True, final_CPU_closure_source='lots/server_formal_job_fields.png'))
    shutil.copy2(output/'server_formal_job_fields.png', target/'server_formal_job_fields.png')
    write(target/'final_job_closure.json', closure)
    # 固定哈希普通案例；高Leak/低IoU尾部为结果后选择，保留原生方图。
    ordinary = sorted(inputs, key=lambda i: hashlib.sha256(('T3PP_LOTS_FORMAL/'+i).encode()).hexdigest())[:4]
    failure = list(dict.fromkeys(sorted(metrics, key=lambda i: -float(metrics[i]['leak']))[:3]+
                                 sorted(metrics, key=lambda i: float(metrics[i]['iou']))[:3]))
    provenance = []
    figures = target/'figures'; figures.mkdir(exist_ok=True)
    for name, identities in [('ordinary_hash4', ordinary), ('failure_tail', failure)]:
        canvas = Image.new('RGB', (1050, 280*len(identities)), 'white'); draw = ImageDraw.Draw(canvas)
        for index, identity in enumerate(identities):
            row = inputs[identity]
            images = [('Sketch', root/'server_results/inputs'/Path(row['sketch']).name),
                ('GT (evaluation only)', root/'server_results/inputs'/Path(row['gt']).name),
                ('LOTS native512', target/'formal/native'/(identity+'.png')),
                ('LOTS eval384x512', target/'formal/eval384x512'/(identity+'.png'))]
            y = index*280
            draw.text((8,y+4),f'{name} | ID {identity} | IoU {float(metrics[identity]["iou"]):.3f} | Leak {float(metrics[identity]["leak"]):.3f}',fill='black')
            for col, (label, path) in enumerate(images):
                draw.text((col*262+8,y+25), label, fill='black')
                image = ImageOps.contain(Image.open(path).convert('RGB'), (248, 225))
                canvas.paste(image, (col*262+(262-image.width)//2,y+50+(225-image.height)//2))
                provenance.append(dict(panel=name,id=identity,label=label,path=str(path),sha256=sha(path),
                    selection='input hash before viewing results' if name=='ordinary_hash4' else 'result-selected LOTS high Leak/low IoU tails; not unbiased examples'))
        canvas.save(figures/(name+'.png'))
    all_ids = list(inputs)
    canvas = Image.new('RGB', (12*136, 8*156), 'white'); draw = ImageDraw.Draw(canvas)
    for index, identity in enumerate(all_ids):
        x=(index%12)*136; y=(index//12)*156
        draw.text((x+3,y+2),identity,fill='black')
        image=Image.open(target/'formal/native'/(identity+'.png')).convert('RGB').resize((132,132),Image.Resampling.BILINEAR)
        canvas.paste(image,(x+2,y+20))
    canvas.save(figures/'all96_native_contact.png')
    csv_write(target/'panel_provenance.csv', provenance)
    summary = dict(status='FORMAL96_GENERATION_AND_FIVE_METRICS_COMPLETE', images=96,
        smoke4_reused=sum(bool(r.get('reused_from_smoke')) for r in generated.values()),
        additional_images_after_smoke=92, missing_generation=0, metric_means=means, valid_n={k:96 for k in fields},
        not_applicable=['tcf_lab','dists_margin','color_margin'], reference_texture_used=False,
        native_resolution=[512,512], evaluation_resolution=[384,512], scheduler='EulerDiscreteScheduler',
        steps=30,cfg=5,scale=.8, training=0,human_reviewers=0,
        generation_commits=sorted({r['code_commit'] for r in generated.values()}),
        audit_commit=read(server/'candidate_status.json')['reproduction_commit'],
        metric_commit='c9c89ea47bed9a6efe2dc41d3b50dc54aacd702e',
        bundle=integrity, minimum_task_gate='passed by AI actual image inspection; not human scoring or quality superiority',
        limitation='official pretrained zero-shot text+sketch baseline; not fine-tuned on the current dataset; background/color/structure failures retained')
    write(target/'followup_summary.json',summary)
    write(output/'candidate_status.json',read(server/'candidate_status.json'))
    status=read(root/'execution_status.json')
    status.update(status='AUTOMATIC_EXPERIMENTS_COMPLETE_HUMAN_PENDING',date='2026-10-29',
        LOTS_commit=summary['audit_commit'],LOTS_images=96,new_images=544,
        LOTS_bundle_members=integrity['members'],LOTS_result=str(target),
        pending=[p for p in status['pending'] if 'LOTS' not in p])
    write(root/'execution_status.json',status)
    automatic=read(root/'automatic_results_status.json')
    automatic.update(new_images=544,LOTS='FORMAL96_COMPLETE_WITH_TASK_LIMITATIONS',LOTS_images=96,
        LOTS_metrics='structure4+LPIPS complete; reference metrics N/A',human_reviewers=0)
    write(root/'automatic_results_status.json',automatic)
    for name in ['t3_p0p1_lots_closeout.py','t3_p0p1_evaluate.py']:
        shutil.copy2(Path(__file__).with_name(name),root/'code/tools'/name)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--project',type=Path,required=True)
    parser.add_argument('--lots',type=Path,required=True)
    args=parser.parse_args();closeout(args.project,args.lots)
