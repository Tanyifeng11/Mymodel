"""扫描原划分，冻结 caption/cohort/donor；重复执行只核验哈希。"""
from pathlib import Path
from data.e33tm_caption_audit import audit_row
from tools.e33tm_protocol import *

def prepare():
    ready = OUT/'manifests/frozen.json'
    if ready.exists():
        freeze_check()
        return read(OUT/'manifests/cohorts.json')
    split = read(RF/'split_manifest.json')
    assert [len(split[k]) for k in ('train', 'dev')] == [45126, 256]
    captions = {Path(r['cloth']).stem: r['caption'] for r in read('data/train_bf_texture.json')}
    rows = [audit_row(r, captions[r['id']], key) for key in ('train', 'dev') for r in split[key]]
    index = {r['id']: r for r in rows}
    enriched = {r['id']: dict(r, **{k: index[r['id']][k] for k in ('caption', 'category', 'pattern', 'cohort')})
                for key in ('train','dev') for r in split[key]}
    dev = [enriched[r['id']] for r in split['dev']]
    primary = [r for r in dev if r['cohort'] == 'primary']
    conflict = [r for r in dev if r['cohort'] == 'conflict']
    assert primary and len(primary)+len(conflict) == 256
    donors = {}
    for row in dev:
        candidates = order([r for r in dev if r['id'] != row['id'] and r['category'] == row['category']],
                           'E33TM/donor/'+row['id'])
        if not candidates:
            candidates = order([r for r in enriched.values() if r['id'] != row['id'] and r['category'] == row['category']],
                               'E33TM/donor/'+row['id'])
        assert candidates, 'same-category donor unavailable: '+row['id']
        donors[row['id']] = dict(shuffled_text=next((r['id'] for r in candidates if r['caption'] != row['caption']), None),
                                 wrong_sketch=candidates[0]['id'], wrong_texture=candidates[-1]['id'])
        assert donors[row['id']]['shuffled_text'], 'distinct caption donor unavailable'
    used = {sid for choices in donors.values() for sid in choices.values()}
    cohort = dict(dev=dev, primary=primary, conflict=conflict, donors=donors,
        donor_rows={sid:enriched[sid] for sid in used},
        smoke16=[r['id'] for r in order(primary, 'E33TM/smoke')[:16]],
        robust64=[r['id'] for r in order(primary, 'E33TM/robust')[:64]],
        hash16=[r['id'] for r in order(primary, 'E33TM/visual')[:16]])
    folder = OUT/'caption_audit'
    write(folder/'caption_geometry_audit.json', rows)
    for name, predicate in [('neutral_text_ids', lambda r: r['cohort']=='primary'),
                            ('conflict_text_ids', lambda r: r['cohort']=='conflict'),
                            ('scale_sensitive_ids', lambda r: bool(r['scale_words']))]:
        write(folder/(name+'.json'), {key: [r['id'] for r in rows if r['split']==key and predicate(r)] for key in ('train','dev')})
    summary = {key: dict(total=sum(r['split']==key for r in rows),
                        primary=sum(r['split']==key and r['cohort']=='primary' for r in rows),
                        conflict=sum(r['split']==key and r['cohort']=='conflict' for r in rows),
                        scale_sensitive=sum(r['split']==key and bool(r['scale_words']) for r in rows)) for key in ('train','dev')}
    write(folder/'caption_audit_summary.json', summary)
    write(OUT/'manifests/cohorts.json', cohort)
    protocol = dict(PROTOCOL)
    write(OUT/'protocol.json', dict(protocol=protocol, git_commit=commit()))
    paths = [RF/'split_manifest.json', RF/'controlled_manifest.json', E5, Path('data/train_bf_texture.json'),
             OUT/'manifests/cohorts.json', OUT/'protocol.json']
    for seed in SEEDS:
        paths += [RF/('seed%d'%seed)/'RF2/checkpoint_final.pt',
                  RF/('seed%d'%seed)/'RF2/real/summary.json', RF/('seed%d'%seed)/'RF2/controlled/summary.json']
    write(OUT/'frozen_check.json', dict(before={str(p): sha(p) for p in paths}, after=None, **{'pass':None}, training_steps=0))
    write(ready, dict(cohort_sha256=sha(OUT/'manifests/cohorts.json'), selection_before_generation=True))
    decision(caption_audit_complete=True, primary_cohort_size=len(primary), conflict_cohort_size=len(conflict),
             field_precheck_pass=None, trimodal_reference_causality_pass=None,
             modality_role_disentanglement_pass=None, next_route='field_precheck')
    freeze_check()
    print('[E33TM caption]', summary, flush=True)
    return cohort

if __name__ == '__main__':
    prepare()
