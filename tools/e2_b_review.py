"""记录实际查看过的输入图板决策；不自动推断语义，不接触模型输出。"""
import argparse
from pathlib import Path

from tools.e2_fft_experiment import read, write


def main(args):
    root=Path(args.out)
    registry=read(root/'inputs/registry.json')
    rows={r['index']:r for r in registry['records']}
    page=next(p for p in read(root/'inputs/review_pages.json')['pages'] if p['file'].endswith('review_%02d.png'%args.page))
    accepted={}
    if args.accept:
        for item in args.accept.split(','):
            index,pattern,target=item.split(':')
            accepted[int(index)]=(pattern,target=='1')
    assert set(accepted)<=set(page['indices'])
    path=Path(args.review)
    value=read(path) if path.exists() else dict(review_type='AI_visual_input_only',human_annotation=False,
        model_outputs_seen_for_selection=False,reviewed_pages=[],records={})
    for index in page['indices']:
        decision=accepted.get(index)
        value['records'][str(index)]=dict(clean=decision is not None,
            pattern=decision[0] if decision else 'uncertain_or_unsuitable',
            target_eligible=decision[1] if decision else False,hashes=rows[index]['hashes'],
            note='AI input review: visible regular texture; target eligibility checked separately' if decision
                 else 'Not admitted: clean regular-pattern eligibility not sufficiently established; no generated output used')
    if page['file'] not in value['reviewed_pages']:value['reviewed_pages'].append(page['file'])
    value['bound_input_map_sha256']=registry['input_map_sha256']
    write(path,value)
    print('REVIEWED',args.page,'ACCEPTED',len(accepted),'TOTAL_REVIEWED',len(value['records']))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True);p.add_argument('--review',required=True)
    p.add_argument('--page',type=int,required=True);p.add_argument('--accept',default='')
    main(p.parse_args())
