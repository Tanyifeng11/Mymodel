"""确认前定位历史跨环境差异；逐项验证输入，绝不放宽确认终点。"""
import hashlib
import shutil
import numpy as np
from PIL import Image
from tools.t3_response_common import *
from tools.t3_response_run import array


def digest(value):return hashlib.sha256(value).hexdigest()


def probe():
    fixture=read(SOURCE/'tools/t3_response_replay_fixture.json')
    preserved=ROOT/'audit/initial_120716';preserved.mkdir(parents=True,exist_ok=True)
    for name in ['development_replay.csv','development_replay.json']:
        target=preserved/name
        if not target.exists():shutil.copy2(ROOT/'audit'/name,target)
    rows={r['id']:r for r in read(MATCHED/'splits/P2b.json')}
    originals={};references={};outputs=[]
    rc=[(int(x),int(y)) for y in np.linspace(0,448,4).round() for x in np.linspace(0,320,4).round()]
    for stage in ['P2b','reference_swap']:
        for row in read(MATCHED/f'splits/{stage}.json') if stage=='P2b' else read(MATCHED/'splits/reference_swap.json'):
            key='reference' if stage=='P2b' else 'reference_alternative'
            originals[sha(row[key])]=row[key]
    for key,path in originals.items():
        image=Image.open(path).convert('RGB');canvas=np.asarray(image.resize((384,512),Image.Resampling.LANCZOS))
        old=fixture['reference'][key]
        new=dict(file_sha256=key,path=path,raw_rgb_sha256=digest(image.tobytes()),canvas_rgb_sha256=digest(canvas.tobytes()),
                 patch_concat_sha256=digest(b''.join(canvas[y:y+64,x:x+64].tobytes() for x,y in rc)))
        for field in ['raw_rgb_sha256','canvas_rgb_sha256','patch_concat_sha256']:
            new[field+'_matches_local']=new[field]==old[field]
        references[key]=new
    for old in fixture['outputs']:
        identity=old['identity'];path=legacy(old['stage'],old['arm'])/'images'/(identity+'.png')
        mask=np.asarray(Image.open(SUP/f'masks/evaluation/{identity}/garment.png').convert('L'))>127
        cs,n=patches(mask);a=array(path)
        new=dict(identity=identity,stage=old['stage'],arm=old['arm'],output_sha256=sha(path),
                 coords=cs,patch_concat_sha256=digest(b''.join(a[y:y+64,x:x+64].tobytes() for x,y in cs)))
        new['output_file_matches']=new['output_sha256']==old['output_sha256']
        new['coords_match']=[list(v) for v in cs]==old['coords']
        new['generated_patch_pixels_match']=new['patch_concat_sha256']==old['patch_concat_sha256']
        outputs.append(new)
    result=dict(reference_count=len(references),output_count=len(outputs),
        raw_reference_mismatch=sum(not r['raw_rgb_sha256_matches_local'] for r in references.values()),
        resized_reference_mismatch=sum(not r['canvas_rgb_sha256_matches_local'] for r in references.values()),
        reference_patch_mismatch=sum(not r['patch_concat_sha256_matches_local'] for r in references.values()),
        generated_file_mismatch=sum(not r['output_file_matches'] for r in outputs),
        coordinate_mismatch=sum(not r['coords_match'] for r in outputs),
        generated_patch_pixel_mismatch=sum(not r['generated_patch_pixels_match'] for r in outputs),
        local_environment=fixture['environment'],reference_details=references,output_details=outputs,
        fixture_sha256=sha(SOURCE/'tools/t3_response_replay_fixture.json'),commit=commit())
    write(ROOT/'audit/input_equivalence_probe.json',result)
    print('INPUT_EQUIVALENCE', {k:v for k,v in result.items() if k not in ['reference_details','output_details']},flush=True)


if __name__=='__main__':probe()
