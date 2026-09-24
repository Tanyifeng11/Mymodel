"""113719专项轮廓复核：在192×256目标预览上标注外轮廓，保留原结果。"""
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from tools.e14_denoising_regions import partition,regional_summary
from tools.e14_denoising_complete import analyze
from tools.e14_pattern_probe import write_json

# 坐标是 pairs 首列去掉24像素标题后的目标图；用于latent粗分区，不是像素级真值。
POLYGONS={
425:[(27,43),(34,23),(61,10),(68,1),(93,9),(128,2),(154,12),(164,26),(168,80),(166,152),(143,156),(141,166),(153,252),(128,255),(64,255),(47,252),(48,204),(53,155),(31,149)],
1639:[(34,14),(57,5),(72,8),(110,9),(131,5),(152,12),(151,38),(158,87),(160,160),(161,243),(150,251),(137,245),(125,253),(112,248),(102,253),(89,248),(76,253),(64,246),(51,251),(42,242),(32,248),(32,172),(31,94),(38,47)],
5697:[(28,59),(38,32),(66,16),(77,4),(110,3),(120,17),(145,28),(156,54),(159,106),(163,252),(149,254),(142,158),(138,116),(137,185),(119,188),(56,185),(54,120),(49,151),(46,250),(32,252),(28,152)],
6140:[(62,10),(86,6),(101,3),(116,7),(129,11),(121,55),(122,88),(134,131),(144,178),(158,241),(147,247),(121,252),(81,254),(52,249),(33,241),(45,198),(58,153),(70,101),(69,65)],
13031:[(25,236),(31,144),(36,83),(48,30),(70,16),(76,4),(108,2),(118,15),(142,30),(152,82),(157,145),(165,236),(145,239),(136,153),(135,217),(138,252),(109,255),(97,253),(94,208),(91,252),(77,255),(53,251),(54,184),(50,151),(43,240)],
14446:[(30,163),(36,104),(47,52),(62,20),(80,7),(108,2),(125,14),(140,35),(151,60),(161,113),(162,164),(145,167),(140,113),(133,79),(133,150),(145,223),(130,240),(115,252),(89,254),(70,243),(51,222),(56,150),(58,86),(49,117),(47,168)],
15247:[(20,225),(26,171),(39,123),(45,75),(42,25),(52,23),(61,46),(74,64),(91,74),(107,71),(121,56),(130,27),(141,28),(143,76),(140,105),(147,135),(159,172),(167,226),(137,228),(104,224),(72,227),(44,229)],
27651:[(33,245),(41,195),(42,137),(43,118),(52,97),(56,69),(57,41),(62,6),(88,11),(111,9),(132,16),(141,44),(144,77),(138,97),(146,120),(150,174),(157,246),(137,250),(107,253),(69,253),(46,250)],
29439:[(13,236),(24,174),(31,111),(41,42),(49,16),(62,8),(78,14),(99,15),(128,8),(144,17),(152,45),(159,97),(168,175),(177,236),(153,243),(137,243),(132,187),(111,198),(94,194),(75,199),(56,192),(52,239),(35,242)],
35741:[(4,232),(14,191),(23,141),(33,96),(43,42),(55,25),(79,16),(83,7),(105,5),(115,17),(137,28),(147,54),(155,98),(165,145),(176,195),(189,235),(162,241),(134,242),(133,205),(132,163),(131,129),(107,146),(65,145),(61,122),(53,168),(56,239),(31,242)],
39453:[(0,90),(13,70),(28,65),(42,39),(48,18),(63,8),(74,16),(100,20),(121,8),(136,19),(147,45),(160,65),(175,71),(190,94),(183,116),(166,144),(153,124),(155,162),(163,237),(153,245),(121,249),(73,249),(45,246),(26,238),(34,180),(34,126),(26,142),(14,121)]}


def run():
    root=Path('eval_outputs/e14_complete_denoising/113719')
    out=root/'mask_corrected';out.mkdir(exist_ok=True);(out/'regions').mkdir(exist_ok=True)
    data=json.loads((root/'denoising_report.json').read_text(encoding='utf-8'))
    contact=Image.new('RGB',(6*192,6*280),'white');draw=ImageDraw.Draw(contact)
    changes=[]
    for k,s in enumerate(data['samples']):
        i=s['sample_index']
        with Image.open(root/'regions'/('%05d_mask.png'%i)) as im:old=im.convert('L')
        if i in POLYGONS:
            mask=Image.new('L',(384,512));ImageDraw.Draw(mask).polygon([(x*2,y*2) for x,y in POLYGONS[i]],fill=255)
        else:mask=old.copy()
        mask.save(out/'regions'/('%05d_mask.png'%i));regions=partition(mask,(64,48))
        np.savez_compressed(out/'regions'/('%05d_masks.npz'%i),**regions)
        s['region_diagnostics']['latent_region_pixels']={name:int(m.sum()) for name,m in regions.items()}
        with Image.open(root/'pairs'/('%05d.png'%i)) as im:target=im.crop((0,24,192,280)).resize((384,512))
        arr=np.asarray(target).astype(float)
        for name,color in [('interior',[0,180,0]),('boundary',[255,180,0]),('background',[0,80,255])]:
            m=np.asarray(Image.fromarray(regions[name]).resize((384,512),Image.NEAREST)).astype(bool)
            arr[m]=arr[m]*.65+np.array(color)*.35
        overlay=Image.fromarray(arr.astype('uint8'));overlay.save(out/'regions'/('%05d_overlay.png'%i))
        contact.paste(overlay.resize((192,256)),((k%6)*192,(k//6)*280+24));draw.text(((k%6)*192,(k//6)*280),str(i)+(' corrected' if i in POLYGONS else ''),fill='black')
        if i in POLYGONS:
            changes.append(dict(sample_index=i,polygon=POLYGONS[i],old_area=float((np.asarray(old)>127).mean()),new_area=float((np.asarray(mask)>127).mean())))
            s['region_diagnostics'].update(mask_low_confidence=False,manual_review='assistant_visual_polygon_correction_not_ground_truth')
    for r in data['records']:
        r['error_maps']='../'+r['error_maps']
        with np.load(out/r['error_maps']) as errors,np.load(out/'regions'/('%05d_masks.npz'%r['sample_index'])) as masks:
            r['region_losses']={name:{c:float(errors[c][masks[name]].mean()) if masks[name].any() else None
                                     for c in r['losses']} for name in masks.files}
    data['mask_revision']='11 assistant visually traced silhouettes from 192x256 previews; original results retained'
    data['region_summary']=regional_summary([r for r in data['records'] if r['timestep'] in [1,181,481,781,981]])
    write_json(out/'denoising_report.json',data)
    write_json(out/'mask_corrections.json',dict(changes=changes,limits=['轮廓仅按预览人工描绘；蕾丝、细小孔洞未逐像素分割。','其余21张保持原mask，不代表全部为真值。']))
    contact.save(out/'mask_contact_sheet.png')
    analyze(out)
    # 离线报告置于子目录，配对预览仍引用上一级原始图。
    anomalies=json.loads((out/'anomaly_index.json').read_text(encoding='utf-8'))
    for item in anomalies:item['pair_image']='../'+item['pair_image']
    write_json(out/'anomaly_index.json',anomalies)
    weighted={}
    for label,steps in [('base_steps',[1,181,481,781,981]),('t181',[181])]:
        values={name:[] for name in ['interior','boundary','background']}
        for r in data['records']:
            if r['timestep'] not in steps:continue
            with np.load(out/r['error_maps']) as error,np.load(out/'regions'/('%05d_masks.npz'%r['sample_index'])) as masks:
                delta=(error['wrong_1']+error['wrong_2'])/2-error['matched']
                for name in values:values[name].append(float((delta*masks[name]).mean()))
        weighted[label]={name:float(np.mean(v)) for name,v in values.items()}
    write_json(out/'area_weighted_deltas.json',weighted)


if __name__=='__main__':run()
