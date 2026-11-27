"""从审核过的源图和人工标注重现草图通道诊断的四个输入。"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from garment_mask_utils import build_sketch_garment_mask


# 源文件及 SHA 摘自冻结 strict_pairs.json，避免要求安装整套诊断归档。
records = [{'case_id': 'validation_5735',
  'index': 5735,
  'identity_group': 'validation/195397475',
  'sketch': 'validation/sketch/195397475.jpg',
  'gt': 'validation/gt/195397475.jpg',
  'hashes': {'sketch': '97f5d79d1b6e98c4121090760354edace7a7d2a926bd4d7f4c58b2b86c3bb812',
             'gt': '079998f98d346b0dad1a159016203dfbc429088f403238a5d3b601146bd1e4d8'}},
 {'case_id': 'validation_1455',
  'index': 1455,
  'identity_group': 'validation/139252531',
  'sketch': 'validation/sketch/139252531.jpg',
  'gt': 'validation/gt/139252531.jpg',
  'hashes': {'sketch': 'd84b829fac79232b97a9be44f0ee90a798ce2c95005e15785e33e8c8a3dd8d42',
             'gt': '73eafad12cb399d4d3f6e04abd7a356b501c3115a307b242be0b860a5c3a0907'}},
 {'case_id': 'validation_2744',
  'index': 2744,
  'identity_group': 'validation/159180238',
  'sketch': 'validation/sketch/159180238.jpg',
  'gt': 'validation/gt/159180238.jpg',
  'hashes': {'sketch': '70131df932a58507acda8d90534ef732bd359b2272626f7b5552332fc45c7bab',
             'gt': 'b44f2a0440c97f6e6b31da17dc3b76f97b7c0d5181739c2abcae242742f0bd7a'}},
 {'case_id': 'validation_7871',
  'index': 7871,
  'identity_group': 'validation/214322442',
  'sketch': 'validation/sketch/214322442.jpg',
  'gt': 'validation/gt/214322442.jpg',
  'hashes': {'sketch': '439a6fd816b8cb6178fa7643ee054d107ceb3d9081ee7748ea06391c8bb20fc8',
             'gt': 'e12ef698aee8470bebd3380744846fc66519d3633455d5055a5156c08b1044dd'}}]

# Coordinates were reviewed on the 384 x 512 BILINEAR rendering.
# The first polyline outlines the visible garment. Later polylines protect
# genuine details inside the silhouette. Missing ink is never reconstructed.
annotations = {
    5735: {
        'description': 'long-sleeved crew-neck top', 'family': 'horizontal stripes',
        'outline': [(134,2),(113,10),(88,19),(78,29),(73,39),(68,62),(66,94),(68,145),(69,200),(69,233),(67,263),(66,311),(63,353),(61,392),(55,432),(49,480),(47,499),(88,509),(93,504),(94,488),(102,500),(142,503),(219,503),(282,502),(292,500),(293,511),(333,504),(331,464),(325,415),(321,368),(318,321),(315,274),(312,241),(314,203),(313,146),(313,92),(312,64),(308,38),(299,22),(286,16),(252,10),(241,2),(226,14),(211,21),(185,24),(158,22),(143,13)],
        'seams': [[(136,4),(138,20),(147,34),(154,46),(169,55),(187,60),(199,59),(216,53),(229,42),(237,27),(243,8)],
                  [(137,14),(145,25),(157,33),(177,38),(193,39),(211,36),(226,28),(240,13)],
                  [(109,142),(111,165),(113,186),(114,214),(117,247),(117,267),(114,291),(110,317),(106,344),(104,370),(101,394),(99,422),(97,450),(94,478),(92,492)],
                  [(272,142),(269,159),(271,177),(273,197),(270,212),(269,238),(268,259),(269,280),(272,300),(277,324),(280,352),(284,382),(286,407),(289,435),(291,464),(293,499)]],
        'keep_polys': [[(135,0),(143,18),(160,27),(185,32),(209,29),(229,18),(242,0),(242,22),(235,36),(225,49),(212,57),(190,63),(168,58),(151,49),(139,34),(133,22)]],
        'edit': [(58,10),(132,4),(144,21),(167,34),(193,40),(221,31),(239,10),(284,13),(318,21),(335,509),(293,508),(284,494),(119,496),(97,490),(89,503),(47,498)],
        'notes': 'Crew neckline, sleeve/torso boundaries, cuff tips and hem preserved. A small eye applique visible in the paired photo was removed with the horizontal print. Ambiguous ink touching seams kept.'
    },
    1455: {
        'description': 'sleeveless strappy top', 'family': 'horizontal stripes',
        'outline': [(67,0),(81,0),(87,27),(96,48),(118,55),(146,62),(178,67),(201,66),(227,62),(254,55),(276,48),(285,18),(290,0),(302,1),(306,12),(299,36),(294,51),(297,74),(303,99),(309,130),(315,151),(322,182),(328,204),(337,231),(346,256),(352,283),(356,314),(358,343),(356,368),(352,390),(350,416),(347,449),(345,481),(341,507),(41,511),(38,475),(35,442),(32,413),(29,394),(24,366),(22,338),(24,306),(26,277),(31,252),(39,231),(48,206),(55,185),(62,161),(68,135),(71,110),(74,78),(77,54),(69,34),(65,16)],
        'seams': [[(93,64),(92,85),(89,110),(86,129),(80,153),(74,174),(66,194),(59,217),(50,241),(42,265),(34,294),(28,322)],
                  [(279,64),(283,85),(288,107),(293,130),(300,154),(307,179),(315,201),(321,221),(329,247),(337,271),(345,299),(351,322)]],
        'keep_polys': [[(56,0),(314,0),(314,48),(283,50),(264,58),(244,65),(213,71),(188,74),(162,73),(141,70),(118,64),(97,58),(59,54)]],
        'edit': [(68,50),(100,59),(138,70),(178,76),(216,73),(249,66),(278,56),(308,54),(369,345),(347,506),(38,506),(18,343)],
        'notes': 'Straps, outer silhouette and armhole binding were protected. Bottom is faded and cropped in the original and ambiguous marks at the side binding were kept.'
    },
    2744: {
        'description': 'short-sleeved dress', 'family': 'large dots',
        'outline': [(147,0),(165,0),(176,9),(184,19),(194,16),(208,7),(217,0),(242,0),(255,10),(268,20),(275,30),(282,45),(288,63),(295,82),(285,99),(273,114),(260,134),(260,151),(263,178),(266,208),(268,235),(269,267),(271,310),(271,354),(271,399),(269,454),(266,493),(265,511),(116,511),(113,487),(111,450),(111,414),(111,369),(112,324),(112,280),(114,236),(117,198),(120,165),(121,142),(115,131),(107,115),(98,100),(87,83),(91,70),(96,56),(101,43),(106,31),(112,22),(126,11)],
        'seams': [[(150,4),(148,20),(155,31),(165,40),(175,51),(182,45),(192,35),(204,25),(217,12),(225,2)],
                  [(104,84),(112,92),(121,112),(124,132)],[(279,88),(271,100),(260,121),(255,141)]],
        'keep_polys': [[(85,0),(303,0),(305,85),(267,112),(255,139),(250,145),(241,139),(238,119),(242,97),(232,90),(139,90),(131,102),(132,125),(132,138),(119,132),(92,110)]],
        'edit': [(127,92),(147,86),(218,86),(237,87),(238,120),(241,140),(250,146),(257,142),(267,245),(270,366),(268,491),(266,511),(113,511),(108,414),(108,230),(119,147),(134,140)],
        'notes': 'Neck opening, sleeves, underarm folds and silhouette protected. Only circular interior motifs were removed. Ambiguous shading intersecting folds and cropped hem retained.'
    },
    7871: {
        'description': 'wide-sleeved blouse', 'family': 'no visible interior print',
        'notes': 'Already clean. Identical original and clean RGB images serve as the no-op control.'
    }
}


def polygon(points):
    mask = np.zeros((512,384),np.uint8)
    cv2.fillPoly(mask,[np.asarray(points,np.int32)],255)
    return mask


def stroke(points,width):
    mask = np.zeros((512,384),np.uint8)
    cv2.polylines(mask,[np.asarray(points,np.int32)],False,255,width)
    return mask


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, default=Path('F:/fuxian/dataset/datasets/BF'))
    parser.add_argument('--output', type=Path, default=PROJECT / 'output_eval/e2_sketch_route_abcd_20261127/inputs')
    args = parser.parse_args()
    dataset_root, output = args.dataset_root, args.output
    output.mkdir(parents=True, exist_ok=True)
    cases=[]
    for idx, a in annotations.items():
        r=next(q for q in records if q['index']==idx)
        dest=output/r['case_id']
        dest.mkdir(exist_ok=True)
        path=dataset_root/r['sketch']
        assert sha(path)==r['hashes']['sketch']
        assert sha(dataset_root/r['gt'])==r['hashes']['gt']
        orig=Image.open(path).convert('RGB').resize((384,512),Image.Resampling.BILINEAR)
        rgb=np.array(orig)
        gray=np.asarray(orig.convert('L'))
        o=np.zeros_like(gray)
        if 'outline' in a:
            fg=polygon(a['outline'])
            protective=stroke(a['outline']+a['outline'][:1],11)
            # Edges running along the image frame are crop boundaries, not garment lines.
            if idx in (2744,1455):
                protective[505:,55:330]=0
            for pts in a['seams']:
                protective |= stroke(pts,8)
            for pts in a['keep_polys']:
                protective |= polygon(pts)
            erase=(polygon(a['edit'])>0)&(fg>0)&(protective==0)
            structure=(protective>0)&(gray<170)
        else:
            fg=np.asarray(build_sketch_garment_mask(orig,384,512)[0])
            erase=(o>0)
            structure=(gray<170)&(cv2.dilate(fg,np.ones((9,9),np.uint8))>0)
        clean=rgb.copy()
        clean[erase]=255
        edited=np.any(clean!=rgb,axis=2)
        structure[:2,:]=False
        structure[-2:,:]=False
        structure[:,:2]=False
        structure[:,-2:]=False
        orig.save(dest/'original.png')
        Image.fromarray(clean).save(dest/'clean.png')
        Image.fromarray(fg).save(dest/'mask.png')
        Image.fromarray(structure.astype('uint8')*255).save(dest/'structure_mask.png')
        Image.fromarray(edited.astype('uint8')*255).save(dest/'edit_mask.png')
        assert not np.any(edited & structure)
        assert not np.any(edited & (fg==0))
        cases.append(dict(case_id=r['case_id'],identity_group=r['identity_group'],
            sample_type='clean_control' if idx==7871 else 'patterned',
            category=a['description'],prompt='A '+a['description']+' following the garment sketch.',
            seed=42,source=r['sketch'],source_dataset_root=r'F:\fuxian\dataset\datasets\BF',source_sketch_sha256=sha(path),
            source_photo_for_annotation_review_only=r['gt'],source_photo_sha256=r['hashes']['gt'],
            source_split='BF official validation; previously used diagnostic identities',
            preprocessing='PIL RGB BILINEAR resize from 256x256 to 384x512 before annotation, lossless PNG, unchanged framing.',
            annotation=a['notes'],source_pattern=a['family'],
            mask_method='manual outline drawn on original sketch' if idx!=7871 else 'original sketch flood fill',
            structure_method='confident original dark lines in reviewed outline/detail regions; partial positive annotation, not exhaustive',
            changed_pixels=int(edited.sum()),changed_structural_pixels=int((edited&structure).sum()),
            original_ink=int((gray<180).sum()),removed_ink=int((edited&(gray<180)).sum()),
            hashes={name:sha(dest/name) for name in ('original.png','clean.png','mask.png','structure_mask.png','edit_mask.png')}))
    # 固定序列化换行，与归档中的 Windows 文件一致。
    manifest_text = json.dumps(cases,indent=2,ensure_ascii=False)+'\n'
    (output/'cases.json').write_bytes(manifest_text.replace('\n','\r\n').encode('utf-8'))
    preview=Image.new('RGB',(4*240,4*(270)),'white')
    d=ImageDraw.Draw(preview)
    for i,r in enumerate(cases):
        dest=output/r['case_id']
        for j,name in enumerate(('original','clean','edit_mask','structure_mask')):
            im=Image.open(dest/(name+'.png')).convert('RGB').resize((192,256))
            preview.paste(im,(i*240+24,j*270+14))
            d.text((i*240+24,j*270),r['case_id']+' '+name,fill='black')
    preview.save(output/'review.png')


if __name__ == '__main__':
    main()
