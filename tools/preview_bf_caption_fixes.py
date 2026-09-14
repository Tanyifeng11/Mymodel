"""把已完成的文本修订排版为核验图；不修改任何数据图像。"""
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def wrap(draw, text, font, width):
    lines=['']
    for word in text.split():
        trial=(lines[-1]+' '+word).strip()
        if draw.textlength(trial,font=font)>width and lines[-1]: lines.append(word)
        else: lines[-1]=trial
    return '\n'.join(lines)


def main():
    base=Path('eval_outputs/bf_annotation_review_v1')
    out=base/'caption_comparisons';out.mkdir(exist_ok=True)
    patches=[json.loads(s) for s in (base/'caption_corrections.jsonl').read_text(encoding='utf-8').splitlines()]
    records={r['sample_id']:r for r in (json.loads(s) for s in (base/'selected_100.jsonl').read_text(encoding='utf-8').splitlines())}
    selected=[patches[i] for i in (0,2,5,6,8,10)]
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',22)
    cn=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',21)
    title=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',25)
    for page in range(3):
        sheet=Image.new('RGB',(1280,820),'#f4f5f7');d=ImageDraw.Draw(sheet)
        for j,p in enumerate(selected[page*2:page*2+2]):
            y=j*410
            d.text((24,y+15),p['sample_id']+'  |  原图未改，仅修订文本',font=title,fill='#202838')
            with Image.open(Path('F:/fuxian/dataset/datasets/BF')/records[p['sample_id']]['cloth']) as im:
                sheet.paste(im.convert('RGB'),(24,y+70))
            for x,label,key,color in [(310,'修复前','original_caption','#9c3535'),(790,'修复后','revised_caption','#20663f')]:
                d.rounded_rectangle((x,y+62,x+460,y+320),radius=10,fill='white')
                d.text((x+16,y+76),label,font=title,fill=color)
                text=wrap(d,p[key],font,425)
                assert d.multiline_textbbox((0,0),text,font=font,spacing=7)[3]<195
                d.multiline_text((x+16,y+122),text,font=font,fill='#202838',spacing=7)
            reason=p['reason']
            # 中文按字符折行，保留源修订理由。
            lines=['']
            for c in reason:
                if d.textlength(lines[-1]+c,font=cn)>1180: lines.append(c)
                else: lines[-1]+=c
            d.multiline_text((24,y+342),'\n'.join(lines),font=cn,fill='#394252',spacing=4)
        sheet.save(out/f'comparison_{page+1:02d}.png')
    print(out.resolve())


if __name__=='__main__': main()
