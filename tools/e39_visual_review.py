"""2026-10-11 单一 AI 对输入 contact sheet 的视觉记录，先于总体统计锁定。

这不是人工标注或独立验证集。只描述 reference 裁块可见表面，不照抄服装 caption。
"""
LABELS = {
    '174397224':'heather', '193488637':'floral', '53887902':'solid', '127609356':'denim',
    '154758093':'denim', '5074971':'graphic', '185842747':'solid', '213285235':'floral',
    '73024600':'solid', '140540361':'lace', '204446309':'floral', '149190311':'solid',
    '149003380':'solid', '180633937':'solid', '172600892':'solid', '194305575':'indeterminate',
    '210013766':'graphic', '53328647':'solid', '160097987':'floral', '167344818':'floral',
    '54445206':'solid', '170162876':'abstract', '193312302':'solid', '204012322':'heather',
    '137601298':'solid', '184940860':'denim', '195833180':'plaid', '93881823':'solid',
    '177871861':'ribbed', '208152706':'denim', '162607053':'floral', '77103854':'solid',
}
COLOR = {
    '127609356':dict(verified=True,notes='blue-gray worn denim vs blue-gray plaid; palette only approximately matched'),
    '149190311':dict(verified=False,notes='both near-black; donor horizontal bands too faint to establish different motif'),
    '210013766':dict(verified=True,notes='both black/white graphic fragments, distinguishable glyph arrangement; not different broad texture class'),
    '53328647':dict(verified=True,notes='near-uniform cream reference vs cream muted floral donor; original caption graphic not visible'),
    '204012322':dict(verified=False,notes='both gray mottled fragments; different motif not visually established'),
}
INPUT_LIMITS = {
    '174397224':'reference shows heather fabric and a tiny dark edge; full skull/text motif absent',
    '5074971':'reference contains only a blurry dark/white graphic fragment, not identifiable woman motif',
    '53328647':'caption graphic not identifiable in near-uniform light reference',
    '193312302':'caption floral embroidery not visible in black crop',
    '204012322':'caption Alice graphic not identifiable in gray heather crop',
    '127609356':'reference includes jeans/white background boundary rather than fabric-only swatch',
    '154758093':'reference contains placket/button structure',
    '180633937':'reference contains button/garment structure',
    '184940860':'reference contains seam/button structure',
}
