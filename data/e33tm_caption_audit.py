"""Caption 只决定预先冻结的队列，不读取实验输出。"""
import re
from tools.e33tm_protocol import ORIENTATION, PATTERN, SCALE, category, pattern

def audit_row(row, caption, split):
    words = lambda expr: [m.group().lower() for m in re.finditer(expr, caption, flags=re.I)]
    orientation = words(ORIENTATION)
    return dict(id=row['id'], split=split, caption=caption, category=category(caption),
                pattern=pattern(caption), pattern_words=words(PATTERN),
                orientation_words=orientation, scale_words=words(SCALE),
                cohort='conflict' if orientation else 'primary')
