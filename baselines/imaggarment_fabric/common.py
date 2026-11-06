"""独立 Baseline 的路径约束和数据接口；不导入 Mymodel 模型模块。"""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / 'config.json').read_text(encoding='utf-8'))
ROOT = Path(CONFIG['run'])
PROJECT = Path(CONFIG['project'])
OLD = Path(CONFIG['comparison'])
NAME = CONFIG['name']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def output(path):
    """所有服务器写操作必须留在本次目录，拒绝符号链接指向旧实验。"""
    path = Path(path).resolve()
    path.relative_to(ROOT.resolve())
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write(path, value):
    path = output(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def identity(path):
    return Path(path).stem.split('_')[0]


def training_row(row, base, split):
    return dict(sample_id=identity(row['cloth']), split=split,
                sketch_path=str(base / row['sketch']), fabric_path=str(base / row['texture']),
                caption=row['caption'], target_path=str(base / row['cloth']),
                category=row.get('category', 'unknown'))


def generation_row(row):
    """生成入口只接收条件，不接收 GT 路径或 GT 像素。"""
    return {key: row[key] for key in ['id', 'category', 'condition', 'seed', 'caption',
                                    'sketch', 'reference', 'sketch_sha256', 'reference_sha256', 'split']}
