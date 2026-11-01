"""本地核验下载包并写入用户指定 docs；不连接服务器或启动训练。"""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def extract(archive, destination, expected):
    assert sha(archive) == expected, '服务器与本地压缩包 SHA256 不同'
    destination.mkdir(parents=True, exist_ok=True)
    base = destination.resolve()
    with tarfile.open(archive, 'r:gz') as package:
        for member in package.getmembers():
            target = (base / member.name).resolve()
            assert target == base or base in target.parents, '压缩包路径越界'
            assert member.isfile() or member.isdir(), '结果包不应包含符号链接'
        package.extractall(destination)
    checksums = read(destination / 'checksums.json')
    for name, expected_sha in checksums.items():
        assert sha(destination / name) == expected_sha, ('结果文件校验失败', name)
    return dict(archive_sha256=expected, verified_files=len(checksums), archive_bytes=archive.stat().st_size)


def render(root, docs):
    raw = root / 'server_results'
    decision = read(raw / 'decision.json')
    protocol = read(raw / 'protocol.json')
    summary = read(raw / 'metrics/summary.json')
    training = {arm: read(raw / 'training' / arm / 'complete.json') for arm in ['residual', 'reference']}
    integrity = read(root / 'download_integrity.json')
    lines = [
        '# T3 局部纹理残差与参考一致性监督小验证', '',
        '**本轮判定：%s。** %s' % (decision['result'],
            '达到本轮探索门槛；扩大训练前仍需独立数据确认。' if decision['result'] == 'GO'
            else '未同时达到纹理与结构门槛，不扩大本配置训练。'), '',
        '本轮冻结原 T3 正式最终权重，只训练两个局部纹理残差支路。三组为原 T3、仅残差、残差加参考一致性监督。'
        '新增支路在两个较高分辨率 ResNet 层以零初始化输出接入，仅在 t≤249 的有条件前向中启用。', '',
        '真实训练128件、受控条纹训练32例；开发评价16件真实服装的原参考与换参考，共32条，另有两张留出草图的8条纹样条件。'
        '真实评价身份均来自已查看开发集，按输入哈希及至少两个128²衣内窗口选择，不能称作独立确认。', '',
        '参考监督使用归一化灰度功率谱集合距离，配合噪声预测和衣外/边界同状态T3蒸馏。'
        '频谱、DISTS及合成横竖条纹检验均不等于真实花卉、文字或logo身份准确率。', '',
        '| 组别/条件 | 灰度DISTS↓ | 参考Margin↑ | IoU↑ | Edge F1↑ | 白底偏差↓ |',
        '| --- | ---: | ---: | ---: | ---: | ---: |',
    ]
    labels = {'T3': '原T3', 'residual': '仅残差', 'reference': '残差+参考监督'}
    for record in summary:
        if record['condition'] == 'synthetic':
            continue
        number = lambda key: 'N/A' if record[key] is None else '%.6f' % record[key]
        lines.append('| %s/%s | %s | %s | %s | %s | %s |' % (
            labels[record['arm']], '原参考' if record['condition'] == 'original' else '换参考',
            number('gray_dists'), number('gray_margin'), number('iou'), number('edge_f1'), number('background_mae')))
    lines += ['', '灰度DISTS在本轮128²窗口下计算，不能与此前64²结果直接比较；白底偏差为固定草图mask外对纯白的RGB MAE，'
              '不是历史Leak指标。区间均为身份配对bootstrap描述区间，未作多重检验确认。', '',
              '| 换参考主比较 | 灰度DISTS均值差 | 95%配对区间 | 有效身份 | 通过 |',
              '| --- | ---: | --- | ---: | --- |']
    for record in decision['quality']:
        interval = record['ci95']
        lines.append('| %s | %.6f | [%+.6f, %+.6f] | %d | %s |' % (
            record['comparison'], record['mean'], interval[0], interval[1], record['n'],
            '是' if record['passed'] else '否'))
    lines += ['', '主门槛要求相对原T3及仅残差均下降至少0.002，区间上界低于0，且有效身份至少13/16。'
              '结构门槛分别检查原/换参考下IoU、Edge F1和白底偏差的配对区间；合成检验要求方向准确率至少75%、周期相对误差不超过20%。', '',
              '| 合成条纹 | 方向准确率 | 平均周期相对误差 |', '| --- | ---: | ---: |']
    for arm, record in decision['synthetic'].items():
        lines.append('| %s | %.2f%% | %.2f%% |' % (labels[arm], 100*record['orientation_accuracy'],
                     100*record['period_relative_error']))
    passed = sum(record['passed'] for record in decision['structure'])
    lines += ['', '结构门槛通过%d/%d；合成门槛%s。两组T3冻结检查%s，训练样本与时间步序列%s。' % (
        passed, len(decision['structure']), '通过' if decision['synthetic_pass'] else '未通过',
        '通过' if decision['frozen'] else '未通过', '相同' if decision['same_training_stream'] else '不同'), '',
        '| 训练支路 | 成功更新 | 训练秒数 | 峰值显存GiB |', '| --- | ---: | ---: | ---: |']
    for arm, record in training.items():
        lines.append('| %s | %d | %.1f | %.3f |' % (
            labels[arm], record['updates'], record['seconds'], record['peak_gib']))
    lines += ['', '本轮采用单seed、每臂400步和有效batch2；本表仅为训练函数耗时，完整作业成本需结合原始sacct账单，'
              '不能用它替代分配GPU小时。', '',
              '服务器独立目录：`/share/home/u2515283058/Mymodel/output_eval/%s`。代码分支为`codex/t3-local-texture-pilot-20261101`，'
              '通过本地Git推送、服务器pull部署。正式协议源码提交为`%s`。' % (protocol['name'], protocol['source_commit']), '',
              '本地原包SHA256：`%s`；%d个清单文件逐一校验通过。原包及解压目录保留。' % (
                  integrity['archive_sha256'], integrity['verified_files']), '',
              '[完整判定](%s)、[逐图指标](%s)、[配对统计](%s)。' % (
                  (raw/'decision.json').as_posix(), (raw/'metrics/per_image.json').as_posix(),
                  (raw/'metrics/paired.json').as_posix()), '',
              '图审观察须在实际查看生成图后另补；本程序不自动生成视觉评价或真人结论。', '']
    docs.mkdir(parents=True, exist_ok=True)
    target = docs / 'T3_局部纹理残差与参考监督验证_20261101.md'
    target.write_text('\n'.join(lines), encoding='utf-8')
    return target


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--docs', type=Path, default=Path('F:/fuxian/Mymodel/docs'))
    args = parser.parse_args()
    result = extract(args.archive, args.root/'server_results', args.sha256)
    (args.root/'download_integrity.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(render(args.root, args.docs))
