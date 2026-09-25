# E18 第一轮：BF 与 readout 的方向语义联合对齐

实验日期：2026-09-26。代码：`codex/e18` 分支，提交 `a8a76bc`、`8da626b`。服务器作业 `114003` 已完成，退出码 `0:0`；结果位于 `/share/home/u2515283058/Mymodel/e18/`。

## 问题与停止条件

E17 发现，BF 的部分来源特征有纹样线索，但完整 fused、最终 token 和生成图缺少一致的方向语义。E18 第一轮只检验：**同时训练 BF 后段与 readout，并加入方向对齐目标，能否让 fused 和最终 texture token 在独立验证集上同步获得方向语义？**

预设比较 B0（联合训练对照）与 B1（相同联合训练，加 orientation alignment）。只有 B1 在 fused 和 final token 均明显优于 B0，才进入 E18-C 的生成利用训练；C 成功后再做 E18-D 完整生成。

## 数据与训练

Pattern-Gold 从 E14 已核对缩略图的验证集候选构建，固定为 31 张图：10 个方向明确的条纹参考各含原图和 90° 旋转，另含条纹、格纹、波点等类别参考；标注了 4 对视觉上色调接近、类别不同的候选。构建时检查像素哈希，方向探针对同一原始参考的两种旋转版本整体留出。训练只使用 BF training split，Pattern-Gold 不参与训练。

4 对候选只经视觉检查和灰度直方图比较，**尚不能严格称为逐对同色异纹**；`source_group` 也只是临时样本分组，真实面料来源身份尚未核验。较广的独立验证沿用 E17 协议，横/竖方向任务为 120 个样本、三折评估。因此 Gold 是更明确的小型方向诊断集，不是严格按真实面料来源留出的最终测试集。

B0、B1 均从同一个 E17 `select` direct readout 检查点出发，使用相同训练样本选择、随机种子、500 步和优化设置。联合更新 BF stage3/4、对应 source projection 和 direct readout；冻结 BF stage1/2、CLIP、U-Net、TCPM。两组均通过蒸馏原 E5/GAM texture tokens 保留原有 token 行为；B1 额外对 fused 和 final token 加方向监督对比及两层关系对齐。每个训练参考同时使用原图与 rot90，并施加相同颜色扰动，避免颜色直接标识旋转标签。

训练确实发生：B0 的保留损失从约 0.513 降至 0.041；B1 从约 0.513 降至 0.052，方向损失从约 4.50 降至 3.98。损失下降本身不构成独立方向语义已学成的证据。

## 独立验证结果

表中方向探针均为 balanced accuracy，0.5 约为随机水平。Pattern-Gold 方向任务含 10 个原始参考的 20 个原图/旋转样本。

| Pattern-Gold 方向探针 | E5/GAM | E17 select | B0 | B1 |
| --- | ---: | ---: | ---: | ---: |
| CNN3/4 | 0.60 | 0.60 | 0.60 | 0.55 |
| fused | 0.55 | 0.55 | 0.50 | 0.50 |
| final token | 0.60 | 0.55 | 0.60 | 0.60 |
| 原图 FFT | 0.90 | 0.90 | 0.90 | 0.90 |

| 较广验证集 / 旋转响应 | B0 | B1 |
| --- | ---: | ---: |
| fused 方向 linear | 0.537 | 0.549 |
| fused 方向 MLP | 0.516 | 0.514 |
| final token 方向 linear | 0.562 | 0.514 |
| final token 方向 MLP | 0.526 | 0.495 |
| 原图 FFT 方向 | 0.802 | 0.802 |
| final token `D_rot` | 0.0761 | 0.0928 |
| U-Net `R_rot` | 0.0318 | 0.0237 |

B1 的 fused 线性探针仅比 B0 高 0.012，而 final token 反而低 0.048；Gold 上两层也没有优于 B0。`D_rot` 增大说明 token 对旋转更敏感，但 `R_rot` 未同步提高，更不能由此推断方向语义或生成跟随已经改善。

## 阶段判断

**E18-B 未通过预设门槛。**这次方向对齐目标没有建立 `BF/CNN → fused → final token` 一致的方向语义链路，因此按停止条件未启动 E18-C 的 matched-reference 服装内部 denoising 优势训练，也未运行 E18-D 的完整生成、方向跟随及结构质量评估。本轮没有 E18 生成结果，不能对生成改善或 Sketch IoU 作新结论。

当前只能说：**这组数据、损失权重、训练预算和后段 BF/readout 更新范围下，B1 没有优于联合训练对照。**不能据此证明 BF 架构本身不适合方向纹样；方向目标优化不足、训练标签噪声、Gold 规模以及来源分组未严格核验，仍是可能的限制。下一步应先解决 B 阶段的可学习性与独立验证问题，再按原门槛决定是否进入 C。

服务器可复核文件：`e18/gold/manifest.json`、`e18/train_selection.json`、`e18/b0/train_report.json`、`e18/b1/train_report.json`、`e18/gold_probe_{gam,e17_select,b0,b1}/report.json`、`e18/validation_{b0,b1}/report.json`、`e18/final_probe_{b0,b1}/report.json`、`e18/response_{b0,b1}/report.json`。模型检查点分别为 `e18/b0/joint_model.pt` 和 `e18/b1/joint_model.pt`。
