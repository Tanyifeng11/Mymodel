# E19 阶段总结：显式纹样分支与可学习替代

截至 2026-09-26，E19-A 已通过；E19-B 学会了方向，但未通过频谱泛化门槛。按停止逻辑，未进入 E19-C 属性组合验证和 E19-D 完整生成。

**当前证据：显式方向信息可以经过最终条件接口保留下来，并使冻结 U-Net 对参考旋转产生响应；小型可学习 encoder 的方向能力已形成，但尚不能稳定替代手工分支的频率表示。没有生成方向跟随或结构质量改善的新证据。**

## 1. 实验范围与执行记录

承接 [E18 总结](E18_pattern_joint_alignment_stage1.md)：E18.2 的 fused-only 训练未形成稳定方向几何。本轮保留 BF，新增专门的 pattern branch，先验证已知方向信息能否传过条件接口，再测试小型 encoder 能否替代手工特征。

| 阶段 | 实际执行 | 作业号 | 状态 | 运行代码版本 |
| --- | --- | --- | --- | --- |
| A | 手工特征、最终条件几何、U-Net 单步响应 | 114105 | COMPLETED / 0:0 | `7eac5ca` |
| B | 三个种子的 encoder 训练、留出集特征与最终条件评估 | 114106 | COMPLETED / 0:0 | `a2359b6` |
| B 补充审计 | 从预测频谱主峰读取周期，核查频谱失真 | 114107 | COMPLETED / 0:0 | `c2f1398` |
| C / D | 未开展 | — | B 未通过，停止 | — |

代码位于 `codex/e18` 分支，服务器通过 GitHub `git pull` 同步。结果根目录：`/share/home/u2515283058/Mymodel/e19/`。

## 2. 共用数据与指标口径

继续使用 E18.1 的干净条纹集 `e18_1/clean/`。训练集 96 张，验证集 32 张；各模板均有原图和 rot90，四种配色在横/竖方向上平衡，同一旋转对的颜色像素直方图相同。

| 划分 | 每幅图周期数 | 相位 |
| --- | --- | --- |
| 训练 | 3、5、8、12 | 0、0.23、0.47 |
| 验证 | 4、10 | 0.11、0.36 |

频率、相位均不重叠；验证集不参与 token 幅度标定或 encoder 训练。验证集含四个频率/相位组，每组八张图。此划分验证受控条纹泛化，不代表真实面料来源独立验证。

- **方向 margin**：将 token 的通道均值与标准差拼接并 L2 归一化，计算“同向非自身样本的平均余弦相似度 − 异向样本的平均余弦相似度”。同时报告全验证集和各频率/相位组。
- **final conditioning**：BF 与 pattern tokens 拼接后，经过冻结 TCPM 的完整 texture 条件；不是仅测新增 pattern 段。
- **本轮 `D_rot`**：`||T_matched − T_rot90|| / ||T_matched||`，以全零 tokens 为零点。与使用 zero-image 输出作分母参照的历史指标不直接等价。
- **本轮 `R_rot`**：实际 texture 注入残差的 `RMS(matched − rot90) / RMS(matched − zero_tokens)`，只对启用 texture 的层求平均；GAM 中停用 texture 的 semantic 层不计入。

U-Net 评估使用原固定 32 个 target/sketch 样本，固定文本、目标 latent、噪声和 timestep，取 `181、481、781` 三个时间点。一次将 32 张留出条纹作为参考，另一次使用这 32 个样本原有真实参考，两套结果分别汇总。前者的条纹不是目标服装的真实 matched 纹理，因此该评估只测参考响应，不测 matched denoising 优势。没有运行完整扩散采样。

## 3. E19-A：手工 Pattern Feature 正向对照

### 实现与对照

```text
参考图 → 灰度 128×128 → 四个 64×64 窗口
       → 每窗口 36 维：4 维梯度结构张量 + 32 维 FFT 横/纵频谱
       → 固定正交线性映射 36→768 → 4 个 pattern tokens

当前 BF → 16 个 appearance tokens
concat(appearance, pattern) → 冻结 TCPM → 原 GAM texture attention
```

不训练新参数。正交映射使用种子 42/43/44 检查几何稳定性，预先指定 42 用于 U-Net 评估。每个 pattern token 的 RMS 按训练集 BF token 的平均 RMS 标定，没有扫幅度。attention processor 的 texture token 数随 16/20 个条件更新，避免新增 tokens 被误当作文本。

保留四组对照：E5/GAM 原始模型；**BF-only 为 E18.2 B2 检查点**；BF + 手工 pattern；BF + 恒定 pattern。恒定分支使用训练集平均手工特征归一化后映射，保留新增 token 数与幅度，但不随参考方向变化。BF、CLIP、TCPM、U-Net、sketch reference U-Net 的权重哈希审计全部通过。

### 留出方向几何

| 表示 / 条件 | 全验证集方向 margin |
| --- | ---: |
| 手工 feature | 1.2592 |
| E5/GAM final | −0.0006 |
| BF-only final | −0.0002 |
| 恒定 pattern 对照 final | −0.0001 |
| 手工 pattern tokens，映射种子 42 | 1.2591 |
| 手工分支 final，映射种子 42 | 0.1512 |
| 手工分支 final，映射种子 43 | 0.1455 |
| 手工分支 final，映射种子 44 | 0.1341 |

三个映射种子的全部留出频率/相位组 final margin 均为正且超过 0.01；没有只在某个映射或某个模板上成立。

### 最终 token 与 U-Net 旋转响应

| 对照组 | 留出条纹 `D_rot` | 留出条纹 `R_rot` | 原真实参考 `D_rot` | 原真实参考 `R_rot` |
| --- | ---: | ---: | ---: | ---: |
| E5/GAM | 0.086424 | 0.088211 | 0.035921 | 0.033089 |
| BF-only | 0.020946 | 0.032266 | 0.009886 | 0.013318 |
| BF + 手工 pattern | 0.731356 | 0.473750 | 0.328251 | 0.189110 |
| BF + 恒定 pattern | 0.018729 | 0.031824 | 0.008842 | 0.013013 |

留出条纹上，手工分支相对 BF-only 和恒定分支的样本配对 bootstrap 差值 95% 置信区间下界均大于 0。该区间反映固定参考安排下的 target 样本变异，不是跨真实面料来源的置信区间。真实参考表仅报告旋转敏感度，没有由此验证真实参考的方向几何。

**A 通过本轮预设门槛**：所有映射、所有留出组 final margin > 0.01；全验证集 margin 比 BF-only 提高 > 0.01；手工分支 `R_rot` 对两组对照的配对差值显著为正，且均值超过 BF-only 的 1.2 倍。

该结果说明新增分支的方向信息能够通过拼接及冻结 TCPM，并影响原 GAM 注入路径。它没有证明原 fused 已被修复，也没有证明 U-Net 把这种数值响应转化成了正确纹样。

## 4. E19-B：可学习 Pattern Encoder

### 训练设置

小型灰度 CNN 共 **204,868 个参数**，对四个局部窗口共享卷积编码，输出每窗口 36 维归一化特征。单一训练目标为对 A 的梯度/FFT 特征做 MSE 蒸馏；A 的 token 映射保持冻结，BF、TCPM、CLIP 和生成器也保持冻结。

三个训练种子分别为 42/43/44，均训练 500 步，batch size 16，AdamW 学习率 0.001、weight decay 0.0001。训练数据仅为上述 96 张干净条纹，没有加入真实面料、格纹或波点。冻结模块的审计全部通过。

### 方向成功，频谱泛化不足

| 指标 | 种子 42 | 种子 43 | 种子 44 |
| --- | ---: | ---: | ---: |
| 训练集频谱余弦相似度 | 0.998766 | 0.998970 | 0.998935 |
| 留出集完整特征余弦相似度 | 0.882496 | 0.879148 | 0.881030 |
| 留出集频谱余弦相似度 | 0.638372 | 0.631450 | 0.640155 |
| 留出局部方向直接读出准确率 | 100% | 100% | 100% |
| 留出 final 方向 margin | 0.151741 | 0.151318 | 0.150969 |
| 留出周期主峰准确率 | 37.5% | 0% | 0% |
| 留出周期数 MAE | 4.50 | 6.75 | 8.50 |

“局部方向直接读出”按预测结构张量的横/纵梯度能量大小判断，不是另训 linear probe。三个种子的各留出频率/相位组 final margin 均超过 0.01；因此不能说 B 的方向任务失败。

补充周期审计对每幅图四个局部窗口的对应方向频谱求平均，读取主峰 bin；窗口覆盖半幅图，故全图周期数取 `2 × peak_bin`。本次留出周期为 4/10，均能落在该离散频率网格。**手工 teacher 的同口径周期准确率为 100%，MAE 为 0**；可学习分支明显下降，说明频谱相似度下降也对应实际主峰周期错误。该审计检验当前预测频谱的直接读出，不排除其他表示或解码器存在额外频率线索。

本轮 B 预设要求：三个种子的留出完整特征余弦相似度 ≥ 0.90、频谱余弦相似度 ≥ 0.80；各组 final margin > 0.01，且全体验证 margin 保留 A 主映射的至少 50%。只有这些条件满足后才测 B 的 U-Net 响应。

**方向几何部分通过，特征保真部分未通过，B 整体停止。**训练集频谱拟合接近完美，但未见频率/相位上的泛化不足，下一步应优先处理 pattern representation 的周期学习与留出泛化，尚无依据因此修改 U-Net。

注意原始 `b/report.json` 中 `geometry_pass=false` 是包含特征保真的组合门槛，不能解释为 final 方向 margin 没提高；`response=null`、`response_pass=false` 表示 B 的 U-Net 响应评估未执行，不能解释成 U-Net 实测失败。补充 `comparison.json` 已明确记录 `b_direction_geometry_pass=true`。

## 5. 已回答与尚未回答

- **已回答**：当前拼接 + TCPM + GAM 注入接口能保留显式方向几何并产生旋转响应；一个小型 CNN 能学到并传递干净条纹的方向信息。
- **当前障碍**：这版 CNN 对手工频谱目标的训练集拟合很好，但留出频率/相位上的周期预测不稳定。
- **尚未验证**：真实面料方向泛化、同色异纹和异色异纹的属性分工、matched 相对 wrong reference 的 denoising 优势、生成方向跟随、pattern identity 跟随、主色保持、Sketch IoU、Edge F1 和 background leakage。

因此目前属于 **A 通过、B 部分能力形成但整体未通过**，不属于 A/B/C/D 全链路成功。下一轮应先解决小型 pattern encoder 的频率泛化，再重测 B 的 U-Net 响应，按门槛推进 C/D。

## 6. 复核入口

以下结果路径均相对于服务器 `/share/home/u2515283058/Mymodel/`：

| 内容 | 路径 |
| --- | --- |
| 总览与停止状态 | `e19/comparison.json` |
| A 协议、几何和总报告 | `e19/a/protocol.json`、`geometry.json`、`report.json` |
| A 留出条纹 / 真实参考逐样本响应 | `e19/a/response.json`、`response_real.json` |
| A 固定映射与幅度标定 | `e19/a/pattern_branch.pt` |
| B 协议、训练与几何 | `e19/b/protocol.json`、`training.json`、`geometry.json` |
| B 总报告及周期审计 | `e19/b/report.json`、`frequency_audit.json` |
| B 三个 encoder | `e19/b/encoder_42.pt`、`encoder_43.pt`、`encoder_44.pt` |

实现入口：[手工分支](../models/explicit_pattern.py)、[可学习 encoder](../models/learned_pattern.py)、[A 评估](../tools/e19_handcrafted.py)、[B 训练评估](../tools/e19_learned.py)、[周期审计](../tools/e19_frequency_audit.py)。对应提交脚本位于 `submit/e19_a_handcrafted.sh`、`submit/e19_b_learned.sh`、`submit/e19_frequency_audit.sh`。
