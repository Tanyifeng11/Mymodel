# E25：显式空间纹样可控性诊断结果

## 结论

在受控合成条纹上，已知正确的空间纹样经过 E5 VAE 和冻结 E5 diffusion refinement 后，仍能稳定跟随参考图的 90° 旋转。低噪声时，Texture Path OFF/ON 的 32 案例成对翻转率均为 **100%**；高噪声时分别为 **95.31%** 和 **89.06%**。因此，现有证据支持把下一阶段的纹样几何问题定位在 **reference → target-aligned pattern field** 的建立，而非认为 E5 完全无法保留已对齐的条纹。

**E25 的完整 Gate 未通过。**四个确认组的背景 leakage 与 Sketch IoU 满足预定安全界限，但 Edge F1 相对 E5 均下降，`strong_pass_groups=[]`。`decision_summary.json` 中的 `next_route=explicit_pattern_correspondence` 是基于纹样方向机制的路线判断，不表示已有可用的安全纹样控制方法，也不表示真实面料、任意 motif 或任意周期控制已解决。

Slurm 作业 `114790` 正常完成，退出码 `0:0`，耗时 `20:51`。实验运行提交为 `9099162`，训练步数为 **0**，冻结权重哈希检查通过。全部原始数据、逐图 PNG/JSON 和报告位于服务器 `/share/home/u2515283058/Mymodel/output_eval/e25/`。

## 实验内容与协议

- 沿用 E24 的 32 对验证参考，频率为 6、7、9、11、13、15；每对含 original/rot90。固定两张 sketch、同一 prompt、颜色和 garment mask，种子为 42/43，50 步 DDIM、CFG 7、OpenCV mask 后端。统计先按参考案例汇总，再以案例为单位 bootstrap；同一案例的旋转与种子不被当作独立样本。
- **A，oracle scaffold：**用参考 metadata 和原合成 renderer 逐像素重建条纹，把它按目标衣身二值 mask 合成到白底。校验渲染图与参考像素相等、mask 外全白，并测方向、成对翻转和六候选频率的闭集 FFT 周期读数。
- **B，VAE audit：**用冻结 E5 VAE 的 posterior mean 编码 scaffold，再解码和测量。A 未通过则停止；B 未通过则不进入 C。
- **C，冻结 E5 refinement：**从 scaffold latent 加噪，按 strength `0.15/0.35/0.55` 运行剩余 DDIM 步。Texture Path OFF 关闭参考 texture token，ON 使用匹配参考的 E5 token；同一 strength、案例、seed、旋转变体的 OFF/ON 使用相同噪声和初始 latent 哈希。先跑六组各 8 案例的 pilot，再按预设选择规则只扩四组到 32 案例。
- A、B 各保存 64 张图；C 保存 576 张图及 576 份逐图 JSON，其中 pilot 为 `8×2×2×6=192` 张，四组确认新增 `24×2×2×4=384` 张。

## Gate 与实际结果

A 预定方向、成对翻转、闭集周期准确率都必须为 100%。B 预定方向和成对翻转不低于 95%，闭集周期不低于 90%。两阶段实际三项均为 **100%**，故按顺序进入 C；VAE 平均方向误差为 **0.00485°**。这里的周期读数使用六个已知候选频率与目标方向轴，不能解释为独立的任意周期生成能力。

C 的方向 Gate 要求全样本 paired flip ≥ 75%，且案例级 bootstrap 95% CI 下界 > 50%。安全 Gate 相对 E5 要求 leakage 差值 CI 上界 ≤ `+0.02`，Sketch IoU 与 Edge F1 差值 CI 下界均 ≥ `−0.02`。六组 pilot 的 OFF low/mid/high 翻转率为 **100% / 100% / 93.75%**，ON 为 **100% / 100% / 87.50%**；因此确认 low/high 的 ON/OFF 配对，mid 只保留 8 案例 pilot，不能当作 32 案例确认。

| 32 案例确认组 | Paired flip [95% CI] | 方向读出覆盖率 | 平均方向误差 | ΔEdge F1，均值；CI 下界 | 方向 Gate | 完整 Gate |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| OFF，0.15 | 100% [100%, 100%] | 100% | 0.0247° | −0.1255；−0.1465 | 通过 | **失败** |
| ON，0.15 | 100% [100%, 100%] | 100% | 0.0249° | −0.1330；−0.1542 | 通过 | **失败** |
| OFF，0.55 | 95.31% [89.06%, 100%] | 95.31% | 0.2790° | −0.0534；−0.0866 | 通过 | **失败** |
| ON，0.55 | 89.06% [81.25%, 95.31%] | 89.06% | 1.0409° | −0.0366；−0.0686 | 通过 | **失败** |

四个确认组的闭集周期准确率均为 100%。Leakage 相对 E5 的均值变化为 `−0.00590` 至 `−0.00317`，Sketch IoU 为 `+0.07029` 至 `+0.09505`；它们各自的 CI 也满足安全条件。**Edge F1 是四组共同的正式失败项**，不能因为方向、周期与其他安全指标较好而跳过。

## 图像核对与指标解释

用服务器终端的 32×32 真彩预览抽查了 `c00_s42` 的 low ON 原图/rot90、oracle 与 high ON，以及 `c01_s42` 的 high ON 原图/rot90。两组中横竖条纹的切换与统计一致。低噪声图更接近简单的条纹 scaffold，缺少 E5 基线中的部分材质层次；高噪声图仍可读出方向，但局部肩部和边缘有变化。此预览只作定性核对，不能替代全组逐图统计或高分辨率视觉评测。

探索性轮廓核对显示：E5 baseline 的 `contour_f1` 均值约 **0.489**，OFF/ON low 分别约 **0.967/0.959**，OFF/ON high 约 **0.803/0.735**。相反，连按精确 mask 生成的无噪 oracle 图，按既有全图边缘对 sketch 计算的 Edge F1 也只有 **0.283**。结合 Edge F1 的定义，衣身内部大量规则条纹边缘会降低对 sketch 边缘的精确率，因而它可能把纹样边缘当作结构错误。这是对指标差异的解释性推断；`contour_f1` 是事后诊断，**不修改预注册安全 Gate 的失败判定**。

## 后续位置与原始证据

E25 是冻结模型的机制上界实验。ON 与 OFF 都保留了显式方向，低噪声达到 100% 翻转，高噪声仍过方向 Gate；目前没有“Texture Path ON 单独破坏纹样”或“VAE 已丢失方向”的证据。下一阶段可研究从参考纹样建立目标服装坐标中的空间对应，同时在开展新模型验证前预先区分**服装轮廓边缘**与**衣身内部纹样边缘**，继续保留 leakage、Sketch IoU 和完整图像检查。E25 到此停止，不在本阶段训练新模块。

原始协议与权重哈希：`output_eval/e25/protocol.json`；A/B 报告：`A_oracle/report.json`、`B_vae/report.json`；C 的六组 pilot 与四组确认：`metrics/pilot.json`、`metrics/confirmation.json` 和各 `*_8.json`/`*_32.json`；最终判定：`decision_summary.json`。对应实现见 `models/pattern_scaffold.py`、`tools/e25_spatial_diagnosis.py` 与 `submit/e25_spatial_diagnosis.sh`。
