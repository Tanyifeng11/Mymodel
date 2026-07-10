# 消融实验评估指南：每个实验应该产生什么数字

## 快速使用

### 1. 训练每个 checkpoint 后跑 fixed benchmark（已有工具，已升级）

```bash
python tools/run_fixed_benchmark.py \
    --dataset_json data/test.json \
    --data_root data/ \
    --gam_ckpt checkpoints/joint_model.pt \
    --texture_ckpt checkpoints/texture_adapter.pt \
    --modes token,spatial,hybrid \
    --output_dir eval_outputs \
    --run_name exp_e0_token_baseline
```

每张生成图会自动调用 `evaluate_full()`，得到 **TCF+TPF+Leak+Struct** 全部指标。

### 2. 汇总所有实验生成消融表格

```bash
python -m eval.ablation_report \
    --experiments_dir eval_outputs/ \
    --real_images_dir data/test_real_images/ \
    --output_dir eval_outputs/report
```

输出文件：
- `comprehensive_table.md` — **论文用综合表格**（可直接截图放进论文）
- `ablation_tables.md` — 按类别分组的详细表格
- `ablation_results.csv` — Excel可用
- `ablation_results.json` — 供后续分析
- `radar_chart.html` — 可视化雷达图

---

## 论文消融表格模板

运行 `python -m eval.ablation_report` 后，你的 `comprehensive_table.md` 会是这样的格式：

```
| Experiment | FID ↓ | CLIP-I ↑ | TCF-LAB ↓ | TCF-HSV ↓ | TPF-Patch ↑ | TPF-Gram ↓ | LR-Colored ↓ | LR-Sat ↓ | BAS ↓ | Edge F1 ↑ | IoU ↑ | Edge L1 ↓ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E0_token_baseline | 28.43 | 0.8214 | 18.32 | 0.2412 | 0.6734 | 1.4523 | 0.0421 | 0.1124 | 3.21 | 0.6234 | 0.7812 | 4.32 |
| E1_token_grouped  | 27.98 | 0.8301 | 14.21 | 0.1856 | 0.7102 | 1.2134 | 0.0512 | 0.1245 | 3.18 | 0.6312 | 0.7891 | 4.21 |
| E2_+align_loss    | 28.12 | 0.8287 | 13.89 | 0.1801 | 0.7234 | 1.1856 | 0.0187 | 0.0823 | 2.87 | 0.6356 | 0.7934 | 4.18 |
| E3_+spatial       | 29.51 | 0.8156 | 12.45 | 0.1623 | 0.7589 | 0.9876 | 0.0267 | 0.0956 | 3.45 | 0.6102 | 0.7654 | 4.89 |
```

### 每个数字代表什么（论文里怎么解释）

| 列名 | 含义 | 好的方向 | 对应你的研究问题 |
|------|------|---------|----------------|
| **FID ↓** | Fréchet Inception Distance | 越低越好 | 生成质量没有退化 |
| **CLIP-I ↑** | CLIP Image相似度 | 越高越好 | 语义一致性 |
| **TCF-LAB ↓** | 纹理颜色保真度 | 越低越好 | 生成衣服颜色是否跟随texture图 |
| **TCF-HSV ↓** | 纹理色调分布距离 | 越低越好 | 色调是否匹配texture |
| **TPF-Patch ↑** | 纹理图案保真度 | 越高越好 | 局部纹理pattern是否匹配 |
| **TPF-Gram ↓** | VGG Gram纹理距离 | 越低越好 | 整体纹理风格是否匹配 |
| **LR-Colored ↓** | 纹理溢出比例 | 越低越好 | 背景被纹理"染色"的程度 |
| **LR-Sat ↓** | 背景平均饱和度 | 越低越好 | 背景是否出现不应有的颜色 |
| **BAS ↓** | 边界伪影评分 | 越低越好 | 服装边界是否清晰 |
| **Edge F1 ↑** | 草图边缘一致性 | 越高越好 | 生成图是否保持草图结构 |
| **IoU ↑** | 前景重叠度 | 越高越好 | 服装轮廓是否匹配草图 |
| **Edge L1 ↓** | 边缘差异 | 越低越好 | 结构偏差程度 |

---

## 每个消融实验应该"赢"在哪些指标上

### E0 → E1（加Ti-MGD分组）

分组的目的：让纹理token在高分辨率层不受文本token稀释。

**预期改善（数字应该变好）**：
- **TCF-LAB ↓↓** — 颜色更贴近texture（核心预期）
- **TCF-HSV ↓↓** — 色调分布更匹配
- **TPF-Patch ↑** — 局部纹理相似度提升
- **CLIP-I →** — 不退化

**应该不变或轻微波动的**：
- **FID →** — 生成质量不受影响
- **Edge F1 →** — 结构保持

**可能轻微恶化（需关注但不一定致命）**：
- **LR-Colored 可能轻微↑** — 纹理强了可能溢出多一点点

### E1 → E2（加对齐loss + 三区域约束）

对齐loss的目的：显式惩罚纹理溢出和不匹配。

**预期改善**：
- **LR-Colored ↓↓** — 溢出大幅减少（核心预期）
- **LR-Sat ↓↓** — 背景颜色更干净
- **BAS ↓** — 边界更清晰
- **Edge L1 → 或 ↓** — 结构可能更稳定

**应该不变**：
- **TCF-LAB →** — 对齐loss不应削弱纹理控制
- **TPF-Patch → 或 ↑** — 三区域约束让纹理正确集中在服装内部

### E2 → E3（加spatial分支）

spatial的目的：高频细节增强。

**预期改善**：
- **TPF-Patch ↑↑** — 这是spatial最该提升的指标（核心预期）
- **TPF-Gram ↓↓** — Gram纹理匹配更好
- **TCF-LAB → 或 ↓** — 颜色保持

**风险指标（需要严密监控）**：
- **LR-Colored** — 如果spatial导致溢出，这个会升高
- **FID** — 如果spatial破坏生成分布，FID会恶化
- **Edge F1** — 如果spatial干扰结构

**判断spatial是否值得加的关键**：如果 TPF-Patch 有 ≥0.03 的提升，同时 LR-Colored 没有 ≥0.01 的恶化，spatial 就是有价值的。否则没必要。

---

## 论文中怎么呈现

### 主表（Table 1）：各实验的完整指标对比

直接复制 `comprehensive_table.md` 的内容到论文，横排所有实验、竖排所有指标。

### 纹理敏感度分析（单独一节）

用 `tools/eval_texture_sensitivity.py` 的结果：

```
同一草图 + 3种不同纹理 → 生成3张图 → 计算两两CLIP距离

| 实验 | TSS-CLIP ↑ | 说明 |
|------|-----------|------|
| E0   | 0.034     | 换纹理后输出几乎不变 → 纹理控制弱 |
| E1   | 0.089     | 换纹理后输出差异明显 → 分组有效 |
| E2   | 0.092     | 对齐loss不削弱纹理敏感度 |
```

### 纹理依赖度分析

用 `tools/analyze_texture_reliance.py` 的结果：

```
同一草图 + 真实纹理 vs 纯灰纹理 → 输出差异越大 = 越依赖纹理

| 实验 | 真实vs纯灰 LPIPS ↑ | 说明 |
|------|-------------------|------|
| E0   | 0.12              | 纹理影响有限 |
| E1   | 0.23              | 分组后纹理影响翻倍 |
| E2   | 0.25              | 对齐loss进一步增强 |
```

### 定性展示（Figure）

- 同一草图 + 不同纹理的实验结果网格图（你已有的 `run_fixed_benchmark` 会自动生成 grid）
- 局部放大对比：E0 vs E2 在纹理细节区域（条纹方向、格纹密度）的 crop

---

## 离线评估完整流程（推荐）

```bash
# === 步骤1：每个checkpoint生成评估图像 ===
for ckpt in checkpoint-2000 checkpoint-4000 checkpoint-6000; do
    python tools/run_fixed_benchmark.py \
        --dataset_json data/val.json \
        --data_root data/ \
        --gam_ckpt output_dir/$ckpt/joint_model.pt \
        --texture_ckpt checkpoints/texture_adapter.pt \
        --modes token,spatial,hybrid \
        --num_samples 100 \
        --output_dir eval_outputs/per_checkpoint \
        --run_name $ckpt
done

# === 步骤2：汇总所有checkpoint生成消融报告 ===    
python -m eval.ablation_report \
    --experiments_dir eval_outputs/per_checkpoint \
    --real_images_dir data/test_real_images/ \
    --output_dir eval_outputs/final_report

# === 步骤3：纹理敏感度 ===
python tools/eval_texture_sensitivity.py \
    --GAM_model_ckpt output_dir/checkpoint-6000/joint_model.pt \
    --texture_ckpt checkpoints/texture_adapter.pt \
    --sketch_path data/sketch_samples/sketch001.png \
    --prompt "a blue denim jacket" \
    --texture_paths data/textures/denim.png data/textures/stripe.png data/textures/plaid.png

# === 步骤4：纹理依赖度 ===
python tools/analyze_texture_reliance.py \
    --gam_ckpt output_dir/checkpoint-6000/joint_model.pt \
    --texture_ckpt checkpoints/texture_adapter.pt \
    --sketch_path data/sketch_samples/sketch001.png \
    --real_texture_path data/textures/denim.png \
    --prompt "a blue denim jacket" \
    --modes token,spatial,hybrid
```

---

## 依赖安装

```bash
pip install scipy torchmetrics scikit-image --break-system-packages
# CLIP-I 需要 transformers
pip install transformers --break-system-packages
# 如果已有 torch torchvision（你的环境肯定有），FID使用的 InceptionV3 在 torchvision 里自带
```
