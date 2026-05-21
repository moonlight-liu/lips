# LipFD 轻量化实验结果

> **目的**：记录所有实验的详细数据，便于对比和分析

---

## 实验环境

**硬件配置**：
- GPU: 4 × NVIDIA RTX 3090 (24GB)，本轮优先使用 GPU 3（运行前设置 `CUDA_VISIBLE_DEVICES=3`）
- CPU: Intel Xeon Gold 6133 @ 2.50GHz，2 sockets × 20 cores × 2 threads = 80 threads
- 内存: 125 GiB

**软件环境**：
- PyTorch: 2.1.0+cu121
- CUDA: 可用（PyTorch CUDA 12.1）
- Python: 3.10.12 (`/root/anaconda3/envs/lips/bin/python`)

**数据集**：
- 训练集: AVLips (待确认样本数)
- 验证集: AVLips validation set
- 测试集: [待填写]

---

## 实验 1: 原始模型性能测量

**日期**：2026-05-15

**模型配置**：
- CLIP Encoder: ViT-L/14
- Region Awareness: ResNet50
- 输入: 5 帧 × 3 区域

### 1.1 模型规模

| 指标 | 数值 | 备注 |
|------|------|------|
| 总参数量 (M) | 451.13 | 包含 CLIP、Region Awareness、下采样卷积等所有层 |
| CLIP 参数量 (M) | 427.62 | ViT-L/14，占总参数量 94.8% |
| ResNet50 参数量 (M) | 23.51 | Region Awareness，占总参数量 5.2% |
| 其他参数量 (M) | 0.000228 | Conv1 下采样层约 0.23K，几乎可忽略 |
| 模型文件大小 (MB) | 979.05 | 临时保存 `state_dict` 测得 |

### 1.2 计算复杂度

| 指标 | 数值 | 备注 |
|------|------|------|
| FLOPs (G) | 61.975 | 仅 Region Awareness 分支，单样本；CLIP Encoder 未计入 |
| MACs (G) | 待精确测量 | 当前脚本只输出 FLOPs，后续统一补测 |

### 1.3 推理性能

**GPU 推理 (RTX 3090)**：
| Batch Size | 推理时间 (ms) | FPS | 备注 |
|------------|---------------|-----|------|
| 1 | 106.38 | 9.4 | 单样本，RTX 3090，临时代码将模型整体放到 GPU 后测得 |
| 8 | [待测量] | [待计算] | 小批量 |
| 16 | [待测量] | [待计算] | 中批量 |

**CPU 推理**：
| Batch Size | 推理时间 (ms) | FPS | 备注 |
|------------|---------------|-----|------|
| 1 | [待测量] | [待计算] | 单样本 |

**阶段性分析**：
- 参数角度：CLIP ViT-L/14 是模型体积的绝对瓶颈，占 94.8% 参数量；单纯替换或压缩它，理论上最容易带来模型大小下降。
- 计算角度：Region Awareness 的参数占比不高，但由于 15 个 crop 重复走 ResNet50，单样本区域分支 FLOPs 已达 61.975G；它是推理延迟的重要来源。
- 实时性角度：当前 106.38 ms/样本约等于 9.4 FPS，尚不能满足 30 FPS 实时检测，轻量化目标应至少达到 3.2× 加速。
- 脚本注意：现有 `measure_model.py` / `benchmark_inference.py` 在 GPU 测试时需要确保模型整体 `.to(device)`，否则会出现 CPU/GPU tensor 类型不一致。

### 1.4 准确率

**验证集性能**：
| 指标 | 数值 | 备注 |
|------|------|------|
| Accuracy | [待测量] | 总体准确率 |
| Precision | [待测量] | 精确率 |
| Recall | [待测量] | 召回率 |
| F1-Score | [待测量] | F1 分数 |
| AUC | [待测量] | ROC 曲线下面积 |
| AP | [待测量] | Average Precision |

**混淆矩阵**：
```
[待填写]
```

### 1.5 测量脚本

```bash
# 参数量和 FLOPs（后续运行统一优先使用物理 GPU 3）
CUDA_VISIBLE_DEVICES=3 python lightweight/scripts/measure_model.py --model original

# 推理时间
CUDA_VISIBLE_DEVICES=3 python lightweight/scripts/benchmark_inference.py --model original

# 准确率
CUDA_VISIBLE_DEVICES=3 python validate.py --ckpt ./checkpoints/ckpt.pth --gpu 0
```

---

## 实验 2: 轻量化模型（直接训练）

**日期**：[待填写]

**模型配置**：
- CLIP Encoder: ViT-B/16
- Region Awareness: ResNet34
- 输入: 5 帧 × 3 区域

### 2.1 模型规模

| 指标 | 原始模型 | 轻量化模型 | 减少比例 |
|------|----------|------------|----------|
| 总参数量 (M) | [实验1] | [待测量] | [待计算] |
| CLIP 参数量 (M) | [实验1] | [待测量] | [待计算] |
| ResNet 参数量 (M) | [实验1] | [待测量] | [待计算] |
| 模型文件大小 (MB) | [实验1] | [待测量] | [待计算] |

### 2.2 计算复杂度

| 指标 | 原始模型 | 轻量化模型 | 减少比例 |
|------|----------|------------|----------|
| FLOPs (G) | [实验1] | [待测量] | [待计算] |
| MACs (G) | [实验1] | [待测量] | [待计算] |

### 2.3 推理性能

**GPU 推理 (RTX 3090)**：
| Batch Size | 原始模型 (ms) | 轻量化模型 (ms) | 加速比 |
|------------|---------------|-----------------|--------|
| 1 | [实验1] | [待测量] | [待计算] |
| 8 | [实验1] | [待测量] | [待计算] |
| 16 | [实验1] | [待测量] | [待计算] |

### 2.4 训练过程

**训练配置**：
```python
batch_size = 16
learning_rate = 1e-4
optimizer = Adam
epochs = 30
gpu_ids = [0, 1]
```

**训练曲线**：
```
Epoch | Train Loss | Val Acc | Val AP | Time (s)
------|------------|---------|--------|----------
1     | [待记录]   | [待记录] | [待记录] | [待记录]
2     | [待记录]   | [待记录] | [待记录] | [待记录]
...
```

### 2.5 最终性能

**验证集性能**：
| 指标 | 原始模型 | 轻量化模型 | 差异 |
|------|----------|------------|------|
| Accuracy | [实验1] | [待测量] | [待计算] |
| AP | [实验1] | [待测量] | [待计算] |
| F1-Score | [实验1] | [待测量] | [待计算] |

---

## 实验 3: 轻量化模型（知识蒸馏）

**日期**：[待填写]

**模型配置**：
- Student: ViT-B/16 + ResNet34
- Teacher: ViT-L/14 + ResNet50
- 蒸馏方法: 软标签 + 特征蒸馏

### 3.1 蒸馏配置

```python
# 损失函数权重
alpha = 0.5  # 硬标签权重
beta = 0.3   # 软标签权重
gamma = 0.2  # 特征蒸馏权重
temperature = 4.0  # 蒸馏温度

# 训练配置
batch_size = 16
learning_rate = 1e-4
epochs = 40
```

### 3.2 训练过程

**训练曲线**：
```
Epoch | Total Loss | KD Loss | Hard Loss | Val Acc | Val AP
------|------------|---------|-----------|---------|--------
1     | [待记录]   | [待记录] | [待记录]  | [待记录] | [待记录]
2     | [待记录]   | [待记录] | [待记录]  | [待记录] | [待记录]
...
```

### 3.3 最终性能对比

| 指标 | 原始模型 | 轻量化(直接) | 轻量化(蒸馏) | 蒸馏提升 |
|------|----------|--------------|--------------|----------|
| Accuracy | [实验1] | [实验2] | [待测量] | [待计算] |
| AP | [实验1] | [实验2] | [待测量] | [待计算] |
| F1-Score | [实验1] | [实验2] | [待测量] | [待计算] |

---

## 实验总结

### 综合对比表

| 模型 | 参数量(M) | FLOPs(G) | GPU延迟(ms) | Accuracy(%) | 综合评分 |
|------|-----------|----------|-------------|-------------|----------|
| 原始模型 | [实验1] | [实验1] | [实验1] | [实验1] | - |
| 轻量化(直接) | [实验2] | [实验2] | [实验2] | [实验2] | [待计算] |
| 轻量化(蒸馏) | [实验3] | [实验3] | [实验3] | [实验3] | [待计算] |

**综合评分计算**：
```
score = (参数减少比例 × 0.3) + (速度提升比例 × 0.3) + (准确率保持比例 × 0.4)
```

### 关键发现

1. **参数量减少**：
   - 原始模型总参数量为 451.13M，其中 CLIP ViT-L/14 为 427.62M，占 94.8%。这说明模型体积瓶颈高度集中在 CLIP Encoder，后续轻量化应优先考虑更小的视觉编码器、冻结/缓存全局特征、低秩适配或蒸馏等方向。

2. **推理速度提升**：
   - 原始模型 batch size 1 的单样本延迟为 106.38 ms，仅约 9.4 FPS，距离 30 FPS 实时目标（约 33 ms/样本）还有明显差距。Region Awareness 分支虽然只有 23.51M 参数，但需要对 3 个尺度 × 5 个位置的 15 个 crop 重复运行 ResNet50，单该分支即达到 61.975G FLOPs，因此后续也要优化区域分支的重复计算。

3. **准确率变化**：
   - [待分析]

4. **知识蒸馏效果**：
   - [待分析]

### 结论

[待填写]

---

## 可视化结果

### 训练曲线对比

**Loss 曲线**：
```
[待生成图表]
```

**Accuracy 曲线**：
```
[待生成图表]
```

### 性能对比图

**参数量对比**：
```
[待生成柱状图]
```

**推理时间对比**：
```
[待生成柱状图]
```

**准确率对比**：
```
[待生成柱状图]
```

---

## 附录：详细日志

### 实验 1 详细日志
```
[待填写]
```

### 实验 2 详细日志
```
[待填写]
```

### 实验 3 详细日志
```
[待填写]
```
---

## 2026-05-21 实验记录：官方预处理训练集上的 ResNet18 Region Awareness

### 实验背景

此前旧 `datasets/AVLips` 由偏离官方语义的 `preprocess.py` 生成，其中最关键的问题是使用了：

```python
librosa.load(audio_file, sr=None)
```

这会保留原始音频采样率。由于当前 AVLips 原始数据中 real/fake 音频采样率不同，mel 频谱时间轴会发生类别相关偏移，导致训练集输入分布污染。

随后已按官方语义重新预处理训练集，核心要求是保留：

```python
librosa.load(audio_file)
cv2.COLOR_BGR2RGBA
plt.imsave / plt.imread
```

重新预处理后，官方权重在训练集 `datasets/AVLips` 上推理结果为：

| Model | Dataset | Acc | AP | FPR | FNR |
|---|---|---:|---:|---:|---:|
| Official LipFD, CLIP ViT-L/14 + ResNet50 | new `datasets/AVLips` | 0.8194 | 0.8451 | 0.0213 | 0.3092 |

该结果说明新训练集与官方权重基本匹配，旧训练集实验结果作废。

### 本轮轻量化设置

目标：只替换 Region Awareness 的 ResNet 分支，暂不改 CLIP 和数据输入逻辑。

| Component | Original | This Run |
|---|---|---|
| Global encoder | CLIP ViT-L/14 | CLIP ViT-L/14 |
| Global encoder weights | official ckpt | loaded from official ckpt |
| Conv1 | official ckpt | loaded from official ckpt |
| Region Awareness backbone | ResNet50 | ResNet18 |
| Trainable part | original full model | ResNet18 Region Awareness only |
| Frozen part | - | conv1 + CLIP encoder |
| Dataset | official-preprocessed `datasets/AVLips` | same |
| Validation | `datasets/val` | `datasets/val` |
| LR | - | 1e-4 |
| Batch size | - | 16 |
| Epochs | - | 5 |
| RA loss weight | 1.0 | 1.0 |

训练命令：

```bash
CUDA_VISIBLE_DEVICES=3 python lightweight/scripts/train_region_light.py \
  --backbone resnet18 \
  --teacher_ckpt ./checkpoints/ckpt.pth \
  --real_list_path ./datasets/AVLips/0_real \
  --fake_list_path ./datasets/AVLips/1_fake \
  --val_real_list_path ./datasets/val/0_real \
  --val_fake_list_path ./datasets/val/1_fake \
  --batch_size 16 \
  --num_workers 8 \
  --epochs 5 \
  --lr 1e-4 \
  --max_train_batches -1 \
  --max_val_batches -1 \
  --loss_freq 100 \
  --log_loss_every 20 \
  --name region_resnet18_official_preprocess_ra1
```

### 参数量与速度

| Model | Total Params | Backbone Params | Synthetic bs=1 Latency | Throughput |
|---|---:|---:|---:|---:|
| Original ResNet50 | 451.13M | 23.51M | 98.71 ms/sample | 10.13 samples/s |
| Region-light ResNet18 | 438.80M | 11.18M | 43.72 ms/sample | 22.87 samples/s |
| Region-light ResNet34 | 448.90M | 21.29M | 71.34 ms/sample | 14.02 samples/s |

阶段性结论：只替换 ResNet50 -> ResNet18 后，参数量下降不大，因为 CLIP 仍占绝大多数参数；但单样本纯模型速度明显提升，说明实时性瓶颈确实主要在 15 个 crop 重复通过 Region Awareness backbone。

### 训练结果

| Epoch | Train Loss | Cls Loss | RA Loss | Val Acc | Val AP | FPR | FNR |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 35.8367 | 0.5936 | 35.2869 | 0.7590 | 0.7876 | 0.2344 | 0.2478 |
| 2 | 35.2081 | 0.4379 | 34.7591 | 0.6192 | 0.8771 | 0.7481 | 0.0065 |
| 3 | 35.0682 | 0.3079 | 34.7475 | 0.7607 | 0.9314 | 0.4503 | 0.0243 |
| 4 | 34.9839 | 0.2380 | 34.7392 | **0.8764** | 0.9591 | **0.1966** | 0.0492 |
| 5 | 34.9332 | 0.1899 | 34.7392 | 0.8632 | **0.9695** | 0.2356 | **0.0362** |

Best checkpoint by validation accuracy:

```text
lightweight/results/checkpoints/region_resnet18_official_preprocess_ra1/best.pth
best epoch: 4
acc: 0.8763945977686436
ap: 0.9591492795404047
fpr: 0.1966259453170448
fnr: 0.04919976289270895
```

自动生成图像：

```text
lightweight/results/checkpoints/region_resnet18_official_preprocess_ra1/loss_curve.png
lightweight/results/checkpoints/region_resnet18_official_preprocess_ra1/val_metrics_curve.png
```

### 结果分析

1. 新训练集修复后，ResNet18 轻量化路线明显变好。旧训练集上 8 轮仅约 0.705 acc；新训练集上第 4 轮已达 0.876 acc。

2. `RA loss` 仍然基本不下降，稳定在约 34.739。当前 total loss 主要由 RA loss 主导，因此 total loss 不能直接反映分类学习质量。

3. `classification loss` 持续下降，从约 0.5936 降到约 0.1899，说明 ResNet18 分支确实在学习分类任务。

4. Epoch 5 的 AP 高于 epoch 4，但 acc 和 FPR 变差，说明模型排序能力继续增强，但默认阈值 0.5 不是必然最优。下一步应做阈值扫描，而不是只看 0.5 阈值下的 acc。

### 下一步计划

1. 对 epoch 4 best checkpoint 和 epoch 5 checkpoint 做阈值扫描，观察是否存在更优阈值，使 acc 提升且 FPR 降低。

2. 如果 ResNet18 在最佳阈值下 acc >= 0.90 且 FPR 明显下降，则保留 ResNet18 路线，进入速度/部署和蒸馏实验。

3. 如果 ResNet18 最佳阈值后仍 acc < 0.90 或 FPR 偏高，则训练 ResNet34 作为精度优先对照。

4. 暂时不修改 RA loss。只有当 ResNet18/ResNet34 对照完成后仍受限，再考虑 `ra_loss_weight`、RA warmup 或知识蒸馏。

---

## 2026-05-21 精确记录：ResNet18 Region-light 结果与 ResNet34 下一步

### ResNet18 参数量

本轮只替换 Region Awareness 分支：原始 ResNet50 -> ResNet18。CLIP ViT-L/14 与 `conv1` 仍加载官方权重并冻结，ResNet18 分支重新训练。

| Model | Total Params | Total Params (M) | CLIP Encoder Params (M) | Region Backbone Params (M) | Conv1 Params |
|---|---:|---:|---:|---:|---:|
| Region-light ResNet18 | 438,795,815 | 438.795815 | 427.616513 | 11.179074 | 228 |

结论：

- 原始模型总参数约 451.13M，ResNet18 版本为 438.80M，总参数减少约 12.33M，约减少 2.73%。
- 总参数下降不大是正常现象，因为 CLIP ViT-L/14 仍占主要部分：427.62M，约占 ResNet18 总参数的 97.45%。
- 只替换 Region Awareness 分支时，参数量收益主要体现在局部分支；如果后续目标是大幅压缩模型体积，需要进一步处理 CLIP encoder。

### ResNet18 合成测速

测速命令使用 `benchmark_region_light.py`，加载 ResNet18 最优权重 `best.pth`，输入为合成张量，用于观察模型前向计算速度，不包含真实图片读取和预处理 I/O。

| Model | Batch Size | Mean Latency / Batch | Latency / Sample | Throughput |
|---|---:|---:|---:|---:|
| Region-light ResNet18 | 1 | 45.25 ms | 45.25 ms/sample | 22.10 samples/s |
| Region-light ResNet18 | 4 | 70.64 ms | 17.66 ms/sample | 56.62 samples/s |
| Region-light ResNet18 | 8 | 128.14 ms | 16.02 ms/sample | 62.43 samples/s |
| Region-light ResNet18 | 16 | 241.92 ms | 15.12 ms/sample | 66.14 samples/s |

与原始 ResNet50 分支的早期合成测速对比：

| Model | Batch Size | Mean Latency / Batch | Latency / Sample | Throughput |
|---|---:|---:|---:|---:|
| Original ResNet50 | 1 | 98.71 ms | 98.71 ms/sample | 10.13 samples/s |
| Original ResNet50 | 4 | 132.62 ms | 33.16 ms/sample | 30.16 samples/s |
| Original ResNet50 | 8 | 190.14 ms | 23.77 ms/sample | 42.07 samples/s |
| Original ResNet50 | 16 | 357.64 ms | 22.35 ms/sample | 44.74 samples/s |
| Region-light ResNet18 | 1 | 45.25 ms | 45.25 ms/sample | 22.10 samples/s |
| Region-light ResNet18 | 4 | 70.64 ms | 17.66 ms/sample | 56.62 samples/s |
| Region-light ResNet18 | 8 | 128.14 ms | 16.02 ms/sample | 62.43 samples/s |
| Region-light ResNet18 | 16 | 241.92 ms | 15.12 ms/sample | 66.14 samples/s |

速度结论：

- Batch size 1 时，ResNet18 从 98.71 ms/sample 降到 45.25 ms/sample，约 2.18x 加速。
- Batch size 16 时，ResNet18 从 22.35 ms/sample 降到 15.12 ms/sample，约 1.48x 加速。
- 这说明 Region Awareness 分支确实是单样本实时性的重要瓶颈，替换 ResNet50 可以明显提速。

### ResNet18 验证集结果

官方阈值保持 `0.5`，即 `sigmoid(score) >= 0.5` 判为 fake，否则判为 real。

| Model | Threshold | Acc | AP | FPR | FNR |
|---|---:|---:|---:|---:|---:|
| Region-light ResNet18 best | 0.5 | 0.8763945978 | 0.9591492795 | 0.1966259453 | 0.0491997629 |

验证结论：

- ResNet18 的 AP 达到 0.9591，说明样本排序能力较好。
- 但官方阈值 0.5 下 FPR=0.1966，real 被误判成 fake 的比例偏高。
- 因为官方训练/推理脚本使用固定阈值 0.5，后续主结果不通过调阈值修正，只把阈值扫描作为诊断工具。
- ResNet18 可以作为“速度收益明显，但官方阈值下精度/FPR 不足”的轻量化 baseline。

### 下一步：ResNet34

ResNet34 是下一轮更合适的对照模型：它比 ResNet50 更轻，但容量明显高于 ResNet18。下一步目标是验证 FPR 偏高是否主要来自 ResNet18 表达能力不足。

已新增流程脚本：

```text
lightweight/scripts/run_resnet34_pipeline.sh
```

推荐先运行完整训练：

```bash
cd /root/lx/LipFD
bash lightweight/scripts/run_resnet34_pipeline.sh train
```

训练完成后依次运行：

```bash
bash lightweight/scripts/run_resnet34_pipeline.sh validate
bash lightweight/scripts/run_resnet34_pipeline.sh params
bash lightweight/scripts/run_resnet34_pipeline.sh speed
bash lightweight/scripts/run_resnet34_pipeline.sh scan
```

也可以直接运行全部流程：

```bash
bash lightweight/scripts/run_resnet34_pipeline.sh all
```

注意：`scan` 只读取 `val_scores_best.csv`，不重新加载模型，不重新读取图片，只用于分析分数分布；正式对比仍以阈值 0.5 的验证结果为准。
