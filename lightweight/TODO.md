# LipFD 轻量化改造 - 完整学习计划

> **目标**：深入理解 LipFD 项目，实现轻量化改造，并能够应对老师的提问
> **时间**：不限，以学习和理解为主
> **方法**：渐进式学习 + 完整实验 + 知识蒸馏

---

## 📚 阶段一：理解原始项目（预计 2-3 天）

### 任务 1.1：理解项目整体架构 ✅
**目标**：搞清楚 LipFD 是如何工作的

**学习内容**：
- [ ] 阅读 `ARCHITECTURE.md`，理解三大组件
- [ ] 理解输入输出：视频帧 + 音频 → 真假判断
- [ ] 理解数据流：CLIP 编码器 → Region Awareness → 分类器

**验证方式**：
- 能画出模型的架构图
- 能用自己的话解释每个组件的作用

**老师可能的提问**：
- Q: "你的模型是如何处理音视频多模态的？"
- A: "模型使用 CLIP ViT 提取视频的全局特征（包含语义信息），然后通过 Region Awareness 模块提取嘴巴、头部等局部区域的特征。音频特征通过频谱图转换后也输入到模型中。最后通过注意力机制加权融合这些特征，判断音视频是否同步。"

---

### 任务 1.2：测量原始模型的性能指标 ⏳
**目标**：建立 Baseline，知道原模型的"底子"

**需要测量的指标**：
1. **参数量（Parameters）**：模型有多少个可训练参数
2. **计算量（FLOPs）**：前向传播需要多少次浮点运算
3. **推理时间（Latency）**：处理一个样本需要多少时间
4. **准确率（Accuracy）**：在验证集上的表现

**执行步骤**：
```bash
# 1. 运行测量脚本
cd /root/lx/LipFD/lightweight
python scripts/measure_model.py --model original

# 2. 记录结果到 EXPERIMENT_RESULTS.md
```

**预期结果**：
```
原始模型 (LipFD with ViT-L/14 + ResNet50):
- Parameters: ~XXX M
- FLOPs: ~XXX G
- Latency (GPU): ~XXX ms
- Latency (CPU): ~XXX ms
- Accuracy: ~XX.X%
```

**老师可能的提问**：
- Q: "你说原模型参数量很大，具体是多少？"
- A: "原模型使用 CLIP ViT-L/14 作为编码器，加上 ResNet50 作为 Region Awareness 模块，总参数量约为 XXX M。其中 CLIP ViT-L/14 占了大部分，约 XXX M。"

---

### 任务 1.3：运行原始模型的推理和训练 ⏳
**目标**：确保你能够完整地运行原始模型

**执行步骤**：
```bash
# 1. 在验证集上测试原始模型
python validate.py --real_list_path ./datasets/val/0_real \
                   --fake_list_path ./datasets/val/1_fake \
                   --ckpt ./checkpoints/ckpt.pth

# 2. 尝试训练几个 epoch（如果有训练数据）
python train.py --epoch 2 --batch_size 8
```

**需要记录**：
- 验证集准确率
- 训练时的 loss 变化
- 每个 epoch 的时间

**老师可能的提问**：
- Q: "你复现了原论文的结果吗？"
- A: "是的，我在 AVLips 验证集上测试了原模型，准确率达到了 XX.X%，与论文中报告的 95.3% 相近/有差距（如果有差距，解释原因：数据集版本、预训练权重等）。"

---

## 🔧 阶段二：设计轻量化方案（预计 1-2 天）

### 任务 2.1：分析模型的瓶颈 ⏳
**目标**：找出哪些部分最"重"，优化的优先级

**分析方法**：
```python
# 使用 scripts/analyze_bottleneck.py
# 会输出每个模块的参数量和计算量占比
```

**预期发现**：
- CLIP ViT-L/14：占总参数量的 XX%
- ResNet50：占总参数量的 XX%
- 其他层：占总参数量的 XX%

**结论**：
- 优化 CLIP 编码器是最重要的
- 优化 ResNet50 是第二重要的

**老师可能的提问**：
- Q: "你为什么选择这个轻量化方案？"
- A: "我通过分析发现，CLIP ViT-L/14 占了模型参数量的 XX%，是最大的瓶颈。因此我选择将其替换为同系列的轻量级版本 ViT-B/16，这样可以保留预训练模型的语义信息，同时大幅减少参数量。"

---

### 任务 2.2：设计轻量化架构 ⏳
**目标**：确定具体的改进方案

**方案 A：渐进式轻量化（推荐，风险低）**
```
原版：ViT-L/14 (768维) + ResNet50
  ↓
轻量版：ViT-B/16 (512维) + ResNet34
```
- 优点：改动小，风险低，3-5天可完成
- 缺点：参数减少约 30-40%，不够"震撼"

**方案 B：激进式轻量化（高风险，高回报）**
```
原版：ViT-L/14 (768维) + ResNet50
  ↓
轻量版：ViT-B/32 (512维) + MobileNetV3
```
- 优点：参数减少约 60-70%，更震撼
- 缺点：需要更多调试，可能准确率下降明显

**我的建议**：
- 先做方案 A，跑通整个流程
- 如果时间充足，再尝试方案 B

**老师可能的提问**：
- Q: "为什么不直接用 MobileNet 替换 CLIP？"
- A: "CLIP 是在大规模图文对上预训练的，包含丰富的语义信息，这对于理解唇形和音频的对应关系很重要。如果直接换成 MobileNet（只在 ImageNet 上预训练），会损失这些语义信息，导致准确率大幅下降。因此我选择了同系列的轻量级 CLIP 模型。"

---

### 任务 2.3：实现轻量化模型 ⏳
**目标**：编写代码，实现轻量化架构

**需要修改的文件**：
- `lightweight/models/LipFD_light.py` - 轻量化模型定义
- `lightweight/models/region_awareness_light.py` - 轻量化 Region Awareness
- `lightweight/train_light.py` - 训练脚本
- `lightweight/validate_light.py` - 验证脚本

**检查清单**：
- [ ] 模型能够正常初始化
- [ ] 输入输出维度匹配
- [ ] 能够进行前向传播
- [ ] 能够计算 loss 和反向传播

**验证方式**：
```bash
# 测试模型是否能正常运行
python scripts/test_model.py --model lightweight
```

---

## 🚀 阶段三：训练轻量化模型（预计 3-5 天）

### 任务 3.1：直接训练轻量化模型（不使用蒸馏）⏳
**目标**：先看看轻量化模型"裸奔"的效果

**训练配置**：
```bash
python lightweight/train_light.py \
    --arch "CLIP:ViT-B/16" \
    --batch_size 16 \
    --epoch 20 \
    --gpu_ids "0,1" \
    --name "lipfd_light_baseline"
```

**需要记录**：
- 每个 epoch 的 train loss
- 每个 epoch 的 validation accuracy
- 训练时间
- 最佳模型的性能

**预期结果**：
- 准确率可能会下降 5-10%（这是正常的）

**老师可能的提问**：
- Q: "轻量化之后准确率下降了多少？"
- A: "直接训练轻量化模型，准确率从 XX.X% 下降到了 XX.X%，下降了约 X%。这是因为模型容量减小了，表达能力有所下降。"

---

### 任务 3.2：实现知识蒸馏 ⏳
**目标**：用原模型指导轻量化模型，提升准确率

**知识蒸馏原理**（用人话讲）：
- 把原模型（大模型）当作"老师"
- 把轻量化模型（小模型）当作"学生"
- 学生不仅要学习正确答案（真/假），还要学习老师的"思考过程"
- 这样学生可以学到更多知识，准确率更高

**蒸馏的三个层次**：
1. **硬标签蒸馏**：学习最终的分类结果（真/假）
2. **软标签蒸馏**：学习老师的概率分布（比如 0.95 真，0.05 假）
3. **特征蒸馏**：学习老师的中间特征（Region Awareness 的注意力权重）

**实现步骤**：
```bash
# 1. 实现蒸馏 loss
# 见 lightweight/models/distillation.py

# 2. 修改训练脚本
# 见 lightweight/train_distill.py

# 3. 开始蒸馏训练
python lightweight/train_distill.py \
    --teacher_ckpt ./checkpoints/ckpt.pth \
    --arch "CLIP:ViT-B/16" \
    --batch_size 16 \
    --epoch 30 \
    --gpu_ids "0,1" \
    --name "lipfd_light_distill"
```

**需要记录**：
- 蒸馏 loss 的变化
- 准确率的提升
- 与直接训练的对比

**老师可能的提问**：
- Q: "你是如何实现知识蒸馏的？"
- A: "我实现了三层蒸馏：1) 软标签蒸馏，让学生模型学习教师模型的概率分布；2) 特征蒸馏，让学生模型学习教师模型在 Region Awareness 模块中的注意力权重分布；3) 硬标签蒸馏，学习真实标签。最终的 loss 是这三部分的加权和。"

---

## 📊 阶段四：实验对比与分析（预计 1-2 天）

### 任务 4.1：全面对比三个模型 ⏳
**目标**：制作完整的对比表格

**三个模型**：
1. 原始模型（ViT-L/14 + ResNet50）
2. 轻量化模型 - 直接训练（ViT-B/16 + ResNet34）
3. 轻量化模型 - 知识蒸馏（ViT-B/16 + ResNet34 + KD）

**对比维度**：
| 指标 | 原始模型 | 轻量化(直接训练) | 轻量化(知识蒸馏) |
|------|----------|------------------|------------------|
| 参数量 (M) | XXX | XXX | XXX |
| FLOPs (G) | XXX | XXX | XXX |
| GPU 推理时间 (ms) | XXX | XXX | XXX |
| CPU 推理时间 (ms) | XXX | XXX | XXX |
| 准确率 (%) | XX.X | XX.X | XX.X |
| FPS (GPU) | XX | XX | XX |
| 模型大小 (MB) | XXX | XXX | XXX |

**执行脚本**：
```bash
python scripts/compare_models.py
```

---

### 任务 4.2：分析实验结果 ⏳
**目标**：理解数据背后的含义

**需要回答的问题**：
1. 参数量减少了多少？（百分比）
2. 推理速度提升了多少？（倍数）
3. 准确率下降了多少？（百分比）
4. 知识蒸馏带来了多少提升？
5. 这个 trade-off 是否值得？

**写入 EXPERIMENT_RESULTS.md**

---

### 任务 4.3：可视化分析 ⏳
**目标**：用图表展示实验结果

**需要的图表**：
1. 训练曲线对比（loss 和 accuracy）
2. 参数量对比（柱状图）
3. 推理时间对比（柱状图）
4. 准确率对比（柱状图）

**执行脚本**：
```bash
python scripts/visualize_results.py
```

---

## 🎯 阶段五：部署优化（可选，预计 2-3 天）

### 任务 5.1：导出 ONNX 模型 ⏳
**目标**：将 PyTorch 模型转换为 ONNX 格式，便于部署

```bash
python scripts/export_onnx.py --model lightweight
```

### 任务 5.2：测试 ONNX 推理速度 ⏳
**目标**：验证 ONNX 是否能进一步加速

---

## 📝 阶段六：总结与答辩准备（预计 1 天）

### 任务 6.1：整理项目文档 ⏳
**需要准备的文档**：
- [x] ARCHITECTURE.md - 架构详解
- [ ] EXPERIMENT_RESULTS.md - 实验结果
- [ ] LEARNING_LOG.md - 学习历程
- [ ] README.md - 项目说明

### 任务 6.2：准备答辩 PPT ⏳
**PPT 结构建议**：
1. 背景与动机（为什么要做轻量化）
2. 原始模型分析（瓶颈在哪里）
3. 轻量化方案设计（为什么这样设计）
4. 实验结果对比（数据说话）
5. 知识蒸馏的作用（技术深度）
6. 总结与展望（还能怎么优化）

### 任务 6.3：模拟答辩 ⏳
**高频问题清单**：见 `INTERVIEW_QUESTIONS.md`

---

## 📌 重要提醒

### 学习建议：
1. **不要急于求成**：每个阶段都要真正理解，不要只是跑代码
2. **记录学习过程**：在 LEARNING_LOG.md 中记录每天的学习内容和遇到的问题
3. **多问为什么**：不懂的地方一定要搞清楚
4. **动手实践**：自己写代码，不要只是复制粘贴

### 遇到问题时：
1. 先查看 LEARNING_LOG.md 中是否有类似问题
2. 查看代码注释和文档
3. 使用 `python scripts/debug.py` 进行调试
4. 记录问题和解决方案

---

## 🎓 学习资源

### 推荐阅读：
1. CLIP 论文：Learning Transferable Visual Models From Natural Language Supervision
2. 知识蒸馏论文：Distilling the Knowledge in a Neural Network
3. ResNet 论文：Deep Residual Learning for Image Recognition

### 代码参考：
- `lightweight/examples/` - 示例代码
- `lightweight/scripts/` - 工具脚本

---

**开始时间**：2026-05-08
**预计完成时间**：根据学习进度调整
**当前进度**：阶段一 - 任务 1.1 ✅
