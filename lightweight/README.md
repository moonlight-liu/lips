# LipFD 轻量化项目

> **作者**：YuwanZ  
> **开始日期**：2026-05-08  
> **目标**：实现 LipFD 模型的轻量化，使其能够在资源受限的环境中实时运行

---

## 📁 项目结构

```
lightweight/
├── README.md                    # 项目说明（本文件）
├── TODO.md                      # 详细任务清单
├── LEARNING_LOG.md              # 学习日志
├── ARCHITECTURE.md              # 架构详解
├── EXPERIMENT_RESULTS.md        # 实验结果记录
│
├── scripts/                     # 工具脚本
│   ├── measure_model.py         # 测量模型参数量和FLOPs
│   ├── benchmark_inference.py   # 测试推理速度
│   ├── compare_models.py        # 对比多个模型
│   ├── visualize_results.py     # 可视化实验结果
│   └── test_model.py            # 测试模型是否正常工作
│
├── models/                      # 轻量化模型定义
│   ├── LipFD_light.py          # 轻量化模型
│   ├── region_awareness_light.py # 轻量化Region Awareness
│   └── distillation.py         # 知识蒸馏相关代码
│
├── data/                        # 数据处理
│   └── (数据加载相关代码)
│
├── results/                     # 实验结果
│   ├── checkpoints/            # 模型权重
│   ├── logs/                   # 训练日志
│   └── figures/                # 可视化图表
│
└── logs/                        # 运行日志
```

---

## 🎯 项目目标

### 主要目标
1. **理解原始 LipFD 模型**的架构和工作原理
2. **实现轻量化版本**，减少参数量和计算量
3. **使用知识蒸馏**提升轻量化模型的准确率
4. **完整记录实验过程**，便于答辩和展示

### 性能目标
- 参数量减少：≥ 50%
- 推理速度提升：≥ 2×
- 准确率下降：≤ 5%

---

## 🚀 快速开始

### 1. 环境准备

```bash
# 激活环境
conda activate LipFD

# 进入工作目录
cd /root/lx/LipFD/lightweight
```

### 2. 测量原始模型

```bash
# 测量参数量和FLOPs
python scripts/measure_model.py --model original

# 测试推理速度
python scripts/benchmark_inference.py --model original
```

### 3. 训练轻量化模型

```bash
# 直接训练
python train_light.py --gpu_ids 0,1 --batch_size 16

# 知识蒸馏训练
python train_distill.py --teacher_ckpt ../checkpoints/ckpt.pth --gpu_ids 0,1
```

### 4. 对比结果

```bash
# 生成对比表格
python scripts/compare_models.py

# 生成可视化图表
python scripts/visualize_results.py
```

---

## 📚 学习路径

### 阶段一：理解项目（2-3天）
- [ ] 阅读 `ARCHITECTURE.md`，理解模型架构
- [ ] 运行原始模型，测量性能指标
- [ ] 分析模型瓶颈

### 阶段二：设计方案（1-2天）
- [ ] 设计轻量化架构
- [ ] 实现轻量化模型代码
- [ ] 测试模型是否正常工作

### 阶段三：训练模型（3-5天）
- [ ] 直接训练轻量化模型
- [ ] 实现知识蒸馏
- [ ] 蒸馏训练轻量化模型

### 阶段四：实验对比（1-2天）
- [ ] 对比三个模型的性能
- [ ] 分析实验结果
- [ ] 制作可视化图表

### 阶段五：总结答辩（1天）
- [ ] 整理项目文档
- [ ] 准备答辩材料
- [ ] 模拟答辩

---

## 📊 实验记录

所有实验结果记录在 `EXPERIMENT_RESULTS.md` 中。

### 当前进度

- [x] 创建项目框架
- [ ] 测量原始模型性能
- [ ] 实现轻量化模型
- [ ] 训练轻量化模型
- [ ] 实现知识蒸馏
- [ ] 对比实验结果

---

## 🤔 常见问题

### Q1: 为什么要做轻量化？
A: 原始模型参数量大（~300M），推理速度慢，无法在移动端或实时场景中部署。轻量化可以在保持较高准确率的同时，大幅减少计算量和内存占用。

### Q2: 为什么选择 ViT-B/16 而不是 MobileNet？
A: CLIP ViT 是在大规模图文对上预训练的，包含丰富的语义信息。如果换成 MobileNet，会损失这些语义信息，导致准确率大幅下降。ViT-B/16 是同系列的轻量级版本，可以保留预训练优势。

### Q3: 什么是知识蒸馏？
A: 知识蒸馏是用大模型（教师）指导小模型（学生）训练的方法。学生不仅学习正确答案，还学习教师的"思考过程"（概率分布、中间特征），从而达到接近教师的性能。

### Q4: 如何应对老师的提问？
A: 参考 `TODO.md` 中的"老师可能的提问"部分，提前准备答案。关键是要理解原理，而不是死记硬背。

---

## 📖 参考资料

### 论文
1. **LipFD 原论文**：Lips Are Lying: Spotting the Temporal Inconsistency between Audio and Visual in Lip-syncing DeepFakes (NeurIPS 2024)
2. **CLIP 论文**：Learning Transferable Visual Models From Natural Language Supervision
3. **知识蒸馏论文**：Distilling the Knowledge in a Neural Network

### 代码
- 原始 LipFD 代码：`/root/lx/LipFD/`
- 轻量化代码：`/root/lx/LipFD/lightweight/`

---

## 📝 日志

详细的学习日志记录在 `LEARNING_LOG.md` 中。

---

## 🙏 致谢

感谢 LipFD 原作者开源代码，为本项目提供了基础。

---

**最后更新**：2026-05-08
