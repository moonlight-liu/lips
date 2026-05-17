# 🎉 LipFD 轻量化项目框架已搭建完成！

---

## 📁 项目结构总览

```
lightweight/
├── 📖 文档（学习和记录）
│   ├── README.md                 # 项目总览
│   ├── GETTING_STARTED.md        # 快速入门指南 ⭐ 从这里开始！
│   ├── TODO.md                   # 详细任务清单（6个阶段）
│   ├── LEARNING_LOG.md           # 学习日志（每天记录）
│   ├── ARCHITECTURE.md           # 架构详解（深入理解模型）
│   ├── EXPERIMENT_RESULTS.md     # 实验结果记录
│   └── INTERVIEW_GUIDE.md        # 面试问题准备（已有）
│
├── 🔧 工具脚本
│   ├── scripts/measure_model.py        # 测量参数量和FLOPs
│   ├── scripts/benchmark_inference.py  # 测试推理速度
│   └── start.sh                        # 快速启动脚本
│
├── 🧠 模型代码（待实现）
│   └── models/
│       ├── LipFD_light.py              # 轻量化模型
│       ├── region_awareness_light.py   # 轻量化Region Awareness
│       └── distillation.py             # 知识蒸馏
│
└── 📊 实验结果（运行后生成）
    └── results/
        ├── checkpoints/                # 模型权重
        ├── logs/                       # 训练日志
        └── figures/                    # 可视化图表
```

---

## 🎯 你现在的位置：阶段一 - 任务 1.2

### ✅ 已完成
- [x] 任务 1.1：理解项目整体架构
  - 阅读了 ARCHITECTURE.md
  - 理解了三大组件：CLIP、Region Awareness、分类器
  - 理解了数据流

### ⏳ 当前任务：任务 1.2 - 测量原始模型

**目标**：建立 Baseline，知道原始模型的"底子"

**需要做的事情**：
1. 测量参数量和 FLOPs
2. 测量推理速度
3. 运行原始模型的推理（如果有验证集）
4. 记录所有数据到 EXPERIMENT_RESULTS.md

---

## 🚀 现在开始第一个实验！

### 步骤 1：激活环境并进入目录

```bash
# 激活 conda 环境
conda activate lips

# 进入项目目录
cd /root/lx/LipFD

# 验证环境
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

### 步骤 2：测量原始模型的参数量

```bash
# 运行测量脚本
python lightweight/scripts/measure_model.py --model original
```

**预期输出**：
```
==============================================================
测量原始模型 (ViT-L/14 + ResNet50)
==============================================================

【参数量统计】
总参数量: XXX,XXX,XXX (XXX.XXM)
可训练参数: XXX,XXX,XXX (XXX.XXM)

【模块参数量分布】
CLIP ViT-L/14: XXX,XXX,XXX (XXX.XXM) - XX.X%
ResNet50 (Region Awareness): XX,XXX,XXX (XX.XXM) - XX.X%
Conv1 (下采样层): XXX (X.XXK) - X.X%

【计算复杂度】
正在计算 FLOPs (这可能需要一些时间)...
FLOPs (Region Awareness): XX.XXX GFLOPs
...
```

**重要**：将这些数据复制下来，我们要记录到实验结果中！

### 步骤 3：测量推理速度

```bash
# 测试 GPU 推理速度（推荐）
python lightweight/scripts/benchmark_inference.py --model original --device cuda

# 如果想快速测试，可以减少迭代次数
python lightweight/scripts/benchmark_inference.py --model original --device cuda --num_iterations 50
```

**预期输出**：
```
==============================================================
测试原始模型推理速度 (ViT-L/14 + ResNet50)
==============================================================
设备: cuda

Batch Size   平均时间(ms)      标准差(ms)      FPS       
------------------------------------------------------------
1            XXX.XX          XX.XX          XX.X      
4            XXX.XX          XX.XX          XX.X      
8            XXX.XX          XX.XX          XX.X      
16           XXX.XX          XX.XX          XX.X      
```

**同样记录这些数据！**

### 步骤 4：记录实验结果

```bash
# 打开实验结果文档
vim lightweight/EXPERIMENT_RESULTS.md
# 或使用你喜欢的编辑器
```

将刚才的数据填入"实验 1"部分的表格中。

### 步骤 5：更新学习日志

```bash
# 打开学习日志
vim lightweight/LEARNING_LOG.md
```

在"2026-05-08 - Day 1"部分记录：
- 今天完成了什么？
- 测量的数据是多少？
- 有什么发现？
- 有什么疑问？

---

## 📊 数据分析指导

### 当你拿到测量结果后，思考这些问题：

1. **参数量分析**：
   - 总参数量是多少？（应该在 100M - 400M 之间）
   - CLIP 占了多少比例？（应该 > 80%）
   - 结论：CLIP 是最大的瓶颈

2. **推理速度分析**：
   - batch_size=1 时推理时间是多少？
   - 能达到 30 FPS（< 33ms）吗？
   - 如果不能，说明需要轻量化

3. **瓶颈总结**：
   - 哪个模块最需要优化？
   - 优化的优先级是什么？

---

## 🎓 面试准备

### 当老师问："你为什么要做轻量化？"

**你的回答模板**：

"我在研究 LipFD 这篇论文时，发现虽然它的准确率很高（95.3%），但模型非常大。我实际测量了一下，发现：

1. **参数量**：总参数量约为 **[填入你的数据]** M，其中 CLIP ViT-L/14 占了 **[填入百分比]**%

2. **推理速度**：在 RTX 3090 上，处理一个样本需要 **[填入你的数据]** ms，相当于 **[计算 FPS]** FPS

3. **问题**：论文的目标是在微信视频通话等实时场景中检测伪造，但这个推理速度远远无法满足实时要求（30 FPS 需要 < 33ms）

因此，我决定对模型进行轻量化改造，目标是：
- 参数量减少 ≥ 50%
- 推理速度提升 ≥ 2×
- 准确率下降 ≤ 5%

这样才能真正将模型应用到实际场景中。"

---

## 📝 检查清单

在继续下一步之前，确保你已经：

- [ ] 成功激活了 lips 环境
- [ ] 成功运行了 `measure_model.py`
- [ ] 成功运行了 `benchmark_inference.py`
- [ ] 将所有数据记录到 `EXPERIMENT_RESULTS.md`
- [ ] 更新了 `LEARNING_LOG.md`
- [ ] 理解了数据的含义
- [ ] 分析了模型的瓶颈

---

## 🤔 遇到问题？

### 常见问题：

**Q1: 运行脚本时报错 "No module named 'thop'"**
```bash
# 解决方案：安装 thop
conda activate lips
pip install thop
```

**Q2: 运行脚本时报错 "CUDA out of memory"**
```bash
# 解决方案：减少 batch size 或使用 CPU
python lightweight/scripts/benchmark_inference.py --model original --device cpu
```

**Q3: 找不到模型权重文件**
```bash
# 检查权重文件是否存在
ls -lh checkpoints/ckpt.pth

# 如果不存在，需要先下载或训练模型
```

---

## 🎯 下一步

完成实验 1 后，你将：

1. **理解原始模型的性能**：知道它有多大、多慢
2. **找到优化的方向**：知道哪里是瓶颈
3. **建立 Baseline**：有了对比的基准

然后我们将进入**阶段二：设计轻量化方案**，我会帮你：
- 实现轻量化模型代码
- 训练轻量化模型
- 实现知识蒸馏
- 对比实验结果

---

## 💪 加油！

记住：
- **理解比速度更重要**
- **记录一切**
- **多问为什么**
- **动手实践**

你已经有了完整的学习框架，现在开始第一个实验吧！

有任何问题随时问我 😊
