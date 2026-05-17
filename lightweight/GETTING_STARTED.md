# LipFD 轻量化项目 - 快速入门指南

> **欢迎！** 这份指南将带你一步步完成整个轻量化项目

---

## 🚀 第一步：环境准备

### 1.1 激活 conda 环境

```bash
# 激活环境
conda activate lips

# 进入项目目录
cd /root/lx/LipFD

# 验证环境
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

### 1.2 检查必要的库

```bash
# 检查是否安装了 thop (用于计算 FLOPs)
python -c "import thop; print('thop 已安装')"

# 如果没有安装，运行:
# pip install thop
```

---

## 📊 第二步：测量原始模型（实验 1）

这是我们学习的第一步，也是最重要的一步。我们需要知道原始模型的"底子"。

### 2.1 测量参数量和 FLOPs

```bash
# 运行测量脚本
python lightweight/scripts/measure_model.py --model original
```

**你会看到什么？**
- 总参数量（单位：M，百万）
- 各模块的参数量分布（CLIP、ResNet50 等）
- FLOPs（计算复杂度）
- 模型文件大小

**记录这些数据！** 打开 `lightweight/EXPERIMENT_RESULTS.md`，将结果填入"实验 1"部分。

### 2.2 测量推理速度

```bash
# 测试 GPU 推理速度
python lightweight/scripts/benchmark_inference.py --model original --device cuda

# 如果想测试 CPU 速度（会很慢）
python lightweight/scripts/benchmark_inference.py --model original --device cpu
```

**你会看到什么？**
- 不同 batch size 下的推理时间（单位：ms，毫秒）
- FPS（每秒处理多少个样本）

**同样记录这些数据！**

### 2.3 运行原始模型的推理

```bash
# 在验证集上测试原始模型
python validate.py \
    --real_list_path ./datasets/val/0_real \
    --fake_list_path ./datasets/val/1_fake \
    --ckpt ./checkpoints/ckpt.pth
```

**你会看到什么？**
- Accuracy（准确率）
- AP（Average Precision）
- FPR（False Positive Rate）
- FNR（False Negative Rate）

**记录准确率！** 这是我们的 baseline。

---

## 🔍 第三步：理解数据的含义

### 3.1 参数量（Parameters）

**定义**：模型中所有可训练参数的数量

**为什么重要？**
- 参数越多，模型越大，占用内存越多
- 参数越多，训练和推理越慢
- 参数越多，模型越难部署到移动端

**典型值**：
- 小模型：< 10M（如 MobileNet）
- 中等模型：10M - 100M（如 ResNet50）
- 大模型：> 100M（如 CLIP ViT-L/14）

### 3.2 FLOPs（Floating Point Operations）

**定义**：前向传播需要的浮点运算次数

**为什么重要？**
- FLOPs 越大，计算量越大
- FLOPs 越大，推理越慢，耗电越多
- FLOPs 是衡量模型复杂度的标准指标

**单位**：
- K = 千（1,000）
- M = 百万（1,000,000）
- G = 十亿（1,000,000,000）

**典型值**：
- 轻量级模型：< 1G FLOPs
- 中等模型：1G - 10G FLOPs
- 重型模型：> 10G FLOPs

### 3.3 推理时间（Latency）

**定义**：处理一个样本需要的时间

**为什么重要？**
- 推理时间直接影响用户体验
- 实时应用要求推理时间 < 33ms（30 FPS）
- 移动端要求更低的推理时间

**单位**：ms（毫秒）

**典型值**：
- 实时应用：< 33ms（30 FPS）
- 准实时：33ms - 100ms
- 离线处理：> 100ms

### 3.4 准确率（Accuracy）

**定义**：模型预测正确的样本占总样本的比例

**为什么重要？**
- 准确率是模型性能的核心指标
- 轻量化的目标是在保持准确率的同时减少参数量

**典型值**：
- 优秀：> 95%
- 良好：90% - 95%
- 一般：85% - 90%

---

## 📝 第四步：记录实验结果

### 4.1 打开实验结果文档

```bash
# 使用你喜欢的编辑器打开
vim lightweight/EXPERIMENT_RESULTS.md
# 或
nano lightweight/EXPERIMENT_RESULTS.md
```

### 4.2 填写"实验 1"部分

将刚才测量的数据填入对应的表格中：

```markdown
### 1.1 模型规模

| 指标 | 数值 | 备注 |
|------|------|------|
| 总参数量 (M) | 123.45 | 填入你测量的值 |
| CLIP 参数量 (M) | 100.00 | 填入你测量的值 |
| ResNet50 参数量 (M) | 23.45 | 填入你测量的值 |
...
```

### 4.3 更新学习日志

```bash
# 打开学习日志
vim lightweight/LEARNING_LOG.md
```

记录今天的学习内容：
- 学到了什么？
- 遇到了什么问题？
- 有什么疑问？

---

## 🎯 第五步：分析原始模型的瓶颈

### 5.1 查看参数量分布

根据测量结果，回答以下问题：

1. **哪个模块的参数量最多？**
   - 答案应该是：CLIP ViT-L/14
   - 占比应该在 80% 以上

2. **第二大的模块是什么？**
   - 答案应该是：ResNet50
   - 占比应该在 10-15%

3. **结论是什么？**
   - CLIP 编码器是最大的瓶颈
   - 优化 CLIP 编码器是最重要的

### 5.2 查看推理时间

根据测量结果，回答以下问题：

1. **batch_size=1 时，推理时间是多少？**
   - 记录这个值

2. **能达到实时要求吗？**
   - 实时要求：< 33ms（30 FPS）
   - 如果推理时间 > 33ms，说明无法实时

3. **结论是什么？**
   - 原始模型太慢，无法实时
   - 需要轻量化

---

## 🤔 第六步：思考轻量化方案

### 6.1 我们有哪些选择？

**选项 1：替换 CLIP 编码器**
- ViT-L/14 (768维) → ViT-B/16 (512维)
- 优点：参数量减少 ~60%
- 缺点：准确率可能下降

**选项 2：替换 Region Awareness**
- ResNet50 → ResNet34
- 优点：参数量减少 ~30%
- 缺点：效果有限

**选项 3：两者都替换**
- 优点：参数量减少最多
- 缺点：准确率下降风险最大

**选项 4：使用知识蒸馏**
- 用原模型指导轻量化模型
- 优点：可以弥补准确率损失
- 缺点：需要额外的训练时间

### 6.2 我们的方案

**阶段 1：渐进式轻量化**
- ViT-L/14 → ViT-B/16
- ResNet50 → ResNet34
- 直接训练，看看效果

**阶段 2：知识蒸馏**
- 如果准确率下降太多，使用知识蒸馏
- 用原模型作为教师，指导轻量化模型

---

## ✅ 第七步：检查清单

在进入下一步之前，确保你已经完成：

- [ ] 成功运行了 `measure_model.py`
- [ ] 成功运行了 `benchmark_inference.py`
- [ ] 成功运行了 `validate.py`（如果有验证集）
- [ ] 将所有数据记录到 `EXPERIMENT_RESULTS.md`
- [ ] 更新了 `LEARNING_LOG.md`
- [ ] 理解了参数量、FLOPs、推理时间的含义
- [ ] 分析了原始模型的瓶颈
- [ ] 思考了轻量化方案

---

## 🎓 第八步：准备答辩话术

### 8.1 老师可能会问的问题

**Q1: "你为什么要做轻量化？"**

**参考答案**：
"我在研究 LipFD 这篇论文时发现，虽然它在检测唇形伪造方面效果很好，准确率达到了 95.3%，但是模型参数量非常大，约为 XXX M（填入你测量的值）。我实际测试了推理速度，发现处理一个样本需要 XXX ms（填入你测量的值），远远无法满足实时检测的要求（30 FPS 需要 < 33ms）。

论文的初衷是为了在微信视频通话等真实场景中检测伪造，但这么大的模型根本无法在移动端部署。因此，我决定对模型进行轻量化改造，目标是在保持较高准确率的同时，大幅减少参数量和推理时间，使其真正能够应用到实际场景中。"

**Q2: "你测量了原始模型的哪些指标？"**

**参考答案**：
"我测量了四个关键指标：

1. **参数量**：总参数量约为 XXX M，其中 CLIP ViT-L/14 占了 XX%，ResNet50 占了 XX%。

2. **FLOPs**：计算复杂度约为 XXX G FLOPs，说明模型的计算量很大。

3. **推理时间**：在 RTX 3090 GPU 上，batch_size=1 时推理时间约为 XXX ms，相当于 XX FPS，无法满足实时要求。

4. **准确率**：在验证集上的准确率为 XX.X%，这是我们的 baseline。

通过这些测量，我发现 CLIP 编码器是最大的瓶颈，占了总参数量的 XX%，因此优化 CLIP 编码器是最重要的。"

---

## 📚 下一步

完成了第一步的测量和分析后，你可以：

1. **继续学习**：阅读 `ARCHITECTURE.md`，深入理解模型架构
2. **开始实现**：实现轻量化模型（我会帮你）
3. **训练模型**：训练轻量化模型并对比结果

---

## 💡 提示

- **不要着急**：理解比速度更重要
- **记录一切**：所有的数据、问题、思考都要记录
- **多问为什么**：不懂的地方一定要搞清楚
- **动手实践**：自己运行代码，看看结果

---

**准备好了吗？让我们开始第一个实验！**

运行以下命令：

```bash
# 激活环境
conda activate lips

# 进入项目目录
cd /root/lx/LipFD

# 测量原始模型
python lightweight/scripts/measure_model.py --model original
```
