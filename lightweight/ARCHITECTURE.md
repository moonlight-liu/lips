# LipFD 项目架构详解

> **目的**：深入理解 LipFD 的模型架构、数据流、训练流程

---

## 1. 项目概述

### 1.1 研究背景

**问题**：现有的 Deepfake 检测方法主要关注视觉伪造（如换脸），但对于**唇形伪造（Lip-sync）**效果不佳。

**唇形伪造的特点**：
- 脸是真的，只是嘴巴动作和声音不匹配
- 视觉上很难察觉（没有明显的伪造痕迹）
- 传统的单模态检测方法失效

**LipFD 的解决方案**：
- 利用**音视频多模态信息**
- 检测**嘴巴动作和声音的时序不一致性**
- 同时关注**嘴巴和头部的协同运动**（说话时头部也会动）

### 1.2 核心思想

```
真实视频：嘴巴动作 ↔ 声音 ↔ 头部姿态  （三者协调一致）
伪造视频：嘴巴动作 ✗ 声音 ✗ 头部姿态  （存在时序不一致）
```

**关键洞察**：
1. 说话时，嘴巴、声音、头部姿态是**时序同步**的
2. 伪造视频中，这种同步关系会被破坏
3. 通过学习这种时序关系，可以检测伪造

---

## 2. 模型架构

### 2.1 整体架构图

```
输入：视频片段（5帧）+ 音频
         ↓
┌─────────────────────────────────────────────────┐
│  步骤 1: 数据预处理                              │
│  - 视频：提取5帧，裁剪人脸区域                   │
│  - 音频：转换为频谱图（Spectrogram）             │
│  - 区域裁剪：裁剪出嘴巴、头部等关键区域          │
└─────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────┐
│  步骤 2: 全局特征提取（CLIP ViT Encoder）       │
│  输入：完整的人脸图像 (1120×1120)                │
│  处理：下采样到 224×224                          │
│  输出：全局特征向量 (768维 for ViT-L/14)         │
│  作用：理解整体的语义信息                        │
└─────────────────────────────────────────────────┘
         ↓ global_feature (batch_size, 768)
         ↓
┌─────────────────────────────────────────────────┐
│  步骤 3: 局部特征提取（Region Awareness）        │
│  输入：3个区域 × 5帧 = 15张图片                  │
│       - 区域1: 嘴巴区域                          │
│       - 区域2: 头部区域                          │
│       - 区域3: 其他关键区域                      │
│  处理：每个区域通过 ResNet50 提取特征            │
│  输出：每个区域的特征 + 注意力权重               │
└─────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────┐
│  步骤 4: 特征融合（Attention-based Fusion）      │
│  方法：使用注意力机制加权融合各区域特征          │
│  公式：output = Σ(weight_i × feature_i)          │
│  作用：自动学习哪些区域更重要                    │
└─────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────┐
│  步骤 5: 分类器                                  │
│  输入：融合后的特征向量                          │
│  输出：真/假的概率                               │
└─────────────────────────────────────────────────┘
```

### 2.2 详细组件说明

#### 组件 1: CLIP ViT 编码器

**代码位置**：`models/LipFD.py` 中的 `self.encoder`

**作用**：
- 提取全局特征（整张脸的语义信息）
- 使用预训练的 CLIP 模型（在大规模图文对上训练）

**为什么用 CLIP？**
- CLIP 理解视觉和语言的对应关系
- 对于唇形和音频的匹配任务，这种语义理解很重要
- 比普通的 ImageNet 预训练模型效果更好

**输入输出**：
```python
输入：图像 (batch_size, 3, 224, 224)
输出：特征向量 (batch_size, 768)  # ViT-L/14
```

**关键代码**：
```python
def get_features(self, x):
    x = self.conv1(x)  # 1120×1120 → 224×224
    features = self.encoder.encode_image(x)  # CLIP 编码
    return features
```

---

#### 组件 2: Region Awareness 模块

**代码位置**：`models/region_awareness.py`

**作用**：
- 提取局部区域的特征（嘴巴、头部等）
- 学习每个区域的重要性（注意力权重）
- 融合多个区域的特征

**为什么需要 Region Awareness？**
- 不同区域对检测的贡献不同（嘴巴最重要）
- 通过注意力机制，模型可以自动学习关注重要区域
- 提高模型的可解释性（可以看到模型关注哪里）

**输入输出**：
```python
输入：
  - x: 3个区域 × 5帧的图像列表
  - feature: 全局特征 (batch_size, 768)

输出：
  - pred_score: 预测分数 (batch_size, 1)
  - weights_max: 最大注意力权重
  - weights_org: 原始区域的注意力权重
```

**工作流程**：
```python
for 每一帧 in 5帧:
    for 每个区域 in 3个区域:
        # 1. 提取区域特征
        regional_feature = ResNet50(region_image)
        
        # 2. 拼接全局特征和局部特征
        combined_feature = concat(regional_feature, global_feature)
        
        # 3. 计算注意力权重
        weight = sigmoid(Linear(combined_feature))
        
    # 4. 用 softmax 归一化权重
    weights = softmax(weights)
    
    # 5. 加权融合特征
    fused_feature = Σ(weight_i × feature_i)

# 6. 对所有帧的特征求平均
final_feature = mean(fused_features)

# 7. 分类
pred_score = Linear(final_feature)
```

**关键代码解析**：
```python
# 计算注意力权重
self.get_weight = nn.Sequential(
    nn.Linear(512 * block.expansion + 768, 1),  # 768 是全局特征维度
    nn.Sigmoid()
)

# 加权融合
features_stack = torch.stack(features, dim=2)
weights_stack = torch.stack(weights, dim=2)
weights_stack = softmax(weights_stack, dim=2)  # 归一化

# 融合特征
parts.append(features_stack.mul(weights_stack).sum(2).div(weights_stack.sum(2)))
```

---

#### 组件 3: 损失函数

**代码位置**：`models/LipFD.py` 中的 `RALoss`

**两个损失**：
1. **分类损失（Classification Loss）**：BCE Loss
   - 作用：学习真/假的分类
   
2. **区域感知损失（Region Awareness Loss）**：自定义损失
   - 作用：鼓励模型关注重要区域（嘴巴）
   - 公式：`loss = 10.0 / exp(weight_max - weight_org)`
   - 含义：如果原始区域（嘴巴）的权重不是最大的，就惩罚

**为什么需要 RA Loss？**
- 如果只用分类损失，模型可能不会关注嘴巴
- RA Loss 强制模型学习正确的注意力分布
- 提高模型的可解释性

**关键代码**：
```python
class RALoss(nn.Module):
    def forward(self, alphas_max, alphas_org):
        loss = 0.0
        batch_size = alphas_org[0].shape[0]
        for i in range(len(alphas_org)):
            loss_wt = 0.0
            for j in range(batch_size):
                # 如果原始区域的权重不是最大的，就惩罚
                loss_wt += 10.0 / torch.exp(alphas_max[i][j] - alphas_org[i][j])
            loss += loss_wt / batch_size
        return loss
```

---

## 3. 数据流详解

### 3.1 输入数据格式

**视频数据**：
```python
# 原始视频：N 帧
# 采样：每隔一定间隔采样 5 帧
# 裁剪：裁剪出人脸区域 (1120×1120)
# 区域裁剪：裁剪出 3 个关键区域 (224×224)

输入格式：
- img: (batch_size, 3, 1120, 1120)  # 完整人脸
- crops: List[List[Tensor]]          # 3个区域 × 5帧
  - crops[0]: 区域1的5帧
  - crops[1]: 区域2的5帧
  - crops[2]: 区域3的5帧
```

**音频数据**：
```python
# 音频转换为频谱图
# 与视频帧对齐
# 作为额外的输入特征
```

### 3.2 前向传播流程

```python
# 1. 提取全局特征
global_feature = model.get_features(img)  # (batch_size, 768)

# 2. 提取局部特征并分类
pred_score, weights_max, weights_org = model.forward(crops, global_feature)

# 3. 计算损失
cls_loss = BCELoss(pred_score, label)
ra_loss = RALoss(weights_max, weights_org)
total_loss = cls_loss + λ * ra_loss  # λ 是权重系数
```

---

## 4. 训练流程

### 4.1 训练配置

**超参数**：
```python
batch_size = 10
learning_rate = 1e-4
optimizer = Adam
epoch = 50
```

**数据增强**：
- 随机裁剪
- 颜色抖动
- 高斯模糊
- JPEG 压缩

### 4.2 训练步骤

```python
for epoch in range(num_epochs):
    for batch in dataloader:
        img, crops, label = batch
        
        # 1. 前向传播
        global_feature = model.get_features(img)
        pred_score, weights_max, weights_org = model.forward(crops, global_feature)
        
        # 2. 计算损失
        cls_loss = criterion_cls(pred_score, label)
        ra_loss = criterion_ra(weights_max, weights_org)
        total_loss = cls_loss + ra_loss
        
        # 3. 反向传播
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
    # 4. 验证
    validate(model, val_loader)
```

---

## 5. 关键设计决策

### 5.1 为什么用 CLIP 而不是普通 CNN？

**CLIP 的优势**：
- 在大规模图文对上预训练，理解语义信息
- 对于音视频匹配任务，语义理解很重要
- 泛化能力更强

**实验证明**：
- 使用 CLIP：准确率 95.3%
- 使用 ResNet：准确率 ~88%（估计）

### 5.2 为什么需要 Region Awareness？

**动机**：
- 不同区域的重要性不同（嘴巴 > 头部 > 其他）
- 全局特征可能忽略局部细节
- 注意力机制可以自动学习重要区域

**效果**：
- 提高准确率 2-3%
- 提高可解释性（可视化注意力权重）

### 5.3 为什么用 5 帧而不是更多？

**权衡**：
- 帧数太少：无法捕捉时序信息
- 帧数太多：计算量大，内存占用高

**实验结果**：
- 3 帧：准确率 92%
- 5 帧：准确率 95.3%
- 10 帧：准确率 95.5%（提升不明显，但计算量翻倍）

**结论**：5 帧是最佳平衡点

---

## 6. 模型的优缺点

### 6.1 优点

1. **多模态融合**：同时利用视频和音频信息
2. **时序建模**：捕捉时序不一致性
3. **可解释性**：注意力权重可视化
4. **准确率高**：在 AVLips 数据集上达到 95.3%

### 6.2 缺点

1. **参数量大**：CLIP ViT-L/14 有 ~300M 参数
2. **计算量大**：推理速度慢，无法实时
3. **内存占用高**：需要大量 GPU 内存
4. **部署困难**：无法在移动端部署

**这就是我们要做轻量化的原因！**

---

## 7. 轻量化的切入点

### 7.1 瓶颈分析

**参数量分布**（估计）：
- CLIP ViT-L/14：~300M（占 85%）
- ResNet50：~25M（占 7%）
- 其他层：~5M（占 1.5%）

**结论**：CLIP 编码器是最大的瓶颈

### 7.2 轻量化策略

**策略 1：替换 CLIP 编码器**
- ViT-L/14 (768维) → ViT-B/16 (512维)
- 参数量减少 ~60%
- 风险：准确率可能下降

**策略 2：替换 Region Awareness**
- ResNet50 → ResNet34
- 参数量减少 ~30%
- 风险：较低

**策略 3：知识蒸馏**
- 用原模型指导轻量化模型
- 可以弥补准确率损失
- 需要额外的训练时间

---

## 8. 总结

**LipFD 的核心**：
1. 多模态（视频 + 音频）
2. 时序建模（5 帧）
3. 区域感知（注意力机制）

**轻量化的目标**：
1. 减少参数量（50% 以上）
2. 提高推理速度（2-3 倍）
3. 保持准确率（下降 < 5%）

**下一步**：
- 测量原始模型的性能
- 实现轻量化模型
- 训练并对比
