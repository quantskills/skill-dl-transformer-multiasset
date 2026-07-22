# 模型架构指南

本文档描述 Transformer 多资产建模因子支持的两种模型架构、位置编码方案、默认超参数以及架构切换机制。

## 支持的模型架构

本项目支持两种 Transformer 架构用于多变量时间序列预测：

### 1. PatchTST (默认)

**全称**：Patched Time Series Transformer

**核心思想**：Channel-independent (通道独立) + Patch-based (分段编码)

**架构特点**：

- 将每个变量视为独立通道，分别处理后聚合
- 将时间序列切分为 patches（类似 ViT 对图像的处理）
- 每个 patch 作为一个 token 输入 Transformer encoder
- 通过 patch 降低序列长度，提升训练效率并捕获局部时间模式

**公式**：

```
输入: X ∈ ℝ^(T × C)  (T=lookback, C=features)
分段: P_i = X[i*stride : i*stride+patch_len, :]  (i=0,1,2,...)
编码: E_i = Linear(flatten(P_i)) + PositionalEncoding(i)
Transformer: H = MultiHeadAttention(E)
输出: y = Linear(GlobalAvgPool(H))
```

**优势**：

- 适合高维多变量场景（如 58 维特征）
- 训练速度快，显存占用低
- 对噪声鲁棒性强（patch 平滑了高频噪声）

**参考文献**：

> Nie, Y., Nguyen, N. H., Sinthong, P., & Kalagnanam, J. (2023).  
> A Time Series is Worth 64 Words: Long-term Forecasting with Transformers.  
> *International Conference on Learning Representations (ICLR)*.

### 2. iTransformer

**全称**：Inverted Transformer

**核心思想**：Variate-as-token (变量作为 token)

**架构特点**：

- 将每个变量视为一个 token，时间维度作为 token 的 embedding
- Transformer attention 在**变量维度**上操作，捕获跨变量的关联
- 适合学习多资产间的协同效应与竞争关系

**公式**：

```
输入: X ∈ ℝ^(T × C)  (T=lookback, C=features)
转置: X' = X^T ∈ ℝ^(C × T)  (每个变量成为一个 token)
编码: E_c = Linear(X'[c, :]) + VariateEmbedding(c)  (c=0,1,...,C-1)
Transformer: H = MultiHeadAttention(E)  (attention 在变量维度)
输出: y = Linear(H)
```

**优势**：

- 适合跨品种建模（如商品期货的板块联动）
- 能够学习变量间的因果关系与替代效应
- 对横截面信号更敏感

**参考文献**：

> Liu, Y., Hu, T., Zhang, H., Wu, H., Wang, S., Ma, L., & Long, M. (2024).  
> iTransformer: Inverted Transformers Are Effective for Time Series Forecasting.  
> *International Conference on Learning Representations (ICLR)*.

## 位置编码方案

两种架构均使用 **RoPE (Rotary Position Embedding)** 作为位置编码。

**核心思想**：通过旋转矩阵注入相对位置信息，使模型能够区分不同时间步的 token。

**公式**（简化版）：

```
RoPE(x, pos) = [x_0 * cos(pos*θ_0) - x_1 * sin(pos*θ_0),
                x_0 * sin(pos*θ_0) + x_1 * cos(pos*θ_0),
                ...
                x_{d-2} * cos(pos*θ_{d/2-1}) - x_{d-1} * sin(pos*θ_{d/2-1}),
                x_{d-2} * sin(pos*θ_{d/2-1}) + x_{d-1} * cos(pos*θ_{d/2-1})]

其中 θ_i = 10000^(-2i/d)
```

**优势**：

- 相对位置编码，泛化性强
- 不需要学习额外参数
- 在长序列上表现优于 Sinusoidal Encoding 和 Learnable Embedding

**参考文献**：

> Su, J., Lu, Y., Pan, S., Wen, B., & Liu, Y. (2021).  
> RoFormer: Enhanced Transformer with Rotary Position Embedding.  
> *arXiv preprint arXiv:2104.09864*.

## 默认超参数

以下超参数经过网格搜索和多资产回测验证，适用于商品期货场景：

| 参数名 | 默认值 | 说明 |
|---|---|---|
| `LOOKBACK` | 60 | 回溯窗口（交易日） |
| `PATCH_LEN` | 16 | PatchTST 的 patch 长度 |
| `STRIDE` | 8 | PatchTST 的 patch 滑动步长 |
| `D_MODEL` | 128 | Transformer hidden dimension |
| `N_HEADS` | 8 | Multi-head attention 头数 |
| `N_LAYERS` | 3 | Transformer encoder 层数 |
| `DROPOUT` | 0.2 | Dropout 比例 |
| `LR` | 1e-4 | 学习率 (AdamW optimizer) |
| `BATCH_SIZE` | 64 | 训练批次大小 |
| `MAX_EPOCHS` | 50 | 最大训练轮数 |
| `EARLY_STOP_PATIENCE` | 10 | Early stopping 的耐心值（连续 10 epoch val loss 不下降则停止） |
| `SEED` | 42 | 随机种子（保证可复现） |
| `LOSS_ALPHA` | 0.5 | 混合损失权重：`Loss = alpha * rank_ic_loss + (1-alpha) * mse` |

**超参数调优建议**：

- **小数据集**（< 10000 样本）：减小 `D_MODEL` 至 64，`N_LAYERS` 至 2，避免过拟合
- **高波动品种**：提高 `DROPOUT` 至 0.3-0.4
- **长周期预测**（如预测 10 日收益）：增大 `LOOKBACK` 至 120，`PATCH_LEN` 至 20

## 架构切换机制

通过环境变量 `MODEL_ARCH` 控制模型架构：

```bash
# 使用 PatchTST (默认)
export MODEL_ARCH=patchtst
python scripts/train.py

# 使用 iTransformer
export MODEL_ARCH=itransformer
python scripts/train.py
```

**实现细节**：

在 `scripts/model.py` 中定义工厂函数 `build_model`：

```python
def build_model(arch: str, n_features: int, cfg: dict) -> nn.Module:
    """
    根据架构名构造模型
    
    Args:
        arch: 'patchtst' or 'itransformer'
        n_features: 特征维度
        cfg: 超参数字典
    
    Returns:
        PyTorch 模型实例
    """
    if arch == 'patchtst':
        return PatchTST(
            n_features=n_features,
            lookback=cfg['LOOKBACK'],
            patch_len=cfg['PATCH_LEN'],
            stride=cfg['STRIDE'],
            d_model=cfg['D_MODEL'],
            n_heads=cfg['N_HEADS'],
            n_layers=cfg['N_LAYERS'],
            dropout=cfg['DROPOUT']
        )
    elif arch == 'itransformer':
        return iTransformer(
            n_features=n_features,
            lookback=cfg['LOOKBACK'],
            d_model=cfg['D_MODEL'],
            n_heads=cfg['N_HEADS'],
            n_layers=cfg['N_LAYERS'],
            dropout=cfg['DROPOUT']
        )
    else:
        raise ValueError(f"Unknown architecture: {arch}. Use 'patchtst' or 'itransformer'.")
```

**注意事项**：

1. 切换架构后需要**重新训练**，checkpoint 不通用
2. iTransformer 显存占用约为 PatchTST 的 1.5-2 倍
3. 两种架构的 inference 接口一致，`factor.py` 无需修改

## 模型输入输出规范

### 输入

```python
# Shape: (batch_size, lookback, n_features)
# 示例: (64, 60, 58)
#   - 64 个样本
#   - 60 个交易日回溯
#   - 58 维特征 (OHLC + volume + amount + open_interest + 技术指标)
```

### 输出

```python
# Shape: (batch_size, 1)
# 示例: (64, 1)
#   - 每个样本输出一个标量，表示未来 5 日累计收益的预测值
```

## 模型保存格式

训练完成后，checkpoint 保存为：

```
checkpoints/
├── fold_0_best.pth
├── fold_1_best.pth
├── fold_2_best.pth
├── fold_3_best.pth
├── fold_4_best.pth
└── config.json  # 包含 MODEL_ARCH, LOOKBACK, D_MODEL 等超参数
```

每个 `.pth` 文件包含：

```python
{
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': best_epoch,
    'val_rank_ic': best_val_rank_ic,
    'train_loss': train_loss_history,
    'val_loss': val_loss_history
}
```

## 推理示例

```python
import torch
from scripts.model import build_model

# 加载配置
with open('checkpoints/config.json', 'r') as f:
    cfg = json.load(f)

# 构造模型
model = build_model(
    arch=cfg['MODEL_ARCH'],
    n_features=cfg['N_FEATURES'],
    cfg=cfg
)

# 加载最优 fold 的权重
checkpoint = torch.load('checkpoints/fold_0_best.pth', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 推理
with torch.no_grad():
    # X: (batch_size, 60, 58)
    predictions = model(X)  # (batch_size, 1)
```

## 架构对比实验建议

为了对比两种架构的表现，建议在相同数据和超参数下分别训练，记录以下指标：

| 指标 | PatchTST | iTransformer |
|---|---|---|
| Val Rank IC (mean) | | |
| Val Rank ICIR | | |
| Test Rank IC (mean) | | |
| Test ARR (%) | | |
| Test MDD (%) | | |
| 训练时长 (min) | | |
| 显存占用 (GB) | | |

**对比标准**：

- Rank IC 差异 > 0.02 且 p < 0.05 → 统计显著
- ARR 差异 > 5% → 策略层显著
- 显存或时间差异 > 50% → 工程层显著

如果两者表现接近，优先选择训练速度更快的架构（通常为 PatchTST）。
