# 训练流程指南

本文档描述 Transformer 多资产建模因子的训练流程、设备策略、混合精度训练、训练时长基线和常见故障排查。

## 设备策略

训练脚本通过环境变量 `TRAIN_DEVICE` 控制运行设备：

| 环境变量值 | 行为 | 优先级 |
|---|---|---|
| `auto` (默认) | 自动检测，按优先级选择 | cuda > mps > cpu |
| `cuda` | 强制使用 NVIDIA GPU (CUDA) | - |
| `mps` | 强制使用 Apple Silicon GPU (MPS) | - |
| `cpu` | 强制使用 CPU | - |

**自动检测逻辑**：

```python
def get_device(device_str: str = 'auto') -> torch.device:
    if device_str == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    elif device_str == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        return torch.device('cuda')
    elif device_str == 'mps':
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS not available")
        return torch.device('mps')
    elif device_str == 'cpu':
        return torch.device('cpu')
    else:
        raise ValueError(f"Unknown device: {device_str}")
```

**使用示例**：

```bash
# 自动检测设备
python scripts/train.py

# 强制使用 CUDA
export TRAIN_DEVICE=cuda
python scripts/train.py

# 强制使用 MPS (Apple Silicon Mac)
export TRAIN_DEVICE=mps
python scripts/train.py

# 强制使用 CPU (不推荐，仅用于调试小数据集)
export TRAIN_DEVICE=cpu
export ALLOW_CPU_TRAIN=1  # 需要显式允许
python scripts/train.py
```

## MPS 特殊处理

Apple Silicon (M1/M2/M3) 的 MPS 后端存在以下限制，需要特殊处理：

### 1. 关闭 cuDNN 确定性

```python
if device.type == 'mps':
    # MPS 不支持 torch.backends.cudnn.deterministic
    # 使用 torch.manual_seed 保证可复现性
    torch.manual_seed(cfg['SEED'])
```

### 2. 算子 Fallback

部分算子在 MPS 上未实现，会自动 fallback 到 CPU。如果遇到警告：

```
[W MPSFallback.mm:11] Warning: The operator 'aten::some_op' is not currently supported on the MPS backend...
```

设置环境变量允许 fallback：

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
python scripts/train.py
```

**常见 fallback 算子**：

- `torch.argsort` (stable=True)
- `torch.istft`
- 部分高级索引操作

### 3. Checkpoint 存储为 CPU State Dict

MPS tensor 无法直接保存到磁盘，需要先移动到 CPU：

```python
# 保存 checkpoint
checkpoint = {
    'model_state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
    'optimizer_state_dict': {k: v.cpu() if isinstance(v, torch.Tensor) else v 
                              for k, v in optimizer.state_dict().items()},
    ...
}
torch.save(checkpoint, 'checkpoints/fold_0_best.pth')
```

加载时再移动回 MPS：

```python
checkpoint = torch.load('checkpoints/fold_0_best.pth', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
```

## 训练时长基线

基于 5 折 walk-forward 训练，每折 50 epochs（实际会 early stopping）：

| 设备类型 | 样本量 | 单 fold 时长 (分钟) | 5 fold 总时长 (分钟) |
|---|---|---|---|
| NVIDIA RTX 4090 (24GB) | 50000 | 5-8 | 25-40 |
| NVIDIA RTX 3090 (24GB) | 50000 | 8-12 | 40-60 |
| Apple M2 Max (MPS) | 50000 | 15-25 | 75-125 |
| Apple M1 Pro (MPS) | 50000 | 25-40 | 125-200 |
| Intel i9-12900K (CPU) | 10000 | 60-90 | 300-450 |

**影响因素**：

- 样本量：线性关系，样本量翻倍训练时长翻倍
- Batch size：越大越快，但显存占用越高（建议 64-128）
- `D_MODEL` 和 `N_LAYERS`：越大越慢，显存占用越高

**预估公式**：

```
单 fold 时长 (分钟) ≈ (样本量 / 10000) × 设备基准时长 × (D_MODEL / 128)^2
```

## CPU 拦截规则

为避免 CPU 训练耗时过长，设置以下拦截规则：

```python
def check_cpu_training(device, n_samples):
    if device.type == 'cpu' and n_samples > 10000:
        if os.environ.get('ALLOW_CPU_TRAIN') != '1':
            raise RuntimeError(
                f"Training on CPU with {n_samples} samples may take hours. "
                f"If you really want to proceed, set ALLOW_CPU_TRAIN=1"
            )
        else:
            print(f"[WARNING] Training on CPU with {n_samples} samples. This may take a long time.")
```

**使用示例**：

```bash
# 小数据集 (<10000 样本) 可以直接在 CPU 上训练
python scripts/train.py  # 自动检测，如果只有 CPU 会使用 CPU

# 大数据集 (>10000 样本) 需要显式允许
export ALLOW_CPU_TRAIN=1
python scripts/train.py
```

## 混合精度训练

不同设备的混合精度支持不同：

| 设备类型 | 混合精度方案 | 加速比 | 显存节省 |
|---|---|---|---|
| CUDA (Ampere 及以上) | `autocast` + `GradScaler` | 1.5-2× | 30-40% |
| CUDA (Volta/Turing) | `autocast` + `GradScaler` | 1.2-1.5× | 20-30% |
| MPS | `autocast` only (no `GradScaler`) | 1.1-1.3× | 10-20% |
| CPU | 不支持 (强制 fp32) | - | - |

**实现细节**：

```python
def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    
    # CUDA: 使用 autocast + GradScaler
    if device.type == 'cuda':
        scaler = torch.cuda.amp.GradScaler()
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                pred = model(X)
                loss = compute_loss(pred, y)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
    
    # MPS: 只使用 autocast (不支持 GradScaler)
    elif device.type == 'mps':
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            
            with torch.amp.autocast('cpu'):  # MPS 需要使用 'cpu' 标识
                pred = model(X)
                loss = compute_loss(pred, y)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    
    # CPU: 不使用混合精度
    else:
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(X)
            loss = compute_loss(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    
    return total_loss / len(dataloader)
```

**注意事项**：

1. MPS 的 `autocast` 需要使用 `torch.amp.autocast('cpu')` 而非 `torch.cuda.amp.autocast()`
2. MPS 不支持 `GradScaler`，梯度累积需要手动实现
3. 混合精度可能导致数值不稳定，如果出现 NaN/Inf，尝试关闭混合精度

## 常见故障排查

### 1. OOM (Out of Memory)

**现象**：

```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB ...
```

**解决方案**（按优先级）：

1. 减小 `BATCH_SIZE`：
   ```bash
   export BATCH_SIZE=32  # 默认 64
   python scripts/train.py
   ```

2. 减小 `D_MODEL`：
   ```bash
   export D_MODEL=64  # 默认 128
   python scripts/train.py
   ```

3. 减少 `N_LAYERS`：
   ```bash
   export N_LAYERS=2  # 默认 3
   python scripts/train.py
   ```

4. 使用梯度累积（等效于更大 batch size）：
   ```python
   accumulation_steps = 4
   for i, (X, y) in enumerate(dataloader):
       loss = compute_loss(model(X), y) / accumulation_steps
       loss.backward()
       if (i + 1) % accumulation_steps == 0:
           optimizer.step()
           optimizer.zero_grad()
   ```

### 2. Val Rank IC 一直为负

**现象**：

```
Fold 0 - Epoch 10: train_loss=0.15, val_loss=0.18, val_rank_ic=-0.12
Fold 0 - Epoch 20: train_loss=0.12, val_loss=0.19, val_rank_ic=-0.15
```

**可能原因**：

1. **标签未来函数**：标签使用了 `t+1` 日及以后才能获得的数据
2. **标签构造错误**：5 日累计收益计算错误（如使用了收盘价而非开盘价）
3. **数据泄漏**：特征工程时使用了未来数据（如全局归一化而非滚动归一化）

**排查步骤**：

```python
# 检查标签构造
df['label_5d'] = df.groupby('underlying_symbol')['close'].pct_change(5).shift(-5)
# 正确: shift(-5) 表示使用未来 5 日收益作为标签

# 检查特征归一化
df['feature_norm'] = df.groupby('date')['feature'].rank(pct=True)
# 正确: 按日期分组，横截面 rank normalization

# 检查数据泄漏
assert df['label_5d'].isna().sum() > 0, "标签不应全部有值 (最后 5 日应为 NaN)"
```

### 3. 训练不收敛

**现象**：

```
Fold 0 - Epoch 10: train_loss=0.45, val_loss=0.50
Fold 0 - Epoch 20: train_loss=0.44, val_loss=0.49
Fold 0 - Epoch 30: train_loss=0.43, val_loss=0.49
```

**可能原因**：

1. **特征未归一化**：量价特征尺度差异大（如成交额 1e8，收益率 1e-2）
2. **学习率过高**：导致梯度震荡
3. **Batch size 过小**：梯度估计噪声大

**解决方案**：

1. 检查特征归一化（必须按日横截面归一化）：
   ```python
   # 错误: 全局 StandardScaler
   scaler = StandardScaler()
   df[features] = scaler.fit_transform(df[features])  # ❌
   
   # 正确: 按日横截面 rank normalization
   for col in features:
       df[col] = df.groupby('date')[col].rank(pct=True)  # ✓
   ```

2. 降低学习率：
   ```bash
   export LR=5e-5  # 默认 1e-4
   python scripts/train.py
   ```

3. 增大 Batch size：
   ```bash
   export BATCH_SIZE=128  # 默认 64
   python scripts/train.py
   ```

### 4. MPS 算子不支持

**现象**：

```
NotImplementedError: The operator 'aten::argsort' is not currently implemented for the MPS device.
```

**解决方案**：

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
python scripts/train.py
```

如果 fallback 后仍报错，切换到 CPU：

```bash
export TRAIN_DEVICE=cpu
export ALLOW_CPU_TRAIN=1
python scripts/train.py
```

### 5. CUDA 版本不兼容

**现象**：

```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

**解决方案**：

1. 检查 PyTorch 和 CUDA 版本匹配：
   ```bash
   python -c "import torch; print(torch.version.cuda)"
   nvcc --version  # 应匹配
   ```

2. 重新安装匹配的 PyTorch：
   ```bash
   # CUDA 11.8
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   
   # CUDA 12.1
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

## 训练监控

训练过程中建议监控以下指标：

```python
# 每 epoch 输出
print(f"Fold {fold} - Epoch {epoch}: "
      f"train_loss={train_loss:.4f}, "
      f"val_loss={val_loss:.4f}, "
      f"val_rank_ic={val_rank_ic:.4f}, "
      f"lr={optimizer.param_groups[0]['lr']:.2e}, "
      f"time={epoch_time:.1f}s")
```

**正常训练的标志**：

- `train_loss` 和 `val_loss` 同时下降
- `val_rank_ic` 在前 10 epoch 内变为正值
- `val_rank_ic` 最终稳定在 0.05-0.15 之间（商品期货典型范围）
- Early stopping 在 20-40 epoch 触发

**异常训练的标志**：

- `train_loss` 下降但 `val_loss` 上升 → 过拟合
- `val_rank_ic` 始终为负 → 标签或特征问题
- Loss 出现 NaN/Inf → 数值不稳定（降低学习率或关闭混合精度）

## 调试小数据集

如果需要快速验证代码逻辑，可以使用小数据集：

```bash
export PANDA_DATA_START_DATE="2023-01-01"
export PANDA_DATA_END_DATE="2023-06-30"
export TRAIN_DEVICE=cpu
export ALLOW_CPU_TRAIN=1
python scripts/train.py
```

预期输出：

```
[INFO] Loaded 5000 samples, 58 features
[INFO] Walk-forward 5 folds, train sizes: [2000, 2500, 3000, 3500, 4000]
Fold 0 - Epoch 5: train_loss=0.25, val_loss=0.28, val_rank_ic=0.08
Fold 0 - Best val_rank_ic=0.12 at epoch 15
...
[INFO] All folds completed. Best fold: 2 (val_rank_ic=0.15)
```
