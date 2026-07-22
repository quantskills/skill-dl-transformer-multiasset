# DL Transformer Multi-Asset - 研究目录

这是**研究和训练**目录，用于开发和验证模型。

## 目录结构

```
dl-transformer-multiasset/              # 研究目录
├── data/
│   ├── features.parquet               # 特征工程输出
│   └── database.parquet               # 因子推理临时输出（可选）
├── checkpoints/                       # 训练好的模型检查点
│   ├── fold_0_best.pth
│   ├── fold_1_best.pth
│   └── fold_2_best.pth
├── scripts/
│   ├── features.py                    # 特征工程
│   ├── train.py                       # 模型训练
│   ├── factor.py                      # 因子推理
│   ├── model.py                       # 模型定义
│   └── utils.py                       # 工具函数
├── run_factor.py                      # 一键因子推理（输出到production/data/）
└── run_pipeline.sh                    # 完整pipeline

dl-transformer-multiasset-production/  # 生产目录（只读查询）
└── data/
    ├── features.parquet               # 特征数据副本
    └── database.parquet               # 最终因子表
```

## 使用流程

### 1. 特征工程
```bash
cd dl-transformer-multiasset
export PANDA_DATA_START_DATE="20220401"
export PANDA_DATA_END_DATE="20241231"
export PANDA_DATA_USERNAME="your_username"
export PANDA_DATA_PASSWORD="your_password"

python -m scripts.features
```

输出: `data/features.parquet`

### 2. 模型训练
```bash
python -m scripts.train
```

输出: `checkpoints/fold_*_best.pth`

### 3. 因子推理（推荐）
```bash
# 自动输出到 ../dl-transformer-multiasset-production/data/database.parquet
python run_factor.py
```

或手动指定输出目录：
```bash
python -m scripts.factor \
    data/features.parquet \
    ../dl-transformer-multiasset-production/data \
    checkpoints/fold_0_best.pth \
    checkpoints/fold_1_best.pth \
    checkpoints/fold_2_best.pth
```

输出: 
- `../dl-transformer-multiasset-production/data/database.parquet` - 因子表
- `../dl-transformer-multiasset-production/data/features.parquet` - 特征数据副本（自动复制）

## 模型配置

当前使用**轻量化参数**以降低计算负载：

```python
DEFAULT_CFG = {
    "LOOKBACK": 40,      # 时间窗口长度
    "PATCH_LEN": 10,     # Patch大小
    "STRIDE": 5,         # Patch步长
    "D_MODEL": 64,       # 隐藏层维度
    "N_HEADS": 4,        # 注意力头数
    "N_LAYERS": 2,       # Transformer层数
    "BATCH_SIZE": 32,    # 批量大小
}
```

这些参数相比默认配置减少了约70%的计算量，适合在笔记本电脑上训练。

## 生产使用

训练完成后，生产系统应该从 `dl-transformer-multiasset-production/data/` 读取数据：
- `database.parquet` - 因子预测数据
- `features.parquet` - 特征数据（如需要）

参考 `dl-transformer-multiasset-production/SKILL.md` 了解如何查询因子数据。
