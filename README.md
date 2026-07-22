# skill-dl-transformer-multiasset

Transformer 多资产商品期货因子,通过 PatchTST 或 iTransformer 预测未来 5 日横截面收益率。

## 概述

本仓库为量化研究工具库,提供基于 Transformer 的多资产建模能力,针对商品期货横截面建模。

**核心功能**:
- 使用 PatchTST / iTransformer 对多品种期货进行联合建模
- 预测未来 5 日横截面收益率
- 生成因子值供后续策略使用

## 目录结构

```
skill-dl-transformer-multiasset/
│
├── dl-transformer-multiasset/              # 研究目录
│   ├── data/ -> ../production/data/        # 符号链接
│   ├── checkpoints/                        # 训练好的模型
│   ├── output/                             # 验证临时目录
│   ├── scripts/                            # 脚本
│   ├── run_factor.py                       # 因子推理
│   ├── run_validate.py                     # 验证
│   ├── README.md                           # 详细文档
│   └── VALIDATION.md                       # 验证指南
│
└── dl-transformer-multiasset-production/   # 生产目录 (只读)
    └── data/
        ├── features.parquet               # 特征数据
        ├── database.parquet               # 因子数据 ✓
        └── README.md                       # 数据说明
```

## 快速开始

### 1. 特征工程 (一次性)
```bash
cd dl-transformer-multiasset

export PANDA_DATA_START_DATE="20220401"
export PANDA_DATA_END_DATE="20241231"
export PANDA_DATA_USERNAME="your_username"
export PANDA_DATA_PASSWORD="your_password"

python -m scripts.features
```
**输出**: `data/features.parquet` (通过符号链接指向production/data/)

### 2. 模型训练 (一次性或定期更新)
```bash
python -m scripts.train
```
**输出**: `checkpoints/fold_*_best.pth` (3个模型文件)

### 3. 因子推理
```bash
python run_factor.py
```
**输出**: `../dl-transformer-multiasset-production/data/database.parquet`

### 4. 验证
```bash
python run_validate.py
```
**验证通过**: 因子数据可以安全使用

## 数据使用

### 在Python中查询因子

```python
import pandas as pd

# 读取因子数据
df = pd.read_parquet(
    'dl-transformer-multiasset-production/data/database.parquet'
)

# 查看结构
print(df.head())
print(df.columns)

# 筛选买入信号
buy_signals = df[df['signal'] == 'buy'].sort_values(
    ['trade_date', 'rank'], ascending=[True, False]
)

# 获取最新日期的top10
latest_date = df['trade_date'].max()
top10 = df[
    (df['trade_date'] == latest_date) & 
    (df['signal'] == 'buy')
].nlargest(10, 'factor_value')

print(top10[['symbol', 'factor_value', 'score', 'rank']])
```

## 因子数据格式

| 列名 | 类型 | 说明 | 示例 |
|------|------|------|------|
| trade_date | string | 交易日期 | "20241231" |
| asset_type | string | 资产类型 | "futures" |
| symbol | string | 股票代码 | "AG_DOMINANT.SHF" |
| factor_id | string | 因子ID | "DLTX" |
| factor_name | string | 因子名称 | "Deep Learning Transformer Multi-Asset" |
| factor_value | float | 因子值 | 0.123 |
| score | float | 归一化分数 [0,1] | 0.856 |
| rank | int | 每日排名 | 65 |
| signal | string | 交易信号 | "buy" / "sell" / "hold" |
| confidence | float | 置信度 | 1.0 |
| data_version | string | 数据版本 | "v1" |
| update_time | datetime | 更新时间 | "2026-07-21 20:40:03" |

## 模型参数 (轻量化配置)

当前使用的模型参数经过优化，适合在笔记本电脑上训练：

- **LOOKBACK**: 40 (时间窗口)
- **D_MODEL**: 64 (隐藏层维度)
- **N_HEADS**: 4 (注意力头数)
- **N_LAYERS**: 2 (Transformer层数)
- **BATCH_SIZE**: 32

这些参数相比默认配置减少了约70%的计算量，同时保持了良好的预测效果。

## 性能统计

- **训练数据**: 41,033行 × 68特征
- **因子输出**: 38,243行预测
- **股票数量**: 71个
- **日期范围**: 2022-08-26 至 2024-12-31
- **信号分布**: 11% buy, 9.5% sell, 79.5% hold

## 常见问题

### Q: 如何更新因子数据？
A: 重新运行步骤3和4：
```bash
python run_factor.py   # 重新推理
python run_validate.py # 验证
```

### Q: data目录的符号链接是什么？
A: 为了避免数据重复，研究目录的`data/`直接链接到production的`data/`：
```bash
ln -s ../dl-transformer-multiasset-production/data data
```

### Q: 验证失败怎么办？
A: 查看 `VALIDATION.md` 了解详细的验证流程和故障排查。

## 依赖

- Python 3.8+
- panda-data SDK
- PyTorch
- 其他依赖见各子目录

## 常量配置

- `FACTOR_ID = "DLTX"`
- `FACTOR_NAME = "Transformer多资产联合建模"`
- `DATA_VERSION = "real-v1"`

## 许可证

GPL-3.0-only

## 边界声明

本仓库为研究与工程材料。**不构成投资建议、不承诺收益、不代表 QuantSkills / Panda data / Codex / Claude Code / Cursor / Hermes / OpenClaw 的官方背书。** 不得记录或提交 Panda data 凭据。

## 下一步

验证通过后，可以：
1. 在回测系统中使用 `database.parquet`
2. 接入实盘交易系统
3. 与其他因子进行组合优化
4. 定期重新训练模型（建议每季度一次）
