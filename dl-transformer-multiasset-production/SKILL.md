---
name: dl-transformer-multiasset-production
description: Use when reading the local deep learning transformer multi-asset factor parquet output without recomputing the factor.
license: GPL-3.0-only
tags: [quant, alpha, production, deep-learning, multi-asset]
---

# Deep Learning Transformer Multi-Asset Factor 生产结果

## 适用场景

- 当用户需要查询深度学习 Transformer 多资产因子最新结果时
- 当交易 agent 需要使用该因子辅助股票、期货、指数等多资产交易判断时

## 结果文件

- 文件路径：`claude_code_skills/skill-dl-transformer-multiasset/dl-transformer-multiasset-production/database.parquet`（相对于项目根目录）
- 数据格式：Parquet
- 更新频率：每日收盘后
- 生成方式：开发产物 `scripts/factor.py` 通过验证后生成

## 主键

- `trade_date`
- `asset_type`
- `symbol`
- `factor_id`

## 字段说明

| 字段 | 类型 | 说明 |
|---|---|---|
| trade_date | string | 交易日期，`YYYY-MM-DD` |
| asset_type | string | 资产类型：`stock` / `future` / `index` |
| symbol | string | 资产代码（股票代码 / 期货品种代码 / 指数代码） |
| factor_id | string | 固定为 `dl_transformer_multiasset` |
| factor_name | string | 固定为 `Deep Learning Transformer Multi-Asset` |
| factor_value | float | 因子预测值（模型推理输出） |
| score | float | 当日横截面百分位评分，范围 `[0, 100]` |
| rank | int | 当日横截面排名 |
| signal | string | `buy` / `sell` / `hold` |
| confidence | float | 固定为 `1.0` |
| data_version | string | 固定为 `v1` |
| update_time | string | ISO8601 生成时间 |

## 读取规则

交易 agent 读取 `database.parquet`，使用 `scripts/query.py` 筛选目标资产类型、日期范围、品种和信号。优先使用最新有效交易日结果。若最新结果不存在，可回退最近有效交易日，但必须说明数据日期。

### 查询接口

```bash
# 查询所有数据
python scripts/query.py

# 按日期范围筛选
python scripts/query.py --start 2024-01-01 --end 2024-12-31

# 按品种筛选
python scripts/query.py --symbols 000001.SZ,IF2401,600000.SH

# 按信号筛选
python scripts/query.py --signals buy,sell

# 组合筛选
python scripts/query.py --start 2024-01-01 --symbols 000001.SZ --signals buy
```

### Python API

```python
from scripts.query import query

# 筛选数据
result = query(
    db_path="database.parquet",
    start="2024-01-01",
    end="2024-12-31",
    symbols=["000001.SZ", "IF2401"],
    signals=["buy", "sell"]
)
```

## 信号解释

- `buy`: 模型预测未来收益为正，横截面排名靠前（top 30%），建议做多。
- `sell`: 模型预测未来收益为负，横截面排名靠后（bottom 30%），建议做空或规避。
- `hold`: 模型预测信号不明确或横截面排名居中（30%-70%），建议观望。

## 模型架构

- 基础模型：PatchTST (Patched Time Series Transformer)
- 输入特征：60 日技术指标序列（收益率、波动率、成交量等）
- 预测目标：未来 1 日收益率
- 训练方式：Walk-forward cross-validation（6 折）
- 横截面标准化：每日对所有资产进行排名和打分

## 禁止行为

- 不允许在 agent 调用时重新拉取原始行情。
- 不允许在 agent 调用时重新计算因子。
- 不允许手工修改 Parquet 结果。
- 不允许将结果表述为投资建议、收益承诺或官方背书。
