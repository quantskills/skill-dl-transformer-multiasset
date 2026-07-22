---
name: dl-transformer-multiasset
description: Use when developing, training or validating Transformer-based multi-asset commodity futures factor in a local Panda data environment.
license: GPL-3.0-only
tags: [quant, deep-learning, transformer, development, future]
---

# Transformer 多资产联合建模 Alpha

## 适用场景

- 当用户需要训练 Transformer 模型预测商品期货横截面收益时
- 当用户需要对比 PatchTST 与 iTransformer 两种架构在多资产建模场景下的表现时
- 当用户需要生成商品期货横截面 `buy` / `sell` / `hold` 候选信号时

## 因子逻辑

- 核心假设：商品期货价格存在跨品种动量与反转的非线性交互模式,Transformer 架构能够学习多变量时序中的长距离依赖与横截面关联。
- 模型架构：默认使用 **PatchTST** (channel-independent patch-based Transformer)，可通过环境变量 `MODEL_ARCH=itransformer` 切换为 **iTransformer** (variate-as-token 架构)。
- 输入窗口：60 交易日 lookback 的量价特征和工程特征（约 58 维），包含 OHLC + volume + amount + open_interest + MA/EMA/RSI/ATR 等技术指标。
- 标签构造：5 日累计收益的横截面 rank normalization (0-1 归一化)，即 `(rank - rank.min()) / (rank.max() - rank.min())`。
- 损失函数：`Loss = 0.5 × rank_ic_loss + 0.5 × mse`，其中 `rank_ic_loss = 1 - spearmanr(pred, label)`。
- 样本外验证：Walk-forward 5 折交叉验证，每折训练集递增，保证无未来函数。
- 信号生成：每日横截面对预测值排序，rank ≤ 前 10% → `buy`，rank ≥ 后 10% → `sell`，其余 → `hold`（`BUY_QUANTILE=SELL_QUANTILE=0.1`）。
- 排序方向：预测值越大越好（横截面降序排名，rank=1 为最高）。

## 输入数据

因子计算必须使用 Panda data SDK，不读取本地表格作为正式输入。

| 字段 | 说明 | 来源 |
|---|---|---|
| date | 交易日期 | `panda_data.get_future_daily_post` |
| underlying_symbol | 期货品种代码 | `panda_data.get_future_detail` (过滤 `product='Commodity'`) |
| open | 开盘价 (后复权) | `panda_data.get_future_daily_post` |
| high | 最高价 (后复权) | `panda_data.get_future_daily_post` |
| low | 最低价 (后复权) | `panda_data.get_future_daily_post` |
| close | 收盘价 (后复权) | `panda_data.get_future_daily_post` |
| volume | 成交量 | `panda_data.get_future_daily_post` |
| amount | 成交额 | `panda_data.get_future_daily_post` |
| open_interest | 持仓量 | `panda_data.get_future_daily_post` |

**说明**：Panda data 已处理主力换月逻辑，返回的是后复权价格，不需要自行拼合约。

## 输出结果

| 字段 | 说明 |
|---|---|
| trade_date | `YYYY-MM-DD` 交易日 |
| asset_type | 固定为 `future` |
| symbol | 期货品种代码 |
| factor_id | 固定为 `DLTX` |
| factor_name | 固定为 `Transformer多资产联合建模` |
| factor_value | Transformer 模型输出的预测值（5日累计收益预测），不限制在 [-1, 1] |
| score | 当日横截面百分位评分，范围 `[0, 100]` |
| rank | 当日横截面整数排名，`1` 为最优 |
| signal | `buy` / `sell` / `hold` |
| confidence | `score / 100` |
| data_version | 固定为 `real-v1` |
| update_time | ISO8601 生成时间 |

## 因子评价标准

Alpha 任务需要同时报告因子预测能力和策略层表现。

| 分类 | 指标 | 方向 | 说明 |
|---|---|---|---|
| Factor Predictive Power | `IC` | 越高越好 | 因子值与下一期收益的 Pearson 相关 |
| Factor Predictive Power | `ICIR` | 越高越好 | IC 均值 / IC 标准差 |
| Factor Predictive Power | `Rank IC` | 越高越好 | 因子排名与下一期收益排名的 Spearman 相关 |
| Factor Predictive Power | `Rank ICIR` | 越高越好 | Rank IC 均值 / Rank IC 标准差 |
| Strategy Performance | `IR(SHR*)` | 越高越好 | 多空组合收益的信息比率/年化 Sharpe 近似 |
| Strategy Performance | `CR` | 越高越好 | 累计收益 / 最大回撤绝对值 |
| Strategy Performance | `ARR(%)` | 越高越好 | 年化收益率 |
| Strategy Performance | `MDD(%)` | 越低越好 | 最大回撤 |

硬性要求：

- 不允许未来函数：因子在 `t` 日形成时，只能使用 `t` 日及以前可获得的 OHLC + volume + amount + open_interest。
- 回测收益使用 Method A：`t` 日主力同一合约的 `t+1 close / t close - 1`，换月价差不计入收益。
- MVP 不计手续费、滑点、保证金占用和换仓成本。

## 使用方式

```bash
# Step 1: 生成特征工程表 (含标签)
python scripts/features.py

# Step 2: 训练模型 (默认 PatchTST, 5 折 walk-forward)
python scripts/train.py

# Step 3: 生成因子表 (使用最优 checkpoint)
python scripts/factor.py

# Step 4: 验证因子表规范性
python scripts/validate.py

# Step 5: 回测因子表现
python scripts/backtest.py
```

运行前需要设置 `PANDA_DATA_USERNAME` 和 `PANDA_DATA_PASSWORD`。可选设置 `PANDA_DATA_START_DATE`、`PANDA_DATA_END_DATE`、`MODEL_ARCH`（`patchtst` 或 `itransformer`）、`TRAIN_DEVICE`（`auto`/`cuda`/`mps`/`cpu`）。

## Agent 执行规则

1. 先运行 `scripts/features.py`，确认 Panda data 返回真实 OHLC + volume + amount + open_interest，且特征表非空。报告数据日期范围、品种数量、特征维度。
2. 再运行 `scripts/train.py`，确认 5 折 walk-forward 训练完成，每折 val Rank IC > 0。报告最优 fold、最优 val Rank IC、checkpoint 路径、训练设备、训练时长。
3. 再运行 `scripts/factor.py`，确认因子结果包含 12 个必需字段，`factor_id=DLTX`，`data_version=real-v1`。报告因子表行数、日期范围、品种数量。
4. 再运行 `scripts/validate.py`，确认无未来函数、字段完整、取值范围合法、信号枚举合法、样本外切片可用。报告验证通过信息。
5. 最后运行 `scripts/backtest.py`，输出两组指标：
   - **研究口径**：`IC`/`ICIR`/`IC_p`/`Rank IC`/`Rank ICIR`/`Bootstrap IC 95% CI`/多周期 IC 衰减（1D/2D/3D/5D/10D/20D）/`ARR(%)`/`MDD(%)`/`CR`/`分层收益`/`多空收益`/`换手率`/`换月次数`/`成本敏感性`（0/0.05%/0.15%/0.30% 四档）。
   - **可交易口径**：`tradeable_ARR(%)`/`tradeable_IR`（data_lag=1 日，换月成本 5bps，交易成本 5bps）。
   - **基线对比**：与横截面动量基线（5日收益 rank）、随机信号基线的 IC/IR/ARR 差异，要求 Transformer 在 IC 和 ARR 上均优于两个基线且差异统计显著（p < 0.05）。
6. 如果任一步失败，必须报告失败命令、错误信息和数据日期，不得进入生产。

## 成功标准

- `features.py` 输出非空，包含至少 58 维特征和 1 维标签列 `label_5d_rank`。
- `train.py` 输出 5 折 checkpoint，每折 val Rank IC > 0，最优 fold val Rank IC > 0.05。
- `factor.py` 输出包含 12 个必需字段，`factor_id=DLTX`，`data_version=real-v1`。
- `validate.py` 输出验证通过信息。
- `backtest.py` 输出研究口径和可交易口径两组指标，且在 IC 和 ARR 上显著优于横截面动量基线和随机信号基线。
- 数据来源为 Panda data SDK，不读取本地文件作为正式输入。

## 边界

本仓库为研究与工程材料。**不构成投资建议、不承诺收益、不代表 QuantSkills / Panda data / Codex / Claude Code / Cursor / Hermes / OpenClaw 的官方背书。** 不得记录或提交 Panda data 凭据。

## 验收要求

- 不允许未来函数。
- 必须有样本外验证（walk-forward 5 折满足此要求）。
- 必须有回测指标。
- 必须使用 Panda data 或项目指定数据源实现。
- 不通过 `validate.py` 验证不得进入生产。

## 依赖

- Python 3.10+
- panda-data
- torch >= 2.1
- pandas
- numpy
- scipy
- pyarrow 或 fastparquet
- matplotlib

## 与生产产物的关系

开发产物用于训练、验证和回测；生产产物用于读取已生成结果。在生产查询时应使用 `dl-transformer-multiasset-production`，不要临时重算因子。
