# 数据来源指南

本文档描述 Transformer 多资产建模因子所需的数据来源、接口规范、限流策略和数据清洗口径。

## 数据来源

**唯一数据源**：Panda data SDK。

不允许从本地文件或其他第三方数据源读取 OHLC、成交量、持仓量等作为正式输入。所有数据必须通过 Panda data SDK 实时获取。

## 环境变量

运行前必须设置以下环境变量：

```bash
export PANDA_DATA_USERNAME="your_username"
export PANDA_DATA_PASSWORD="your_password"
```

可选设置数据日期范围：

```bash
export PANDA_DATA_START_DATE="2020-01-01"  # 默认 2018-01-01
export PANDA_DATA_END_DATE="2024-12-31"    # 默认当前日期
```

**安全要求**：不得在代码中硬编码凭据，不得提交包含凭据的 `.env` 文件到版本控制系统。

## 接口规范

### 1. 品种清单接口

```python
from panda_data import PandaDataClient

client = PandaDataClient(username=os.environ['PANDA_DATA_USERNAME'],
                          password=os.environ['PANDA_DATA_PASSWORD'])

# 获取所有商品期货品种
df = client.get_future_detail()
commodity_symbols = df[df['product'] == 'Commodity']['underlying_symbol'].tolist()
```

**返回字段**：

| 字段 | 类型 | 说明 |
|---|---|---|
| underlying_symbol | str | 期货品种代码，如 `RB`（螺纹钢）、`CU`（铜） |
| product | str | 产品类型，过滤条件为 `'Commodity'` |
| exchange | str | 交易所代码，如 `SHFE`（上期所）、`DCE`（大商所）、`CZCE`（郑商所） |
| name_cn | str | 品种中文名称 |

### 2. 日线行情接口

```python
# 批量获取多个品种的日线行情 (后复权)
df = client.get_future_daily_post(
    underlying_symbol=commodity_symbols,
    start_date=start_date,
    end_date=end_date
)
```

**返回字段**：

| 字段 | 类型 | 说明 |
|---|---|---|
| date | datetime64 | 交易日期 |
| underlying_symbol | str | 期货品种代码 |
| open | float64 | 开盘价（后复权，已处理主力换月） |
| high | float64 | 最高价（后复权） |
| low | float64 | 最低价（后复权） |
| close | float64 | 收盘价（后复权） |
| volume | float64 | 成交量（手） |
| amount | float64 | 成交额（万元） |
| open_interest | float64 | 持仓量（手） |

**重要说明**：

- `get_future_daily_post` 返回的是**后复权价格**，已内置处理主力合约换月逻辑。
- 不需要自行拼接不同合约的价格序列。
- 换月时刻的价格跳跃已通过复权系数消除，保证价格序列连续性。

## 频率与限制

### 限流策略

Panda data API 存在速率限制，必须遵守以下规则：

1. **批量查询**：每次最多查询 5 个品种，超过 5 个需要分批。
2. **请求间隔**：每批请求之间 sleep 2 秒。
3. **错误重试**：遇到以下错误码时，sleep 5 秒后重试（最多 3 次）：
   - `500010`：速率限制 (rate limit exceeded)
   - `200004`：token 过期 (token expired, need re-login)

示例代码：

```python
import time

def fetch_daily_data(client, symbols, start_date, end_date):
    batch_size = 5
    all_data = []
    
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i+batch_size]
        retry_count = 0
        max_retries = 3
        
        while retry_count < max_retries:
            try:
                df = client.get_future_daily_post(
                    underlying_symbol=batch,
                    start_date=start_date,
                    end_date=end_date
                )
                all_data.append(df)
                break
            except Exception as e:
                error_code = getattr(e, 'code', None)
                if error_code in ['500010', '200004']:
                    retry_count += 1
                    if retry_count < max_retries:
                        print(f"Retry {retry_count}/{max_retries} after error {error_code}")
                        time.sleep(5)
                    else:
                        raise
                else:
                    raise
        
        # 批次间隔
        if i + batch_size < len(symbols):
            time.sleep(2)
    
    return pd.concat(all_data, ignore_index=True)
```

## 数据清洗口径

### 1. 上市时间过滤

品种上市不足 250 个交易日时，**剔除该品种的所有数据**。

```python
symbol_counts = df.groupby('underlying_symbol')['date'].nunique()
valid_symbols = symbol_counts[symbol_counts >= 250].index.tolist()
df = df[df['underlying_symbol'].isin(valid_symbols)]
```

**理由**：新上市品种交易不活跃，价格发现机制不完善，会引入噪声。

### 2. 滚动窗口过滤

构造 Transformer 输入时，需要 60 个交易日的 lookback 窗口。如果某个品种在某日的回溯窗口内数据不足 60 个交易日，**丢弃该样本**。

```python
def filter_by_lookback(df, lookback=60):
    valid_samples = []
    for symbol in df['underlying_symbol'].unique():
        symbol_df = df[df['underlying_symbol'] == symbol].sort_values('date')
        if len(symbol_df) < lookback:
            continue
        # 保留从第 lookback 个交易日开始的所有样本
        valid_samples.append(symbol_df.iloc[lookback-1:])
    return pd.concat(valid_samples, ignore_index=True)
```

**理由**：窗口不足会导致特征计算错误或需要 forward fill，违反无未来函数原则。

### 3. 异常值处理

对于 OHLC 价格字段，检测以下异常：

- **价格为 0 或负数**：丢弃该行
- **涨跌幅超过 ±30%**（单日）：标记为异常，但保留（期货涨跌停板可达 10-15%，极端行情可能触发）

对于 volume、amount、open_interest：

- **为 0**：保留（可能是节假日前后的正常现象）
- **为负数**：丢弃该行

```python
# 价格异常值
df = df[(df['open'] > 0) & (df['high'] > 0) & (df['low'] > 0) & (df['close'] > 0)]

# 成交量/持仓量异常值
df = df[(df['volume'] >= 0) & (df['amount'] >= 0) & (df['open_interest'] >= 0)]
```

### 4. 节假日与停牌

Panda data 返回的数据**不包含**停牌日和节假日，因此不需要单独过滤。

### 5. 主力合约换月

**Panda data 已处理**，`get_future_daily_post` 返回的是主力合约后复权价格，换月逻辑已内置。不需要自行判断合约切换时间点或手动拼接价格序列。

## 数据更新频率

- **日线数据**：T+1 日早上 9:00 前更新前一交易日数据
- **回测场景**：假设因子在 `t` 日收盘后生成，使用 `t` 日及以前的数据，`t+1` 日开盘可交易

## 常见问题

### Q1: 如何验证数据无未来函数？

A: 构造标签时，使用 `t+1` 至 `t+5` 日的收益，确保 `t` 日的特征只使用 `≤ t` 日的数据。

```python
# 正确：使用 t 日及以前的 60 日数据
features_t = df[(df['date'] <= t) & (df['date'] > t - pd.Timedelta(days=100))].tail(60)

# 错误：使用了 t+1 日的数据
features_t = df[(df['date'] <= t + pd.Timedelta(days=1))].tail(60)  # ❌
```

### Q2: 为什么要过滤 product='Commodity'？

A: Panda data 返回的 `get_future_detail` 包含股指期货、国债期货等，本因子专注于商品期货，因此需要过滤。

### Q3: 数据缺失如何处理？

A: **不允许 forward fill 或 backward fill**，缺失日期直接跳过。如果回溯窗口内缺失过多，丢弃该样本。

### Q4: 如何处理极端行情（涨跌停）？

A: 保留涨跌停数据，这是真实市场行为的一部分。Transformer 模型应学习到涨跌停后的价格行为模式。
