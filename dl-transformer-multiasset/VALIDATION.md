# 验证指南

## 验证流程

验证脚本会检查生成的因子数据是否符合规范，包括：

1. ✅ **必填字段检查** - 确保12列都存在
2. ✅ **数值范围检查** - 确保score在[0,1]，confidence等于1
3. ✅ **信号枚举检查** - 确保signal只包含buy/sell/hold
4. ✅ **样本外切片检查** - 确保有足够的OOS数据
5. ✅ **无未来函数检查** - 确保不使用未来数据
6. ✅ **检查点存在检查** - 确保模型文件存在

## 运行验证

### 方法1: 使用辅助脚本（推荐）

```bash
cd dl-transformer-multiasset
python run_validate.py
```

这个脚本会自动：
1. 创建 `output/` 目录
2. 从 `../dl-transformer-multiasset-production/data/` 复制必要文件
3. 运行所有验证检查

### 方法2: 手动准备并验证

```bash
cd dl-transformer-multiasset

# 1. 创建output目录并复制文件
mkdir -p output
cp ../dl-transformer-multiasset-production/data/database.parquet output/factor.parquet
cp ../dl-transformer-multiasset-production/data/features.parquet output/features.parquet

# 2. 运行验证
python -m scripts.validate
```

## 验证输出示例

成功的验证输出：

```
Running validation checks...
Check 1/6: Required fields...
Check 2/6: Value ranges...
Check 3/6: Signal enum...
Check 4/6: Out-of-sample slice...
Check 5/6: No future function...
Check 6/6: Checkpoints exist...

✓ All validation checks passed!
```

## 常见问题

### 1. FileNotFoundError: Factor output not found

**问题**: 没有找到 `output/factor.parquet`

**解决**: 
```bash
# 确保已经运行过因子推理
python run_factor.py

# 然后运行验证
python run_validate.py
```

### 2. ValueError: All factor_id must be 'DLTX'

**问题**: factor_id 不匹配

**解决**: 确保 `scripts/factor.py` 使用了正确的 `FACTOR_ID` 常量：
```python
from scripts.utils import FACTOR_ID
# ...
df["factor_id"] = FACTOR_ID  # 应该是 "DLTX"
```

### 3. AssertionError: 存在未来函数嫌疑

**问题**: 检测到使用了未来数据

**解决**: 
- 检查特征工程代码，确保所有特征只使用历史数据
- 检查rolling window计算是否正确
- 确保没有使用shift(-1)等前视操作

### 4. No such file: data/features.parquet

**问题**: 研究目录缺少 data 符号链接

**解决**:
```bash
cd dl-transformer-multiasset
ln -s ../dl-transformer-multiasset-production/data data
```

## 验证详细说明

### Check 1: Required fields
确保输出包含12个必填列：
- trade_date, asset_type, symbol
- factor_id, factor_name, factor_value
- score, rank, signal
- confidence, data_version, update_time

### Check 2: Value ranges
- `score`: 必须在 [0, 1] 范围内
- `confidence`: 必须等于 1.0
- `rank`: 必须 >= 0

### Check 3: Signal enum
`signal` 列只能包含: "buy", "sell", "hold"

### Check 4: Out-of-sample slice
确保至少有 90 个交易日的样本外数据

### Check 5: No future function
通过截断测试验证不存在未来函数：
- 删除最后5个交易日的数据
- 重新计算前面日期的因子
- 确保因子值没有改变（误差 < 1e-6）

### Check 6: Checkpoints exist
确保所有 fold 的模型checkpoint都存在：
- checkpoints/fold_0_best.pth
- checkpoints/fold_1_best.pth
- checkpoints/fold_2_best.pth

## 目录结构要求

验证需要以下目录结构：

```
dl-transformer-multiasset/
├── output/                              # 验证输入
│   ├── factor.parquet                  # 待验证的因子数据
│   └── features.parquet                # 特征数据（用于未来函数检查）
├── checkpoints/                         # 模型文件
│   ├── fold_0_best.pth
│   ├── fold_1_best.pth
│   └── fold_2_best.pth
├── data/ -> ../production/data/         # 符号链接（可选）
└── run_validate.py                      # 验证辅助脚本
```

## 验证通过后

验证通过意味着：
- ✅ 数据格式完全符合规范
- ✅ 因子值在合理范围内
- ✅ 不存在未来函数问题
- ✅ 可以安全地用于回测和实盘

验证通过后，production目录的数据就可以被其他系统使用了：
```
dl-transformer-multiasset-production/data/database.parquet
```
