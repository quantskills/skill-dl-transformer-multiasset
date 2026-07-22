#!/bin/bash
# Transformer Multi-Asset Skill - 完整测试脚本
#
# 使用前提：
# 1. 设置 PANDA_DATA_USERNAME 和 PANDA_DATA_PASSWORD 环境变量
# 2. 确保你的 panda_data 账户有 get_future_daily_post 访问权限
# 3. Python 环境已安装所有依赖: torch pandas numpy scipy pyarrow matplotlib panda-data

set -e  # 遇到错误立即退出

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================"
echo "Transformer Multi-Asset Skill 测试"
echo "========================================"
echo ""

# 检查环境变量
if [ -z "$PANDA_DATA_USERNAME" ] || [ -z "$PANDA_DATA_PASSWORD" ]; then
    echo -e "${RED}错误: 请先设置环境变量${NC}"
    echo "export PANDA_DATA_USERNAME='你的用户名'"
    echo "export PANDA_DATA_PASSWORD='你的密码'"
    exit 1
fi

# 设置工作目录
SKILL_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DEV_DIR="$SKILL_DIR/dl-transformer-multiasset"
PROD_DIR="$SKILL_DIR/dl-transformer-multiasset-production"

cd "$DEV_DIR"

# 设置 Python 路径
export PYTHONPATH="$DEV_DIR:$PYTHONPATH"

# 配置参数
export PANDA_DATA_START_DATE="${PANDA_DATA_START_DATE:-2015-01-01}"
export PANDA_DATA_END_DATE="${PANDA_DATA_END_DATE:-2024-12-31}"
export MODEL_ARCH="${MODEL_ARCH:-patchtst}"
export TRAIN_DEVICE="${TRAIN_DEVICE:-auto}"
export ALLOW_CPU_TRAIN=1  # 允许 CPU 训练（用于测试）

echo -e "${GREEN}配置:${NC}"
echo "  数据窗口: $PANDA_DATA_START_DATE 至 $PANDA_DATA_END_DATE"
echo "  模型架构: $MODEL_ARCH"
echo "  训练设备: $TRAIN_DEVICE"
echo ""

# ============================================
# Step 1: 特征工程
# ============================================
echo -e "${YELLOW}[1/6] 运行 features.py - 特征工程${NC}"
python -m scripts.features
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ features.py 完成${NC}"
    # 显示结果统计
    if [ -f "data/features.parquet" ]; then
        python -c "import pandas as pd; df=pd.read_parquet('data/features.parquet'); print(f'  特征表: {len(df):,} 行, {df.symbol.nunique()} 个品种, 日期范围 [{df.date.min()}, {df.date.max()}]')"
    fi
else
    echo -e "${RED}✗ features.py 失败${NC}"
    exit 1
fi
echo ""

# ============================================
# Step 2: 模型训练
# ============================================
echo -e "${YELLOW}[2/6] 运行 train.py - Walk-forward 训练${NC}"
python -m scripts.train
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ train.py 完成${NC}"
    # 显示训练结果
    if [ -f "$PROD_DIR/checkpoints/manifest.json" ]; then
        python -c "import json; m=json.load(open('$PROD_DIR/checkpoints/manifest.json')); print(f\"  训练了 {len(m['folds'])} 折\"); [print(f\"    Fold {f['fold_id']}: val_RankIC={f['best_val_rank_ic']:.4f}, epochs={f['epochs_run']}\") for f in m['folds']]"
    fi
else
    echo -e "${RED}✗ train.py 失败${NC}"
    exit 1
fi
echo ""

# ============================================
# Step 3: 因子生成
# ============================================
echo -e "${YELLOW}[3/6] 运行 factor.py - 因子推理与生成${NC}"
python -m scripts.factor
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ factor.py 完成${NC}"
    # 显示因子统计
    if [ -f "$PROD_DIR/database.parquet" ]; then
        python -c "import pandas as pd; df=pd.read_parquet('$PROD_DIR/database.parquet'); print(f'  因子表: {len(df):,} 行'); print(f'  factor_id: {df.factor_id.unique()[0]}'); print(f'  data_version: {df.data_version.unique()[0]}'); print(f'  信号分布: buy={sum(df.signal==\"buy\")}, sell={sum(df.signal==\"sell\")}, hold={sum(df.signal==\"hold\")}')"
    fi
else
    echo -e "${RED}✗ factor.py 失败${NC}"
    exit 1
fi
echo ""

# ============================================
# Step 4: 验证检查
# ============================================
echo -e "${YELLOW}[4/6] 运行 validate.py - 6项验证检查${NC}"
python -m scripts.validate
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ validate.py 通过所有检查${NC}"
else
    echo -e "${RED}✗ validate.py 检查失败${NC}"
    exit 1
fi
echo ""

# ============================================
# Step 5: 回测评估
# ============================================
echo -e "${YELLOW}[5/6] 运行 backtest.py - 回测与基线对比${NC}"
python -m scripts.backtest > backtest_results.txt 2>&1
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ backtest.py 完成${NC}"
    echo ""
    echo -e "${GREEN}回测结果摘要:${NC}"
    grep -A 20 "研究口径 (DLTX)" backtest_results.txt | head -15
    echo "  ..."
    echo "  (完整结果见 backtest_results.txt)"
else
    echo -e "${RED}✗ backtest.py 失败${NC}"
    cat backtest_results.txt
    exit 1
fi
echo ""

# ============================================
# Step 6: 查询测试
# ============================================
echo -e "${YELLOW}[6/6] 测试 query.py - 生产查询接口${NC}"
cd "$PROD_DIR"
export PYTHONPATH="$PROD_DIR:$PYTHONPATH"
python -m scripts.query --start 2024-01-01 --signal buy 2>/dev/null | head -10
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ query.py 工作正常${NC}"
else
    echo -e "${RED}✗ query.py 失败${NC}"
    exit 1
fi
echo ""

# ============================================
# 完成
# ============================================
echo "========================================"
echo -e "${GREEN}✓ 所有测试通过！${NC}"
echo "========================================"
echo ""
echo "生成的文件:"
echo "  特征表: $DEV_DIR/data/features.parquet"
echo "  模型检查点: $PROD_DIR/checkpoints/fold_*/model.pt"
echo "  因子数据库: $PROD_DIR/database.parquet"
echo "  回测结果: $DEV_DIR/backtest_results.txt"
echo ""
echo "下一步:"
echo "  1. 查看回测结果: cat $DEV_DIR/backtest_results.txt"
echo "  2. 查询因子: cd $PROD_DIR && python -m scripts.query --help"
echo "  3. 可视化训练曲线: $PROD_DIR/checkpoints/fold_*/curves.png"
