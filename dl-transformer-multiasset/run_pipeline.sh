#!/bin/bash
# Complete pipeline: features -> train -> factor inference

set -e  # Exit on error

# Set environment variables if not already set
export PANDA_DATA_START_DATE="${PANDA_DATA_START_DATE:-20220401}"
export PANDA_DATA_END_DATE="${PANDA_DATA_END_DATE:-20241231}"
export PANDA_DATA_USERNAME="${PANDA_DATA_USERNAME}"
export PANDA_DATA_PASSWORD="${PANDA_DATA_PASSWORD}"

# Check if credentials are provided
if [ -z "$PANDA_DATA_USERNAME" ] || [ -z "$PANDA_DATA_PASSWORD" ]; then
    echo "Error: Please set PANDA_DATA_USERNAME and PANDA_DATA_PASSWORD environment variables"
    echo ""
    echo "Usage:"
    echo "  export PANDA_DATA_USERNAME='your_username'"
    echo "  export PANDA_DATA_PASSWORD='your_password'"
    echo "  bash run_pipeline.sh"
    exit 1
fi

echo "Using date range: $PANDA_DATA_START_DATE to $PANDA_DATA_END_DATE"
echo ""

echo "========================================="
echo "Step 1: Generate Features (if not exists)"
echo "========================================="
if [ -f "../dl-transformer-multiasset-production/data/features.parquet" ]; then
    echo "Features already exist at production/data/features.parquet"
else
    echo "Running feature engineering..."
    python -m scripts.features
fi

echo ""
echo "========================================="
echo "Step 2: Train Models"
echo "========================================="
python -m scripts.train

echo ""
echo "========================================="
echo "Step 3: Generate Factor Predictions"
echo "========================================="
python run_factor.py

echo ""
echo "========================================="
echo "Step 4: Validate Results"
echo "========================================="
python run_full_validate.py

echo ""
echo "========================================="
echo "✅ 完整流程运行完成！"
echo "========================================="
echo ""
echo "输出文件位置:"
echo "  📊 因子数据: ../dl-transformer-multiasset-production/data/database.parquet"
echo "  📈 特征数据: ../dl-transformer-multiasset-production/data/features.parquet"
echo "  🤖 模型文件: checkpoints/"
echo ""
