#!/bin/bash
# 一键运行完整流程的启动脚本

cd "$(dirname "$0")/dl-transformer-multiasset"

# 检查环境变量
if [ -z "$PANDA_DATA_USERNAME" ] || [ -z "$PANDA_DATA_PASSWORD" ]; then
    echo "========================================="
    echo "Panda Data API 认证信息"
    echo "========================================="
    echo ""
    echo "请设置以下环境变量:"
    echo ""
    echo "  export PANDA_DATA_USERNAME='your_username'"
    echo "  export PANDA_DATA_PASSWORD='your_password'"
    echo ""
    echo "然后再运行本脚本:"
    echo "  bash run_pipeline.sh"
    echo ""
    exit 1
fi

echo "✓ 认证信息已设置"
echo ""

# 运行pipeline
bash run_pipeline.sh

echo ""
echo "========================================="
echo "✓ 全流程运行完成！"
echo "========================================="
echo ""
echo "验证数据:"
echo "  cd dl-transformer-multiasset"
echo "  python run_validate.py"
