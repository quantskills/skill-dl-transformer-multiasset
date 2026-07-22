#!/usr/bin/env python
"""
验证辅助脚本 - 直接从 production 目录读取数据并运行验证
"""
from pathlib import Path

def main():
    print("开始验证...")
    print("=" * 50)

    # 检查必要文件是否存在
    production_data = Path("../dl-transformer-multiasset-production/data")

    database_path = production_data / "database.parquet"
    features_path = production_data / "features.parquet"

    if not database_path.exists():
        raise FileNotFoundError(
            f"未找到因子数据: {database_path}\n"
            f"请先运行: python run_factor.py"
        )

    if not features_path.exists():
        raise FileNotFoundError(
            f"未找到特征数据: {features_path}\n"
            f"请先运行: python -m scripts.features"
        )

    print(f"✓ 因子数据: {database_path}")
    print(f"✓ 特征数据: {features_path}")
    print()

    # 运行验证
    import scripts.validate
    scripts.validate.main()

if __name__ == "__main__":
    main()
