#!/usr/bin/env python
"""
完整验证脚本 - 包含数据格式验证 + 因子质量指标
"""
from pathlib import Path

def main():
    print()
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 15 + "完整验证流程" + " " * 32 + "║")
    print("╚" + "=" * 58 + "╝")
    print()

    # Step 1: 基本数据格式验证
    print("[1/2] 数据格式验证中...")
    print("-" * 60)
    import scripts.validate
    scripts.validate.main()

    print()
    print("[2/2] 因子质量指标计算中...")
    print("-" * 60)

    # Step 2: 因子质量指标
    import run_metrics
    run_metrics.main()

    print()
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 12 + "🎉 完整验证已通过，因子可以使用！" + " " * 10 + "║")
    print("╚" + "=" * 58 + "╝")
    print()

    # 输出下一步建议
    print("下一步建议:")
    print("  1. 查看因子输出: ../dl-transformer-multiasset-production/data/database.parquet")
    print("  2. 在回测系统中使用因子数据")
    print("  3. 定期重新训练模型（建议每季度一次）")
    print()

if __name__ == "__main__":
    main()
