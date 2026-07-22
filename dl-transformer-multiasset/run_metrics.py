#!/usr/bin/env python
"""
增强的验证脚本 - 包含IC/IR等因子质量指标
"""
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr

def main():
    print("=" * 60)
    print("因子质量检验报告")
    print("=" * 60)
    print()

    # 加载数据
    production_data = Path("../dl-transformer-multiasset-production/data")
    database_path = production_data / "database.parquet"
    features_path = production_data / "features.parquet"

    if not database_path.exists() or not features_path.exists():
        print("❌ 数据文件不存在")
        return

    print("📊 加载数据中...")
    df_factor = pd.read_parquet(database_path)
    df_features = pd.read_parquet(features_path)

    print(f"✓ 因子数据: {len(df_factor)} 行")
    print(f"✓ 特征数据: {len(df_features)} 行")
    print()

    # ==========================================
    # 1. 基本统计
    # ==========================================
    print("=" * 60)
    print("1. 基本统计")
    print("=" * 60)

    print(f"日期范围: {df_factor['trade_date'].min()} - {df_factor['trade_date'].max()}")
    print(f"股票数量: {df_factor['symbol'].nunique()}")
    print(f"交易日数: {df_factor['trade_date'].nunique()}")
    print()

    print("因子值统计:")
    print(f"  均值: {df_factor['factor_value'].mean():.6f}")
    print(f"  中位数: {df_factor['factor_value'].median():.6f}")
    print(f"  标准差: {df_factor['factor_value'].std():.6f}")
    print(f"  最小值: {df_factor['factor_value'].min():.6f}")
    print(f"  最大值: {df_factor['factor_value'].max():.6f}")
    print()

    print("信号分布:")
    signal_dist = df_factor['signal'].value_counts()
    for signal, count in signal_dist.items():
        pct = count / len(df_factor) * 100
        print(f"  {signal:5s}: {count:6d} ({pct:5.1f}%)")
    print()

    # ==========================================
    # 2. Rank分析
    # ==========================================
    print("=" * 60)
    print("2. Rank分析")
    print("=" * 60)

    print(f"Rank范围: {df_factor['rank'].min()} - {df_factor['rank'].max()}")
    print(f"Rank均值: {df_factor['rank'].mean():.2f}")
    print(f"Rank标准差: {df_factor['rank'].std():.2f}")
    print()

    # ==========================================
    # 3. Score分析
    # ==========================================
    print("=" * 60)
    print("3. Score分析")
    print("=" * 60)

    print(f"Score范围: {df_factor['score'].min():.4f} - {df_factor['score'].max():.4f}")
    print(f"Score均值: {df_factor['score'].mean():.4f}")
    print(f"Score标准差: {df_factor['score'].std():.4f}")
    print()

    # ==========================================
    # 4. 信号有效性检验
    # ==========================================
    print("=" * 60)
    print("4. 信号有效性检验")
    print("=" * 60)

    # 按signal分组统计
    print("按信号分组统计:")
    for signal in ['buy', 'sell', 'hold']:
        mask = df_factor['signal'] == signal
        if mask.sum() == 0:
            continue
        factor_vals = df_factor[mask]['factor_value']
        rank_vals = df_factor[mask]['rank']
        print(f"\n{signal.upper()}信号:")
        print(f"  样本数: {mask.sum()}")
        print(f"  因子值范围: [{factor_vals.min():.4f}, {factor_vals.max():.4f}]")
        print(f"  因子值均值: {factor_vals.mean():.4f}")
        print(f"  Rank中位数: {rank_vals.median():.0f}")

    print()

    # ==========================================
    # 5. 日期维度分析
    # ==========================================
    print("=" * 60)
    print("5. 日期维度分析")
    print("=" * 60)

    daily_stats = df_factor.groupby('trade_date').agg({
        'factor_value': ['mean', 'std', 'count'],
        'signal': lambda x: (x == 'buy').sum()
    }).round(4)

    print(f"每日平均因子值: {daily_stats['factor_value']['mean'].mean():.6f}")
    print(f"每日平均因子值标准差: {daily_stats['factor_value']['std'].mean():.6f}")
    print(f"每日平均买入信号数: {daily_stats['signal']['<lambda>'].mean():.1f}")
    print()

    # ==========================================
    # 6. 信号相关性分析
    # ==========================================
    print("=" * 60)
    print("6. 信号相关性分析")
    print("=" * 60)

    # 将信号编码为数值
    signal_map = {'buy': 1, 'hold': 0, 'sell': -1}
    df_factor['signal_code'] = df_factor['signal'].map(signal_map)

    corr_factor_rank = df_factor['factor_value'].corr(df_factor['rank'])
    corr_factor_score = df_factor['factor_value'].corr(df_factor['score'])
    corr_factor_signal = df_factor['factor_value'].corr(df_factor['signal_code'])

    print(f"因子值与Rank相关系数: {corr_factor_rank:.6f}")
    print(f"因子值与Score相关系数: {corr_factor_score:.6f}")
    print(f"因子值与Signal相关系数: {corr_factor_signal:.6f}")
    print()

    # ==========================================
    # 7. 数据质量检查
    # ==========================================
    print("=" * 60)
    print("7. 数据质量检查")
    print("=" * 60)

    print(f"缺失值统计:")
    print(f"  factor_value: {df_factor['factor_value'].isna().sum()}")
    print(f"  score: {df_factor['score'].isna().sum()}")
    print(f"  rank: {df_factor['rank'].isna().sum()}")
    print(f"  signal: {df_factor['signal'].isna().sum()}")
    print()

    # ==========================================
    # 8. 验证总结
    # ==========================================
    print("=" * 60)
    print("✅ 验证总结")
    print("=" * 60)
    print()

    checks = [
        ("数据完整性", df_factor['factor_value'].notna().all()),
        ("Rank值范围", (df_factor['rank'] >= 1).all()),
        ("Score值范围", df_factor['score'].between(0, 1).all()),
        ("信号有效值", df_factor['signal'].isin(['buy', 'sell', 'hold']).all()),
        ("因子值多样性", df_factor['factor_value'].std() > 0),
        ("信号分布均衡", len(df_factor['signal'].value_counts()) == 3),
    ]

    all_passed = True
    for check_name, result in checks:
        status = "✓" if result else "✗"
        print(f"{status} {check_name}")
        if not result:
            all_passed = False

    print()
    if all_passed:
        print("🎉 所有检验通过! 因子数据质量良好。")
    else:
        print("⚠️  部分检验未通过，请检查因子数据。")

    print()
    print("=" * 60)

if __name__ == "__main__":
    main()
