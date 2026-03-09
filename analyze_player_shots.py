"""
球员射门统计分析脚本 - 完整数据集版本

统计内容:
1. 所有射门的球员列表
2. 每名球员的实际进球数
3. 每名球员的xG总和
4. 每名球员的射门数总和
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
import sys
import io

# 设置标准输出编码为UTF-8
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


def load_all_shot_data(raw_data_dir="data/raw"):
    """
    从所有原始CSV文件中加载射门数据

    Args:
        raw_data_dir: 原始数据目录路径

    Returns:
        DataFrame: 包含所有射门数据的DataFrame
    """
    raw_dir = Path(raw_data_dir)
    csv_files = list(raw_dir.rglob("*.csv"))

    print(f"找到 {len(csv_files)} 个CSV文件")

    # 检查哪些文件包含射门数据
    shot_columns = [
        'player', 'player_id', 'team', 'team_id',
        'location_x', 'location_y',
        'shot_outcome', 'shot_statsbomb_xg'
    ]

    all_data = []
    files_with_shots = []

    for csv_file in csv_files:
        try:
            # 读取前几行检查列名
            df_sample = pd.read_csv(csv_file, nrows=1)

            # 检查是否包含射门相关列
            if any(col in df_sample.columns for col in ['shot_outcome', 'shot_statsbomb_xg']):
                files_with_shots.append(csv_file)
                print(f"  ✓ {csv_file.relative_to(raw_dir)}")

        except Exception as e:
            print(f"  ✗ {csv_file.relative_to(raw_dir)}: {e}")
            continue

    print(f"\n发现 {len(files_with_shots)} 个包含射门数据的文件\n")

    # 读取所有射门数据
    total_records = 0
    for csv_file in files_with_shots:
        try:
            df = pd.read_csv(csv_file, usecols=shot_columns + ['shot_outcome', 'shot_statsbomb_xg'])
            all_data.append(df)
            num_records = len(df)
            total_records += num_records
            print(f"  加载 {csv_file.name}: {num_records} 条记录")

        except Exception as e:
            print(f"  警告: 读取 {csv_file.name} 时出错: {e}")
            continue

    print(f"\n总共加载 {total_records} 条记录")

    # 合并所有数据
    if all_data:
        df_all = pd.concat(all_data, ignore_index=True)
        return df_all
    else:
        return None


def analyze_player_stats(df_shots):
    """
    分析球员射门统计

    Args:
        df_shots: 射门数据DataFrame

    Returns:
        DataFrame: 包含球员统计数据的DataFrame
    """
    print(f"数据集总记录数: {len(df_shots)}")

    # 检查必要的列
    required_columns = ['player', 'player_id', 'shot_statsbomb_xg', 'shot_outcome']
    missing_columns = [col for col in required_columns if col not in df_shots.columns]

    if missing_columns:
        print(f"警告: 数据中缺少必要的列: {missing_columns}")
        return None

    # 筛选射门记录（排除无效记录）
    df_valid = df_shots.dropna(subset=['player_id', 'player']).copy()
    print(f"有效射门记录数: {len(df_valid)}")

    # 标记进球
    df_valid['is_goal'] = (df_valid['shot_outcome'] == 'Goal').astype(int)

    # 统计每个球员的射门数据
    player_stats = df_valid.groupby(['player_id', 'player']).agg(
        shots=('player_id', 'count'),
        goals=('is_goal', 'sum'),
        total_xg=('shot_statsbomb_xg', 'sum')
    ).reset_index()

    # 计算每球平均xG
    player_stats['avg_xg_per_shot'] = player_stats['total_xg'] / player_stats['shots']
    player_stats['goal_rate'] = player_stats['goals'] / player_stats['shots']

    # 计算xG vs 实际进球的差异
    player_stats['xg_minus_goals'] = player_stats['total_xg'] - player_stats['goals']

    # 重命名列
    player_stats.columns = [
        'player_id',
        '球员名称',
        '射门数',
        '进球数',
        'xG总和',
        '每球平均xG',
        '进球率',
        'xG减进球差'
    ]

    # 按射门数降序排列
    player_stats = player_stats.sort_values('射门数', ascending=False).reset_index(drop=True)

    return player_stats


def print_summary(player_stats, top_n=20):
    """
    打印统计摘要

    Args:
        player_stats: 球员统计数据DataFrame
        top_n: 显示前N名球员
    """
    print("\n" + "=" * 100)
    print("球员射门统计摘要")
    print("=" * 100)

    print(f"\n数据集中总共包含 {len(player_stats)} 名球员的射门记录")

    # 总体统计
    total_shots = player_stats['射门数'].sum()
    total_goals = player_stats['进球数'].sum()
    total_xg = player_stats['xG总和'].sum()

    print(f"\n总体统计:")
    print(f"  总射门数: {total_shots:,}")
    print(f"  总进球数: {total_goals:,}")
    print(f"  总xG: {total_xg:.2f}")
    print(f"  整体进球率: {total_goals/total_shots*100:.2f}%")

    # 显示前N名球员
    print(f"\n{'='*100}")
    print(f"前 {top_n} 名射门球员（按射门数排序）")
    print(f"{'='*100}")

    display_cols = ['球员名称', '射门数', '进球数', 'xG总和', '每球平均xG', '进球率', 'xG减进球差']
    top_players = player_stats[display_cols].head(top_n)

    # 打印表头
    print(f"{'球员名称':40s} | {'射门数':^8s} | {'进球数':^8s} | {'xG总和':^10s} | {'每球平均xG':^12s} | {'进球率':^10s} | {'xG-进球':^10s}")
    print("-" * 100)

    for _, row in top_players.iterrows():
        name = row['球员名称'][:40] if pd.notna(row['球员名称']) else f"ID: {row['player_id']}"
        print(f"{name:40s} | {row['射门数']:8d} | {row['进球数']:8d} | "
              f"{row['xG总和']:10.2f} | {row['每球平均xG']:12.3f} | {row['进球率']*100:9.2f}% | "
              f"{row['xG减进球差']:+10.2f}")

    # 按进球数排名
    print(f"\n{'='*100}")
    print(f"前 {top_n} 名射手（按进球数排序）")
    print(f"{'='*100}")

    top_scorers = player_stats.nlargest(top_n, '进球数')[display_cols]

    print(f"{'球员名称':40s} | {'射门数':^8s} | {'进球数':^8s} | {'xG总和':^10s} | {'每球平均xG':^12s} | {'进球率':^10s} | {'xG-进球':^10s}")
    print("-" * 100)

    for _, row in top_scorers.iterrows():
        name = row['球员名称'][:40] if pd.notna(row['球员名称']) else f"ID: {row['player_id']}"
        print(f"{name:40s} | {row['射门数']:8d} | {row['进球数']:8d} | "
              f"{row['xG总和']:10.2f} | {row['每球平均xG']:12.3f} | {row['进球率']*100:9.2f}% | "
              f"{row['xG减进球差']:+10.2f}")

    # xG表现分析
    print(f"\n{'='*100}")
    print("xG表现分析")
    print(f"{'='*100}")

    overperformers = player_stats[player_stats['射门数'] >= 10].nlargest(5, 'xG减进球差')
    print(f"\n低于预期表现球员 (射门>=10, xG减进球差最大 - 射门机会好但进球少):")
    print("-" * 100)
    display_analysis_cols = ['球员名称', '射门数', '进球数', 'xG总和', 'xG减进球差', '进球率']
    for _, row in overperformers.iterrows():
        name = row['球员名称'][:40] if pd.notna(row['球员名称']) else f"ID: {row['player_id']}"
        print(f"{name:40s} | 射门: {row['射门数']:4d} | 进球: {row['进球数']:3d} | "
              f"xG: {row['xG总和']:8.2f} | xG-进球: {row['xG减进球差']:+8.2f} | 进球率: {row['进球率']*100:6.2f}%")

    underperformers = player_stats[player_stats['射门数'] >= 10].nsmallest(5, 'xG减进球差')
    print(f"\n超常发挥球员 (射门>=10, xG减进球差最小 - 射门机会差但进球多):")
    print("-" * 100)
    for _, row in underperformers.iterrows():
        name = row['球员名称'][:40] if pd.notna(row['球员名称']) else f"ID: {row['player_id']}"
        print(f"{name:40s} | 射门: {row['射门数']:4d} | 进球: {row['进球数']:3d} | "
              f"xG: {row['xG总和']:8.2f} | xG-进球: {row['xG减进球差']:+8.2f} | 进球率: {row['进球率']*100:6.2f}%")


def save_results(player_stats, output_dir="data/processed"):
    """
    保存统计结果到文件

    Args:
        player_stats: 球员统计数据DataFrame
        output_dir: 输出目录路径
    """
    output_path = Path(output_dir)

    # 创建时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 保存CSV
    csv_file = output_path / f"player_shots_all_data_{timestamp}.csv"
    player_stats.to_csv(csv_file, index=False, encoding='utf-8-sig')
    print(f"\n结果已保存到: {csv_file}")

    # 保存JSON
    json_file = output_path / f"player_shots_all_data_{timestamp}.json"
    player_stats.to_json(json_file, orient='records', force_ascii=False, indent=2)
    print(f"结果已保存到: {json_file}")


def main():
    """主函数"""
    print("=" * 100)
    print("球员射门统计分析 - 完整数据集")
    print("=" * 100)
    print()

    # 加载所有射门数据
    print("正在加载原始射门数据...")
    df_shots = load_all_shot_data()

    if df_shots is None:
        print("错误: 未能加载射门数据")
        return None

    print()

    # 分析球员统计
    print("正在分析球员射门统计...")
    player_stats = analyze_player_stats(df_shots)

    if player_stats is None:
        print("错误: 统计分析失败")
        return None

    print()

    # 打印摘要
    print_summary(player_stats, top_n=20)

    # 保存结果
    save_results(player_stats)

    print("\n" + "=" * 100)
    print("分析完成!")
    print("=" * 100)

    return player_stats


if __name__ == "__main__":
    player_stats = main()
