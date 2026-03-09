"""
球员射门统计分析脚本

统计内容:
1. 所有射门的球员列表
2. 每名球员的实际进球数
3. 每名球员的xG总和
4. 每名球员的射门数总和
"""

import pandas as pd
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import sys
import io

# 设置标准输出编码为UTF-8
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


def analyze_player_stats(data_dir="data/processed"):
    """
    分析球员射门统计

    Args:
        data_dir: 数据目录路径

    Returns:
        DataFrame: 包含球员统计数据的DataFrame
    """
    # 读取处理后的射门数据
    data_path = Path(data_dir) / "processed_shots.csv"

    if not data_path.exists():
        print(f"错误: 找不到数据文件 {data_path}")
        return None

    print(f"正在读取数据: {data_path}")
    df = pd.read_csv(data_path)

    print(f"数据集总记录数: {len(df)}")

    # 检查必要的列
    required_columns = ['player_id', 'goal', 'shot_statsbomb_xg']
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        print(f"警告: 数据中缺少必要的列: {missing_columns}")
        return None

    # 筛选射门记录（排除无效记录）
    df_shots = df.dropna(subset=['player_id'])
    print(f"有效射门记录数: {len(df_shots)}")

    # 统计每个球员的射门数据
    player_stats = df_shots.groupby('player_id').agg(
        shots=('player_id', 'count'),
        goals=('goal', 'sum'),
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
        '射门数',
        '进球数',
        'xG总和',
        '每球平均xG',
        '进球率',
        'xG-进球差'
    ]

    # 按射门数降序排列
    player_stats = player_stats.sort_values('射门数', ascending=False).reset_index(drop=True)

    return player_stats


def analyze_player_stats_with_names(raw_data_dir="data/raw"):
    """
    分析球员射门统计并包含球员名称

    Args:
        raw_data_dir: 原始数据目录路径

    Returns:
        DataFrame: 包含球员统计数据的DataFrame（含球员名称）
    """
    raw_dir = Path(raw_data_dir)

    # 收集所有原始数据文件中的球员ID到名称的映射
    player_name_map = {}

    print("正在从原始数据中收集球员名称映射...")
    csv_files = list(raw_dir.rglob("*.csv"))

    for csv_file in csv_files:
        try:
            # 读取部分数据来获取球员信息
            df_chunk = pd.read_csv(csv_file, usecols=['player', 'player_id'])

            # 构建映射
            chunk_map = dict(zip(
                df_chunk['player_id'].astype(str),
                df_chunk['player']
            ))

            # 只保留非空的映射
            chunk_map = {k: v for k, v in chunk_map.items() if pd.notna(v)}

            player_name_map.update(chunk_map)

        except Exception as e:
            print(f"警告: 读取文件 {csv_file} 时出错: {e}")
            continue

    print(f"收集到 {len(player_name_map)} 个球员名称映射")

    # 使用基本统计函数
    player_stats = analyze_player_stats()

    if player_stats is None:
        return None

    # 添加球员名称
    player_stats['球员名称'] = player_stats['player_id'].astype(str).map(player_name_map)

    # 重新排列列
    cols = ['球员名称', 'player_id', '射门数', '进球数', 'xG总和',
            '每球平均xG', '进球率', 'xG-进球差']
    player_stats = player_stats[cols]

    return player_stats


def print_summary(player_stats, top_n=20):
    """
    打印统计摘要

    Args:
        player_stats: 球员统计数据DataFrame
        top_n: 显示前N名球员
    """
    print("\n" + "=" * 80)
    print("球员射门统计摘要")
    print("=" * 80)

    print(f"\n数据集中总共包含 {len(player_stats)} 名球员的射门记录")

    # 总体统计
    total_shots = player_stats['射门数'].sum()
    total_goals = player_stats['进球数'].sum()
    total_xg = player_stats['xG总和'].sum()

    print(f"\n总体统计:")
    print(f"  总射门数: {total_shots}")
    print(f"  总进球数: {total_goals}")
    print(f"  总xG: {total_xg:.2f}")
    print(f"  整体进球率: {total_goals/total_shots*100:.2f}%")

    # 显示前N名球员
    print(f"\n前 {top_n} 名射门球员:")
    print("-" * 80)

    if '球员名称' in player_stats.columns:
        display_cols = ['球员名称', '射门数', '进球数', 'xG总和', '进球率', 'xG-进球差']
    else:
        display_cols = ['player_id', '射门数', '进球数', 'xG总和', '进球率', 'xG-进球差']

    top_players = player_stats[display_cols].head(top_n)

    for idx, row in top_players.iterrows():
        if '球员名称' in player_stats.columns:
            name = row['球员名称'] if pd.notna(row['球员名称']) else f"ID: {row['player_id']}"
        else:
            name = f"Player ID: {row['player_id']}"

        print(f"{name:40s} | 射门: {row['射门数']:4d} | 进球: {row['进球数']:3d} | "
              f"xG: {row['xG总和']:6.2f} | 进球率: {row['进球率']*100:5.1f}% | "
              f"xG-进球: {row['xG-进球差']:+6.2f}")

    # xG表现分析
    print(f"\n" + "=" * 80)
    print("xG表现分析")
    print("=" * 80)

    overperformers = player_stats[player_stats['射门数'] >= 10].nlargest(5, 'xG-进球差')
    print("\n超常发挥球员 (射门>=10, xG-进球差最高):")
    print("-" * 80)
    for idx, row in overperformers.iterrows():
        if '球员名称' in player_stats.columns:
            name = row['球员名称'] if pd.notna(row['球员名称']) else f"ID: {row['player_id']}"
        else:
            name = f"Player ID: {row['player_id']}"
        print(f"{name:40s} | xG: {row['xG总和']:6.2f} | 进球: {row['进球数']:3d} | 差值: {row['xG-进球差']:+6.2f}")

    underperformers = player_stats[player_stats['射门数'] >= 10].nsmallest(5, 'xG-进球差')
    print("\n低于预期球员 (射门>=10, xG-进球差最低):")
    print("-" * 80)
    for idx, row in underperformers.iterrows():
        if '球员名称' in player_stats.columns:
            name = row['球员名称'] if pd.notna(row['球员名称']) else f"ID: {row['player_id']}"
        else:
            name = f"Player ID: {row['player_id']}"
        print(f"{name:40s} | xG: {row['xG总和']:6.2f} | 进球: {row['进球数']:3d} | 差值: {row['xG-进球差']:+6.2f}")


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
    csv_file = output_path / f"player_statistics_{timestamp}.csv"
    player_stats.to_csv(csv_file, index=False, encoding='utf-8-sig')
    print(f"\n结果已保存到: {csv_file}")

    # 保存JSON
    json_file = output_path / f"player_statistics_{timestamp}.json"
    player_stats.to_json(json_file, orient='records', force_ascii=False, indent=2)
    print(f"结果已保存到: {json_file}")


def main():
    """主函数"""
    print("=" * 80)
    print("球员射门统计分析")
    print("=" * 80)

    # 分析球员统计（含球员名称）
    print("\n正在分析球员统计（含球员名称）...")
    player_stats = analyze_player_stats_with_names()

    if player_stats is None:
        print("\n尝试使用基本统计（不含球员名称）...")
        player_stats = analyze_player_stats()

    if player_stats is not None:
        # 打印摘要
        print_summary(player_stats, top_n=20)

        # 保存结果
        save_results(player_stats)

        print("\n" + "=" * 80)
        print("分析完成!")
        print("=" * 80)

        return player_stats

    return None


if __name__ == "__main__":
    player_stats = main()
