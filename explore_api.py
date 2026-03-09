"""
StatsBomb API 数据探索脚本

用于了解 StatsBomb Open Data 的数据结构：
- competitions(): 获取可用赛事列表
- matches(competition_id, season_id): 获取比赛列表
- events(match_id): 获取比赛事件数据

运行方式: python src/explore_api.py
"""

from statsbombpy import sb
from pathlib import Path

# 导入配置
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import COMPETITIONS, SEASONS


def explore_competitions():
    """探索可用赛事列表"""
    print("=" * 80)
    print("Step 1: 探索可用赛事 (Competitions)")
    print("=" * 80)

    # 获取所有可用赛事
    all_competitions = sb.competitions()

    print(f"\n总共可用赛事数量: {len(all_competitions)}")
    print("\n赛事列表 (前10个):")
    print(all_competitions.head(10).to_string())

    # 过滤我们关注的赛事
    target_comps = all_competitions[all_competitions['competition_id'].isin(COMPETITIONS.keys())]
    print("\n\n我们将使用以下赛事:")
    print(target_comps[['competition_id', 'country_name', 'competition_name']].to_string())

    return all_competitions


def explore_matches(competition_id=2, season_id=37):
    """
    探索指定赛事和赛季的比赛列表

    Args:
        competition_id: 赛事ID (2 = Premier League)
        season_id: 赛季ID (37 = 2020/21)
    """
    print("\n" + "=" * 80)
    print(f"Step 2: 探索比赛列表 (Matches) - Competition {competition_id}, Season {season_id}")
    print("=" * 80)

    matches = sb.matches(competition_id=competition_id, season_id=season_id)

    print(f"\n该赛季总比赛数: {len(matches)}")
    print("\n比赛列表 (前5场):")
    print(matches[['match_id', 'match_date', 'home_team', 'away_team', 'home_score', 'away_score']].head())

    return matches


def explore_events(match_id=None):
    """
    探索指定比赛的事件数据

    Args:
        match_id: 比赛ID (如果为None，自动选择一场比赛)
    """
    print("\n" + "=" * 80)
    print(f"Step 3: 探索事件数据 (Events) - Match {match_id}")
    print("=" * 80)

    events = sb.events(match_id=match_id)

    print(f"\n该比赛总事件数: {len(events)}")
    print(f"\n事件类型列表:")
    print(events['type'].value_counts().to_string())

    print(f"\n事件数据列名:")
    print(events.columns.tolist())

    return events


def explore_shot_events(match_id=None):
    """
    深入探索射门事件的详细数据结构

    Args:
        match_id: 比赛ID
    """
    print("\n" + "=" * 80)
    print(f"Step 4: 深入探索射门事件 (Shots)")
    print("=" * 80)

    events = sb.events(match_id=match_id)
    shots = events[events['type'] == 'Shot'].copy()

    print(f"\n该比赛射门次数: {len(shots)}")
    print(f"进球数: {shots['shot_outcome'].eq('Goal').sum()}")

    # 显示射门事件的列结构
    print("\n射门事件的主要列:")
    shot_columns = [c for c in shots.columns if 'shot' in c or c in ['location', 'player', 'team']]
    for col in shot_columns:
        print(f"  - {col}")

    # 展示一个射门事件的完整结构
    if len(shots) > 0:
        print("\n\n第一个射门事件的详细信息:")
        first_shot = shots.iloc[0]
        print(f"  位置 (location): {first_shot['location']}")
        print(f"  射门结果 (shot_outcome): {first_shot['shot_outcome']}")
        print(f"  身体部位 (shot_body_part): {first_shot['shot_body_part']}")
        print(f"  射门类型 (shot_type): {first_shot['shot_type']}")

        if 'shot_freeze_frame' in first_shot and first_shot['shot_freeze_frame'] is not None:
            print(f"  Freeze Frame 数据存在: 是 (包含 {len(first_shot['shot_freeze_frame'])} 个球员位置)")

    # 统计各类射门特征
    print("\n\n射门特征统计:")
    print(f"\n身体部位分布:")
    print(shots['shot_body_part'].value_counts().to_string())

    print(f"\n射门类型分布:")
    print(shots['shot_type'].value_counts().to_string())

    print(f"\n射门结果分布:")
    print(shots['shot_outcome'].value_counts().to_string())

    # 进球率
    goal_rate = (shots['shot_outcome'] == 'Goal').mean()
    print(f"\n总进球率: {goal_rate:.2%}")

    return shots


def main():
    """主函数：执行所有探索步骤"""
    print("\n")
    print("*" * 80)
    print("*" + " " * 20 + "StatsBomb API 数据探索" + " " * 34 + "*")
    print("*" * 80)

    # Step 1: 探索赛事
    comps = explore_competitions()

    # Step 1.5: 获取 Premier League 可用的 season_id
    print("\n" + "=" * 80)
    print("获取 Premier League 可用赛季...")
    print("=" * 80)
    pl_comps = comps[comps['competition_id'] == 2]
    print("\nPremier League 可用赛季:")
    print(pl_comps[['season_id', 'season_name']].to_string())

    # 获取第一个可用的 season_id
    first_season_id = pl_comps.iloc[0]['season_id']
    first_season_name = pl_comps.iloc[0]['season_name']
    print(f"\n将使用赛季: {first_season_name} (ID: {first_season_id})")

    # Step 2: 探索比赛列表 (以英超的第一个可用赛季为例)
    matches = explore_matches(competition_id=2, season_id=first_season_id)

    # 获取第一场比赛的ID
    first_match_id = matches.iloc[0]['match_id']
    print(f"\n将使用比赛 ID: {first_match_id} 进行进一步探索")

    # Step 3: 探索事件
    events = explore_events(match_id=first_match_id)

    # Step 4: 深入探索射门事件
    shots = explore_shot_events(match_id=first_match_id)

    print("\n" + "=" * 80)
    print("探索完成！")
    print("=" * 80)
    print("\n关键发现:")
    print("1. Shot 事件包含 location 坐标 (x, y)")
    print("2. shot_outcome 区分 Goal/非进球")
    print("3. shot_body_part 区分脚/头")
    print("4. shot_type 区分普通射门/任意球/点球")
    print("5. shot_freeze_frame 包含射门瞬间所有球员位置（进阶特征）")
    print("\nMVP 计划:")
    print("- 使用 location 计算 distance_to_goal 和 shot_angle")
    print("- One-hot 编码 shot_body_part 和 shot_type")
    print("- 排除 Penalty 类型射门")
    print("=" * 80)


if __name__ == "__main__":
    main()
