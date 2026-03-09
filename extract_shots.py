"""
提取 StatsBomb 射门事件数据

功能：
- 遍历所有配置的联赛和赛季
- 获取每场比赛的事件数据
- 提取所有射门事件
- 按赛季组织存储到 data/raw/season/league.csv

运行方式: python src/extract_shots.py
"""

import os
import sys
from pathlib import Path
import pandas as pd
import time
from tqdm import tqdm
from statsbombpy import sb
import requests

# 设置代理
PROXY = "http://127.0.0.1:7890"
os.environ['HTTP_PROXY'] = PROXY
os.environ['HTTPS_PROXY'] = PROXY
os.environ['http_proxy'] = PROXY
os.environ['https_proxy'] = PROXY

# 配置requests使用代理
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    COMPETITIONS,
    DATA_DIR,
    RAW_DATA_DIR,
    RATE_LIMIT,
    MAX_RETRIES,
    RETRY_DELAY,
    EXCLUDED_SHOT_TYPES
)


def get_available_seasons_from_api():
    """
    从 StatsBomb API 获取所有联赛的可用赛季数据
    一次性获取所有数据，避免重复API调用
    """
    print("正在获取所有联赛的可用赛季数据...")
    all_comps = sb.competitions()
    competitions_dict = {}

    # sb.competitions() 返回的每一行是一个 competition-season 对
    for _, row in all_comps.iterrows():
        comp_id = int(row['competition_id'])
        comp_name = row['competition_name']
        season_id = int(row['season_id'])
        season_name = row['season_name']

        # 如果这个联赛还没初始化，先初始化
        if comp_id not in competitions_dict:
            competitions_dict[comp_id] = {
                'competition_id': comp_id,
                'competition_name': comp_name,
                'seasons': []
            }

        # 添加赛季信息
        competitions_dict[comp_id]['seasons'].append({
            'season_id': season_id,
            'season_name': season_name
        })

    return competitions_dict


def flatten_shot_events(events_df):
    """将射门事件的嵌套字段展平"""
    shots_df = events_df[events_df['type'] == 'Shot'].copy()

    if len(shots_df) == 0:
        return pd.DataFrame()

    # 提取基本字段
    basic_cols = [
        'id', 'index', 'minute', 'second', 'timestamp', 'period',
        'match_id', 'player', 'player_id', 'team', 'team_id',
        'position', 'location', 'under_pressure', 'play_pattern'
    ]
    basic_cols = [c for c in basic_cols if c in shots_df.columns]
    result_df = shots_df[basic_cols].copy()

    # 辅助函数：安全提取位置
    def extract_location(loc, index):
        if loc is None or not isinstance(loc, (list, tuple)) or len(loc) <= index:
            return None
        return loc[index]

    # 提取位置坐标
    result_df['location_x'] = shots_df['location'].apply(lambda x: extract_location(x, 0))
    result_df['location_y'] = shots_df['location'].apply(lambda x: extract_location(x, 1))
    result_df['location_z'] = shots_df['location'].apply(lambda x: extract_location(x, 2))

    # 辅助函数：安全获取列值
    def safe_get_column(df, col, default=None):
        if col in df.columns:
            if default is not None:
                return df[col].fillna(default)
            return df[col]
        return default

    # 射门结果
    result_df['shot_outcome'] = safe_get_column(shots_df, 'shot_outcome', 'Unknown')
    result_df['shot_body_part'] = safe_get_column(shots_df, 'shot_body_part', 'Unknown')
    result_df['shot_type'] = safe_get_column(shots_df, 'shot_type', 'Unknown')
    result_df['shot_technique'] = safe_get_column(shots_df, 'shot_technique', 'Unknown')
    result_df['shot_aerial_won'] = safe_get_column(shots_df, 'shot_aerial_won', False)
    result_df['shot_first_time'] = safe_get_column(shots_df, 'shot_first_time', False)
    result_df['shot_one_on_one'] = safe_get_column(shots_df, 'shot_one_on_one', False)

    # 射门结束位置
    def extract_end_location(loc, index):
        if loc is None or not isinstance(loc, (list, tuple)) or len(loc) <= index:
            return None
        return loc[index]

    result_df['shot_end_location_x'] = shots_df['shot_end_location'].apply(lambda x: extract_end_location(x, 0))
    result_df['shot_end_location_y'] = shots_df['shot_end_location'].apply(lambda x: extract_end_location(x, 1))
    result_df['shot_end_location_z'] = shots_df['shot_end_location'].apply(lambda x: extract_end_location(x, 2))

    # StatsBomb xG
    result_df['shot_statsbomb_xg'] = safe_get_column(shots_df, 'shot_statsbomb_xg', -1)

    # 关键传球ID
    result_df['shot_key_pass_id'] = safe_get_column(shots_df, 'shot_key_pass_id')

    # Freeze Frame 提取
    def count_freeze_frame(x):
        if x is None or not isinstance(x, list):
            return 0
        return len(x)

    def serialize_freeze_frame(x):
        """将 freeze frame 序列化为 JSON 字符串"""
        import json
        if x is None or not isinstance(x, list):
            return None
        try:
            return json.dumps(x, ensure_ascii=False)
        except:
            return None

    def count_teammates(x):
        """统计队友数量"""
        if x is None or not isinstance(x, list):
            return 0
        return sum(1 for p in x if isinstance(p, dict) and p.get('teammate') is True)

    def count_opponents(x):
        """统计对手数量"""
        if x is None or not isinstance(x, list):
            return 0
        return sum(1 for p in x if isinstance(p, dict) and p.get('teammate') is False)

    result_df['shot_freeze_frame_count'] = shots_df['shot_freeze_frame'].apply(count_freeze_frame)
    result_df['shot_freeze_frame_raw'] = shots_df['shot_freeze_frame'].apply(serialize_freeze_frame)
    result_df['freeze_teammate_count'] = shots_df['shot_freeze_frame'].apply(count_teammates)
    result_df['freeze_opponent_count'] = shots_df['shot_freeze_frame'].apply(count_opponents)

    # 进球标记 (0/1)
    result_df['goal'] = (result_df['shot_outcome'] == 'Goal').astype(int)

    return result_df


def fetch_match_events(match_id, max_retries=MAX_RETRIES):
    """获取指定比赛的事件数据，带重试机制"""
    for attempt in range(max_retries):
        try:
            events = sb.events(match_id=match_id)
            return events
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  警告: 获取比赛 {match_id} 失败 (尝试 {attempt + 1}/{max_retries}), 重试中... 错误: {e}")
                time.sleep(RETRY_DELAY)
            else:
                print(f"  警告: 获取比赛 {match_id} 失败，已重试 {max_retries} 次")
                return None


def extract_shots_for_season(competition_id, competition_name, season_id, season_name):
    """提取指定赛季的所有射门事件"""
    print(f"\n{'='*60}")
    print(f"处理: {competition_name} - {season_name} (ID: {season_id})")
    print(f"{'='*60}")

    # 获取比赛列表
    try:
        matches = sb.matches(competition_id=competition_id, season_id=season_id)
    except Exception as e:
        print(f"  警告: 无法获取比赛列表: {e}")
        return pd.DataFrame()

    print(f"  找到 {len(matches)} 场比赛")

    all_shots = []
    match_ids = matches['match_id'].tolist()

    for match_id in tqdm(match_ids, desc=f"  提取射门"):
        time.sleep(1.0 / RATE_LIMIT)  # 遵守速率限制
        events = fetch_match_events(match_id)
        if events is None:
            continue

        # 提取射门事件
        shots = flatten_shot_events(events)
        if len(shots) > 0:
            all_shots.append(shots)

    # 合并所有射门事件
    result_df = pd.concat(all_shots, ignore_index=True)

    # 过滤掉点球（根据配置）
    if EXCLUDED_SHOT_TYPES:
        result_df = result_df[~result_df['shot_type'].isin(EXCLUDED_SHOT_TYPES)]

    print(f"  提取到 {len(result_df)} 次射门 (过滤点球后)")

    # 添加联赛和赛季信息
    result_df['competition_id'] = competition_id
    result_df['competition_name'] = competition_name
    result_df['season_id'] = season_id
    result_df['season_name'] = season_name

    return result_df


def save_shots_data(shots_df, season_name, competition_name):
    """保存射门数据到 CSV 文件"""
    if len(shots_df) == 0:
        return

    # 创建目录结构: data/raw/season_name/league_name.csv
    season_dir = Path(RAW_DATA_DIR) / season_name
    season_dir.mkdir(parents=True, exist_ok=True)

    # 清理文件名（替换特殊字符）
    safe_competition_name = competition_name.replace('/', '-').replace(' ', '_')
    output_file = season_dir / f"{safe_competition_name}.csv"

    shots_df.to_csv(output_file, index=False, encoding='utf-8')

    # 每处理完一个CSV文件后输出提示
    print(f"\n✅ 已处理: {competition_name} - {season_name}")
    print(f"   文件路径: {output_file}")
    print(f"   射门次数: {len(shots_df)}")
    print(f"   进球次数: {shots_df['goal'].sum()}")
    print(f"   进球率: {shots_df['goal'].sum() / len(shots_df) * 100:.1f}%")


def main():
    """主函数：正确跳过已处理的赛季"""
    print("\n" + "="*80)
    print("*" + " " * 25 + "射门数据提取脚本")
    print("="*80)

    # 获取所有联赛的可用赛季数据（从API一次性获取）
    competitions_dict = get_available_seasons_from_api()

    # 构建一个字典来跟踪已处理的赛季
    processed_seasons = set()

    # 遍历所有联赛
    total_shots = 0
    skipped = 0

    for comp_id, comp_name in COMPETITIONS.items():
        print(f"\n{'#'*60}")
        print(f"联赛: {comp_name} (ID: {comp_id})")
        print(f"{'#'*60}")

        # 获取该联赛的可用赛季
        seasons_data = competitions_dict.get(comp_id, {}).get('seasons', [])

        if not seasons_data:
            print(f"  警告: 联赛 {comp_name} 没有可用数据")
            continue

        for season in seasons_data:
            season_id = season['season_id']
            season_name = season['season_name']

            # 检查文件是否已存在
            season_dir = Path(RAW_DATA_DIR) / season_name
            safe_competition_name = comp_name.replace('/', '-').replace(' ', '_')
            output_file = season_dir / f"{safe_competition_name}.csv"

            if output_file.exists():
                print(f"  跳过: {comp_name} - {season_name} (文件已存在)")
                skipped += 1
                continue

            # 提取射门事件
            shots_df = extract_shots_for_season(comp_id, comp_name, season_id, season_name)

            # 保存数据
            if len(shots_df) > 0:
                save_shots_data(shots_df, season_name, comp_name)
                total_shots += len(shots_df)
                # 标记为已处理
                processed_seasons.add(f"{comp_id}:{season_id}")

    print(f"\n{'='*80}")
    print(f"提取完成！")
    print(f" 总计:")
    print(f"  - 提取射门: {total_shots} 次")
    print(f"  - 跳过已存在: {skipped} 个赛季-联赛组合")
    print(f" - 新提取: {(total_shots + skipped)} 个赛季-联赛组合")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
