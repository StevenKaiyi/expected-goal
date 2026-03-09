"""
射门特征提取与几何特征工程

功能：
- 从原始CSV中提取和清洗基础字段
- 计算几何特征（距离、角度、区域判断）
- 解析Freeze Frame数据生成空间特征
- 输出processed_shots.csv

运行方式: python src/extract_features.py
"""

import os
import sys
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any
from datetime import datetime

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR

# =============================================================================
# 几何常量定义 (StatsBomb 120x80 坐标系)
# =============================================================================

# 门柱位置
GOAL_POST_LEFT = (120, 36)      # A: 左门柱
GOAL_POST_RIGHT = (120, 44)     # B: 右门柱
GOAL_CENTER = (120, 40)        # G: 球门中心

# 区域定义
PENALTY_AREA_MIN_X = 102       # 大禁区左边界
PENALTY_AREA_MAX_X = 120       # 大禁区右边界
PENALTY_AREA_MIN_Y = 18        # 大禁区下边界
PENALTY_AREA_MAX_Y = 62        # 大禁穿上边界

SIX_YARD_AREA_MIN_X = 114      # 小禁区左边界
SIX_YARD_AREA_MAX_X = 120      # 小禁区右边界
SIX_YARD_AREA_MIN_Y = 30       # 小禁区下边界
SIX_YARD_AREA_MAX_Y = 50       # 小禁穿上边界

# =============================================================================
# 辅助函数
# =============================================================================

def point_in_rect(point: Tuple[float, float], rect: Tuple[float, float, float, float]) -> bool:
    """判断点是否在矩形内"""
    x, y = point
    x_min, y_min, x_max, y_max = rect
    return x_min <= x <= x_max and y_min <= y <= y_max

def point_in_triangle(p: Tuple[float, float],
                     a: Tuple[float, float],
                     b: Tuple[float, float],
                     c: Tuple[float, float]) -> bool:
    """判断点是否在三角形内（使用叉乘法）"""
    def cross_product(o: Tuple[float, float],
                     a: Tuple[float, float],
                     b: Tuple[float, float]) -> float:
        """计算向量OA和OB的叉积"""
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    # 计算三个叉积
    cp1 = cross_product(a, b, p)
    cp2 = cross_product(b, c, p)
    cp3 = cross_product(c, a, p)

    # 检查符号是否一致（允许在边界上）
    return (cp1 >= 0 and cp2 >= 0 and cp3 >= 0) or (cp1 <= 0 and cp2 <= 0 and cp3 <= 0)

def safe_parse_json(json_str: str) -> List[Dict]:
    """安全解析JSON字符串"""
    if pd.isna(json_str) or json_str == 'null':
        return []
    try:
        return json.loads(json_str)
    except:
        return []

# =============================================================================
# 特征工程函数
# =============================================================================

def calculate_distance_to_goal(location: Tuple[float, float]) -> float:
    """计算射门点到球门中心的欧式距离"""
    return np.sqrt((location[0] - GOAL_CENTER[0])**2 +
                   (location[1] - GOAL_CENTER[1])**2)

def calculate_shot_angle(location: Tuple[float, float]) -> float:
    """计算射门可见角（弧度）"""
    # 使用余弦定理
    d = calculate_distance_to_goal(location)

    # 计算到两门柱的距离
    d_left = np.sqrt((location[0] - GOAL_POST_LEFT[0])**2 +
                     (location[1] - GOAL_POST_LEFT[1])**2)
    d_right = np.sqrt((location[0] - GOAL_POST_RIGHT[0])**2 +
                      (location[1] - GOAL_POST_RIGHT[1])**2)

    # 门柱间距
    goal_width = abs(GOAL_POST_RIGHT[1] - GOAL_POST_LEFT[1])

    # 余弦定理：cos(C) = (a² + b² - c²) / (2ab)
    cos_angle = (d_left**2 + d_right**2 - goal_width**2) / (2 * d_left * d_right)

    # 避免数值误差导致cos_angle超出[-1,1]范围
    cos_angle = np.clip(cos_angle, -1, 1)

    # 返回角度（弧度）
    return np.arccos(cos_angle)

def parse_freeze_frame_features(freeze_frame_data: List[Dict],
                             shot_location: Tuple[float, float]) -> Dict[str, Any]:
    """
    解析Freeze Frame数据，提取空间特征

    返回：
    - opponents_in_box: 大禁区内对手人数
    - teammates_in_box: 大禁区内队友人数
    - opponents_in_six_yard: 小禁区内对手人数
    - gk_in_six_yard: 小禁区内是否有对方门将
    - opponents_in_shot_cone: 射门锥体内对手人数
    """
    if not freeze_frame_data:
        return {
            'opponents_in_box': 0,
            'teammates_in_box': 0,
            'opponents_in_six_yard': 0,
            'gk_in_six_yard': False,
            'opponents_in_shot_cone': 0
        }

    teammates_in_box = 0
    opponents_in_box = 0
    opponents_in_six_yard = 0
    gk_in_six_yard = False
    opponents_in_shot_cone = 0

    # 定义射门三角形（射门点、左门柱、右门柱）
    shot_triangle = (shot_location, GOAL_POST_LEFT, GOAL_POST_RIGHT)

    for player in freeze_frame_data:
        # 跳过无效数据
        if not isinstance(player, dict) or 'location' not in player:
            continue

        player_location = tuple(player['location'])

        # 检查是小禁区
        in_six_yard = point_in_rect(player_location,
                                   (SIX_YARD_AREA_MIN_X, SIX_YARD_AREA_MIN_Y,
                                    SIX_YARD_AREA_MAX_X, SIX_YARD_AREA_MAX_Y))

        # 检查是大禁区
        in_box = point_in_rect(player_location,
                              (PENALTY_AREA_MIN_X, PENALTY_AREA_MIN_Y,
                               PENALTY_AREA_MAX_X, PENALTY_AREA_MAX_Y))

        # 根据队友/对手分类统计
        if player.get('teammate', True):
            if in_box:
                teammates_in_box += 1
        else:
            if in_box:
                opponents_in_box += 1
            if in_six_yard:
                opponents_in_six_yard += 1
                # 检查是否是门将
                if player.get('position') == 'Goalkeeper':
                    gk_in_six_yard = True

            # 检查是否在射门锥体内
            if point_in_triangle(player_location, *shot_triangle):
                opponents_in_shot_cone += 1

    return {
        'opponents_in_box': opponents_in_box,
        'teammates_in_box': teammates_in_box,
        'opponents_in_six_yard': opponents_in_six_yard,
        'gk_in_six_yard': gk_in_six_yard,
        'opponents_in_shot_cone': opponents_in_shot_cone
    }

def process_boolean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """处理布尔值列，将NaN填充为False"""
    bool_columns = [
        'under_pressure',
        'shot_aerial_won',
        'shot_first_time',
        'shot_one_on_one'
    ]

    for col in bool_columns:
        if col in df.columns:
            df[col] = df[col].fillna(False).astype(bool)

    return df

def load_all_raw_data() -> pd.DataFrame:
    """加载所有原始CSV文件并合并"""
    print("Loading raw shot data...")

    all_data = []
    total_files = 0
    skipped_files = 0
    total_records = 0
    loaded_records = 0

    # 递归查找所有CSV文件
    all_csv_files = []
    for csv_file in Path(RAW_DATA_DIR).rglob("*.csv"):
        if csv_file.is_file():
            print(f"  Found: {csv_file.relative_to(RAW_DATA_DIR)}")
            all_csv_files.append(csv_file)

    print(f"\nTotal found {len(all_csv_files)} CSV files across all directories")

    # 遍历所有CSV文件
    for csv_file in all_csv_files:
        try:
            # 尝试读取文件
            df = pd.read_csv(csv_file)

            # 检查文件是否为空
            if len(df) == 0:
                skipped_files += 1
                print(f"  Skipped: {csv_file.name} (empty file)")
                continue

            # 检查是否有我们需要的列
            required_columns = ['location_x', 'location_y', 'shot_outcome']
            if not any(col in df.columns for col in required_columns):
                skipped_files += 1
                print(f"  Skipped: {csv_file.name} (missing required columns)")
                continue

            # 如果有type列且包含Shot，则筛选
            if 'type' in df.columns:
                shot_df = df[df['type'] == 'Shot'].copy()
                if len(shot_df) == 0:
                    skipped_files += 1
                    print(f"  Skipped: {csv_file.name} (no shot events)")
                    continue
            else:
                # 没有type列，假设所有数据都是射门
                shot_df = df.copy()

            # 检查是否有足够的射门数据
            if len(shot_df) > 0:
                all_data.append(shot_df)
                total_files += 1
                loaded_records += len(shot_df)
                total_records += len(df)
                print(f"  Loaded: {csv_file.name} ({len(shot_df)} shots from {len(df)} records)")
            else:
                skipped_files += 1
                print(f"  Skipped: {csv_file.name} (no shot data)")

        except Exception as e:
            skipped_files += 1
            print(f"  Error: Cannot load {csv_file.name}: {e}")

    if not all_data:
        print("Error: No shot data found!")
        return pd.DataFrame()

    # 合并所有数据
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"\nSummary:")
    print(f"  Total records in all files: {total_records}")
    print(f"  Shot records loaded: {loaded_records}")
    print(f"  Files with shots: {total_files}")
    print(f"  Skipped files: {skipped_files}")
    print(f"  Final dataset: {len(combined_df)} shot records")

    return combined_df

def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """提取所有特征"""
    print("\nExtracting features...")

    # Step 1: 基础字段过滤
    base_columns = [
        'id', 'index', 'minute', 'second', 'timestamp', 'period',
        'match_id', 'player_id', 'team_id', 'position', 'under_pressure',
        'play_pattern', 'location_x', 'location_y', 'shot_outcome',
        'shot_body_part', 'shot_type', 'shot_technique', 'shot_aerial_won',
        'shot_first_time', 'shot_one_on_one', 'shot_end_location_x',
        'shot_end_location_y', 'shot_end_location_z', 'shot_statsbomb_xg',
        'shot_key_pass_id', 'shot_freeze_frame_raw', 'goal'
    ]

    # 只保留存在的列
    available_columns = [col for col in base_columns if col in df.columns]
    df = df[available_columns].copy()

    # 处理布尔值
    df = process_boolean_columns(df)

    # Step 2: 几何特征工程
    print("  - Calculating distance and angle features...")

    # 计算射门距离和角度
    df['distance_to_goal'] = np.sqrt((df['location_x'] - GOAL_CENTER[0])**2 +
                                     (df['location_y'] - GOAL_CENTER[1])**2)

    # 计算射门角度（使用向量化操作提高性能）
    x = df['location_x'].values
    y = df['location_y'].values

    # 计算到两门柱的距离
    d_left = np.sqrt((x - GOAL_POST_LEFT[0])**2 + (y - GOAL_POST_LEFT[1])**2)
    d_right = np.sqrt((x - GOAL_POST_RIGHT[0])**2 + (y - GOAL_POST_RIGHT[1])**2)

    # 门柱间距
    goal_width = abs(GOAL_POST_RIGHT[1] - GOAL_POST_LEFT[1])

    # 余弦定理
    cos_angle = (d_left**2 + d_right**2 - goal_width**2) / (2 * d_left * d_right)
    cos_angle = np.clip(cos_angle, -1, 1)
    df['shot_angle'] = np.arccos(cos_angle)

    # Step 3: 解析Freeze Frame特征
    print("  - Parsing freeze frame features...")
    print(f"    Processing {len(df)} records...")

    freeze_features = []
    batch_size = 1000
    total_batches = (len(df) + batch_size - 1) // batch_size

    for i in range(total_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(df))

        batch_df = df.iloc[start_idx:end_idx]
        print(f"    Processing batch {i+1}/{total_batches} (records {start_idx+1}-{end_idx})")

        batch_features = []
        for _, row in batch_df.iterrows():
            # 解析Freeze Frame数据
            freeze_data = safe_parse_json(row.get('shot_freeze_frame_raw', '[]'))

            # 提取射门位置
            shot_location = (row['location_x'], row['location_y'])

            # 计算空间特征
            features = parse_freeze_frame_features(freeze_data, shot_location)
            batch_features.append(features)

        freeze_features.extend(batch_features)

    # 将新特征转换为DataFrame并添加到原始数据
    freeze_features_df = pd.DataFrame(freeze_features)
    df = pd.concat([df, freeze_features_df], axis=1)

    # Step 4: 清理和输出
    print("  - Cleaning data...")

    # 删除原始的freeze_frame列
    if 'shot_freeze_frame_raw' in df.columns:
        df = df.drop('shot_freeze_frame_raw', axis=1)

    # 确保所有数值列都是float类型
    numeric_columns = [
        'distance_to_goal', 'shot_angle', 'shot_statsbomb_xg',
        'opponents_in_box', 'teammates_in_box', 'opponents_in_six_yard',
        'opponents_in_shot_cone'
    ]

    for col in numeric_columns:
        if col in df.columns:
            df[col] = df[col].astype(float)

    return df

def main():
    """主函数"""
    print("=" * 80)
    print("Shot Feature Extraction and Engineering Tool")
    print("=" * 80)

    # 创建输出目录
    Path(PROCESSED_DATA_DIR).mkdir(parents=True, exist_ok=True)

    # 加载数据
    raw_df = load_all_raw_data()
    if len(raw_df) == 0:
        print("\nError: No data found!")
        return

    # 提取特征
    processed_df = extract_features(raw_df)

    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = Path(PROCESSED_DATA_DIR) / f'processed_shots_{timestamp}.csv'

    processed_df.to_csv(output_file, index=False)

    print(f"\n[SUCCESS] Success! Processed {len(processed_df)} records")
    print(f"Output saved to: {output_file}")

    # 保存文件信息
    info_file = Path(PROCESSED_DATA_DIR) / 'processed_shots_info.txt'
    with open(info_file, 'w', encoding='utf-8') as f:
        f.write("Processed Shots Report\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Records: {len(processed_df)}\n")
        f.write(f"Output File: {output_file.name}\n\n")
        f.write("New Features Added:\n")
        f.write("- distance_to_goal: Distance from shot location to goal center\n")
        f.write("- shot_angle: Shooting angle in radians\n")
        f.write("- opponents_in_box: Number of opponents in penalty area\n")
        f.write("- teammates_in_box: Number of teammates in penalty area\n")
        f.write("- opponents_in_six_yard: Number of opponents in six-yard box\n")
        f.write("- gk_in_six_yard: Whether opponent's goalkeeper is in six-yard box\n")
        f.write("- opponents_in_shot_cone: Number of opponents in shooting triangle\n")
    print(f"\nNew features added:")
    print(f"  - distance_to_goal: Distance from shot location to goal center")
    print(f"  - shot_angle: Shooting angle in radians")
    print(f"  - opponents_in_box: Number of opponents in penalty area")
    print(f"  - teammates_in_box: Number of teammates in penalty area")
    print(f"  - opponents_in_six_yard: Number of opponents in six-yard box")
    print(f"  - gk_in_six_yard: Whether opponent's goalkeeper is in six-yard box")
    print(f"  - opponents_in_shot_cone: Number of opponents in shooting triangle")
    print("=" * 80)

if __name__ == "__main__":
    main()