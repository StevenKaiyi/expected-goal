"""
XGBoost xG模型训练与可解释性分析
目标：破解 StatsBomb xG 的内部权重逻辑

数据来源：完整数据集（60,118次射门）
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import io
import json
from datetime import datetime

# 设置标准输出编码为UTF-8
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 尝试导入 SHAP
SHAP_AVAILABLE = False
try:
    import shap
    SHAP_AVAILABLE = True
    print("SHAP 库可用，将进行深度可解释性分析")
except ImportError:
    print("SHAP 库不可用，将使用 XGBoost 内置特征重要性分析")


class XGModelExplainer:
    """XG模型训练与解释器"""

    def __init__(self, raw_data_dir="data/raw", output_dir="data/processed"):
        self.raw_data_dir = Path(raw_data_dir)
        self.output_dir = Path(output_dir)
        self.df = None
        self.preferred_foot_map = {}
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.explainer = None
        self.shap_values = None

    def load_all_shot_data(self):
        """从所有原始CSV文件中加载射门数据"""
        print("=" * 100)
        print("步骤 0: 加载完整射门数据集")
        print("=" * 100)

        csv_files = list(self.raw_data_dir.rglob("*.csv"))
        print(f"找到 {len(csv_files)} 个CSV文件\n")

        # 需要读取的列
        required_columns = [
            'player_id', 'player',
            'shot_body_part', 'under_pressure', 'shot_technique',
            'shot_first_time', 'shot_one_on_one', 'play_pattern',
            'shot_statsbomb_xg', 'shot_outcome',
            'location_x', 'location_y', 'location_z',
            'shot_end_location_x', 'shot_end_location_y', 'shot_end_location_z',
            'shot_freeze_frame_raw'
        ]

        all_data = []
        files_with_shots = []

        # 首先检查哪些文件包含所需列
        for csv_file in csv_files:
            try:
                df_sample = pd.read_csv(csv_file, nrows=1)

                # 检查是否包含射门相关列
                if 'shot_outcome' in df_sample.columns:
                    files_with_shots.append(csv_file)
            except:
                continue

        print(f"发现 {len(files_with_shots)} 个包含射门数据的文件\n")

        # 读取所有射门数据
        for csv_file in files_with_shots:
            try:
                df = pd.read_csv(csv_file)

                # 检查所需列是否存在
                available_cols = [col for col in required_columns if col in df.columns]

                if len(available_cols) < 10:  # 至少需要基本列
                    continue

                df_subset = df[available_cols].copy()
                all_data.append(df_subset)
                print(f"  加载 {csv_file.name}: {len(df_subset)} 条记录")

            except Exception as e:
                print(f"  警告: 读取 {csv_file.name} 时出错: {e}")
                continue

        print(f"\n总共加载 {sum(len(d) for d in all_data):,} 条记录")

        if all_data:
            self.df = pd.concat(all_data, ignore_index=True)
            print(f"合并后数据集大小: {len(self.df):,} 行 × {len(self.df.columns)} 列\n")
        else:
            raise ValueError("未能加载任何数据")

        return self.df

    def calculate_shot_features(self):
        """计算射门特征（距离、角度、防守球员等）"""
        print("=" * 100)
        print("步骤 0.5: 计算射门特征")
        print("=" * 100)

        # 转换为数值类型
        for col in ['location_x', 'location_y', 'location_z',
                    'shot_end_location_x', 'shot_end_location_y', 'shot_end_location_z']:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

        # StatsBomb 坐标系：球场长度为120码，宽度为80码
        # 球门中心在 (120, 40)
        GOAL_X = 120
        GOAL_Y = 40

        # 计算到球门的距离（转换为米：1码 = 0.9144米）
        self.df['distance_to_goal'] = np.sqrt(
            (GOAL_X - self.df['location_x'])**2 + (GOAL_Y - self.df['location_y'])**2
        ) * 0.9144  # 转换为米

        # 计算射门角度
        def calculate_angle(row):
            shot_x = row['location_x']
            shot_y = row['location_y']

            # 到左门柱和右门柱的角度
            # 左门柱: (120, 36.8), 右门柱: (120, 43.2)
            dy_left = 36.8 - shot_y
            dy_right = 43.2 - shot_y
            dx = 120 - shot_x

            # 使用反正切函数计算角度
            angle_left = np.arctan2(dy_left, dx)
            angle_right = np.arctan2(dy_right, dx)

            # 角度差
            angle_diff = abs(angle_left - angle_right)

            # 转换为度
            return abs(angle_diff * 180 / np.pi)

        self.df['shot_angle'] = self.df.apply(calculate_angle, axis=1)

        print(f"  计算了 distance_to_goal 和 shot_angle")

        # 处理 shot_freeze_frame_raw 来计算防守球员信息
        print(f"  处理 shot_freeze_frame_raw 计算防守球员信息...")

        # 初始化列
        self.df['opponents_in_box'] = 0
        self.df['teammates_in_box'] = 0
        self.df['opponents_in_six_yard'] = 0
        self.df['gk_in_six_yard'] = False
        self.df['opponents_in_shot_cone'] = 0

        # StatsBomb 禁区区域：x > 102, y 在 [18, 62]
        PENALTY_AREA_X = 102
        PENALTY_AREA_Y_MIN = 18
        PENALTY_AREA_Y_MAX = 62

        # 六码区：x > 114, y 在 [30, 50]
        SIX_YARD_X = 114
        SIX_YARD_Y_MIN = 30
        SIX_YARD_Y_MAX = 50

        # 处理 freeze_frame 数据
        freeze_data_count = 0
        for idx, row in self.df.iterrows():
            freeze_raw = row.get('shot_freeze_frame_raw')

            if pd.notna(freeze_raw) and freeze_raw:
                try:
                    freeze_data = json.loads(freeze_raw)

                    opponents_in_box = 0
                    teammates_in_box = 0
                    opponents_in_six_yard = 0
                    gk_in_six_yard = False
                    opponents_in_shot_cone = 0

                    shot_x = row['location_x']
                    shot_y = row['location_y']

                    for player in freeze_data:
                        # 正确的数据结构解析
                        player_id = player.get('player', {}).get('id')
                        position_info = player.get('position', {})
                        position_name = position_info.get('name', '')

                        # 球员位置坐标在 location 字段中，格式为 [x, y]
                        player_location = player.get('location')
                        if player_location is None or len(player_location) < 2:
                            continue

                        player_x = player_location[0]
                        player_y = player_location[1]

                        is_teammate = player.get('teammate', False)
                        is_goalkeeper = position_name == 'Goalkeeper'

                        # 判断是否在禁区
                        if player_x > PENALTY_AREA_X and PENALTY_AREA_Y_MIN < player_y < PENALTY_AREA_Y_MAX:
                            if is_teammate:
                                teammates_in_box += 1
                            else:
                                opponents_in_box += 1

                        # 判断是否在六码区
                        if player_x > SIX_YARD_X and SIX_YARD_Y_MIN < player_y < SIX_YARD_Y_MAX:
                            if is_goalkeeper:
                                gk_in_six_yard = True
                            elif not is_teammate:
                                opponents_in_six_yard += 1

                        # 判断是否在射门锥形区域
                        if not is_teammate:
                            # 简单的锥形判断：在射门点和球门线之间的扇形区域
                            # 使用简化的判断方法
                            goal_y_min = 36.8
                            goal_y_max = 43.2
                            shot_to_goal_dx = GOAL_X - shot_x
                            shot_to_goal_dy = GOAL_Y - shot_y

                            # 计算球员到射线的距离
                            player_dx = player_x - shot_x
                            player_dy = player_y - shot_y

                            # 计算垂直距离
                            cross_product = abs(player_dx * shot_to_goal_dy - player_dy * shot_to_goal_dx)
                            shot_to_goal_dist = np.sqrt(shot_to_goal_dx**2 + shot_to_goal_dy**2)

                            if shot_to_goal_dist > 0:
                                perpendicular_dist = cross_product / shot_to_goal_dist

                                # 如果垂直距离小于5码，且在射门线前方
                                dot_product = player_dx * shot_to_goal_dx + player_dy * shot_to_goal_dy
                                if perpendicular_dist < 5 and dot_product > 0:
                                    opponents_in_shot_cone += 1

                    self.df.at[idx, 'opponents_in_box'] = opponents_in_box
                    self.df.at[idx, 'teammates_in_box'] = teammates_in_box
                    self.df.at[idx, 'opponents_in_six_yard'] = opponents_in_six_yard
                    self.df.at[idx, 'gk_in_six_yard'] = gk_in_six_yard
                    self.df.at[idx, 'opponents_in_shot_cone'] = opponents_in_shot_cone

                    freeze_data_count += 1

                except (json.JSONDecodeError, KeyError, TypeError):
                    pass

        print(f"  处理了 {freeze_data_count} 条 freeze_frame 数据")

        print(f"\n特征统计:")
        print(f"  distance_to_goal: 均值={self.df['distance_to_goal'].mean():.2f}m, "
              f"最小值={self.df['distance_to_goal'].min():.2f}m, "
              f"最大值={self.df['distance_to_goal'].max():.2f}m")
        print(f"  shot_angle: 均值={self.df['shot_angle'].mean():.2f}°, "
              f"最小值={self.df['shot_angle'].min():.2f}°, "
              f"最大值={self.df['shot_angle'].max():.2f}°")
        print(f"  opponents_in_box: 均值={self.df['opponents_in_box'].mean():.2f}")
        print(f"  teammates_in_box: 均值={self.df['teammates_in_box'].mean():.2f}")
        print(f"  opponents_in_six_yard: 均值={self.df['opponents_in_six_yard'].mean():.2f}")
        print(f"  gk_in_six_yard: {self.df['gk_in_six_yard'].sum()} 次")
        print(f"  opponents_in_shot_cone: 均值={self.df['opponents_in_shot_cone'].mean():.2f}\n")

        return self.df

    def step1_feature_engineering_preferred_foot(self):
        """Step 1: 惯用脚特征工程"""
        print("=" * 100)
        print("步骤 1: 惯用脚特征工程 (Preferred Foot Logic)")
        print("=" * 100)

        # 筛选有效射门记录
        df_valid = self.df.dropna(subset=['player_id', 'player', 'shot_body_part']).copy()

        # 统计每个球员使用左右脚射门的次数
        print("\n统计每个球员的惯用脚...")
        foot_stats = df_valid[df_valid['shot_body_part'].isin(['Right Foot', 'Left Foot'])]\
            .groupby(['player_id', 'shot_body_part'])\
            .size()\
            .unstack(fill_value=0)

        # 确定每个球员的惯用脚
        self.preferred_foot_map = {}
        for player_id in foot_stats.index:
            right_count = foot_stats.loc[player_id, 'Right Foot'] if 'Right Foot' in foot_stats.columns else 0
            left_count = foot_stats.loc[player_id, 'Left Foot'] if 'Left Foot' in foot_stats.columns else 0

            if right_count > left_count:
                self.preferred_foot_map[player_id] = 'Right Foot'
            elif left_count > right_count:
                self.preferred_foot_map[player_id] = 'Left Foot'
            else:
                # 如果次数相同，设为默认右脚
                self.preferred_foot_map[player_id] = 'Right Foot'

        print(f"  计算了 {len(self.preferred_foot_map)} 名球员的惯用脚")

        # 统计惯用脚分布
        foot_distribution = pd.Series(self.preferred_foot_map).value_counts()
        print(f"\n惯用脚分布:")
        print(f"  右脚: {foot_distribution.get('Right Foot', 0):,} 名球员")
        print(f"  左脚: {foot_distribution.get('Left Foot', 0):,} 名球员")

        # 创建 shot_execution_type 特征
        print("\n创建 shot_execution_type 特征...")

        def determine_execution_type(row):
            body_part = row['shot_body_part']

            # 头球
            if body_part == 'Head':
                return 'Head'

            # 检查是否为脚部射门
            if body_part not in ['Right Foot', 'Left Foot']:
                return 'Other'

            player_id = row['player_id']
            preferred_foot = self.preferred_foot_map.get(player_id, 'Right Foot')

            # 优势脚射门
            if body_part == preferred_foot:
                return 'Strong_Foot'
            else:
                # 弱脚射门
                return 'Weak_Foot'

        self.df['shot_execution_type'] = self.df.apply(determine_execution_type, axis=1)

        # 统计 shot_execution_type 分布
        exec_type_dist = self.df['shot_execution_type'].value_counts()
        print(f"\nshot_execution_type 分布:")
        for exec_type, count in exec_type_dist.items():
            print(f"  {exec_type}: {count:,} 次 ({count/len(self.df)*100:.2f}%)")

        print()

        return self.df

    def step2_data_cleaning_and_encoding(self):
        """Step 2: 数据清理与编码"""
        print("=" * 100)
        print("步骤 2: 数据清理与编码")
        print("=" * 100)

        # 定义特征列
        continuous_features = [
            'distance_to_goal', 'shot_angle',
            'opponents_in_box', 'teammates_in_box',
            'opponents_in_six_yard', 'opponents_in_shot_cone'
        ]

        categorical_features = [
            'shot_execution_type', 'under_pressure', 'shot_technique',
            'shot_first_time', 'shot_one_on_one', 'play_pattern', 'gk_in_six_yard'
        ]

        # 检查哪些特征存在
        available_continuous = [f for f in continuous_features if f in self.df.columns]
        available_categorical = [f for f in categorical_features if f in self.df.columns]

        print(f"\n可用特征:")
        print(f"  连续型特征 ({len(available_continuous)}): {available_continuous}")
        print(f"  类别型特征 ({len(available_categorical)}): {available_categorical}")

        # 创建特征副本
        df_features = self.df[available_continuous + available_categorical + ['shot_statsbomb_xg']].copy()

        # 处理 NaN 值
        print("\n处理缺失值...")
        for col in available_continuous:
            df_features[col] = df_features[col].fillna(0)

        for col in available_categorical:
            df_features[col] = df_features[col].fillna('Unknown')

        # 检查目标变量
        print(f"\n目标变量 shot_statsbomb_xg:")
        print(f"  非空值数量: {df_features['shot_statsbomb_xg'].notna().sum():,}")
        print(f"  缺失值数量: {df_features['shot_statsbomb_xg'].isna().sum():,}")

        # 移除目标变量为 NaN 的行
        df_features = df_features.dropna(subset=['shot_statsbomb_xg'])
        print(f"清理后数据集大小: {len(df_features):,} 行")

        # One-hot 编码类别变量
        print("\n进行 One-hot 编码...")
        df_encoded = pd.get_dummies(df_features, columns=available_categorical, drop_first=True)

        print(f"编码后特征数量: {len(df_encoded.columns) - 1}")  # 减去目标变量

        # 分离特征和目标
        y = df_encoded['shot_statsbomb_xg']
        X = df_encoded.drop('shot_statsbomb_xg', axis=1)

        self.feature_names = X.columns.tolist()

        print(f"\n特征矩阵形状: {X.shape}")
        print(f"目标变量形状: {y.shape}")

        # 数据集基本信息
        print(f"\n目标变量统计:")
        print(f"  均值: {y.mean():.4f}")
        print(f"  标准差: {y.std():.4f}")
        print(f"  最小值: {y.min():.4f}")
        print(f"  中位数: {y.median():.4f}")
        print(f"  最大值: {y.max():.4f}")

        return X, y

    def step3_model_training(self, X, y):
        """Step 3: 模型训练"""
        print("=" * 100)
        print("步骤 3: XGBoost 模型训练")
        print("=" * 100)

        # 划分训练集和测试集
        print("\n划分训练集和测试集 (80/20)...")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        print(f"  训练集大小: {len(self.X_train):,}")
        print(f"  测试集大小: {len(self.X_test):,}")

        # 定义 XGBoost 回归器
        print("\n训练 XGBoost 回归模型...")
        self.model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )

        self.model.fit(self.X_train, self.y_train)

        print("模型训练完成!")

        # 评估模型
        print("\n模型评估:")

        # 训练集预测
        y_train_pred = self.model.predict(self.X_train)
        train_r2 = r2_score(self.y_train, y_train_pred)
        train_mae = mean_absolute_error(self.y_train, y_train_pred)
        train_rmse = np.sqrt(mean_squared_error(self.y_train, y_train_pred))

        print(f"\n训练集性能:")
        print(f"  R² Score: {train_r2:.4f}")
        print(f"  MAE: {train_mae:.4f}")
        print(f"  RMSE: {train_rmse:.4f}")

        # 测试集预测
        y_test_pred = self.model.predict(self.X_test)
        test_r2 = r2_score(self.y_test, y_test_pred)
        test_mae = mean_absolute_error(self.y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_test_pred))

        print(f"\n测试集性能:")
        print(f"  R² Score: {test_r2:.4f}")
        print(f"  MAE: {test_mae:.4f}")
        print(f"  RMSE: {test_rmse:.4f}")

        # 模型还原程度分析
        print(f"\n模型还原度分析:")
        print(f"  测试集 R² = {test_r2:.2%}，说明模型能够解释 StatsBomb xG 变化的 {test_r2*100:.1f}%")
        print(f"  平均绝对误差 MAE = {test_mae:.4f}，说明预测值与真实值的平均偏差约 {test_mae:.4f}")

        # 特征重要性
        print(f"\n前20个最重要特征:")
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        for idx, row in importance_df.head(20).iterrows():
            print(f"  {row['feature']:40s}: {row['importance']:.4f}")

        return importance_df

    def step4_explainability_analysis(self):
        """Step 4: 可解释性分析"""
        print("\n" + "=" * 100)
        print("步骤 4: 可解释性分析")
        print("=" * 100)

        if SHAP_AVAILABLE:
            return self._shap_analysis()
        else:
            return self._xgb_gain_analysis()

    def _shap_analysis(self):
        """使用 SHAP 进行深度可解释性分析"""
        print("\n使用 SHAP 进行深度可解释性分析...")

        # 创建 SHAP 解释器
        print("创建 SHAP TreeExplainer...")
        self.explainer = shap.TreeExplainer(self.model)

        print("计算 SHAP 值...")
        # 使用测试集的一个子集来计算 SHAP 值（加快速度）
        sample_size = min(2000, len(self.X_test))
        X_sample = self.X_test.sample(sample_size, random_state=42)
        self.shap_values = self.explainer.shap_values(X_sample)

        print(f"计算完成! 样本数: {len(X_sample)}")

        # 绘制 SHAP Summary Plot
        print("\n绘制 SHAP Summary Plot...")
        plt.figure(figsize=(14, 10))
        shap.summary_plot(self.shap_values, X_sample, feature_names=self.feature_names,
                         show=False, plot_size=(14, 10))
        plt.title('SHAP Summary Plot - StatsBomb xG 特征贡献度', fontsize=16, pad=20)
        plt.tight_layout()

        summary_plot_path = self.output_dir / 'shap_summary_plot_xg_model.png'
        plt.savefig(summary_plot_path, dpi=150, bbox_inches='tight')
        print(f"  保存到: {summary_plot_path}")
        plt.close()

        # 绘制特征重要性条形图
        print("\n绘制特征重要性条形图...")
        mean_shap = np.abs(self.shap_values).mean(axis=0)
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'mean_shap': mean_shap
        }).sort_values('mean_shap', ascending=False)

        plt.figure(figsize=(14, 10))
        sns.barplot(data=feature_importance_df.head(20), x='mean_shap', y='feature')
        plt.title('Top 20 Features by Mean |SHAP Value|', fontsize=16, pad=20)
        plt.xlabel('Mean |SHAP Value|', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.tight_layout()

        importance_plot_path = self.output_dir / 'shap_feature_importance_xg_model.png'
        plt.savefig(importance_plot_path, dpi=150, bbox_inches='tight')
        print(f"  保存到: {importance_plot_path}")
        plt.close()

        # distance_to_goal Dependence Plot
        if 'distance_to_goal' in self.feature_names:
            print("\n绘制 distance_to_goal Dependence Plot...")
            dist_idx = self.feature_names.index('distance_to_goal')

            fig, ax = plt.subplots(figsize=(12, 8))
            shap.dependence_plot(dist_idx, self.shap_values, X_sample,
                               feature_names=self.feature_names,
                               show=False, ax=ax)
            plt.title('SHAP Dependence Plot: distance_to_goal', fontsize=16, pad=20)
            plt.tight_layout()

            dist_plot_path = self.output_dir / 'shap_dependence_distance_to_goal_xg_model.png'
            plt.savefig(dist_plot_path, dpi=150, bbox_inches='tight')
            print(f"  保存到: {dist_plot_path}")
            plt.close()

        # shot_angle Dependence Plot
        if 'shot_angle' in self.feature_names:
            print("\n绘制 shot_angle Dependence Plot...")
            angle_idx = self.feature_names.index('shot_angle')

            fig, ax = plt.subplots(figsize=(12, 8))
            shap.dependence_plot(angle_idx, self.shap_values, X_sample,
                               feature_names=self.feature_names,
                               show=False, ax=ax)
            plt.title('SHAP Dependence Plot: shot_angle', fontsize=16, pad=20)
            plt.tight_layout()

            angle_plot_path = self.output_dir / 'shap_dependence_shot_angle_xg_model.png'
            plt.savefig(angle_plot_path, dpi=150, bbox_inches='tight')
            print(f"  保存到: {angle_plot_path}")
            plt.close()

        # 距离衰减分析
        print("\n分析距离衰减非线性关系...")
        self._analyze_distance_decay()

        return feature_importance_df

    def _xgb_gain_analysis(self):
        """使用 XGBoost Gain 进行可解释性分析（SHAP 不可用时的替代方案）"""
        print("\n使用 XGBoost 内置特征重要性分析...")

        # 获取多种特征重要性类型
        gain_importance = self.model.get_booster().get_score(importance_type='gain')
        weight_importance = self.model.get_booster().get_score(importance_type='weight')
        cover_importance = self.model.get_booster().get_score(importance_type='cover')

        # 转换为 DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'gain': [gain_importance.get(f'f{i}', 0) for i in range(len(self.feature_names))],
            'weight': [weight_importance.get(f'f{i}', 0) for i in range(len(self.feature_names))],
            'cover': [cover_importance.get(f'f{i}', 0) for i in range(len(self.feature_names))],
            'importance': self.model.feature_importances_
        }).sort_values('gain', ascending=False)

        print(f"\nTop 20 特征重要性 (Gain - 特征对模型的贡献度):")
        print("-" * 100)
        for idx, row in importance_df.head(20).iterrows():
            print(f"{row['feature']:40s} | Gain: {row['gain']:8.2f} | "
                  f"Weight: {row['weight']:6.0f} | Cover: {row['cover']:8.0f}")

        # 绘制特征重要性图
        self._plot_feature_importance(importance_df)

        # 距离衰减分析
        self._analyze_distance_decay()

        # 部分依赖图分析
        self._plot_partial_dependence()

        return importance_df

    def _plot_feature_importance(self, importance_df):
        """绘制特征重要性图"""
        # 创建一个包含多种重要性类型的图表
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Gain 重要性
        sns.barplot(data=importance_df.head(20), x='gain', y='feature', ax=axes[0, 0])
        axes[0, 0].set_title('Feature Importance by Gain', fontsize=12)
        axes[0, 0].set_xlabel('Gain', fontsize=10)
        axes[0, 0].set_ylabel('')

        # Weight 重要性
        sns.barplot(data=importance_df.head(20), x='weight', y='feature', ax=axes[0, 1])
        axes[0, 1].set_title('Feature Importance by Weight', fontsize=12)
        axes[0, 1].set_xlabel('Weight', fontsize=10)
        axes[0, 1].set_ylabel('')

        # Cover 重要性
        sns.barplot(data=importance_df.head(20), x='cover', y='feature', ax=axes[1, 0])
        axes[1, 0].set_title('Feature Importance by Cover', fontsize=12)
        axes[1, 0].set_xlabel('Cover', fontsize=10)
        axes[1, 0].set_ylabel('')

        # Feature importance (默认)
        sns.barplot(data=importance_df.head(20), x='importance', y='feature', ax=axes[1, 1])
        axes[1, 1].set_title('Feature Importance (Default)', fontsize=12)
        axes[1, 1].set_xlabel('Importance', fontsize=10)
        axes[1, 1].set_ylabel('')

        plt.tight_layout()

        plot_path = self.output_dir / 'xgboost_feature_importance_xg_model.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"\n特征重要性图保存到: {plot_path}")
        plt.close()

    def _plot_partial_dependence(self):
        """绘制部分依赖图"""
        from sklearn.inspection import PartialDependenceDisplay

        print("\n绘制部分依赖图 (Partial Dependence Plot)...")

        # 选择最重要的连续特征
        top_continuous_features = [f for f in ['distance_to_goal', 'shot_angle',
                                              'opponents_in_box', 'teammates_in_box']
                                   if f in self.feature_names]

        if top_continuous_features:
            fig, ax = plt.subplots(figsize=(14, 10))

            PartialDependenceDisplay.from_estimator(
                self.model, self.X_test.sample(min(2000, len(self.X_test)), random_state=42),
                features=top_continuous_features[:4], ax=ax,
                n_jobs=-1
            )

            plt.suptitle('Partial Dependence Plots - StatsBomb xG Model', fontsize=16)
            plt.tight_layout()

            pdp_path = self.output_dir / 'partial_dependence_xg_model.png'
            plt.savefig(pdp_path, dpi=150, bbox_inches='tight')
            print(f"部分依赖图保存到: {pdp_path}")
            plt.close()

    def _analyze_distance_decay(self):
        """分析距离衰减的非线性关系"""
        print("\n" + "=" * 100)
        print("距离衰减分析 - StatsBomb 如何处理距离与 xG 的关系")
        print("=" * 100)

        # 计算不同距离区间的平均 xG
        df_analysis = self.df.copy()
        df_analysis = df_analysis.dropna(subset=['distance_to_goal', 'shot_statsbomb_xg'])

        # 创建距离区间
        df_analysis['distance_bin'] = pd.cut(df_analysis['distance_to_goal'],
                                             bins=[0, 5, 10, 15, 20, 25, 30, 40, 50],
                                             labels=['0-5m', '5-10m', '10-15m', '15-20m',
                                                    '20-25m', '25-30m', '30-40m', '40m+'])

        distance_xg = df_analysis.groupby('distance_bin').agg({
            'shot_statsbomb_xg': ['mean', 'count'],
            'shot_outcome': lambda x: (x == 'Goal').sum()
        }).reset_index()
        distance_xg.columns = ['distance_bin', 'avg_xg', 'shots', 'goals']
        distance_xg['conversion_rate'] = distance_xg['goals'] / distance_xg['shots'] * 100

        print("\n距离区间分析:")
        print(f"{'距离区间':<12} | {'射门数':>8} | {'进球数':>8} | {'平均xG':>10} | {'转化率':>10}")
        print("-" * 70)
        for _, row in distance_xg.iterrows():
            print(f"{row['distance_bin']:<12} | {row['shots']:8d} | {row['goals']:8d} | "
                  f"{row['avg_xg']:10.4f} | {row['conversion_rate']:9.2f}%")

        # 绘制距离衰减曲线
        plt.figure(figsize=(14, 8))

        # 左图：平均xG vs 距离
        plt.subplot(1, 2, 1)
        sns.scatterplot(data=df_analysis.sample(min(5000, len(df_analysis)), random_state=42),
                       x='distance_to_goal', y='shot_statsbomb_xg', alpha=0.1, color='blue', s=10)
        sns.scatterplot(data=distance_xg, x='distance_bin', y='avg_xg',
                       s=200, color='red', zorder=5, label='区间平均值')
        plt.xlabel('Distance to Goal (m)', fontsize=12)
        plt.ylabel('Expected Goals (xG)', fontsize=12)
        plt.title('StatsBomb xG vs Distance', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 右图：转化率 vs 距离
        plt.subplot(1, 2, 2)
        sns.barplot(data=distance_xg, x='distance_bin', y='conversion_rate')
        plt.xlabel('Distance Bin', fontsize=12)
        plt.ylabel('Conversion Rate (%)', fontsize=12)
        plt.title('Actual Conversion Rate by Distance', fontsize=14)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        distance_plot_path = self.output_dir / 'distance_decay_analysis_xg_model.png'
        plt.savefig(distance_plot_path, dpi=150, bbox_inches='tight')
        print(f"\n距离衰减分析图保存到: {distance_plot_path}")
        plt.close()

        # 射门角度分析
        print("\n" + "=" * 100)
        print("射门角度分析 - StatsBomb 如何处理角度与 xG 的关系")
        print("=" * 100)

        df_analysis['angle_bin'] = pd.cut(df_analysis['shot_angle'],
                                         bins=[0, 5, 10, 20, 30, 45, 90, 180],
                                         labels=['0-5°', '5-10°', '10-20°', '20-30°',
                                                '30-45°', '45-90°', '90°+'])

        angle_xg = df_analysis.groupby('angle_bin').agg({
            'shot_statsbomb_xg': ['mean', 'count'],
            'shot_outcome': lambda x: (x == 'Goal').sum()
        }).reset_index()
        angle_xg.columns = ['angle_bin', 'avg_xg', 'shots', 'goals']
        angle_xg['conversion_rate'] = angle_xg['goals'] / angle_xg['shots'] * 100

        print("\n角度区间分析:")
        print(f"{'角度区间':<12} | {'射门数':>8} | {'进球数':>8} | {'平均xG':>10} | {'转化率':>10}")
        print("-" * 70)
        for _, row in angle_xg.iterrows():
            print(f"{row['angle_bin']:<12} | {row['shots']:8d} | {row['goals']:8d} | "
                  f"{row['avg_xg']:10.4f} | {row['conversion_rate']:9.2f}%")

        # 绘制角度曲线
        plt.figure(figsize=(14, 8))

        # 左图：平均xG vs 角度
        plt.subplot(1, 2, 1)
        sns.scatterplot(data=df_analysis.sample(min(5000, len(df_analysis)), random_state=42),
                       x='shot_angle', y='shot_statsbomb_xg', alpha=0.1, color='green', s=10)
        sns.scatterplot(data=angle_xg, x='angle_bin', y='avg_xg',
                       s=200, color='red', zorder=5, label='区间平均值')
        plt.xlabel('Shot Angle (degrees)', fontsize=12)
        plt.ylabel('Expected Goals (xG)', fontsize=12)
        plt.title('StatsBomb xG vs Angle', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 右图：转化率 vs 角度
        plt.subplot(1, 2, 2)
        sns.barplot(data=angle_xg, x='angle_bin', y='conversion_rate')
        plt.xlabel('Angle Bin', fontsize=12)
        plt.ylabel('Conversion Rate (%)', fontsize=12)
        plt.title('Actual Conversion Rate by Angle', fontsize=14)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        angle_plot_path = self.output_dir / 'angle_analysis_xg_model.png'
        plt.savefig(angle_plot_path, dpi=150, bbox_inches='tight')
        print(f"\n角度分析图保存到: {angle_plot_path}")
        plt.close()

    def save_results(self, importance_df):
        """保存结果"""
        print("\n" + "=" * 100)
        print("保存结果")
        print("=" * 100)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 保存特征重要性
        importance_path = self.output_dir / f'feature_importance_xg_model_{timestamp}.csv'
        importance_df.to_csv(importance_path, index=False, encoding='utf-8-sig')
        print(f"特征重要性保存到: {importance_path}")

        # 保存模型
        model_path = self.output_dir / f'xgboost_xg_model_{timestamp}.json'
        self.model.save_model(str(model_path))
        print(f"模型保存到: {model_path}")

        # 保存训练数据统计
        y_train_pred = self.model.predict(self.X_train)
        y_test_pred = self.model.predict(self.X_test)
        stats_df = pd.DataFrame({
            'metric': ['train_size', 'test_size', 'train_r2', 'test_r2', 'train_mae', 'test_mae'],
            'value': [len(self.X_train), len(self.X_test),
                     r2_score(self.y_train, y_train_pred),
                     r2_score(self.y_test, y_test_pred),
                     mean_absolute_error(self.y_train, y_train_pred),
                     mean_absolute_error(self.y_test, y_test_pred)]
        })
        stats_path = self.output_dir / f'model_stats_xg_model_{timestamp}.csv'
        stats_df.to_csv(stats_path, index=False, encoding='utf-8-sig')
        print(f"模型统计保存到: {stats_path}")

    def run_full_pipeline(self):
        """运行完整流程"""
        print("=" * 100)
        print("XGBoost xG 模型训练与可解释性分析")
        print("目标：破解 StatsBomb xG 的内部权重逻辑")
        print("=" * 100)
        print()

        # Step 0: 加载数据
        self.load_all_shot_data()

        # Step 0.5: 计算射门特征
        self.calculate_shot_features()

        # Step 1: 惯用脚特征工程
        self.step1_feature_engineering_preferred_foot()

        # Step 2: 数据清理与编码
        X, y = self.step2_data_cleaning_and_encoding()

        # Step 3: 模型训练
        xgb_importance = self.step3_model_training(X, y)

        # Step 4: 可解释性分析
        shap_importance = self.step4_explainability_analysis()

        # 保存结果
        self.save_results(shap_importance)

        print("\n" + "=" * 100)
        print("分析完成!")
        print("=" * 100)

        return self.model, xgb_importance, shap_importance


def main():
    """主函数"""
    # 创建分析器实例
    analyzer = XGModelExplainer(
        raw_data_dir="data/raw",
        output_dir="data/processed"
    )

    # 运行完整流程
    model, xgb_importance, shap_importance = analyzer.run_full_pipeline()

    return analyzer


if __name__ == "__main__":
    analyzer = main()
