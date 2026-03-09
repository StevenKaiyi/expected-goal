# 开发日志 (CHANGELOG)

本文档记录项目的开发历史和重要变更。

---

## [2026-03-09] - xG模型训练与SHAP分析

### 新增功能

#### 数据分析
- **球员射门统计脚本** (`src/analyze_player_shots.py`)
  - 统计所有60,118次射门的球员数据
  - 计算每名球员的射门数、进球数、xG总和
  - 分析xG vs 实际进球的差异
  - 识别超常发挥和低于预期的球员

#### 模型训练
- **XGBoost xG模型训练脚本** (`src/train_xgboost_shots_model.py`)
  - 从35个CSV文件（60,118次射门）加载数据
  - 实现完整的特征工程流程
  - 训练XGBoost回归模型拟合StatsBomb xG
  - 使用SHAP/XGBoost内置方法进行可解释性分析

### 特征工程

#### 几何特征
- `distance_to_goal`: 射门点到球门中心的距离（米）
- `shot_angle`: 射门角度（度），基于两门柱夹角计算

#### 防守特征
- `opponents_in_box`: 大禁区内对手人数
- `teammates_in_box`: 大禁区内队友人数
- `opponents_in_six_yard`: 六码区内对手人数
- `gk_in_six_yard`: 对方门将是否在小禁区内
- `opponents_in_shot_cone`: 射门锥形区域内的对手人数

#### 惯用脚特征
- 统计3,727名球员的惯用脚（2,652右脚，1,075左脚）
- `shot_execution_type`: 强脚/弱脚/头球分类

#### 比赛情境特征
- `under_pressure`: 是否受到防守压力
- `shot_first_time`: 是否第一时间射门
- `shot_one_on_one`: 是否单刀
- `shot_technique`: 射门技术（Normal/Volley/Lob等）
- `play_pattern`: 进攻模式（Regular Play/Counter/Free Kick等）

### 模型性能

#### 初始模型（Bug版本）
- 测试集 R²: 72.90%
- 测试集 MAE: 0.0316
- 问题：防守特征计算bug导致所有值为0

#### 修复后模型
- **测试集 R²: 84.29%** ✅ (+11.39%)
- **测试集 MAE: 0.0242** ✅ (-23%)
- **测试集 RMSE: 0.0486**

### 特征重要性分析（修复后）

| 排名 | 特征 | 重要性 | 说明 |
|------|------|--------|------|
| 1 | opponents_in_shot_cone | 15.89% | 射门锥形区域内的对手数量 |
| 2 | gk_in_six_yard_True | 12.93% | 门将在六码区 |
| 3 | distance_to_goal | 10.82% | 到球门的距离 |
| 4 | shot_first_time_True | 10.05% | 第一时间射门 |
| 5 | shot_angle | 9.97% | 射门角度 |
| 6 | shot_execution_type_Strong_Foot | 7.37% | 优势脚射门 |

### 关键发现

1. **防守压力是最重要因素**：射门路径上的防守球员数量（opponents_in_shot_cone）是影响xG的首要特征（15.89%）
2. **门将位置至关重要**：门将是否在六码区（gk_in_six_yard）是第二重要特征（12.93%）
3. **空间特征占主导地位**：前5个最重要特征中有4个是空间/防守相关特征
4. **距离和角度仍为核心**：传统几何特征（distance_to_goal, shot_angle）依然是关键因素

### 数据分析发现

#### 射门距离衰减
| 距离区间 | 射门数 | 进球数 | 平均xG | 转化率 |
|----------|--------|--------|--------|--------|
| 0-5m | 1,137 | 608 | 0.5043 | 53.47% |
| 5-10m | 10,147 | 2,106 | 0.1916 | 20.75% |
| 10-15m | 13,186 | 1,813 | 0.1275 | 13.75% |
| 15-20m | 12,025 | 959 | 0.0762 | 7.98% |
| 20m+ | 21,453 | 694 | 0.0171 | 3.23% |

#### 射门角度影响
| 角度区间 | 射门数 | 进球数 | 平均xG | 转化率 |
|----------|--------|--------|--------|--------|
| 0-10° | 6,393 | 163 | 0.0107 | 2.55% |
| 10-20° | 33,687 | 2,017 | 0.0585 | 5.99% |
| 20-30° | 10,341 | 1,525 | 0.1382 | 14.75% |
| 30-45° | 6,699 | 1,298 | 0.1776 | 19.38% |
| 45°+ | 2,994 | 1,179 | 0.3921 | 39.38% |

### Bug修复

#### 严重问题：freeze_frame_raw 数据解析错误
- **问题描述**: 防守特征计算函数中错误地从 `position` 字典中提取 `x` 和 `y` 坐标
- **实际数据结构**: 球员坐标在 `location` 列表中，格式为 `[x, y]`
- **影响**: 所有防守特征（opponents_in_box, teammates_in_box等）被错误计算为0
- **修复**: 正确从 `player['location']` 获取坐标
- **结果**: 防守特征恢复有效值，模型R²从72.90%提升至84.29%

### 文件组织

创建了清晰的文件夹结构：
```
data/processed/
├── models/                    # 模型相关
│   ├── saved_models/          # 保存的模型文件
│   ├── feature_importance/    # 特征重要性
│   ├── stats/                # 模型统计数据
│   └── visualizations/       # 模型可视化图表
├── statistics/               # 基础统计
│   ├── player/               # 球员统计
│   ├── league/               # 联赛统计
│   └── shot/                # 射门统计
├── visualizations/           # 基础可视化
├── data/                     # 处理后的数据
└── archive/                 # 归档旧文件
```

### 数据集统计

- **总射门数**: 60,118次
- **覆盖赛季**: 1974-2024年
- **包含赛事**: 世界杯、西甲、英超、意甲、德甲、欧冠
- **涉及球员**: 4,005名
- **总进球数**: 6,183个
- **整体转化率**: 10.28%

### 输出文件

#### 模型文件
- `data/processed/models/saved_models/xgboost_xg_model_20260309_142311.json`
- `data/processed/models/feature_importance/feature_importance_xg_model_20260309_142311.csv`
- `data/processed/models/stats/model_stats_xg_model_20260309_142311.csv`

#### 可视化
- `data/processed/models/visualizations/angle_analysis_xg_model.png` - 射门角度分析
- `data/processed/models/visualizations/distance_decay_analysis_xg_model.png` - 距离衰减分析
- `data/processed/models/visualizations/partial_dependence_xg_model.png` - 部分依赖图
- `data/processed/models/visualizations/xgboost_feature_importance_xg_model.png` - 特征重要性

#### 统计数据
- `data/processed/statistics/player/player_shots_all_data_20260309_132136.csv`
- `data/processed/statistics/shot/shot_statistics_report.json`

### 未来计划

- [ ] 实现增强的特征工程（多级防守密度、球员热区等）
- [ ] 集成LightGBM/CatBoost进行模型对比
- [ ] 添加球员技术能力特征
- [ ] 引入比赛时间维度特征
- [ ] 探索混合模型架构（神经网络+树模型）
- [ ] 目标：将R²提升至89%+以挑战StatsBomb基准

---

## [2026-03-07] - 项目初始化

### 初始设置
- 创建项目结构
- 配置StatsBomb API访问
- 设置数据目录
- 创建基础配置文件

---

## 版本说明

- **v0.1.0** (2026-03-09): 初始xG模型实现，R²=84.29%
- **v0.0.0** (2026-03-07): 项目初始化
