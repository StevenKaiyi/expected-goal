# Football Expected Goals (xG) Model

基于 StatsBomb Open Data 的足球预期进球模型，用于量化射门进球概率。

## 项目概述

xG (Expected Goals) 是一个高级足球指标，用于量化每次射门的进球概率。本项目使用 StatsBomb 的公开数据集，构建机器学习模型来分析并预测射门 xG 值。

### 核心目标

1. **破解 StatsBomb xG 模型** - 通过特征工程和可解释性分析理解其内部权重逻辑
2. **构建竞争性 xG 模型** - 开发具有竞争力的自有模型
3. **高级分析** - 提供深入的射门、球员和比赛分析

### 技术栈

- **数据获取**: StatsBomb CSV 数据集 (60,118+ 次射门)
- **数据处理**: pandas, numpy
- **机器学习**: xgboost
- **可视化**: matplotlib, seaborn
- **可解释性**: SHAP (可用时)

## 项目结构

```
expected-goal/
├── data/                      # 数据目录
│   ├── raw/                   # 原始CSV数据 (35个文件，1974-2024年)
│   │   ├── 1958/World_Cup.csv
│   │   ├── 1962/World_Cup.csv
│   │   ├── 2015/2016/La_Liga.csv
│   │   ├── 2022/2023/Champions_League.csv
│   │   └── ...
│   └── processed/             # 处理后的数据
│       ├── models/            # 模型相关
│       │   ├── saved_models/     # 训练好的模型
│       │   ├── feature_importance/ # 特征重要性
│       │   ├── stats/           # 模型统计
│       │   └── visualizations/  # 模型可视化
│       ├── statistics/         # 基础统计
│       │   ├── player/         # 球员统计
│       │   ├── league/         # 联赛统计
│       │   └── shot/          # 射门统计
│       ├── visualizations/     # 基础可视化图表
│       ├── data/             # 处理后的CSV数据
│       └── archive/           # 归档的旧文件
├── src/
│   ├── train_xgboost_shots_model.py  # XGBoost模型训练主脚本
│   ├── analyze_player_shots.py        # 球员射门统计分析
│   └── analyze_shot_stats.py         # 射门数据统计分析
├── config.py                       # 项目配置
├── requirements.txt                 # Python依赖
├── README.md                       # 本文件
└── CHANGELOG.md                    # 开发日志
```

## 当前状态

### 最新成果 (2026-03-09)

| 指标 | 值 |
|------|------|
| 模型 | XGBoost Regressor |
| 测试集 R² | **84.29%** |
| 测试集 MAE | 0.0242 |
| 数据规模 | 60,118 次射门 |
| 特征数量 | 27 (编码后) |
| 评估基准 | StatsBomb xG |

### 特征重要性 Top 6

| 排名 | 特征 | 重要性 | 说明 |
|------|------|--------|------|
| 1 | opponents_in_shot_cone | 15.89% | 射门锥形区域内的对手数量 |
| 2 | gk_in_six_yard_True | 12.93% | 门将在六码区 |
| 3 | distance_to_goal | 10.82% | 到球门的距离 |
| 4 | shot_first_time_True | 10.05% | 第一时间射门 |
| 5 | shot_angle | 9.97% | 射门角度 |
| 6 | shot_execution_type_Strong_Foot | 7.37% | 优势脚射门 |

### 关键发现

1. **防守压力是xG的最重要因素** - 射门路径上的防守球员数量直接影响进球概率
2. **门将位置至关重要** - 门将在小禁区内会显著降低xG
3. **几何特征占主导地位** - 空间/防守相关特征贡献超过40%的特征重要性

## 核心特征

### 空间特征
| 特征 | 类型 | 描述 |
|------|------|------|
| distance_to_goal | 连续 | 射门位置到球门中心的距离（米） |
| shot_angle | 连续 | 射门角度（度），基于两门柱夹角计算 |

### 防守特征
| 特征 | 类型 | 描述 |
|------|------|------|
| opponents_in_box | 连续 | 大禁区内对手人数 |
| teammates_in_box | 连续 | 大禁区内队友人数 |
| opponents_in_six_yard | 连续 | 六码区内对手人数 |
| gk_in_six_yard | 布尔 | 对方门将是否在小禁区内 |
| opponents_in_shot_cone | 连续 | 射门锥形区域内的对手人数 |

### 惯用脚特征
| 特征 | 类型 | 描述 |
|------|------|------|
| shot_execution_type | 分类 | Strong_Foot / Weak_Foot / Head |

### 比赛情境特征
| 特征 | 类型 | 描述 |
|------|------|------|
| under_pressure | 布尔 | 是否受到防守压力 |
| shot_first_time | 布尔 | 是否第一时间射门 |
| shot_one_on_one | 布尔 | 是否单刀 |
| shot_technique | 分类 | Normal / Volley / Lob / Half Volley等 |
| play_pattern | 分类 | Regular Play / Counter / Free Kick等 |

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 训练xG模型

```bash
python src/train_xgboost_shots_model.py
```

这将执行：
- 加载所有原始CSV数据
- 计算几何和防守特征
- 训练XGBoost模型
- 评估模型性能
- 生成可视化图表

### 3. 球员射门统计

```bash
python src/analyze_player_shots.py
```

这将输出：
- 所有射门的球员统计
- 每名球员的射门数、进球数、xG总和
- xG vs 实际进球的差异分析

## 数据集

### 来源

- **赛事**: 世界杯、西甲、英超、意甲、德甲、欧冠
- **时间跨度**: 1974-2024年
- **数据格式**: StatsBomb CSV 格式

### 规模

| 指标 | 数值 |
|------|------|
| 总射门数 | 60,118 |
| 包含比赛 | 35个文件 |
| 涉及球员 | 4,005名 |
| 总进球数 | 6,183 |
| 整体转化率 | 10.28% |

## 模型评估

### 性能指标

| 指标 | 当前值 | 说明 |
|------|--------|------|
| 测试集 R² | 84.29% | 模型能解释84.29%的xG变化 |
| 测试集 MAE | 0.0242 | 平均绝对误差 |
| 测试集 RMSE | 0.0486 | 均方根误差 |

### 距离衰减分析

| 距离区间 | 射门数 | 进球数 | 平均xG | 转化率 |
|----------|--------|--------|--------|--------|
| 0-5m | 1,137 | 608 | 0.5043 | 53.47% |
| 5-10m | 10,147 | 2,106 | 0.1916 | 20.75% |
| 10-15m | 13,186 | 1,813 | 0.1275 | 13.75% |
| 15-20m | 12,025 | 959 | 0.0762 | 7.98% |
| 20-25m | 11,825 | 457 | 0.0371 | 3.86% |
| 25m+ | 11,346 | 240 | 0.0153 | 2.12% |

## 参考资料

- [StatsBomb Open Data](https://github.com/statsbomb/open-data)
- [Expected Goals Explained](https://statsbomb.com/expected-goals-explained/)
- [xG Theory](https://en.wikipedia.org/wiki/Expected_goals)

## 开发进度

- [x] 环境搭建与数据探索
- [x] 数据加载与处理
- [x] 基础特征工程
- [x] XGBoost模型训练
- [x] 模型评估与可解释性分析
- [x] 球员射门统计分析
- [ ] 增强特征工程
- [ ] 多模型集成
- [ ] 模型优化与对比
- [ ] 生产部署

查看 [CHANGELOG.md](CHANGELOG.md) 了解详细的开发历史。

## License

MIT License
