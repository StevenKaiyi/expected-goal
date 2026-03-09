"""
Configuration file for xG Model Project
"""

# =============================================================================
# Data Configuration
# =============================================================================

# StatsBomb Free Data Access (no API key required)
STATSBOMB_CREDENTIALS = {
    "user": None,
    "passwd": None
}

# Competitions to include (past 5 years)
COMPETITIONS = {
    # English Premier League
    2: "Premier League",
    # UEFA Champions League
    7: "Champions League",
    # FIFA World Cup
    43: "World Cup",
    # La Liga
    11: "La Liga",
    # Serie A
    12: "Serie A",
    # Bundesliga
    9: "Bundesliga",
    # Ligue 1
    27: "Ligue 1",
}

# Seasons to include (competition_id: season_list)
SEASONS = {
    2: [37, 44, 90, 181, 282],  # Premier League 2020-2024
    7: [7, 9, 79, 123, 179],   # Champions League 2020-2024
    43: [3, 106],               # World Cup 2018, 2022
    11: [38, 42, 88, 180, 281], # La Liga 2020-2024
    12: [41, 45, 91, 183, 283], # Serie A 2020-2024
    9: [40, 46, 89, 182, 284],  # Bundesliga 2020-2024
    27: [39, 47, 92, 184, 285], # Ligue 1 2020-2024
}

# =============================================================================
# Path Configuration
# =============================================================================

DATA_DIR = "data"
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
CACHE_DIR = "data/cache"
MODELS_DIR = "models"

# =============================================================================
# MVP Feature List
# =============================================================================

# Spatial features (computed from location)
SPATIAL_FEATURES = [
    "distance_to_goal",
    "shot_angle",
]

# Categorical features (one-hot encoded)
CATEGORICAL_FEATURES = {
    "shot_body_part": ["Left Foot", "Right Foot", "Head/Other"],
    "shot_type": [
        "Open Play",
        "Free Kick",
        "Corner",
        "Volley",
        "Half Volley",
        "Other",
    ],
}

# Exclude shot types
EXCLUDED_SHOT_TYPES = ["Penalty"]

# =============================================================================
# Model Configuration
# =============================================================================

# Season split for time-based validation (avoid data leakage)
TRAIN_SEASONS = ["2020", "2021", "2022"]
VAL_SEASONS = ["2023"]
TEST_SEASONS = ["2024"]

# XGBoost parameters (MVP baseline)
XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": ["auc", "logloss"],
    "max_depth": 6,
    "learning_rate": 0.1,
    "n_estimators": 200,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
}

# Scale position weight for imbalanced data (1:9 ratio)
# Formula: total_negative / total_positive
SCALE_POS_WEIGHT = 9.0

# =============================================================================
# Evaluation Metrics
# =============================================================================

METRICS = ["roc_auc", "brier_score", "logloss", "calibration"]

# Target AUC for MVP
TARGET_AUC = 0.75

# =============================================================================
# Visualization Configuration
# =============================================================================

# Pitch dimensions (StatsBomb uses 120x80 meters)
PITCH_LENGTH = 120
PITCH_WIDTH = 80

# Goal post positions (StatsBomb coordinates)
GOAL_CENTER = (120, 40)
LEFT_POST = (120, 36)
RIGHT_POST = (120, 44)

# Shot plot marker size range
MARKER_SIZE_MIN = 20
MARKER_SIZE_MAX = 300

# =============================================================================
# API & Cache Configuration
# =============================================================================

# Cache duration (seconds)
CACHE_EXPIRY = 86400  # 24 hours

# Rate limiting (requests per second)
RATE_LIMIT = 2

# Retry settings
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds
