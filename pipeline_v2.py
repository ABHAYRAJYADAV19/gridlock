#!/usr/bin/env python3
"""
Traffic Demand Prediction - Improved Pipeline v2
=================================================
Key improvements over v1:
1. NO IQR clipping on target (was destroying 8.3% of high-demand signal)
2. Log-transform on target for better regression
3. Richer interaction features and target encoding with smoothing
4. Optuna hyperparameter tuning per model
5. Stacking meta-learner on top of base models
6. Proper temporal awareness (train=day48+early49, test=day49 hours 2-13)
"""

import sys, io, os, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import pygeohash as pgh
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

SEED = 42
N_FOLDS = 5
np.random.seed(SEED)

start_time = time.time()

print("=" * 70)
print("IMPROVED PIPELINE v2 - Traffic Demand Prediction")
print("=" * 70)

# =============================================================
# STEP 1 — LOAD DATA
# =============================================================
print("\n[1/10] Loading data...")
train = pd.read_csv('./work/train.csv')
test  = pd.read_csv('./work/test.csv')
print(f"  Train: {train.shape}, Test: {test.shape}")

y_raw = train['demand'].copy()
n_train = len(train)

print(f"  Demand: mean={y_raw.mean():.6f}, std={y_raw.std():.6f}, "
      f"min={y_raw.min():.6f}, max={y_raw.max():.6f}, skew={y_raw.skew():.2f}")

# =============================================================
# STEP 2 — TARGET TRANSFORM (NO clipping! Use log1p for skew)
# =============================================================
print("\n[2/10] Target transform...")
# Demand is in [0,1], skewed right. Log-transform helps regression.
y = np.log1p(y_raw)
print(f"  log1p(demand): mean={y.mean():.6f}, std={y.std():.6f}, skew={y.skew():.2f}")

# =============================================================
# STEP 3 — COMBINE & BASIC PREPROCESSING
# =============================================================
print("\n[3/10] Combining train+test for feature engineering...")

train['_source'] = 'train'
test['_source']  = 'test'

combined = pd.concat(
    [train.drop('demand', axis=1), test], axis=0
).reset_index(drop=True)

# --- Parse timestamp ---
parts = combined['timestamp'].astype(str).str.split(':', expand=True)
combined['hour']   = parts[0].astype(int)
combined['minute'] = parts[1].astype(int)

# --- Decode geohash ---
def safe_decode(gh):
    try:
        lat, lon = pgh.decode(gh)
        return float(lat), float(lon)
    except:
        return np.nan, np.nan

decoded = combined['geohash'].apply(safe_decode)
combined['lat'] = [d[0] for d in decoded]
combined['lon'] = [d[1] for d in decoded]

# --- Fill missing values (fit on train only) ---
train_mask = combined['_source'] == 'train'
for col in ['RoadType', 'Weather']:
    if combined[col].isnull().any():
        fill_val = combined.loc[train_mask, col].mode()[0]
        combined[col] = combined[col].fillna(fill_val)
        print(f"  Filled {col} NaN with '{fill_val}'")

if combined['Temperature'].isnull().any():
    # Fill temperature by geohash median (more accurate than global median)
    temp_by_geo = combined.loc[train_mask].groupby('geohash')['Temperature'].median()
    global_temp_median = combined.loc[train_mask, 'Temperature'].median()
    
    for idx in combined[combined['Temperature'].isnull()].index:
        geo = combined.loc[idx, 'geohash']
        if geo in temp_by_geo.index and not pd.isna(temp_by_geo[geo]):
            combined.loc[idx, 'Temperature'] = temp_by_geo[geo]
        else:
            combined.loc[idx, 'Temperature'] = global_temp_median
    print(f"  Filled Temperature NaN by geohash-median (fallback: {global_temp_median:.2f})")

print(f"  Remaining NaN: {combined.isnull().sum().sum()}")

# =============================================================
# STEP 4 — FEATURE ENGINEERING
# =============================================================
print("\n[4/10] Feature engineering...")

# --- Time features ---
combined['time_slot'] = combined['hour'] * 4 + combined['minute'] // 15  # 0-95
combined['day_of_week'] = combined['day'] % 7
combined['is_weekend']  = combined['day_of_week'].isin([5, 6]).astype(int)

# Peak hours (based on EDA: hours 9-13 have highest demand on day 48)
combined['is_peak']     = combined['hour'].isin([7,8,9,10,11,12,13]).astype(int)
combined['is_morning']  = combined['hour'].isin([6,7,8,9,10,11]).astype(int)
combined['is_afternoon']= combined['hour'].isin([12,13,14,15,16]).astype(int)
combined['is_night']    = combined['hour'].isin([22,23,0,1,2,3,4,5]).astype(int)
combined['is_early_morning'] = combined['hour'].isin([0,1,2,3,4,5]).astype(int)

def part_of_day(h):
    if 0  <= h < 6:  return 0
    if 6  <= h < 12: return 1
    if 12 <= h < 17: return 2
    if 17 <= h < 21: return 3
    return 4
combined['part_of_day'] = combined['hour'].apply(part_of_day)

# Minutes since midnight
combined['minutes_since_midnight'] = combined['hour'] * 60 + combined['minute']

# --- Cyclical encoding ---
combined['hour_sin'] = np.sin(2 * np.pi * combined['hour'] / 24)
combined['hour_cos'] = np.cos(2 * np.pi * combined['hour'] / 24)
combined['min_sin']  = np.sin(2 * np.pi * combined['minute'] / 60)
combined['min_cos']  = np.cos(2 * np.pi * combined['minute'] / 60)
combined['dow_sin']  = np.sin(2 * np.pi * combined['day_of_week'] / 7)
combined['dow_cos']  = np.cos(2 * np.pi * combined['day_of_week'] / 7)
combined['slot_sin'] = np.sin(2 * np.pi * combined['time_slot'] / 96)
combined['slot_cos'] = np.cos(2 * np.pi * combined['time_slot'] / 96)

# --- Geohash features ---
combined['geohash_len'] = combined['geohash'].str.len()
combined['geohash_4']   = combined['geohash'].str[:4]
combined['geohash_5']   = combined['geohash'].str[:5]
combined['geohash_3']   = combined['geohash'].str[:3]

# --- Binary encoding ---
combined['LargeVehicles_bin'] = (combined['LargeVehicles'] == 'Allowed').astype(int)
combined['Landmarks_bin']     = (combined['Landmarks'] == 'Yes').astype(int)

# --- Temperature features ---
combined['temp_sq'] = combined['Temperature'] ** 2
combined['temp_abs'] = combined['Temperature'].abs()
temp_bins = [-np.inf, 0, 10, 20, 30, 40, np.inf]
combined['temp_bin'] = pd.cut(combined['Temperature'], bins=temp_bins, labels=False)
combined['temp_bin'] = combined['temp_bin'].fillna(2)

# --- Interaction features ---
combined['peak_x_lanes']    = combined['is_peak'] * combined['NumberofLanes']
combined['morning_x_lanes'] = combined['is_morning'] * combined['NumberofLanes']
combined['night_x_lanes']   = combined['is_night'] * combined['NumberofLanes']
combined['weekend_x_peak']  = combined['is_weekend'] * combined['is_peak']
combined['weekend_x_lanes'] = combined['is_weekend'] * combined['NumberofLanes']
combined['large_veh_x_lanes'] = combined['LargeVehicles_bin'] * combined['NumberofLanes']
combined['landmarks_x_peak']  = combined['Landmarks_bin'] * combined['is_peak']
combined['landmarks_x_lanes'] = combined['Landmarks_bin'] * combined['NumberofLanes']
combined['large_veh_x_peak']  = combined['LargeVehicles_bin'] * combined['is_peak']

# --- Lat/Lon interactions ---
combined['lat_x_lon'] = combined['lat'] * combined['lon']
combined['lat_sq']    = combined['lat'] ** 2
combined['lon_sq']    = combined['lon'] ** 2

print(f"  Features after engineering: {combined.shape[1]}")

# =============================================================
# STEP 5 — TARGET ENCODING (with smoothing, train-only)
# =============================================================
print("\n[5/10] Target encoding with smoothing...")

train_rows = combined[combined['_source'] == 'train'].copy()
train_rows['demand_log'] = y.values

global_mean = y.mean()
global_std  = y.std()
SMOOTH_MIN = 20  # minimum samples for full trust

def smoothed_target_encode(df_combined, df_train, group_col, target_col='demand_log', 
                            prefix=None, smooth=SMOOTH_MIN):
    """Target encoding with Bayesian smoothing to prevent overfitting."""
    if prefix is None:
        prefix = group_col
    
    agg = df_train.groupby(group_col)[target_col].agg(['mean', 'std', 'median', 'count']).reset_index()
    agg.columns = [group_col, f'{prefix}_te_mean', f'{prefix}_te_std', 
                   f'{prefix}_te_median', f'{prefix}_te_count']
    
    # Bayesian smoothing: blend group mean with global mean
    agg[f'{prefix}_te_mean_smooth'] = (
        (agg[f'{prefix}_te_count'] * agg[f'{prefix}_te_mean'] + smooth * global_mean) /
        (agg[f'{prefix}_te_count'] + smooth)
    )
    
    df_combined = df_combined.merge(agg, on=group_col, how='left')
    
    # Fill missing (unseen groups)
    df_combined[f'{prefix}_te_mean']   = df_combined[f'{prefix}_te_mean'].fillna(global_mean)
    df_combined[f'{prefix}_te_std']    = df_combined[f'{prefix}_te_std'].fillna(global_std)
    df_combined[f'{prefix}_te_median'] = df_combined[f'{prefix}_te_median'].fillna(global_mean)
    df_combined[f'{prefix}_te_count']  = df_combined[f'{prefix}_te_count'].fillna(0)
    df_combined[f'{prefix}_te_mean_smooth'] = df_combined[f'{prefix}_te_mean_smooth'].fillna(global_mean)
    
    return df_combined

# Per-geohash
combined = smoothed_target_encode(combined, train_rows, 'geohash', prefix='geo')

# Per-geohash_3, geohash_4, geohash_5 (region levels)
combined = smoothed_target_encode(combined, train_rows, 'geohash_3', prefix='geo3')
combined = smoothed_target_encode(combined, train_rows, 'geohash_4', prefix='geo4')
combined = smoothed_target_encode(combined, train_rows, 'geohash_5', prefix='geo5')

# Per-hour
hour_agg = train_rows.groupby('hour')['demand_log'].agg(['mean', 'median', 'std']).reset_index()
hour_agg.columns = ['hour', 'hour_te_mean', 'hour_te_median', 'hour_te_std']
combined = combined.merge(hour_agg, on='hour', how='left')
combined['hour_te_mean']   = combined['hour_te_mean'].fillna(global_mean)
combined['hour_te_median'] = combined['hour_te_median'].fillna(global_mean)
combined['hour_te_std']    = combined['hour_te_std'].fillna(global_std)

# Per-time_slot (15-min granularity)
slot_agg = train_rows.groupby('time_slot')['demand_log'].agg(['mean', 'count']).reset_index()
slot_agg.columns = ['time_slot', 'slot_te_mean', 'slot_te_count']
slot_agg['slot_te_mean_smooth'] = (
    (slot_agg['slot_te_count'] * slot_agg['slot_te_mean'] + SMOOTH_MIN * global_mean) /
    (slot_agg['slot_te_count'] + SMOOTH_MIN)
)
combined = combined.merge(slot_agg, on='time_slot', how='left')
combined['slot_te_mean'] = combined['slot_te_mean'].fillna(global_mean)
combined['slot_te_mean_smooth'] = combined['slot_te_mean_smooth'].fillna(global_mean)
combined['slot_te_count'] = combined['slot_te_count'].fillna(0)

# Per-day
day_agg = train_rows.groupby('day')['demand_log'].agg(['mean', 'std']).reset_index()
day_agg.columns = ['day', 'day_te_mean', 'day_te_std']
combined = combined.merge(day_agg, on='day', how='left')
combined['day_te_mean'] = combined['day_te_mean'].fillna(global_mean)
combined['day_te_std']  = combined['day_te_std'].fillna(global_std)

# --- Interaction target encodings ---
# Geohash x Hour
geo_hour_agg = train_rows.groupby(['geohash', 'hour'])['demand_log'].agg(['mean', 'count']).reset_index()
geo_hour_agg.columns = ['geohash', 'hour', 'geo_hour_te_mean', 'geo_hour_te_count']
geo_hour_agg['geo_hour_te_smooth'] = (
    (geo_hour_agg['geo_hour_te_count'] * geo_hour_agg['geo_hour_te_mean'] + 5 * global_mean) /
    (geo_hour_agg['geo_hour_te_count'] + 5)
)
combined = combined.merge(geo_hour_agg, on=['geohash', 'hour'], how='left')
combined['geo_hour_te_mean'] = combined['geo_hour_te_mean'].fillna(combined['geo_te_mean_smooth'])
combined['geo_hour_te_smooth'] = combined['geo_hour_te_smooth'].fillna(combined['geo_te_mean_smooth'])
combined['geo_hour_te_count'] = combined['geo_hour_te_count'].fillna(0)

# Geohash x time_slot
geo_slot_agg = train_rows.groupby(['geohash', 'time_slot'])['demand_log'].agg(['mean', 'count']).reset_index()
geo_slot_agg.columns = ['geohash', 'time_slot', 'geo_slot_te_mean', 'geo_slot_te_count']
geo_slot_agg['geo_slot_te_smooth'] = (
    (geo_slot_agg['geo_slot_te_count'] * geo_slot_agg['geo_slot_te_mean'] + 3 * global_mean) /
    (geo_slot_agg['geo_slot_te_count'] + 3)
)
combined = combined.merge(geo_slot_agg, on=['geohash', 'time_slot'], how='left')
combined['geo_slot_te_mean'] = combined['geo_slot_te_mean'].fillna(combined['geo_te_mean_smooth'])
combined['geo_slot_te_smooth'] = combined['geo_slot_te_smooth'].fillna(combined['geo_te_mean_smooth'])
combined['geo_slot_te_count'] = combined['geo_slot_te_count'].fillna(0)

# RoadType x Hour
road_hour_agg = train_rows.groupby(['RoadType', 'hour'])['demand_log'].agg(['mean']).reset_index()
road_hour_agg.columns = ['RoadType', 'hour', 'road_hour_te_mean']
combined = combined.merge(road_hour_agg, on=['RoadType', 'hour'], how='left')
combined['road_hour_te_mean'] = combined['road_hour_te_mean'].fillna(global_mean)

# Weather x Hour
weather_hour_agg = train_rows.groupby(['Weather', 'hour'])['demand_log'].agg(['mean']).reset_index()
weather_hour_agg.columns = ['Weather', 'hour', 'weather_hour_te_mean']
combined = combined.merge(weather_hour_agg, on=['Weather', 'hour'], how='left')
combined['weather_hour_te_mean'] = combined['weather_hour_te_mean'].fillna(global_mean)

# Geohash x RoadType
geo_road_agg = train_rows.groupby(['geohash', 'RoadType'])['demand_log'].agg(['mean', 'count']).reset_index()
geo_road_agg.columns = ['geohash', 'RoadType', 'geo_road_te_mean', 'geo_road_te_count']
combined = combined.merge(geo_road_agg, on=['geohash', 'RoadType'], how='left')
combined['geo_road_te_mean'] = combined['geo_road_te_mean'].fillna(combined['geo_te_mean_smooth'])
combined['geo_road_te_count'] = combined['geo_road_te_count'].fillna(0)

# --- Deviation features (how much does this combo deviate from baseline) ---
combined['geo_hour_dev']  = combined['geo_hour_te_mean'] - combined['geo_te_mean_smooth']
combined['geo_slot_dev']  = combined['geo_slot_te_mean'] - combined['geo_te_mean_smooth']
combined['hour_dev']      = combined['hour_te_mean'] - global_mean
combined['geo_road_dev']  = combined['geo_road_te_mean'] - combined['geo_te_mean_smooth']

print(f"  Features after target encoding: {combined.shape[1]}")

# =============================================================
# STEP 6 — ENCODE CATEGORICALS
# =============================================================
print("\n[6/10] Encoding categoricals...")

cat_cols = ['RoadType', 'Weather', 'LargeVehicles', 'Landmarks']
cat_cols = [c for c in cat_cols if c in combined.columns]

encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    le.fit(combined.loc[train_mask, col].astype(str))
    vals = combined[col].astype(str).copy()
    mask_unseen = ~vals.isin(le.classes_)
    if mask_unseen.any():
        vals[mask_unseen] = le.classes_[0]
    combined[col] = le.transform(vals)
    encoders[col] = le
    print(f"  Encoded: {col}")

# =============================================================
# STEP 7 — FINAL FEATURE SELECTION
# =============================================================
print("\n[7/10] Final feature selection...")

drop_cols = ['_source', 'geohash', 'geohash_3', 'geohash_4', 'geohash_5', 'timestamp', 'Index']
drop_cols = [c for c in drop_cols if c in combined.columns]
combined.drop(columns=drop_cols, inplace=True)

# Convert all to numeric
for col in combined.columns:
    combined[col] = pd.to_numeric(combined[col], errors='coerce')
combined = combined.fillna(0)

X      = combined.iloc[:n_train].copy()
X_test = combined.iloc[n_train:].copy()

print(f"  X: {X.shape}, X_test: {X_test.shape}")
print(f"  Features: {X.columns.tolist()}")

assert not X.isnull().any().any(), "NaN in X!"
assert not X_test.isnull().any().any(), "NaN in X_test!"

# Quick feature importance
print("\n  Running quick feature importance check...")
quick_lgb = lgb.LGBMRegressor(n_estimators=300, random_state=SEED, verbose=-1)
quick_lgb.fit(X, y)
imp_df = pd.DataFrame({
    'feature': X.columns,
    'importance': quick_lgb.feature_importances_
}).sort_values('importance', ascending=False)

print("  Top 20 features:")
print(imp_df.head(20).to_string(index=False))

# Drop zero-importance features
zero_imp = imp_df[imp_df['importance'] == 0]['feature'].tolist()
if zero_imp:
    print(f"\n  Dropping {len(zero_imp)} zero-importance features: {zero_imp}")
    X = X.drop(columns=zero_imp)
    X_test = X_test.drop(columns=zero_imp)
    print(f"  X after drop: {X.shape}")

# =============================================================
# STEP 8 — OPTUNA HYPERPARAMETER TUNING
# =============================================================
print("\n[8/10] Optuna hyperparameter tuning...")

# Quick 3-fold tuning
kf_tune = KFold(n_splits=3, shuffle=True, random_state=SEED)

def objective_lgbm(trial):
    params = {
        'n_estimators': 1500,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 4, 9),
        'num_leaves': trial.suggest_int('num_leaves', 20, 127),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'subsample': trial.suggest_float('subsample', 0.6, 0.95),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.95),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10, log=True),
        'random_state': SEED,
        'verbose': -1,
    }
    scores = []
    for tr_idx, val_idx in kf_tune.split(X, y):
        m = lgb.LGBMRegressor(**params)
        m.fit(X.iloc[tr_idx], y.iloc[tr_idx],
              eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
              callbacks=[lgb.early_stopping(30, verbose=False)])
        pred = m.predict(X.iloc[val_idx])
        # Evaluate in original scale
        pred_orig = np.expm1(pred)
        actual_orig = np.expm1(y.iloc[val_idx])
        scores.append(r2_score(actual_orig, pred_orig))
    return np.mean(scores)

def objective_xgb(trial):
    params = {
        'n_estimators': 1500,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 4, 9),
        'subsample': trial.suggest_float('subsample', 0.6, 0.95),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.95),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'random_state': SEED,
        'verbosity': 0,
        'early_stopping_rounds': 30,
    }
    scores = []
    for tr_idx, val_idx in kf_tune.split(X, y):
        m = xgb.XGBRegressor(**params)
        m.fit(X.iloc[tr_idx], y.iloc[tr_idx],
              eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
              verbose=False)
        pred = m.predict(X.iloc[val_idx])
        pred_orig = np.expm1(pred)
        actual_orig = np.expm1(y.iloc[val_idx])
        scores.append(r2_score(actual_orig, pred_orig))
    return np.mean(scores)

def objective_cat(trial):
    params = {
        'iterations': 1500,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'depth': trial.suggest_int('depth', 4, 9),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.1, 10, log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 5),
        'random_strength': trial.suggest_float('random_strength', 0.1, 10, log=True),
        'early_stopping_rounds': 30,
        'random_seed': SEED,
        'verbose': False,
    }
    scores = []
    for tr_idx, val_idx in kf_tune.split(X, y):
        m = CatBoostRegressor(**params)
        m.fit(X.iloc[tr_idx], y.iloc[tr_idx],
              eval_set=(X.iloc[val_idx], y.iloc[val_idx]),
              use_best_model=True)
        pred = m.predict(X.iloc[val_idx])
        pred_orig = np.expm1(pred)
        actual_orig = np.expm1(y.iloc[val_idx])
        scores.append(r2_score(actual_orig, pred_orig))
    return np.mean(scores)

N_OPTUNA_TRIALS = 15

print("  Tuning LightGBM...")
study_lgbm = optuna.create_study(direction='maximize')
study_lgbm.optimize(objective_lgbm, n_trials=N_OPTUNA_TRIALS, show_progress_bar=False)
best_lgbm_params = study_lgbm.best_params
print(f"    Best R2: {study_lgbm.best_value:.4f}")
print(f"    Best params: {best_lgbm_params}")

print("  Tuning XGBoost...")
study_xgb = optuna.create_study(direction='maximize')
study_xgb.optimize(objective_xgb, n_trials=N_OPTUNA_TRIALS, show_progress_bar=False)
best_xgb_params = study_xgb.best_params
print(f"    Best R2: {study_xgb.best_value:.4f}")
print(f"    Best params: {best_xgb_params}")

print("  Tuning CatBoost...")
study_cat = optuna.create_study(direction='maximize')
study_cat.optimize(objective_cat, n_trials=N_OPTUNA_TRIALS, show_progress_bar=False)
best_cat_params = study_cat.best_params
print(f"    Best R2: {study_cat.best_value:.4f}")
print(f"    Best params: {best_cat_params}")

# =============================================================
# STEP 9 — FULL 5-FOLD CV TRAINING WITH TUNED PARAMS + STACKING
# =============================================================
print("\n[9/10] Full 5-fold CV training with tuned hyperparameters...")

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

oof_lgbm = np.zeros(len(X))
oof_xgb  = np.zeros(len(X))
oof_cat  = np.zeros(len(X))

test_lgbm = np.zeros(len(X_test))
test_xgb  = np.zeros(len(X_test))
test_cat  = np.zeros(len(X_test))

lgbm_scores, xgb_scores, cat_scores = [], [], []

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"\n  === FOLD {fold+1}/{N_FOLDS} ===")
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # -- LightGBM --
    lgbm_p = {
        'n_estimators': 3000,
        'random_state': SEED,
        'verbose': -1,
        **best_lgbm_params
    }
    m_lgbm = lgb.LGBMRegressor(**lgbm_p)
    m_lgbm.fit(X_tr, y_tr,
               eval_set=[(X_val, y_val)],
               callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
    oof_lgbm[val_idx] = m_lgbm.predict(X_val)
    test_lgbm += m_lgbm.predict(X_test) / N_FOLDS
    r2_lgbm = r2_score(np.expm1(y_val), np.expm1(oof_lgbm[val_idx]))
    lgbm_scores.append(r2_lgbm)
    print(f"    LGBM R2: {r2_lgbm:.4f} (best_iter={m_lgbm.best_iteration_})")
    
    # -- XGBoost --
    xgb_p = {
        'n_estimators': 3000,
        'random_state': SEED,
        'verbosity': 0,
        'early_stopping_rounds': 100,
        **best_xgb_params
    }
    m_xgb = xgb.XGBRegressor(**xgb_p)
    m_xgb.fit(X_tr, y_tr,
              eval_set=[(X_val, y_val)],
              verbose=False)
    oof_xgb[val_idx] = m_xgb.predict(X_val)
    test_xgb += m_xgb.predict(X_test) / N_FOLDS
    r2_xgb = r2_score(np.expm1(y_val), np.expm1(oof_xgb[val_idx]))
    xgb_scores.append(r2_xgb)
    print(f"    XGB  R2: {r2_xgb:.4f} (best_iter={m_xgb.best_iteration})")
    
    # -- CatBoost --
    cat_p = {
        'iterations': 3000,
        'random_seed': SEED,
        'verbose': False,
        'early_stopping_rounds': 100,
        **best_cat_params
    }
    m_cat = CatBoostRegressor(**cat_p)
    m_cat.fit(X_tr, y_tr,
              eval_set=(X_val, y_val),
              use_best_model=True)
    oof_cat[val_idx] = m_cat.predict(X_val)
    test_cat += m_cat.predict(X_test) / N_FOLDS
    r2_cat = r2_score(np.expm1(y_val), np.expm1(oof_cat[val_idx]))
    cat_scores.append(r2_cat)
    print(f"    CAT  R2: {r2_cat:.4f} (best_iter={m_cat.best_iteration_})")

print(f"\n{'='*60}")
print(f"CV SUMMARY (in original demand scale)")
print(f"{'='*60}")
print(f"  LGBM  R2: {np.mean(lgbm_scores):.4f} +/- {np.std(lgbm_scores):.4f}")
print(f"  XGB   R2: {np.mean(xgb_scores):.4f} +/- {np.std(xgb_scores):.4f}")
print(f"  CAT   R2: {np.mean(cat_scores):.4f} +/- {np.std(cat_scores):.4f}")

# =============================================================
# STEP 10 — ENSEMBLE: Weighted + Stacking
# =============================================================
print("\n[10/10] Ensemble...")

# --- A: Weighted average ---
w_l = np.mean(lgbm_scores)
w_x = np.mean(xgb_scores)
w_c = np.mean(cat_scores)
total_w = w_l + w_x + w_c
w_l /= total_w; w_x /= total_w; w_c /= total_w

oof_weighted = oof_lgbm * w_l + oof_xgb * w_x + oof_cat * w_c
r2_weighted = r2_score(np.expm1(y), np.expm1(oof_weighted))
print(f"  Weighted Average OOF R2: {r2_weighted:.4f}  (weights: L={w_l:.3f} X={w_x:.3f} C={w_c:.3f})")

# --- B: Simple average ---
oof_simple = (oof_lgbm + oof_xgb + oof_cat) / 3
r2_simple = r2_score(np.expm1(y), np.expm1(oof_simple))
print(f"  Simple Average OOF R2 : {r2_simple:.4f}")

# --- C: Stacking with Ridge ---
print("  Training stacking meta-learner (Ridge)...")
stack_X = np.column_stack([oof_lgbm, oof_xgb, oof_cat])
stack_test_X = np.column_stack([test_lgbm, test_xgb, test_cat])

best_stack_r2 = -999
best_alpha = 1.0
for alpha in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
    stack_oof = np.zeros(len(y))
    for tr_idx, val_idx in kf.split(stack_X, y):
        ridge = Ridge(alpha=alpha)
        ridge.fit(stack_X[tr_idx], y.iloc[tr_idx])
        stack_oof[val_idx] = ridge.predict(stack_X[val_idx])
    r2_stack = r2_score(np.expm1(y), np.expm1(stack_oof))
    if r2_stack > best_stack_r2:
        best_stack_r2 = r2_stack
        best_alpha = alpha

print(f"  Stacking OOF R2: {best_stack_r2:.4f} (alpha={best_alpha})")

# Final stacking model
ridge_final = Ridge(alpha=best_alpha)
ridge_final.fit(stack_X, y)
print(f"  Ridge coefficients: {ridge_final.coef_}")
test_stacked = ridge_final.predict(stack_test_X)

# --- D: Optuna ensemble weight search ---
print("  Optuna ensemble weight optimization...")
def objective_ensemble(trial):
    w1 = trial.suggest_float('w_lgbm', 0, 1)
    w2 = trial.suggest_float('w_xgb', 0, 1)
    w3 = trial.suggest_float('w_cat', 0, 1)
    total = w1 + w2 + w3
    if total == 0:
        return -999
    w1 /= total; w2 /= total; w3 /= total
    blend = oof_lgbm * w1 + oof_xgb * w2 + oof_cat * w3
    return r2_score(np.expm1(y), np.expm1(blend))

study_ens = optuna.create_study(direction='maximize')
study_ens.optimize(objective_ensemble, n_trials=200, show_progress_bar=False)
best_ew = study_ens.best_params
ew_total = best_ew['w_lgbm'] + best_ew['w_xgb'] + best_ew['w_cat']
ew_l = best_ew['w_lgbm'] / ew_total
ew_x = best_ew['w_xgb'] / ew_total
ew_c = best_ew['w_cat'] / ew_total

oof_optuna_ens = oof_lgbm * ew_l + oof_xgb * ew_x + oof_cat * ew_c
r2_optuna = r2_score(np.expm1(y), np.expm1(oof_optuna_ens))
print(f"  Optuna Ensemble OOF R2: {r2_optuna:.4f}  (w: L={ew_l:.3f} X={ew_x:.3f} C={ew_c:.3f})")

# --- Choose best ensemble ---
results = {
    'weighted': (r2_weighted, test_lgbm * w_l + test_xgb * w_x + test_cat * w_c),
    'simple': (r2_simple, (test_lgbm + test_xgb + test_cat) / 3),
    'stacking': (best_stack_r2, test_stacked),
    'optuna': (r2_optuna, test_lgbm * ew_l + test_xgb * ew_x + test_cat * ew_c),
}

best_method = max(results, key=lambda k: results[k][0])
best_r2 = results[best_method][0]
final_preds_log = results[best_method][1]

print(f"\n  *** Best ensemble: {best_method} with OOF R2 = {best_r2:.4f} ***")
print(f"  *** Hackathon Score estimate: {max(0, 100 * best_r2):.2f} ***")

# Transform back to original scale
final_preds = np.expm1(final_preds_log)
final_preds = np.clip(final_preds, 0, None)  # Clip negatives

# =============================================================
# GENERATE SUBMISSION
# =============================================================
print(f"\n{'='*60}")
print("GENERATING SUBMISSION")
print(f"{'='*60}")

test_original = pd.read_csv('./work/test.csv')
submission = pd.DataFrame({
    'Index': test_original['Index'],
    'demand': final_preds
})

print(f"  Shape: {submission.shape}")
print(f"  Demand stats:\n{submission['demand'].describe()}")

assert submission.shape[0] == len(test_original)
assert submission.shape[1] == 2
assert list(submission.columns) == ['Index', 'demand']
assert submission.isnull().sum().sum() == 0

submission.to_csv('./work/submission_v2.csv', index=False)
print(f"\n  [OK] submission_v2.csv saved!")

# Also save as submission.csv for easy upload
submission.to_csv('./work/submission.csv', index=False)
print(f"  [OK] submission.csv overwritten with v2 predictions!")

elapsed = time.time() - start_time
print(f"\n{'='*60}")
print(f"DONE! Total time: {elapsed/60:.1f} minutes")
print(f"Estimated hackathon score: {max(0, 100*best_r2):.2f}")
print(f"{'='*60}")
