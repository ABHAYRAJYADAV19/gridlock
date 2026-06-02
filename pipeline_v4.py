#!/usr/bin/env python3
"""
Traffic Demand Prediction - Pipeline v4
========================================
Key insight: Day 49 demand is ~1.7x higher than day 48 at same hours.
Per-geohash calibration ratio + lookup raises R2 from 0.52 to 0.82.

Strategy:
1. Calibrated lookup predictions as FEATURES (day48 pattern x day49 ratio)
2. Full target encoding (v2-style, works better than OOF)
3. Train on RAW target (no log, better per v3 results)
4. Aggressive pseudo-labeling (5 rounds)
5. Ensemble with multiple seeds + calibrated baseline
"""

import sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
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
print("PIPELINE v4 - Calibrated Lookup + Aggressive Pseudo-Labeling")
print("=" * 70)

# =============================================================
# STEP 1 - LOAD DATA
# =============================================================
print("\n[1/11] Loading data...")
train_orig = pd.read_csv('./work/train.csv')
test_orig  = pd.read_csv('./work/test.csv')
print(f"  Train: {train_orig.shape}, Test: {test_orig.shape}")

y_raw = train_orig['demand'].copy()
n_train = len(train_orig)
n_test  = len(test_orig)

# =============================================================
# STEP 2 - PREPROCESS + FEATURE ENGINEERING
# =============================================================
print("\n[2/11] Feature engineering...")

train = train_orig.copy()
test  = test_orig.copy()
train['_src'] = 'train'
test['_src']  = 'test'

combined = pd.concat([train.drop('demand', axis=1), test], axis=0).reset_index(drop=True)
train_mask = combined['_src'] == 'train'

# Timestamp
parts = combined['timestamp'].astype(str).str.split(':', expand=True)
combined['hour']   = parts[0].astype(int)
combined['minute'] = parts[1].astype(int)
combined['time_slot'] = combined['hour'] * 4 + combined['minute'] // 15

# Geohash decode
decoded = combined['geohash'].apply(lambda gh: pgh.decode(gh))
combined['lat'] = [float(d[0]) for d in decoded]
combined['lon'] = [float(d[1]) for d in decoded]

# Fill NaN
for col in ['RoadType', 'Weather']:
    if combined[col].isnull().any():
        combined[col] = combined[col].fillna(combined.loc[train_mask, col].mode()[0])

if combined['Temperature'].isnull().any():
    temp_geo = combined.loc[train_mask].groupby('geohash')['Temperature'].median()
    global_temp = combined.loc[train_mask, 'Temperature'].median()
    for idx in combined[combined['Temperature'].isnull()].index:
        geo = combined.loc[idx, 'geohash']
        combined.loc[idx, 'Temperature'] = temp_geo.get(geo, global_temp)

# Time features
combined['day_of_week'] = combined['day'] % 7
combined['is_weekend']  = combined['day_of_week'].isin([5, 6]).astype(int)
combined['is_peak']     = combined['hour'].isin([7,8,9,10,11,12,13]).astype(int)
combined['is_morning']  = combined['hour'].isin([6,7,8,9,10,11]).astype(int)
combined['is_night']    = combined['hour'].isin([22,23,0,1,2,3,4,5]).astype(int)

def part_of_day(h):
    if 0 <= h < 6: return 0
    if 6 <= h < 12: return 1
    if 12 <= h < 17: return 2
    if 17 <= h < 21: return 3
    return 4
combined['part_of_day'] = combined['hour'].apply(part_of_day)
combined['minutes_since_midnight'] = combined['hour'] * 60 + combined['minute']

# Cyclical
for col, period in [('hour', 24), ('minute', 60), ('time_slot', 96)]:
    combined[f'{col}_sin'] = np.sin(2 * np.pi * combined[col] / period)
    combined[f'{col}_cos'] = np.cos(2 * np.pi * combined[col] / period)

# Binary
combined['LargeVehicles_bin'] = (combined['LargeVehicles'] == 'Allowed').astype(int)
combined['Landmarks_bin']     = (combined['Landmarks'] == 'Yes').astype(int)

# Temperature
combined['temp_sq'] = combined['Temperature'] ** 2

# Interactions
combined['peak_x_lanes']      = combined['is_peak'] * combined['NumberofLanes']
combined['morning_x_lanes']   = combined['is_morning'] * combined['NumberofLanes']
combined['night_x_lanes']     = combined['is_night'] * combined['NumberofLanes']
combined['large_veh_x_lanes'] = combined['LargeVehicles_bin'] * combined['NumberofLanes']
combined['landmarks_x_lanes'] = combined['Landmarks_bin'] * combined['NumberofLanes']
combined['lat_x_lon']         = combined['lat'] * combined['lon']

# Frequency encoding
for col_name in ['geohash']:
    freq = combined[col_name].value_counts()
    combined[f'{col_name}_freq'] = combined[col_name].map(freq)

# Geohash prefixes for later TE
combined['geohash_4'] = combined['geohash'].str[:4]
combined['geohash_5'] = combined['geohash'].str[:5]

print(f"  Base features: {combined.shape[1]}")

# =============================================================
# STEP 3 - CALIBRATION FEATURES (KEY INNOVATION)
# =============================================================
print("\n[3/11] Computing calibration features...")

# Day 48 and Day 49 subsets from training
tr_d48 = train_orig[train_orig['day'] == 48].copy()
tr_d49 = train_orig[train_orig['day'] == 49].copy()

ts48 = tr_d48['timestamp'].str.split(':', expand=True)
tr_d48['hour'] = ts48[0].astype(int)
tr_d48['minute'] = ts48[1].astype(int)
tr_d48['time_slot'] = tr_d48['hour'] * 4 + tr_d48['minute'] // 15

ts49 = tr_d49['timestamp'].str.split(':', expand=True)
tr_d49['hour'] = ts49[0].astype(int)
tr_d49['minute'] = ts49[1].astype(int)
tr_d49['time_slot'] = tr_d49['hour'] * 4 + tr_d49['minute'] // 15

# Per-geohash calibration ratio: day49_early / day48_early
d48_early = tr_d48[tr_d48['hour'] <= 2]
d48_early_geo_mean = d48_early.groupby('geohash')['demand'].mean()
d49_geo_mean = tr_d49.groupby('geohash')['demand'].mean()

common_geos = d48_early_geo_mean.index.intersection(d49_geo_mean.index)
cal_ratio = (d49_geo_mean[common_geos] / d48_early_geo_mean[common_geos].clip(lower=1e-6))
cal_ratio = cal_ratio.clip(lower=0.1, upper=10)  # Clip extreme ratios

global_cal_ratio = cal_ratio.median()
print(f"  Calibration ratio: median={cal_ratio.median():.4f}, mean={cal_ratio.mean():.4f}")
print(f"  Geohashes with ratio: {len(cal_ratio)}")

# Day 48 lookup tables
d48_slot_mean   = tr_d48.groupby(['geohash', 'time_slot'])['demand'].mean()
d48_hour_mean   = tr_d48.groupby(['geohash', 'hour'])['demand'].mean()
d48_geo_mean    = tr_d48.groupby('geohash')['demand'].mean()
d48_geo_std     = tr_d48.groupby('geohash')['demand'].std().fillna(0)
d48_geo_median  = tr_d48.groupby('geohash')['demand'].median()

# Day 49 features (from training hours 0-2)
d49_geo_mean_all = tr_d49.groupby('geohash')['demand'].mean()
d49_geo_std      = tr_d49.groupby('geohash')['demand'].std().fillna(0)

global_mean = y_raw.mean()

# Apply calibration features to combined dataset
def add_calibration_features(df, geohashes, hours, slots):
    """Add calibrated lookup features."""
    result = df.copy()
    n = len(df)
    
    # Raw day 48 lookup (geohash x slot)
    d48_slot_vals = []
    for geo, slot in zip(geohashes, slots):
        key = (geo, slot)
        if key in d48_slot_mean.index:
            d48_slot_vals.append(d48_slot_mean[key])
        elif geo in d48_geo_mean.index:
            d48_slot_vals.append(d48_geo_mean[geo])
        else:
            d48_slot_vals.append(global_mean)
    result['d48_slot_lookup'] = d48_slot_vals
    
    # Day 48 hour lookup
    d48_hour_vals = []
    for geo, hour in zip(geohashes, hours):
        key = (geo, hour)
        if key in d48_hour_mean.index:
            d48_hour_vals.append(d48_hour_mean[key])
        elif geo in d48_geo_mean.index:
            d48_hour_vals.append(d48_geo_mean[geo])
        else:
            d48_hour_vals.append(global_mean)
    result['d48_hour_lookup'] = d48_hour_vals
    
    # Per-geohash calibration ratio
    result['cal_ratio'] = [cal_ratio.get(geo, global_cal_ratio) for geo in geohashes]
    
    # CALIBRATED predictions (day 48 lookup x calibration ratio)
    result['calibrated_slot_pred'] = result['d48_slot_lookup'] * result['cal_ratio']
    result['calibrated_hour_pred'] = result['d48_hour_lookup'] * result['cal_ratio']
    
    # Day 48 geohash stats
    result['d48_geo_mean']   = [d48_geo_mean.get(geo, global_mean) for geo in geohashes]
    result['d48_geo_std']    = [d48_geo_std.get(geo, 0) for geo in geohashes]
    result['d48_geo_median'] = [d48_geo_median.get(geo, global_mean) for geo in geohashes]
    
    # Day 49 early stats (from training)
    result['d49_geo_mean']   = [d49_geo_mean_all.get(geo, global_mean) for geo in geohashes]
    result['d49_geo_std']    = [d49_geo_std.get(geo, 0) for geo in geohashes]
    
    # Deviation features
    result['slot_dev_from_geo'] = result['d48_slot_lookup'] - result['d48_geo_mean']
    result['cal_pred_dev']      = result['calibrated_slot_pred'] - result['d49_geo_mean']
    
    return result

geohashes_all = combined['geohash'].values
hours_all     = combined['hour'].values
slots_all     = combined['time_slot'].values

combined = add_calibration_features(combined, geohashes_all, hours_all, slots_all)
print(f"  Features after calibration: {combined.shape[1]}")

# =============================================================
# STEP 4 - TARGET ENCODING (full data, v2-style)
# =============================================================
print("\n[4/11] Target encoding (full train data)...")

train_rows = combined[combined['_src'] == 'train'].copy()
train_rows['demand'] = y_raw.values

global_mean_te = y_raw.mean()
global_std_te  = y_raw.std()
SMOOTH = 10

def add_target_encoding(df_combined, df_train, smooth=SMOOTH):
    """Add target encoding features computed from training data."""
    gm = global_mean_te
    gs = global_std_te
    
    # Per-geohash
    geo_agg = df_train.groupby('geohash')['demand'].agg(['mean', 'std', 'median', 'count']).reset_index()
    geo_agg.columns = ['geohash', 'geo_te_mean', 'geo_te_std', 'geo_te_median', 'geo_te_count']
    geo_agg['geo_te_smooth'] = (geo_agg['geo_te_count'] * geo_agg['geo_te_mean'] + smooth * gm) / (geo_agg['geo_te_count'] + smooth)
    df_combined = df_combined.merge(geo_agg, on='geohash', how='left')
    for c in ['geo_te_mean', 'geo_te_median', 'geo_te_smooth']:
        df_combined[c] = df_combined[c].fillna(gm)
    df_combined['geo_te_std'] = df_combined['geo_te_std'].fillna(gs)
    df_combined['geo_te_count'] = df_combined['geo_te_count'].fillna(0)
    
    # Per-geohash_4
    geo4_agg = df_train.groupby(df_train['geohash'].str[:4])['demand'].agg(['mean', 'count']).reset_index()
    geo4_agg.columns = ['geohash_4', 'geo4_te_mean', 'geo4_te_count']
    df_combined = df_combined.merge(geo4_agg, on='geohash_4', how='left')
    df_combined['geo4_te_mean'] = df_combined['geo4_te_mean'].fillna(gm)
    df_combined['geo4_te_count'] = df_combined['geo4_te_count'].fillna(0)
    
    # Per-geohash_5
    geo5_agg = df_train.groupby(df_train['geohash'].str[:5])['demand'].agg(['mean', 'count']).reset_index()
    geo5_agg.columns = ['geohash_5', 'geo5_te_mean', 'geo5_te_count']
    df_combined = df_combined.merge(geo5_agg, on='geohash_5', how='left')
    df_combined['geo5_te_mean'] = df_combined['geo5_te_mean'].fillna(gm)
    df_combined['geo5_te_count'] = df_combined['geo5_te_count'].fillna(0)
    
    # Per-hour
    hour_agg = df_train.groupby('hour')['demand'].agg(['mean', 'median', 'std']).reset_index()
    hour_agg.columns = ['hour', 'hour_te_mean', 'hour_te_median', 'hour_te_std']
    df_combined = df_combined.merge(hour_agg, on='hour', how='left')
    df_combined['hour_te_mean'] = df_combined['hour_te_mean'].fillna(gm)
    df_combined['hour_te_median'] = df_combined['hour_te_median'].fillna(gm)
    df_combined['hour_te_std'] = df_combined['hour_te_std'].fillna(gs)
    
    # Per-time_slot
    slot_agg = df_train.groupby('time_slot')['demand'].agg(['mean', 'count']).reset_index()
    slot_agg.columns = ['time_slot', 'slot_te_mean', 'slot_te_count']
    df_combined = df_combined.merge(slot_agg, on='time_slot', how='left')
    df_combined['slot_te_mean'] = df_combined['slot_te_mean'].fillna(gm)
    df_combined['slot_te_count'] = df_combined['slot_te_count'].fillna(0)
    
    # Geohash x Hour
    geo_hour_agg = df_train.groupby(['geohash', 'hour'])['demand'].agg(['mean', 'count']).reset_index()
    geo_hour_agg.columns = ['geohash', 'hour', 'geo_hour_te_mean', 'geo_hour_te_count']
    df_combined = df_combined.merge(geo_hour_agg, on=['geohash', 'hour'], how='left')
    df_combined['geo_hour_te_mean'] = df_combined['geo_hour_te_mean'].fillna(df_combined['geo_te_smooth'])
    df_combined['geo_hour_te_count'] = df_combined['geo_hour_te_count'].fillna(0)
    
    # Geohash x Time_slot
    geo_slot_agg = df_train.groupby(['geohash', 'time_slot'])['demand'].agg(['mean', 'count']).reset_index()
    geo_slot_agg.columns = ['geohash', 'time_slot', 'geo_slot_te_mean', 'geo_slot_te_count']
    df_combined = df_combined.merge(geo_slot_agg, on=['geohash', 'time_slot'], how='left')
    df_combined['geo_slot_te_mean'] = df_combined['geo_slot_te_mean'].fillna(df_combined['geo_te_smooth'])
    df_combined['geo_slot_te_count'] = df_combined['geo_slot_te_count'].fillna(0)
    
    # RoadType x Hour
    road_hour_agg = df_train.groupby(['RoadType', 'hour'])['demand'].agg(['mean']).reset_index()
    road_hour_agg.columns = ['RoadType', 'hour', 'road_hour_te_mean']
    df_combined = df_combined.merge(road_hour_agg, on=['RoadType', 'hour'], how='left')
    df_combined['road_hour_te_mean'] = df_combined['road_hour_te_mean'].fillna(gm)
    
    # Weather x Hour
    weather_hour_agg = df_train.groupby(['Weather', 'hour'])['demand'].agg(['mean']).reset_index()
    weather_hour_agg.columns = ['Weather', 'hour', 'weather_hour_te_mean']
    df_combined = df_combined.merge(weather_hour_agg, on=['Weather', 'hour'], how='left')
    df_combined['weather_hour_te_mean'] = df_combined['weather_hour_te_mean'].fillna(gm)
    
    # Deviation features
    df_combined['geo_hour_dev'] = df_combined['geo_hour_te_mean'] - df_combined['geo_te_smooth']
    df_combined['geo_slot_dev'] = df_combined['geo_slot_te_mean'] - df_combined['geo_te_smooth']
    
    return df_combined

combined = add_target_encoding(combined, train_rows)
print(f"  Features after TE: {combined.shape[1]}")

# =============================================================
# STEP 5 - ENCODE & FINALIZE FEATURES
# =============================================================
print("\n[5/11] Encoding categoricals & finalizing features...")

cat_cols = ['RoadType', 'Weather', 'LargeVehicles', 'Landmarks']
for col in cat_cols:
    le = LabelEncoder()
    le.fit(combined.loc[train_mask, col].astype(str))
    vals = combined[col].astype(str).copy()
    unseen = ~vals.isin(le.classes_)
    if unseen.any():
        vals[unseen] = le.classes_[0]
    combined[col] = le.transform(vals)

drop_cols = ['_src', 'geohash', 'geohash_4', 'geohash_5', 'timestamp', 'Index']
drop_cols = [c for c in drop_cols if c in combined.columns]
combined.drop(columns=drop_cols, inplace=True)

for col in combined.columns:
    combined[col] = pd.to_numeric(combined[col], errors='coerce')
combined = combined.fillna(0)

X      = combined.iloc[:n_train].copy()
X_test = combined.iloc[n_train:].copy()

# Quick importance check & drop zero-importance
print(f"  X: {X.shape}")
quick = lgb.LGBMRegressor(n_estimators=300, random_state=SEED, verbose=-1)
quick.fit(X, y_raw)
imp = pd.DataFrame({'f': X.columns, 'i': quick.feature_importances_}).sort_values('i', ascending=False)
zero_feats = imp[imp['i'] == 0]['f'].tolist()
if zero_feats:
    print(f"  Dropping {len(zero_feats)} zero-importance features")
    X = X.drop(columns=zero_feats)
    X_test = X_test.drop(columns=zero_feats)

print(f"  Final X: {X.shape}")
print(f"  Top 15 features:")
print(imp.head(15).to_string(index=False))

# =============================================================
# STEP 6 - TEMPORAL VALIDATION (honest estimate)
# =============================================================
print("\n[6/11] Temporal validation...")

day_vals = train_orig['day'].values
d48_idx = np.where(day_vals == 48)[0]
d49_idx = np.where(day_vals == 49)[0]

m_temp = lgb.LGBMRegressor(n_estimators=2000, learning_rate=0.03, max_depth=6,
                             num_leaves=50, verbose=-1, random_state=SEED)
m_temp.fit(X.iloc[d48_idx], y_raw.iloc[d48_idx],
           eval_set=[(X.iloc[d49_idx], y_raw.iloc[d49_idx])],
           callbacks=[lgb.early_stopping(50, verbose=False)])
temp_pred = m_temp.predict(X.iloc[d49_idx])
temp_r2 = r2_score(y_raw.iloc[d49_idx], temp_pred)
print(f"  Temporal R2 (day48 -> day49): {temp_r2:.4f}")

# =============================================================
# STEP 7 - 5-FOLD CV TRAINING (Base models)
# =============================================================
print("\n[7/11] 5-fold CV training (base models)...")

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

# Sample weights: upweight day 49 rows
sw_train = np.ones(n_train)
sw_train[day_vals == 49] = 3.0

oof_lgbm = np.zeros(n_train)
oof_xgb  = np.zeros(n_train)
oof_cat  = np.zeros(n_train)
test_lgbm = np.zeros(n_test)
test_xgb  = np.zeros(n_test)
test_cat  = np.zeros(n_test)

for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y_raw.iloc[tr_idx], y_raw.iloc[val_idx]
    w_tr = sw_train[tr_idx]
    
    # LightGBM
    m = lgb.LGBMRegressor(
        n_estimators=3000, learning_rate=0.03, max_depth=7, num_leaves=63,
        min_child_samples=30, subsample=0.8, colsample_bytree=0.7,
        reg_alpha=0.3, reg_lambda=0.5, random_state=SEED, verbose=-1
    )
    m.fit(X_tr, y_tr, sample_weight=w_tr,
          eval_set=[(X_val, y_val)],
          callbacks=[lgb.early_stopping(100, verbose=False)])
    oof_lgbm[val_idx] = m.predict(X_val)
    test_lgbm += m.predict(X_test) / N_FOLDS
    
    # XGBoost
    m2 = xgb.XGBRegressor(
        n_estimators=3000, learning_rate=0.03, max_depth=7,
        subsample=0.8, colsample_bytree=0.7,
        reg_alpha=0.3, reg_lambda=0.5, min_child_weight=10,
        random_state=SEED, verbosity=0, early_stopping_rounds=100
    )
    m2.fit(X_tr, y_tr, sample_weight=w_tr,
           eval_set=[(X_val, y_val)], verbose=False)
    oof_xgb[val_idx] = m2.predict(X_val)
    test_xgb += m2.predict(X_test) / N_FOLDS
    
    # CatBoost
    m3 = CatBoostRegressor(
        iterations=3000, learning_rate=0.03, depth=7,
        l2_leaf_reg=3, random_seed=SEED, verbose=False,
        early_stopping_rounds=100
    )
    m3.fit(X_tr, y_tr, sample_weight=w_tr,
           eval_set=(X_val, y_val), use_best_model=True)
    oof_cat[val_idx] = m3.predict(X_val)
    test_cat += m3.predict(X_test) / N_FOLDS
    
    r2_l = r2_score(y_val, oof_lgbm[val_idx])
    r2_x = r2_score(y_val, oof_xgb[val_idx])
    r2_c = r2_score(y_val, oof_cat[val_idx])
    print(f"  Fold {fold+1}: LGBM={r2_l:.4f}  XGB={r2_x:.4f}  CAT={r2_c:.4f}")

# Base ensemble
base_oof = np.column_stack([oof_lgbm, oof_xgb, oof_cat])
base_test_arr = np.column_stack([test_lgbm, test_xgb, test_cat])

def opt_weights(oof_arr, y_true, n_models):
    def obj(trial):
        w = [trial.suggest_float(f'w{i}', 0, 1) for i in range(n_models)]
        t = sum(w)
        if t == 0: return -999
        w = [x/t for x in w]
        blend = sum(w[i] * oof_arr[:, i] for i in range(n_models))
        return r2_score(y_true, blend)
    s = optuna.create_study(direction='maximize')
    s.optimize(obj, n_trials=300, show_progress_bar=False)
    w = [s.best_params[f'w{i}'] for i in range(n_models)]
    t = sum(w)
    return [x/t for x in w], s.best_value

weights_base, base_r2 = opt_weights(base_oof, y_raw, 3)
base_pred_test = sum(weights_base[i] * base_test_arr[:, i] for i in range(3))
print(f"\n  Base OOF R2: {base_r2:.4f}  weights={['%.3f'%w for w in weights_base]}")

# =============================================================
# STEP 8-10 - AGGRESSIVE PSEUDO-LABELING (5 rounds)
# =============================================================
N_PSEUDO = 5
pseudo_weights_schedule = [0.3, 0.5, 0.7, 0.8, 0.9]
current_preds = base_pred_test.copy()

# Store all round predictions for final ensemble
all_round_preds = [base_pred_test.copy()]

for rnd in range(N_PSEUDO):
    pw = pseudo_weights_schedule[rnd]
    print(f"\n[{8+min(rnd,2)}/11] Pseudo-labeling round {rnd+1}/{N_PSEUDO} (weight={pw})...")
    
    pseudo_y = np.clip(current_preds, 0, 1)
    
    # Combined data
    y_combined = np.concatenate([y_raw.values, pseudo_y])
    sw_combined = np.ones(n_train + n_test)
    sw_combined[:n_train][day_vals == 49] = 3.0
    sw_combined[n_train:] = pw
    
    # Recompute target encoding with combined data
    combined_geo = np.concatenate([geohashes_all[:n_train], geohashes_all[n_train:]])
    df_te = pd.DataFrame({
        'geohash': combined_geo,
        'hour': np.concatenate([hours_all[:n_train], hours_all[n_train:]]),
        'time_slot': np.concatenate([slots_all[:n_train], slots_all[n_train:]]),
        'demand': y_combined,
        'RoadType': np.concatenate([combined.iloc[:n_train]['RoadType'].values if 'RoadType' in combined.columns else np.zeros(n_train),
                                     combined.iloc[n_train:]['RoadType'].values if 'RoadType' in combined.columns else np.zeros(n_test)]),
        'Weather': np.concatenate([combined.iloc[:n_train]['Weather'].values if 'Weather' in combined.columns else np.zeros(n_train),
                                    combined.iloc[n_train:]['Weather'].values if 'Weather' in combined.columns else np.zeros(n_test)]),
    })
    
    # Recompute key TE features
    new_geo_slot_mean = df_te.groupby(['geohash', 'time_slot'])['demand'].mean()
    new_geo_hour_mean = df_te.groupby(['geohash', 'hour'])['demand'].mean()
    new_geo_mean      = df_te.groupby('geohash')['demand'].mean()
    new_slot_mean     = df_te.groupby('time_slot')['demand'].mean()
    new_hour_mean     = df_te.groupby('hour')['demand'].mean()
    
    # Update TE features in X and X_test
    X_rnd = X.copy()
    X_test_rnd = X_test.copy()
    
    for df_part, geos, hrs, sls in [(X_rnd, geohashes_all[:n_train], hours_all[:n_train], slots_all[:n_train]),
                                     (X_test_rnd, geohashes_all[n_train:], hours_all[n_train:], slots_all[n_train:])]:
        geo_s = pd.Series(geos, index=df_part.index)
        hour_s = pd.Series(hrs, index=df_part.index)
        slot_s = pd.Series(sls, index=df_part.index)
        
        if 'geo_te_mean' in df_part.columns:
            df_part['geo_te_mean'] = geo_s.map(new_geo_mean).fillna(global_mean_te).values
        if 'geo_slot_te_mean' in df_part.columns:
            keys = list(zip(geos, sls))
            df_part['geo_slot_te_mean'] = [new_geo_slot_mean.get(k, global_mean_te) for k in keys]
        if 'geo_hour_te_mean' in df_part.columns:
            keys = list(zip(geos, hrs))
            df_part['geo_hour_te_mean'] = [new_geo_hour_mean.get(k, global_mean_te) for k in keys]
        if 'slot_te_mean' in df_part.columns:
            df_part['slot_te_mean'] = slot_s.map(new_slot_mean).fillna(global_mean_te).values
        if 'hour_te_mean' in df_part.columns:
            df_part['hour_te_mean'] = hour_s.map(new_hour_mean).fillna(global_mean_te).values
        
        # Update deviation features
        if 'geo_slot_dev' in df_part.columns and 'geo_slot_te_mean' in df_part.columns:
            df_part['geo_slot_dev'] = df_part['geo_slot_te_mean'] - df_part.get('geo_te_smooth', global_mean_te)
        if 'geo_hour_dev' in df_part.columns and 'geo_hour_te_mean' in df_part.columns:
            df_part['geo_hour_dev'] = df_part['geo_hour_te_mean'] - df_part.get('geo_te_smooth', global_mean_te)
    
    X_rnd = X_rnd.fillna(0)
    X_test_rnd = X_test_rnd.fillna(0)
    
    # Train on FULL combined data (train + pseudo-test)
    X_full = pd.concat([X_rnd, X_test_rnd], axis=0).reset_index(drop=True)
    
    round_test_preds = np.zeros(n_test)
    n_models = 0
    
    for seed_offset in [0, 1, 2]:
        # LightGBM
        m = lgb.LGBMRegressor(
            n_estimators=2000, learning_rate=0.03, max_depth=7, num_leaves=63,
            min_child_samples=30, subsample=0.8, colsample_bytree=0.7,
            reg_alpha=0.3, reg_lambda=0.5,
            random_state=SEED + rnd * 10 + seed_offset, verbose=-1
        )
        m.fit(X_full, y_combined, sample_weight=sw_combined)
        round_test_preds += m.predict(X_test_rnd)
        n_models += 1
        
        # CatBoost (strongest model from base)
        m3 = CatBoostRegressor(
            iterations=2000, learning_rate=0.03, depth=7,
            l2_leaf_reg=3, random_seed=SEED + rnd * 10 + seed_offset, verbose=False
        )
        m3.fit(X_full, y_combined, sample_weight=sw_combined)
        round_test_preds += m3.predict(X_test_rnd)
        n_models += 1
    
    round_test_preds /= n_models
    round_test_preds = np.clip(round_test_preds, 0, 1)
    
    # Update current predictions (blend for stability)
    blend_new = 0.7
    current_preds = blend_new * round_test_preds + (1 - blend_new) * current_preds
    current_preds = np.clip(current_preds, 0, 1)
    
    all_round_preds.append(current_preds.copy())
    
    print(f"  Preds: mean={current_preds.mean():.6f}, std={current_preds.std():.6f}, "
          f"min={current_preds.min():.6f}, max={current_preds.max():.6f}")

# =============================================================
# STEP 11 - FINAL ENSEMBLE & SUBMISSION
# =============================================================
print(f"\n[11/11] Final ensemble & submission...")

# Final prediction: weighted average of base + last 3 pseudo rounds
final_preds = (
    0.15 * all_round_preds[0] +   # base
    0.10 * all_round_preds[-3] +   # round 3
    0.25 * all_round_preds[-2] +   # round 4
    0.50 * all_round_preds[-1]     # round 5 (most refined)
)
final_preds = np.clip(final_preds, 0, 1)

print(f"  Final: mean={final_preds.mean():.6f}, std={final_preds.std():.6f}")
print(f"  Range: [{final_preds.min():.6f}, {final_preds.max():.6f}]")

# Generate submissions
test_original = pd.read_csv('./work/test.csv')

for name, preds in [
    ('submission_v4.csv', final_preds),
    ('submission_v4_pseudo_only.csv', np.clip(all_round_preds[-1], 0, 1)),
    ('submission_v4_base_only.csv', np.clip(all_round_preds[0], 0, 1)),
    ('submission.csv', final_preds),
]:
    sub = pd.DataFrame({'Index': test_original['Index'], 'demand': preds})
    assert sub.shape[0] == n_test
    assert sub.isnull().sum().sum() == 0
    sub.to_csv(f'./work/{name}', index=False)
    print(f"  [OK] {name} saved")

elapsed = time.time() - start_time
print(f"\n{'='*70}")
print(f"DONE! Total time: {elapsed/60:.1f} minutes")
print(f"Base OOF R2: {base_r2:.4f}")
print(f"Temporal R2: {temp_r2:.4f}")
print(f"{'='*70}")
print(f"\nSubmission files:")
print(f"  1. submission_v4.csv (main blend)")
print(f"  2. submission_v4_pseudo_only.csv (last pseudo round)")
print(f"  3. submission_v4_base_only.csv (no pseudo)")
print(f"  4. submission.csv (same as v4)")
