#!/usr/bin/env python3
"""
Traffic Demand Prediction - Pipeline v3
========================================
Key fixes over v2 (83.93 → targeting 99+):
1. OUT-OF-FOLD target encoding (fixes CV leakage + prevents over-reliance on TE)
2. PSEUDO-LABELING (3 rounds to learn test distribution)
3. DAY-49 WEIGHTED training (test is day 49, upweight day 49 train rows)
4. TEMPORAL VALIDATION (honest CV using day49 holdout)
5. Train BOTH with and without log transform, ensemble both
6. More regularization to prevent overfitting
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
print("PIPELINE v3 - OOF Target Encoding + Pseudo-Labeling")
print("=" * 70)

# =============================================================
# STEP 1 — LOAD DATA
# =============================================================
print("\n[1/12] Loading data...")
train = pd.read_csv('./work/train.csv')
test  = pd.read_csv('./work/test.csv')
print(f"  Train: {train.shape}, Test: {test.shape}")

y_raw = train['demand'].copy()
n_train = len(train)
n_test = len(test)

# =============================================================
# STEP 2 — PREPROCESS (no target encoding yet)
# =============================================================
print("\n[2/12] Preprocessing...")

train['_source'] = 'train'
test['_source']  = 'test'

combined = pd.concat(
    [train.drop('demand', axis=1), test], axis=0
).reset_index(drop=True)

# Parse timestamp
parts = combined['timestamp'].astype(str).str.split(':', expand=True)
combined['hour']   = parts[0].astype(int)
combined['minute'] = parts[1].astype(int)

# Decode geohash
def safe_decode(gh):
    try:
        lat, lon = pgh.decode(gh)
        return float(lat), float(lon)
    except:
        return np.nan, np.nan

decoded = combined['geohash'].apply(safe_decode)
combined['lat'] = [d[0] for d in decoded]
combined['lon'] = [d[1] for d in decoded]

# Fill missing values (train-only stats)
train_mask = combined['_source'] == 'train'
for col in ['RoadType', 'Weather']:
    if combined[col].isnull().any():
        fill_val = combined.loc[train_mask, col].mode()[0]
        combined[col] = combined[col].fillna(fill_val)

if combined['Temperature'].isnull().any():
    temp_by_geo = combined.loc[train_mask].groupby('geohash')['Temperature'].median()
    global_temp_med = combined.loc[train_mask, 'Temperature'].median()
    for idx in combined[combined['Temperature'].isnull()].index:
        geo = combined.loc[idx, 'geohash']
        combined.loc[idx, 'Temperature'] = temp_by_geo.get(geo, global_temp_med)

print(f"  NaN remaining: {combined.isnull().sum().sum()}")

# =============================================================
# STEP 3 — FEATURE ENGINEERING (no target encoding here)
# =============================================================
print("\n[3/12] Feature engineering (non-target features)...")

combined['time_slot'] = combined['hour'] * 4 + combined['minute'] // 15
combined['day_of_week'] = combined['day'] % 7
combined['is_weekend']  = combined['day_of_week'].isin([5, 6]).astype(int)
combined['is_peak']     = combined['hour'].isin([7,8,9,10,11,12,13]).astype(int)
combined['is_morning']  = combined['hour'].isin([6,7,8,9,10,11]).astype(int)
combined['is_night']    = combined['hour'].isin([22,23,0,1,2,3,4,5]).astype(int)

def part_of_day(h):
    if 0  <= h < 6:  return 0
    if 6  <= h < 12: return 1
    if 12 <= h < 17: return 2
    if 17 <= h < 21: return 3
    return 4
combined['part_of_day'] = combined['hour'].apply(part_of_day)
combined['minutes_since_midnight'] = combined['hour'] * 60 + combined['minute']

# Cyclical
for col, period in [('hour', 24), ('minute', 60), ('day_of_week', 7), ('time_slot', 96)]:
    combined[f'{col}_sin'] = np.sin(2 * np.pi * combined[col] / period)
    combined[f'{col}_cos'] = np.cos(2 * np.pi * combined[col] / period)

# Geohash
combined['geohash_4'] = combined['geohash'].str[:4]
combined['geohash_5'] = combined['geohash'].str[:5]

# Binary
combined['LargeVehicles_bin'] = (combined['LargeVehicles'] == 'Allowed').astype(int)
combined['Landmarks_bin']     = (combined['Landmarks'] == 'Yes').astype(int)

# Temperature
combined['temp_sq'] = combined['Temperature'] ** 2

# Interactions
combined['peak_x_lanes']    = combined['is_peak'] * combined['NumberofLanes']
combined['morning_x_lanes'] = combined['is_morning'] * combined['NumberofLanes']
combined['night_x_lanes']   = combined['is_night'] * combined['NumberofLanes']
combined['weekend_x_lanes'] = combined['is_weekend'] * combined['NumberofLanes']
combined['large_veh_x_lanes'] = combined['LargeVehicles_bin'] * combined['NumberofLanes']
combined['landmarks_x_peak']  = combined['Landmarks_bin'] * combined['is_peak']
combined['landmarks_x_lanes'] = combined['Landmarks_bin'] * combined['NumberofLanes']

# Lat/Lon
combined['lat_x_lon'] = combined['lat'] * combined['lon']

# FREQUENCY ENCODING (uses both train and test — no leakage)
for col in ['geohash', 'geohash_4', 'geohash_5']:
    freq = combined[col].value_counts()
    combined[f'{col}_freq'] = combined[col].map(freq)

# Encode categoricals
cat_cols = ['RoadType', 'Weather', 'LargeVehicles', 'Landmarks']
for col in cat_cols:
    le = LabelEncoder()
    le.fit(combined.loc[train_mask, col].astype(str))
    vals = combined[col].astype(str).copy()
    mask_unseen = ~vals.isin(le.classes_)
    if mask_unseen.any():
        vals[mask_unseen] = le.classes_[0]
    combined[col] = le.transform(vals)

# Store geohash/timestamp for later, then drop
geo_col = combined['geohash'].copy()
drop_cols = ['_source', 'geohash', 'geohash_4', 'geohash_5', 'timestamp', 'Index']
drop_cols = [c for c in drop_cols if c in combined.columns]
combined.drop(columns=drop_cols, inplace=True)

for col in combined.columns:
    combined[col] = pd.to_numeric(combined[col], errors='coerce')
combined = combined.fillna(0)

X_base = combined.iloc[:n_train].copy()
X_test_base = combined.iloc[n_train:].copy()

print(f"  Base features: {X_base.shape[1]}")

# =============================================================
# STEP 4 — OUT-OF-FOLD TARGET ENCODING FUNCTION
# =============================================================
print("\n[4/12] Setting up OOF target encoding...")

train_geohash = geo_col.iloc[:n_train].values
test_geohash  = geo_col.iloc[n_train:].values

def compute_target_encoding_features(X_tr_idx, y_series, train_geo, X_all_shape,
                                      test_geo, n_test, smooth=20):
    """Compute target encoding features for given train indices.
    Returns TE features for those indices only (for OOF use)."""
    
    y_vals = y_series.values
    global_mean = y_vals[X_tr_idx].mean()
    global_std  = y_vals[X_tr_idx].std()
    
    tr_geo = train_geo[X_tr_idx]
    tr_y   = y_vals[X_tr_idx]
    
    # Build aggregation tables from training subset
    df_tr = pd.DataFrame({'geohash': tr_geo, 'demand': tr_y})
    
    # Add hour and time_slot
    all_hours = X_all_shape['hour'].values if 'hour' in X_all_shape.columns else None
    all_slots = X_all_shape['time_slot'].values if 'time_slot' in X_all_shape.columns else None
    
    tr_hours = all_hours[X_tr_idx] if all_hours is not None else None
    tr_slots = all_slots[X_tr_idx] if all_slots is not None else None
    
    if tr_hours is not None:
        df_tr['hour'] = tr_hours
    if tr_slots is not None:
        df_tr['time_slot'] = tr_slots
    
    features = {}
    
    # Per-geohash
    geo_agg = df_tr.groupby('geohash')['demand'].agg(['mean', 'std', 'median', 'count'])
    geo_agg.columns = ['geo_te_mean', 'geo_te_std', 'geo_te_median', 'geo_te_count']
    geo_agg['geo_te_smooth'] = (
        (geo_agg['geo_te_count'] * geo_agg['geo_te_mean'] + smooth * global_mean) /
        (geo_agg['geo_te_count'] + smooth)
    )
    features['geo_agg'] = geo_agg
    
    # Per-geohash prefix
    df_tr['geo4'] = df_tr['geohash'].str[:4]
    df_tr['geo5'] = df_tr['geohash'].str[:5]
    
    geo4_agg = df_tr.groupby('geo4')['demand'].agg(['mean', 'count'])
    geo4_agg.columns = ['geo4_te_mean', 'geo4_te_count']
    geo4_agg['geo4_te_smooth'] = (
        (geo4_agg['geo4_te_count'] * geo4_agg['geo4_te_mean'] + smooth * global_mean) /
        (geo4_agg['geo4_te_count'] + smooth)
    )
    features['geo4_agg'] = geo4_agg
    
    geo5_agg = df_tr.groupby('geo5')['demand'].agg(['mean', 'count'])
    geo5_agg.columns = ['geo5_te_mean', 'geo5_te_count']
    features['geo5_agg'] = geo5_agg
    
    # Per-hour
    if tr_hours is not None:
        hour_agg = df_tr.groupby('hour')['demand'].agg(['mean', 'median', 'std'])
        hour_agg.columns = ['hour_te_mean', 'hour_te_median', 'hour_te_std']
        features['hour_agg'] = hour_agg
    
    # Per-time_slot
    if tr_slots is not None:
        slot_agg = df_tr.groupby('time_slot')['demand'].agg(['mean', 'count'])
        slot_agg.columns = ['slot_te_mean', 'slot_te_count']
        features['slot_agg'] = slot_agg
    
    # Per-geohash x hour
    if tr_hours is not None:
        geo_hour_agg = df_tr.groupby(['geohash', 'hour'])['demand'].agg(['mean', 'count'])
        geo_hour_agg.columns = ['geo_hour_te_mean', 'geo_hour_te_count']
        geo_hour_agg['geo_hour_te_smooth'] = (
            (geo_hour_agg['geo_hour_te_count'] * geo_hour_agg['geo_hour_te_mean'] + 5 * global_mean) /
            (geo_hour_agg['geo_hour_te_count'] + 5)
        )
        features['geo_hour_agg'] = geo_hour_agg
    
    # Per-geohash x time_slot
    if tr_slots is not None:
        geo_slot_agg = df_tr.groupby(['geohash', 'time_slot'])['demand'].agg(['mean', 'count'])
        geo_slot_agg.columns = ['geo_slot_te_mean', 'geo_slot_te_count']
        geo_slot_agg['geo_slot_te_smooth'] = (
            (geo_slot_agg['geo_slot_te_count'] * geo_slot_agg['geo_slot_te_mean'] + 3 * global_mean) /
            (geo_slot_agg['geo_slot_te_count'] + 3)
        )
        features['geo_slot_agg'] = geo_slot_agg
    
    return features, global_mean, global_std


def apply_te_features(X_df, geohashes, hours, slots, te_features, global_mean, global_std):
    """Apply pre-computed target encoding features to a dataset."""
    X = X_df.copy()
    geo_series = pd.Series(geohashes, index=X.index)
    
    # Geo features
    geo_agg = te_features['geo_agg']
    for col in ['geo_te_mean', 'geo_te_std', 'geo_te_median', 'geo_te_count', 'geo_te_smooth']:
        X[col] = geo_series.map(geo_agg[col]).fillna(global_mean if 'count' not in col else 0)
    X['geo_te_std'] = X['geo_te_std'].fillna(global_std)
    
    # Geo4 features
    geo4_series = geo_series.str[:4]
    geo4_agg = te_features['geo4_agg']
    for col in ['geo4_te_mean', 'geo4_te_count', 'geo4_te_smooth']:
        X[col] = geo4_series.map(geo4_agg[col]).fillna(global_mean if 'count' not in col else 0)
    
    # Geo5 features
    geo5_series = geo_series.str[:5]
    geo5_agg = te_features['geo5_agg']
    for col in ['geo5_te_mean', 'geo5_te_count']:
        X[col] = geo5_series.map(geo5_agg[col]).fillna(global_mean if 'count' not in col else 0)
    
    # Hour features
    if 'hour_agg' in te_features:
        hour_agg = te_features['hour_agg']
        hour_series = pd.Series(hours, index=X.index)
        for col in ['hour_te_mean', 'hour_te_median', 'hour_te_std']:
            X[col] = hour_series.map(hour_agg[col]).fillna(global_mean if 'std' not in col else global_std)
    
    # Slot features
    if 'slot_agg' in te_features:
        slot_agg = te_features['slot_agg']
        slot_series = pd.Series(slots, index=X.index)
        for col in ['slot_te_mean', 'slot_te_count']:
            X[col] = slot_series.map(slot_agg[col]).fillna(global_mean if 'count' not in col else 0)
    
    # Geo x Hour features
    if 'geo_hour_agg' in te_features:
        geo_hour_agg = te_features['geo_hour_agg']
        geo_hour_key = list(zip(geohashes, hours))
        for col in ['geo_hour_te_mean', 'geo_hour_te_count', 'geo_hour_te_smooth']:
            mapping = geo_hour_agg[col].to_dict()
            vals = [mapping.get(k, np.nan) for k in geo_hour_key]
            X[col] = vals
            fill = X['geo_te_smooth'] if 'count' not in col else 0
            X[col] = X[col].fillna(fill)
    
    # Geo x Slot features
    if 'geo_slot_agg' in te_features:
        geo_slot_agg = te_features['geo_slot_agg']
        geo_slot_key = list(zip(geohashes, slots))
        for col in ['geo_slot_te_mean', 'geo_slot_te_count', 'geo_slot_te_smooth']:
            mapping = geo_slot_agg[col].to_dict()
            vals = [mapping.get(k, np.nan) for k in geo_slot_key]
            X[col] = vals
            fill = X['geo_te_smooth'] if 'count' not in col else 0
            X[col] = X[col].fillna(fill)
    
    # Deviation features
    if 'geo_hour_te_mean' in X.columns:
        X['geo_hour_dev'] = X['geo_hour_te_mean'] - X['geo_te_smooth']
    if 'geo_slot_te_mean' in X.columns:
        X['geo_slot_dev'] = X['geo_slot_te_mean'] - X['geo_te_smooth']
    if 'hour_te_mean' in X.columns:
        X['hour_dev'] = X['hour_te_mean'] - global_mean
    
    X = X.fillna(0)
    return X


# Get base hour/slot arrays
train_hours = X_base['hour'].values
train_slots = X_base['time_slot'].values
test_hours  = X_test_base['hour'].values
test_slots  = X_test_base['time_slot'].values

# =============================================================
# STEP 5 — TEMPORAL VALIDATION (honest score estimate)
# =============================================================
print("\n[5/12] Temporal validation (day48 train → day49 val)...")

train_df_orig = pd.read_csv('./work/train.csv')
day48_mask = train_df_orig['day'] == 48
day49_mask = train_df_orig['day'] == 49

day48_idx = np.where(day48_mask)[0]
day49_idx = np.where(day49_mask)[0]

# Quick temporal validation with LightGBM
y_log = np.log1p(y_raw)

# Compute TE on day48 only
te_feats_48, gm_48, gs_48 = compute_target_encoding_features(
    day48_idx, y_log, train_geohash, X_base, test_geohash, n_test
)

X_48_te = apply_te_features(X_base.iloc[day48_idx], train_geohash[day48_idx],
                             train_hours[day48_idx], train_slots[day48_idx],
                             te_feats_48, gm_48, gs_48)
X_49_te = apply_te_features(X_base.iloc[day49_idx], train_geohash[day49_idx],
                             train_hours[day49_idx], train_slots[day49_idx],
                             te_feats_48, gm_48, gs_48)

quick_m = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05, max_depth=6, 
                              num_leaves=50, verbose=-1, random_state=SEED)
quick_m.fit(X_48_te, y_log.iloc[day48_idx],
            eval_set=[(X_49_te, y_log.iloc[day49_idx])],
            callbacks=[lgb.early_stopping(50, verbose=False)])

pred_49 = quick_m.predict(X_49_te)
temporal_r2 = r2_score(y_raw.iloc[day49_idx], np.expm1(pred_49))
print(f"  Temporal R2 (day48→day49): {temporal_r2:.4f}")
print(f"  This is the HONEST estimate of test performance")

# =============================================================
# STEP 6 — TRAIN BASE MODELS WITH OOF TARGET ENCODING
# =============================================================
print("\n[6/12] Training base models with OOF target encoding...")

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

# We'll train with BOTH log and raw targets and ensemble
for target_name, y_target in [('log', y_log), ('raw', y_raw)]:
    print(f"\n  --- Training with {target_name} target ---")
    
    oof_lgbm = np.zeros(n_train)
    oof_xgb  = np.zeros(n_train)
    oof_cat  = np.zeros(n_train)
    
    test_lgbm = np.zeros(n_test)
    test_xgb  = np.zeros(n_test)
    test_cat  = np.zeros(n_test)
    
    scores_lgbm, scores_xgb, scores_cat = [], [], []
    
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_base)):
        # OOF target encoding: compute TE on train fold only
        te_feats, gm, gs = compute_target_encoding_features(
            tr_idx, y_target, train_geohash, X_base, test_geohash, n_test
        )
        
        X_tr = apply_te_features(X_base.iloc[tr_idx], train_geohash[tr_idx],
                                  train_hours[tr_idx], train_slots[tr_idx],
                                  te_feats, gm, gs)
        X_val = apply_te_features(X_base.iloc[val_idx], train_geohash[val_idx],
                                   train_hours[val_idx], train_slots[val_idx],
                                   te_feats, gm, gs)
        X_te = apply_te_features(X_test_base, test_geohash,
                                  test_hours, test_slots,
                                  te_feats, gm, gs)
        
        y_tr  = y_target.iloc[tr_idx]
        y_val = y_target.iloc[val_idx]
        
        # Sample weights: upweight day 49 rows (3x)
        day_vals = train_df_orig['day'].values
        w_tr = np.ones(len(tr_idx))
        w_tr[day_vals[tr_idx] == 49] = 3.0
        
        # LightGBM
        m_lgb = lgb.LGBMRegressor(
            n_estimators=3000, learning_rate=0.03, max_depth=6,
            num_leaves=50, min_child_samples=30, subsample=0.8,
            colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=1.0,
            random_state=SEED, verbose=-1
        )
        m_lgb.fit(X_tr, y_tr, sample_weight=w_tr,
                  eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
        oof_lgbm[val_idx] = m_lgb.predict(X_val)
        test_lgbm += m_lgb.predict(X_te) / N_FOLDS
        
        # XGBoost
        m_xgb = xgb.XGBRegressor(
            n_estimators=3000, learning_rate=0.03, max_depth=6,
            subsample=0.8, colsample_bytree=0.7,
            reg_alpha=0.5, reg_lambda=1.0, min_child_weight=10,
            random_state=SEED, verbosity=0, early_stopping_rounds=100
        )
        m_xgb.fit(X_tr, y_tr, sample_weight=w_tr,
                  eval_set=[(X_val, y_val)], verbose=False)
        oof_xgb[val_idx] = m_xgb.predict(X_val)
        test_xgb += m_xgb.predict(X_te) / N_FOLDS
        
        # CatBoost
        m_cat = CatBoostRegressor(
            iterations=3000, learning_rate=0.03, depth=6,
            l2_leaf_reg=3, random_seed=SEED, verbose=False,
            early_stopping_rounds=100
        )
        m_cat.fit(X_tr, y_tr, sample_weight=w_tr,
                  eval_set=(X_val, y_val), use_best_model=True)
        oof_cat[val_idx] = m_cat.predict(X_val)
        test_cat += m_cat.predict(X_te) / N_FOLDS
        
        # Scores in original scale
        if target_name == 'log':
            r2_l = r2_score(np.expm1(y_val), np.expm1(oof_lgbm[val_idx]))
            r2_x = r2_score(np.expm1(y_val), np.expm1(oof_xgb[val_idx]))
            r2_c = r2_score(np.expm1(y_val), np.expm1(oof_cat[val_idx]))
        else:
            r2_l = r2_score(y_val, oof_lgbm[val_idx])
            r2_x = r2_score(y_val, oof_xgb[val_idx])
            r2_c = r2_score(y_val, oof_cat[val_idx])
        
        scores_lgbm.append(r2_l)
        scores_xgb.append(r2_x)
        scores_cat.append(r2_c)
        print(f"    Fold {fold+1}: LGBM={r2_l:.4f}  XGB={r2_x:.4f}  CAT={r2_c:.4f}")
    
    print(f"  CV Mean: LGBM={np.mean(scores_lgbm):.4f}  XGB={np.mean(scores_xgb):.4f}  CAT={np.mean(scores_cat):.4f}")
    
    # Store results
    if target_name == 'log':
        oof_log = {'lgbm': oof_lgbm.copy(), 'xgb': oof_xgb.copy(), 'cat': oof_cat.copy()}
        test_log = {'lgbm': test_lgbm.copy(), 'xgb': test_xgb.copy(), 'cat': test_cat.copy()}
    else:
        oof_raw_dict = {'lgbm': oof_lgbm.copy(), 'xgb': oof_xgb.copy(), 'cat': oof_cat.copy()}
        test_raw_dict = {'lgbm': test_lgbm.copy(), 'xgb': test_xgb.copy(), 'cat': test_cat.copy()}

# =============================================================
# STEP 7 — BASE ENSEMBLE
# =============================================================
print("\n[7/12] Base ensemble...")

# Convert log predictions to original scale
oof_log_orig = {k: np.expm1(v) for k, v in oof_log.items()}
test_log_orig = {k: np.expm1(v) for k, v in test_log.items()}

# Average all 6 models (3 log + 3 raw)
all_oof = np.column_stack([
    oof_log_orig['lgbm'], oof_log_orig['xgb'], oof_log_orig['cat'],
    oof_raw_dict['lgbm'], oof_raw_dict['xgb'], oof_raw_dict['cat']
])
all_test = np.column_stack([
    test_log_orig['lgbm'], test_log_orig['xgb'], test_log_orig['cat'],
    test_raw_dict['lgbm'], test_raw_dict['xgb'], test_raw_dict['cat']
])

# Optuna weight optimization
def objective_ens(trial):
    weights = [trial.suggest_float(f'w{i}', 0, 1) for i in range(6)]
    total = sum(weights)
    if total == 0: return -999
    weights = [w/total for w in weights]
    blend = sum(w * all_oof[:, i] for i, w in enumerate(weights))
    return r2_score(y_raw, blend)

study = optuna.create_study(direction='maximize')
study.optimize(objective_ens, n_trials=500, show_progress_bar=False)

best_w = [study.best_params[f'w{i}'] for i in range(6)]
tw = sum(best_w)
best_w = [w/tw for w in best_w]

base_oof = sum(w * all_oof[:, i] for i, w in enumerate(best_w))
base_test = sum(w * all_test[:, i] for i, w in enumerate(best_w))

base_r2 = r2_score(y_raw, base_oof)
print(f"  Base OOF R2: {base_r2:.4f}")
print(f"  Weights: {['%.3f' % w for w in best_w]}")
print(f"  (log_lgbm, log_xgb, log_cat, raw_lgbm, raw_xgb, raw_cat)")

# =============================================================
# STEP 8-10 — PSEUDO-LABELING (3 rounds)
# =============================================================
N_PSEUDO_ROUNDS = 3
pseudo_weights = [0.3, 0.5, 0.7]
current_test_preds = base_test.copy()
best_test_preds = base_test.copy()

for rnd in range(N_PSEUDO_ROUNDS):
    pw = pseudo_weights[rnd]
    print(f"\n[{8+rnd}/12] Pseudo-labeling round {rnd+1} (weight={pw})...")
    
    # Create pseudo-labeled dataset
    pseudo_y = current_test_preds.copy()
    pseudo_y = np.clip(pseudo_y, 0, 1)  # Demand is in [0,1]
    
    # Combine train + pseudo-test
    y_combined_log = np.concatenate([y_log.values, np.log1p(pseudo_y)])
    y_combined_raw = np.concatenate([y_raw.values, pseudo_y])
    
    # Sample weights: train=1.0, pseudo=pw, day49 train=3.0
    sw = np.ones(n_train + n_test)
    sw[n_train:] = pw
    day_vals = train_df_orig['day'].values
    sw[:n_train][day_vals == 49] = 3.0
    
    # Combined geohashes
    combined_geohash = np.concatenate([train_geohash, test_geohash])
    combined_hours = np.concatenate([train_hours, test_hours])
    combined_slots = np.concatenate([train_slots, test_slots])
    
    # Full target encoding on ALL data (train + pseudo-test)
    all_idx = np.arange(n_train + n_test)
    
    # Compute TE from combined data
    X_combined_base = pd.concat([X_base, X_test_base], axis=0).reset_index(drop=True)
    
    te_feats_full, gm_full, gs_full = compute_target_encoding_features(
        all_idx, pd.Series(y_combined_log), combined_geohash, X_combined_base,
        test_geohash, n_test
    )
    
    X_combined_te = apply_te_features(
        X_combined_base, combined_geohash, combined_hours, combined_slots,
        te_feats_full, gm_full, gs_full
    )
    
    X_test_te = apply_te_features(
        X_test_base, test_geohash, test_hours, test_slots,
        te_feats_full, gm_full, gs_full
    )
    
    # Train models on combined data
    test_preds_round = np.zeros(n_test)
    n_models = 0
    
    for target_name, y_comb in [('log', y_combined_log), ('raw', y_combined_raw)]:
        for model_type in ['lgbm', 'xgb', 'cat']:
            if model_type == 'lgbm':
                m = lgb.LGBMRegressor(
                    n_estimators=2000, learning_rate=0.03, max_depth=6,
                    num_leaves=50, min_child_samples=30, subsample=0.8,
                    colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=1.0,
                    random_state=SEED + rnd, verbose=-1
                )
                m.fit(X_combined_te, y_comb, sample_weight=sw)
                pred = m.predict(X_test_te)
            elif model_type == 'xgb':
                m = xgb.XGBRegressor(
                    n_estimators=2000, learning_rate=0.03, max_depth=6,
                    subsample=0.8, colsample_bytree=0.7,
                    reg_alpha=0.5, reg_lambda=1.0, min_child_weight=10,
                    random_state=SEED + rnd, verbosity=0
                )
                m.fit(X_combined_te, y_comb, sample_weight=sw)
                pred = m.predict(X_test_te)
            else:
                m = CatBoostRegressor(
                    iterations=2000, learning_rate=0.03, depth=6,
                    l2_leaf_reg=3, random_seed=SEED + rnd, verbose=False
                )
                m.fit(X_combined_te, y_comb, sample_weight=sw)
                pred = m.predict(X_test_te)
            
            if target_name == 'log':
                pred = np.expm1(pred)
            
            test_preds_round += pred
            n_models += 1
    
    test_preds_round /= n_models
    test_preds_round = np.clip(test_preds_round, 0, 1)
    
    # Blend with previous predictions for stability
    blend_ratio = 0.6  # 60% new, 40% old
    current_test_preds = blend_ratio * test_preds_round + (1 - blend_ratio) * current_test_preds
    current_test_preds = np.clip(current_test_preds, 0, 1)
    
    print(f"  Round {rnd+1} predictions: mean={current_test_preds.mean():.6f}, "
          f"std={current_test_preds.std():.6f}, "
          f"min={current_test_preds.min():.6f}, max={current_test_preds.max():.6f}")

# =============================================================
# STEP 11 — FINAL BLEND: base + pseudo-labeled
# =============================================================
print(f"\n[11/12] Final blending...")

# Try different blend ratios of base vs pseudo-labeled
best_blend = None
best_blend_r2 = -999

# We can't evaluate on test (no labels), so use OOF R2 of base as guide
# and trust that pseudo-labeling improves generalization
# Final: use mostly pseudo-labeled predictions
final_preds = 0.3 * base_test + 0.7 * current_test_preds
final_preds = np.clip(final_preds, 0, 1)

print(f"  Final blend: 30% base + 70% pseudo-labeled")
print(f"  Final predictions: mean={final_preds.mean():.6f}, std={final_preds.std():.6f}")
print(f"  Range: [{final_preds.min():.6f}, {final_preds.max():.6f}]")

# =============================================================
# STEP 12 — GENERATE SUBMISSION
# =============================================================
print(f"\n[12/12] Generating submission...")

test_original = pd.read_csv('./work/test.csv')
submission = pd.DataFrame({
    'Index': test_original['Index'],
    'demand': final_preds
})

assert submission.shape[0] == len(test_original)
assert submission.shape[1] == 2
assert list(submission.columns) == ['Index', 'demand']
assert submission.isnull().sum().sum() == 0

submission.to_csv('./work/submission_v3.csv', index=False)
submission.to_csv('./work/submission.csv', index=False)

print(f"  Shape: {submission.shape}")
print(f"  Demand stats:\n{submission['demand'].describe()}")
print(f"\n  [OK] submission_v3.csv saved!")
print(f"  [OK] submission.csv overwritten!")

# Also save pure pseudo-labeled version (might be better)
sub_pseudo = pd.DataFrame({
    'Index': test_original['Index'],
    'demand': np.clip(current_test_preds, 0, 1)
})
sub_pseudo.to_csv('./work/submission_v3_pseudo_only.csv', index=False)
print(f"  [OK] submission_v3_pseudo_only.csv saved (pure pseudo-labeled)")

# And save base-only version for comparison
sub_base = pd.DataFrame({
    'Index': test_original['Index'],
    'demand': np.clip(base_test, 0, 1)
})
sub_base.to_csv('./work/submission_v3_base_only.csv', index=False)
print(f"  [OK] submission_v3_base_only.csv saved (OOF TE only, no pseudo)")

elapsed = time.time() - start_time
print(f"\n{'='*70}")
print(f"DONE! Total time: {elapsed/60:.1f} minutes")
print(f"Base OOF R2: {base_r2:.4f}")
print(f"Temporal R2 (day48→day49): {temporal_r2:.4f}")
print(f"{'='*70}")
print(f"\nTIP: Upload submission_v3.csv first. If not good enough, try")
print(f"submission_v3_pseudo_only.csv and submission_v3_base_only.csv")
