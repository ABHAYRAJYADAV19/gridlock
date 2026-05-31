#!/usr/bin/env python3
"""
Traffic Demand Prediction - Hackathon Pipeline
================================================
Regression: predict 'demand' (traffic demand)
Metric   : score = max(0, 100 * R2(actual, predicted))
Goal     : Maximize R2 on hidden test data - generalization first!
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

import pygeohash as pgh
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

print("=" * 60)
print("STEP 1 - Libraries Imported Successfully")
print("=" * 60)

# =============================================================
# STEP 2 — LOAD & VERIFY DATA
# =============================================================
print("\n" + "=" * 60)
print("STEP 2 - LOAD & VERIFY DATA")
print("=" * 60)

train = pd.read_csv('./work/train.csv')
test  = pd.read_csv('./work/test.csv')

print("Train shape:", train.shape)
print("Test shape :", test.shape)

print("\nTrain columns:", train.columns.tolist())
print("Test columns :", test.columns.tolist())

print("\nTrain dtypes:\n", train.dtypes)

print("\nTrain head:\n", train.head())
print("\nTest head:\n", test.head())

# Missing values
print("\nTrain missing values:")
print(train.isnull().sum())
print("\nTest missing values:")
print(test.isnull().sum())

# Value counts for categorical columns
for col in train.select_dtypes(include=['object']).columns:
    print(f"\nTrain '{col}' value_counts():")
    print(train[col].value_counts())

# Numerical describe
print("\nTrain numerical describe:\n", train.describe())

# Demand distribution
if 'demand' in train.columns:
    print("\n--- Demand Distribution ---")
    print(f"  Mean  : {train['demand'].mean():.6f}")
    print(f"  Std   : {train['demand'].std():.6f}")
    print(f"  Min   : {train['demand'].min():.6f}")
    print(f"  Max   : {train['demand'].max():.6f}")
    print(f"  Skew  : {train['demand'].skew():.6f}")
    
    plt.figure(figsize=(10, 5))
    plt.hist(train['demand'], bins=50, edgecolor='black', alpha=0.7)
    plt.title('Demand Distribution')
    plt.xlabel('Demand')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('./work/demand_distribution.png', dpi=100)
    plt.close()
    print("  -> Saved demand_distribution.png")
else:
    raise ValueError("Target column 'demand' NOT found in train!")

# Verification summary
verified_cols = train.columns.tolist()
print(f"\nVerified columns: {verified_cols}")
print(f"Target column exists in train: {'demand' in train.columns}")
missing_cols = [c for c in train.columns if train[c].isnull().sum() > 0]
print(f"Missing values found in: {missing_cols}")

# =============================================================
# STEP 3 — EXPLORATORY DATA ANALYSIS
# =============================================================
print("\n" + "=" * 60)
print("STEP 3 - EXPLORATORY DATA ANALYSIS")
print("=" * 60)

try:
    # Correlation heatmap
    numeric_cols = train.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) > 1:
        plt.figure(figsize=(10, 8))
        corr = train[numeric_cols].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', center=0)
        plt.title('Correlation Heatmap')
        plt.tight_layout()
        plt.savefig('./work/correlation_heatmap.png', dpi=100)
        plt.close()
        print("  -> Saved correlation_heatmap.png")
except Exception as e:
    print(f"  Correlation heatmap error: {e}")

# Boxplot: demand vs RoadType
if 'RoadType' in train.columns:
    try:
        plt.figure(figsize=(10, 5))
        sns.boxplot(data=train, x='RoadType', y='demand')
        plt.title('Demand vs RoadType')
        plt.tight_layout()
        plt.savefig('./work/demand_vs_roadtype.png', dpi=100)
        plt.close()
        print("  -> Saved demand_vs_roadtype.png")
    except Exception as e:
        print(f"  RoadType boxplot error: {e}")

# Boxplot: demand vs Weather
if 'Weather' in train.columns:
    try:
        plt.figure(figsize=(10, 5))
        sns.boxplot(data=train, x='Weather', y='demand')
        plt.title('Demand vs Weather')
        plt.tight_layout()
        plt.savefig('./work/demand_vs_weather.png', dpi=100)
        plt.close()
        print("  -> Saved demand_vs_weather.png")
    except Exception as e:
        print(f"  Weather boxplot error: {e}")

# Scatter: Temperature vs demand
if 'Temperature' in train.columns:
    try:
        plt.figure(figsize=(8, 5))
        plt.scatter(train['Temperature'].dropna(), 
                    train.loc[train['Temperature'].notna(), 'demand'], 
                    alpha=0.2, s=3)
        plt.xlabel('Temperature')
        plt.ylabel('Demand')
        plt.title('Temperature vs Demand')
        plt.tight_layout()
        plt.savefig('./work/temp_vs_demand.png', dpi=100)
        plt.close()
        print("  -> Saved temp_vs_demand.png")
    except Exception as e:
        print(f"  Temperature scatter error: {e}")

# Demand by hour (extract from timestamp)
if 'timestamp' in train.columns:
    try:
        train_temp = train.copy()
        # timestamp might be in H:M format
        if train_temp['timestamp'].dtype == 'object':
            parts = train_temp['timestamp'].str.split(':', expand=True)
            train_temp['_hour'] = parts[0].astype(int)
        else:
            train_temp['_hour'] = pd.to_datetime(train_temp['timestamp']).dt.hour
        
        hour_demand = train_temp.groupby('_hour')['demand'].mean()
        plt.figure(figsize=(10, 5))
        hour_demand.plot(kind='bar')
        plt.title('Mean Demand by Hour')
        plt.xlabel('Hour')
        plt.ylabel('Mean Demand')
        plt.tight_layout()
        plt.savefig('./work/demand_by_hour.png', dpi=100)
        plt.close()
        print("  -> Saved demand_by_hour.png")
        del train_temp
    except Exception as e:
        print(f"  Hour plot error: {e}")

# Demand by day
if 'day' in train.columns:
    try:
        day_demand = train.groupby('day')['demand'].mean()
        plt.figure(figsize=(10, 5))
        day_demand.plot(kind='bar')
        plt.title('Mean Demand by Day')
        plt.xlabel('Day')
        plt.ylabel('Mean Demand')
        plt.tight_layout()
        plt.savefig('./work/demand_by_day.png', dpi=100)
        plt.close()
        print("  -> Saved demand_by_day.png")
    except Exception as e:
        print(f"  Day plot error: {e}")

# Outlier visualization
try:
    plt.figure(figsize=(8, 5))
    sns.boxplot(y=train['demand'])
    plt.title('Demand Outlier Visualization (IQR)')
    plt.tight_layout()
    plt.savefig('./work/demand_outliers.png', dpi=100)
    plt.close()
    print("  -> Saved demand_outliers.png")
except Exception as e:
    print(f"  Outlier plot error: {e}")

# =============================================================
# STEP 4 — COMBINE TRAIN + TEST FOR PREPROCESSING
# =============================================================
print("\n" + "=" * 60)
print("STEP 4 - COMBINE TRAIN + TEST FOR PREPROCESSING")
print("=" * 60)

y = train['demand'].copy()
n_train = len(train)

train['_source'] = 'train'
test['_source']  = 'test'

# Combine (test won't have demand column)
combined = pd.concat(
    [train.drop('demand', axis=1), test], axis=0
).reset_index(drop=True)

print("Combined shape:", combined.shape)
print("Train rows:", n_train)
print("Test rows :", len(test))

# =============================================================
# STEP 5 — MISSING VALUE TREATMENT
# =============================================================
print("\n" + "=" * 60)
print("STEP 5 - MISSING VALUE TREATMENT")
print("=" * 60)

train_combined = combined[combined['_source'] == 'train']

for col in combined.columns:
    if col in ['_source', 'Index']:
        continue
    if combined[col].isnull().sum() > 0:
        if combined[col].dtype in ['float64', 'int64']:
            fill_val = train_combined[col].median()
        else:
            fill_val = train_combined[col].mode()[0]
        combined[col] = combined[col].fillna(fill_val)
        print(f"  Filled '{col}' with {fill_val}")

print("Missing after treatment:", combined.isnull().sum().sum())

# =============================================================
# STEP 6 — OUTLIER TREATMENT (ONLY ON TRAIN TARGET)
# =============================================================
print("\n" + "=" * 60)
print("STEP 6 - OUTLIER TREATMENT (on target only)")
print("=" * 60)

Q1  = y.quantile(0.25)
Q3  = y.quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

print(f"  Demand before clip: min={y.min():.6f}, max={y.max():.6f}")
y = y.clip(lower=lower, upper=upper)
print(f"  Demand after clip : min={y.min():.6f}, max={y.max():.6f}")
print(f"  Outliers clipped  : {((y == lower) | (y == upper)).sum()}")

# =============================================================
# STEP 7 — FEATURE ENGINEERING
# =============================================================
print("\n" + "=" * 60)
print("STEP 7 - FEATURE ENGINEERING")
print("=" * 60)

# --- Geohash Features ---
if 'geohash' in combined.columns:
    print("  Creating geohash features...")
    
    def safe_decode(gh):
        try:
            decoded = pgh.decode(gh)
            return pd.Series([decoded[0], decoded[1]])
        except Exception:
            return pd.Series([np.nan, np.nan])
    
    combined[['lat', 'lon']] = combined['geohash'].apply(safe_decode)
    combined['geohash_len'] = combined['geohash'].str.len()
    
    # Geohash prefix features for grouping nearby locations
    combined['geohash_4'] = combined['geohash'].str[:4]
    combined['geohash_5'] = combined['geohash'].str[:5]
    
    print(f"    lat range: {combined['lat'].min():.4f} - {combined['lat'].max():.4f}")
    print(f"    lon range: {combined['lon'].min():.4f} - {combined['lon'].max():.4f}")
else:
    print("  WARNING: 'geohash' column NOT found, skipping.")

# --- Timestamp Features ---
if 'timestamp' in combined.columns:
    print("  Creating timestamp features...")
    
    # Detect timestamp format: could be "H:M" or datetime
    sample_ts = str(combined['timestamp'].iloc[0])
    if ':' in sample_ts and len(sample_ts) <= 5:
        # Format is "H:M" like "0:0", "14:30"
        parts = combined['timestamp'].astype(str).str.split(':', expand=True)
        combined['hour']   = parts[0].astype(int)
        combined['minute'] = parts[1].astype(int)
        print("    Detected H:M format")
    else:
        combined['timestamp'] = pd.to_datetime(combined['timestamp'], errors='coerce')
        combined['hour']   = combined['timestamp'].dt.hour
        combined['minute'] = combined['timestamp'].dt.minute
        print("    Detected datetime format")
    
    combined['day_of_week'] = combined['day'] % 7 if 'day' in combined.columns else 0
    combined['is_weekend']  = combined['day_of_week'].isin([5, 6]).astype(int)
    combined['is_peak']     = combined['hour'].isin([7, 8, 9, 17, 18, 19, 20]).astype(int)
    combined['is_night']    = combined['hour'].isin([22, 23, 0, 1, 2, 3, 4]).astype(int)
    
    # Time of day as minutes since midnight
    combined['minutes_since_midnight'] = combined['hour'] * 60 + combined['minute']
    
    def part_of_day(h):
        if 0  <= h < 6:  return 0  # night
        if 6  <= h < 12: return 1  # morning
        if 12 <= h < 17: return 2  # afternoon
        return 3                    # evening
    combined['part_of_day'] = combined['hour'].apply(part_of_day)
    
    print(f"    hour range: {combined['hour'].min()} - {combined['hour'].max()}")
    print(f"    minute range: {combined['minute'].min()} - {combined['minute'].max()}")
else:
    print("  WARNING: 'timestamp' column NOT found, skipping.")

# --- Cyclical Encoding ---
if 'hour' in combined.columns:
    combined['hour_sin'] = np.sin(2 * np.pi * combined['hour'] / 24)
    combined['hour_cos'] = np.cos(2 * np.pi * combined['hour'] / 24)
    print("  Created cyclical hour features")

if 'day_of_week' in combined.columns:
    combined['dow_sin'] = np.sin(2 * np.pi * combined['day_of_week'] / 7)
    combined['dow_cos'] = np.cos(2 * np.pi * combined['day_of_week'] / 7)
    print("  Created cyclical day-of-week features")

if 'minute' in combined.columns:
    combined['min_sin'] = np.sin(2 * np.pi * combined['minute'] / 60)
    combined['min_cos'] = np.cos(2 * np.pi * combined['minute'] / 60)
    print("  Created cyclical minute features")

# --- Geohash Aggregation Features (TRAIN ONLY) ---
if 'geohash' in combined.columns:
    print("  Creating geohash aggregation features (from train only)...")
    
    train_rows = combined[combined['_source'] == 'train'].copy()
    train_rows['demand'] = y.values
    
    global_mean = y.mean()
    global_std  = y.std()
    
    # Per-geohash stats
    geo_agg = train_rows.groupby('geohash')['demand'].agg(
        geohash_mean   = 'mean',
        geohash_median = 'median',
        geohash_std    = 'std',
        geohash_count  = 'count'
    ).reset_index()
    
    combined = combined.merge(geo_agg, on='geohash', how='left')
    combined['geohash_mean']  = combined['geohash_mean'].fillna(global_mean)
    combined['geohash_median'] = combined['geohash_median'].fillna(global_mean)
    combined['geohash_std']   = combined['geohash_std'].fillna(global_std)
    combined['geohash_count'] = combined['geohash_count'].fillna(0)
    
    # Per-geohash_4 (coarser region) stats
    geo4_agg = train_rows.groupby(train_rows['geohash'].str[:4])['demand'].agg(
        geohash4_mean   = 'mean',
        geohash4_count  = 'count'
    ).reset_index()
    geo4_agg.rename(columns={'geohash': 'geohash_4'}, inplace=True)
    combined = combined.merge(geo4_agg, on='geohash_4', how='left')
    combined['geohash4_mean']  = combined['geohash4_mean'].fillna(global_mean)
    combined['geohash4_count'] = combined['geohash4_count'].fillna(0)
    
    # Per-geohash_5 stats
    geo5_agg = train_rows.groupby(train_rows['geohash'].str[:5])['demand'].agg(
        geohash5_mean   = 'mean',
        geohash5_count  = 'count'
    ).reset_index()
    geo5_agg.rename(columns={'geohash': 'geohash_5'}, inplace=True)
    combined = combined.merge(geo5_agg, on='geohash_5', how='left')
    combined['geohash5_mean']  = combined['geohash5_mean'].fillna(global_mean)
    combined['geohash5_count'] = combined['geohash5_count'].fillna(0)
    
    print(f"    Geo agg features: geohash_mean, geohash_median, geohash_std, geohash_count, etc.")

# --- Hour-level Aggregation from Train ---
if 'hour' in combined.columns:
    print("  Creating hour aggregation features (from train only)...")
    
    train_rows_h = combined[combined['_source'] == 'train'].copy()
    train_rows_h['demand'] = y.values
    
    hour_agg = train_rows_h.groupby('hour')['demand'].agg(
        hour_mean_demand   = 'mean',
        hour_median_demand = 'median',
        hour_std_demand    = 'std'
    ).reset_index()
    combined = combined.merge(hour_agg, on='hour', how='left')
    combined['hour_mean_demand']   = combined['hour_mean_demand'].fillna(global_mean)
    combined['hour_median_demand'] = combined['hour_median_demand'].fillna(global_mean)
    combined['hour_std_demand']    = combined['hour_std_demand'].fillna(global_std)

# --- Day-level Aggregation from Train ---
if 'day' in combined.columns:
    print("  Creating day aggregation features (from train only)...")
    
    train_rows_d = combined[combined['_source'] == 'train'].copy()
    train_rows_d['demand'] = y.values
    
    day_agg = train_rows_d.groupby('day')['demand'].agg(
        day_mean_demand = 'mean',
        day_std_demand  = 'std'
    ).reset_index()
    combined = combined.merge(day_agg, on='day', how='left')
    combined['day_mean_demand'] = combined['day_mean_demand'].fillna(global_mean)
    combined['day_std_demand']  = combined['day_std_demand'].fillna(global_std)

# --- Geohash x Hour interaction aggregation ---
if 'geohash' in combined.columns and 'hour' in combined.columns:
    print("  Creating geohash x hour aggregation features (from train only)...")
    
    train_rows_gh = combined[combined['_source'] == 'train'].copy()
    train_rows_gh['demand'] = y.values
    
    geo_hour_agg = train_rows_gh.groupby(['geohash', 'hour'])['demand'].agg(
        geo_hour_mean = 'mean'
    ).reset_index()
    combined = combined.merge(geo_hour_agg, on=['geohash', 'hour'], how='left')
    combined['geo_hour_mean'] = combined['geo_hour_mean'].fillna(
        combined['geohash_mean']  # fallback to geohash mean
    )

# --- RoadType x Hour interaction aggregation ---
if 'RoadType' in combined.columns and 'hour' in combined.columns:
    print("  Creating RoadType x Hour aggregation features (from train only)...")
    
    train_rows_rh = combined[combined['_source'] == 'train'].copy()
    train_rows_rh['demand'] = y.values
    
    road_hour_agg = train_rows_rh.groupby(['RoadType', 'hour'])['demand'].agg(
        road_hour_mean = 'mean'
    ).reset_index()
    combined = combined.merge(road_hour_agg, on=['RoadType', 'hour'], how='left')
    combined['road_hour_mean'] = combined['road_hour_mean'].fillna(global_mean)

# --- Interaction Features ---
if 'NumberofLanes' in combined.columns:
    combined['peak_x_lanes']    = combined['is_peak'] * combined['NumberofLanes']
    combined['weekend_x_peak']  = combined['is_weekend'] * combined['is_peak']
    combined['night_x_lanes']   = combined['is_night'] * combined['NumberofLanes']
    combined['weekend_x_lanes'] = combined['is_weekend'] * combined['NumberofLanes']
    print("  Created interaction features (lanes, peak, weekend, night)")

# --- LargeVehicles binary encoding ---
if 'LargeVehicles' in combined.columns:
    combined['LargeVehicles_bin'] = (combined['LargeVehicles'] == 'Allowed').astype(int)
    print("  Created LargeVehicles_bin")

# --- Landmarks binary encoding ---
if 'Landmarks' in combined.columns:
    combined['Landmarks_bin'] = (combined['Landmarks'] == 'Yes').astype(int)
    print("  Created Landmarks_bin")

# --- Additional interactions with binary features ---
if 'LargeVehicles_bin' in combined.columns and 'NumberofLanes' in combined.columns:
    combined['large_veh_x_lanes'] = combined['LargeVehicles_bin'] * combined['NumberofLanes']

if 'Landmarks_bin' in combined.columns and 'is_peak' in combined.columns:
    combined['landmarks_x_peak'] = combined['Landmarks_bin'] * combined['is_peak']

# --- Temperature features ---
if 'Temperature' in combined.columns:
    combined['temp_sq'] = combined['Temperature'] ** 2
    combined['temp_bin'] = pd.cut(
        combined['Temperature'], 
        bins=[-np.inf, 10, 20, 30, 40, np.inf],
        labels=[0, 1, 2, 3, 4]
    ).astype(float)
    combined['temp_bin'] = combined['temp_bin'].fillna(2)  # Fill with median bin
    print("  Created temperature features")

print(f"\n  Combined shape after features: {combined.shape}")

# =============================================================
# STEP 8 — ENCODING CATEGORICAL COLUMNS
# =============================================================
print("\n" + "=" * 60)
print("STEP 8 - ENCODING CATEGORICAL COLUMNS")
print("=" * 60)

cat_cols = combined.select_dtypes(include=['object']).columns.tolist()
cat_cols = [c for c in cat_cols if c not in ['_source', 'geohash', 'geohash_4', 'geohash_5']]

print("Categorical columns to encode:", cat_cols)

encoders = {}
train_mask = combined['_source'] == 'train'

for col in cat_cols:
    le = LabelEncoder()
    # Fit only on train
    le.fit(combined.loc[train_mask, col].astype(str))
    
    # Handle unseen categories in test gracefully
    vals = combined[col].astype(str).copy()
    mask = ~vals.isin(le.classes_)
    if mask.any():
        print(f"    '{col}': {mask.sum()} unseen values -> mapped to most frequent class")
        vals[mask] = le.classes_[0]
    
    combined[col] = le.transform(vals)
    encoders[col] = le
    print(f"  Encoded: {col}")

# =============================================================
# STEP 9 — FINAL FEATURE SELECTION
# =============================================================
print("\n" + "=" * 60)
print("STEP 9 - FINAL FEATURE SELECTION")
print("=" * 60)

drop_cols = ['_source', 'geohash', 'geohash_4', 'geohash_5', 'timestamp', 'Index']
drop_cols = [c for c in drop_cols if c in combined.columns]
combined.drop(columns=drop_cols, inplace=True)

print("Final combined shape:", combined.shape)
print("Final features:", combined.columns.tolist())

# Split back
X      = combined.iloc[:n_train].copy()
X_test = combined.iloc[n_train:].copy()

print("X shape     :", X.shape)
print("X_test shape:", X_test.shape)
print("y shape     :", y.shape)

# Ensure all numeric
for col in X.columns:
    X[col]      = pd.to_numeric(X[col], errors='coerce')
    X_test[col] = pd.to_numeric(X_test[col], errors='coerce')

# Fill any remaining NaN from conversion
X      = X.fillna(0)
X_test = X_test.fillna(0)

# Final check
assert not X.isnull().any().any(),       "NaN found in X!"
assert not X_test.isnull().any().any(),  "NaN found in X_test!"
assert not np.isinf(X.values).any(),     "Inf found in X!"
assert not np.isinf(X_test.values).any(),"Inf found in X_test!"
print("[OK] No NaN or Inf values in features")

# =============================================================
# STEP 10 — QUICK FEATURE IMPORTANCE CHECK
# =============================================================
print("\n" + "=" * 60)
print("STEP 10 - QUICK FEATURE IMPORTANCE CHECK")
print("=" * 60)

quick_model = lgb.LGBMRegressor(
    n_estimators=200, random_state=42, verbose=-1
)
quick_model.fit(X, y)

importance_df = pd.DataFrame({
    'feature'   : X.columns,
    'importance': quick_model.feature_importances_
}).sort_values('importance', ascending=False)

print(importance_df.to_string(index=False))

# Plot top 20 features
plt.figure(figsize=(10, 8))
sns.barplot(data=importance_df.head(20), x='importance', y='feature')
plt.title('Top 20 Feature Importances')
plt.tight_layout()
plt.savefig('./work/feature_importance.png', dpi=100)
plt.close()
print("  -> Saved feature_importance.png")

# Drop zero-importance features
zero_imp_features = importance_df[
    importance_df['importance'] == 0
]['feature'].tolist()

if zero_imp_features:
    print(f"\n  Dropping zero-importance features: {zero_imp_features}")
    X      = X.drop(columns=zero_imp_features)
    X_test = X_test.drop(columns=zero_imp_features)
    print(f"  X shape after drop: {X.shape}")
else:
    print("  No zero-importance features found - keeping all.")

# =============================================================
# STEP 11 — CROSS-VALIDATED TRAINING
# =============================================================
print("\n" + "=" * 60)
print("STEP 11 - 5-FOLD CROSS-VALIDATED TRAINING")
print("=" * 60)

N_FOLDS = 5
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

# Storage
oof_lgbm = np.zeros(len(X))
oof_xgb  = np.zeros(len(X))
oof_cat  = np.zeros(len(X))

test_lgbm = np.zeros(len(X_test))
test_xgb  = np.zeros(len(X_test))
test_cat  = np.zeros(len(X_test))

lgbm_scores, xgb_scores, cat_scores = [], [], []

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"\n{'=' * 50}")
    print(f"FOLD {fold + 1} / {N_FOLDS}")
    print(f"{'=' * 50}")
    
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # -- LightGBM --
    lgbm_model = lgb.LGBMRegressor(
        n_estimators      = 3000,
        learning_rate     = 0.03,
        max_depth         = 7,
        num_leaves        = 63,
        min_child_samples = 50,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        reg_alpha         = 0.1,
        reg_lambda        = 0.1,
        random_state      = 42,
        verbose           = -1
    )
    lgbm_model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(500)
        ]
    )
    oof_lgbm[val_idx]  = lgbm_model.predict(X_val)
    test_lgbm         += lgbm_model.predict(X_test) / N_FOLDS
    fold_r2            = r2_score(y_val, oof_lgbm[val_idx])
    lgbm_scores.append(fold_r2)
    print(f"  LGBM  -> Val R2: {fold_r2:.4f}")
    
    # -- XGBoost --
    xgb_model = xgb.XGBRegressor(
        n_estimators          = 3000,
        learning_rate         = 0.03,
        max_depth             = 6,
        subsample             = 0.8,
        colsample_bytree      = 0.8,
        reg_alpha             = 0.1,
        reg_lambda            = 0.1,
        early_stopping_rounds = 50,
        random_state          = 42,
        verbosity             = 0
    )
    xgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    oof_xgb[val_idx]  = xgb_model.predict(X_val)
    test_xgb         += xgb_model.predict(X_test) / N_FOLDS
    fold_r2            = r2_score(y_val, oof_xgb[val_idx])
    xgb_scores.append(fold_r2)
    print(f"  XGB   -> Val R2: {fold_r2:.4f}")
    
    # -- CatBoost --
    cat_model = CatBoostRegressor(
        iterations            = 3000,
        learning_rate         = 0.03,
        depth                 = 6,
        l2_leaf_reg           = 3,
        early_stopping_rounds = 50,
        random_seed           = 42,
        verbose               = False
    )
    cat_model.fit(
        X_tr, y_tr,
        eval_set=(X_val, y_val),
        use_best_model=True
    )
    oof_cat[val_idx]  = cat_model.predict(X_val)
    test_cat         += cat_model.predict(X_test) / N_FOLDS
    fold_r2            = r2_score(y_val, oof_cat[val_idx])
    cat_scores.append(fold_r2)
    print(f"  CAT   -> Val R2: {fold_r2:.4f}")

# Summary
print(f"\n{'=' * 60}")
print(f"CV SUMMARY")
print(f"{'=' * 60}")
print(f"LGBM  CV R2: {np.mean(lgbm_scores):.4f} +/- {np.std(lgbm_scores):.4f}")
print(f"XGB   CV R2: {np.mean(xgb_scores):.4f} +/- {np.std(xgb_scores):.4f}")
print(f"CAT   CV R2: {np.mean(cat_scores):.4f} +/- {np.std(cat_scores):.4f}")

# =============================================================
# STEP 12 — ENSEMBLE
# =============================================================
print("\n" + "=" * 60)
print("STEP 12 - ENSEMBLE")
print("=" * 60)

# Weighted ensemble based on CV scores
w_lgbm = np.mean(lgbm_scores)
w_xgb  = np.mean(xgb_scores)
w_cat  = np.mean(cat_scores)
total  = w_lgbm + w_xgb + w_cat

w_lgbm /= total
w_xgb  /= total
w_cat  /= total

print(f"Ensemble weights -> LGBM:{w_lgbm:.3f}  XGB:{w_xgb:.3f}  CAT:{w_cat:.3f}")

# OOF ensemble score
oof_ensemble = (
    oof_lgbm * w_lgbm +
    oof_xgb  * w_xgb  +
    oof_cat  * w_cat
)
oof_r2 = r2_score(y, oof_ensemble)
print(f"Ensemble OOF R2  : {oof_r2:.4f}")
print(f"Hackathon Score  : {max(0, 100 * oof_r2):.2f}")

# Also try simple averaging
oof_simple_avg = (oof_lgbm + oof_xgb + oof_cat) / 3
simple_r2 = r2_score(y, oof_simple_avg)
print(f"\nSimple Average OOF R2: {simple_r2:.4f}")

# Use the better ensemble strategy
if simple_r2 > oof_r2:
    print("  -> Simple average is better, using that.")
    final_preds = (test_lgbm + test_xgb + test_cat) / 3
    best_oof_r2 = simple_r2
else:
    print("  -> Weighted ensemble is better, using that.")
    final_preds = (
        test_lgbm * w_lgbm +
        test_xgb  * w_xgb  +
        test_cat  * w_cat
    )
    best_oof_r2 = oof_r2

# =============================================================
# STEP 13 — OVERFITTING DIAGNOSTIC
# =============================================================
print("\n" + "=" * 60)
print("STEP 13 - OVERFITTING DIAGNOSTIC")
print("=" * 60)

# Retrain quick model on reduced feature set for fair comparison
quick_model_diag = lgb.LGBMRegressor(n_estimators=200, random_state=42, verbose=-1)
quick_model_diag.fit(X, y)
train_r2 = r2_score(y, quick_model_diag.predict(X))

print(f"Quick Model Train R2 : {train_r2:.4f}")
print(f"Ensemble OOF (Val) R2: {best_oof_r2:.4f}")
print(f"Gap                  : {train_r2 - best_oof_r2:.4f}")

if train_r2 - best_oof_r2 > 0.05:
    print("WARNING: Possible Overfitting Detected!")
    print("   -> Increase reg_alpha, reg_lambda")
    print("   -> Reduce max_depth or num_leaves")
    print("   -> Reduce subsample / colsample_bytree")
elif best_oof_r2 < 0.60:
    print("WARNING: Underfitting Detected!")
    print("   -> Add more features")
    print("   -> Try higher n_estimators")
else:
    print("[OK] Model looks healthy - good generalization!")

# OOF vs Actual scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(y, oof_ensemble, alpha=0.3, s=5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel('Actual Demand')
plt.ylabel('OOF Predicted Demand')
plt.title(f'OOF Predictions vs Actual (R2={best_oof_r2:.4f})')
plt.tight_layout()
plt.savefig('./work/oof_vs_actual.png', dpi=100)
plt.close()
print("  -> Saved oof_vs_actual.png")

# =============================================================
# STEP 14 — GENERATE SUBMISSION FILE
# =============================================================
print("\n" + "=" * 60)
print("STEP 14 - GENERATE SUBMISSION FILE")
print("=" * 60)

test_original = pd.read_csv('./work/test.csv')

# Clip negative predictions
final_preds = np.clip(final_preds, 0, None)

submission = pd.DataFrame({
    'Index' : test_original['Index'],
    'demand': final_preds
})

print(f"Submission shape : {submission.shape}")
print(f"Expected shape   : ({len(test_original)}, 2)")
print(f"\nFirst 5 rows:\n{submission.head()}")
print(f"\nDemand stats:\n{submission['demand'].describe()}")

assert submission.shape[0] == len(test_original), f"Row count mismatch! Got {submission.shape[0]}, expected {len(test_original)}"
assert submission.shape[1] == 2, "Column count mismatch!"
assert list(submission.columns) == ['Index', 'demand'], f"Column name mismatch! Got {list(submission.columns)}"
assert submission.isnull().sum().sum() == 0, "NaN in submission!"

submission.to_csv('./work/submission.csv', index=False)
print("\n[OK] submission.csv saved to ./work/submission.csv")

# =============================================================
# FINAL SUMMARY
# =============================================================
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"  [OK] Data loaded and verified from ./work/")
print(f"  [OK] EDA plots generated")
print(f"  [OK] Missing values handled (train-fit only)")
print(f"  [OK] Outliers clipped on target")
print(f"  [OK] Features engineered (geohash, time, cyclical, aggregation)")
print(f"  [OK] Encoders fit on train only")
print(f"  [OK] Zero-importance features dropped")
print(f"  [OK] 5-Fold CV complete for LGBM + XGBoost + CatBoost")
print(f"  [OK] Ensemble weights auto-computed from CV scores")
print(f"  [OK] Overfitting diagnostic printed")
print(f"  [OK] submission.csv saved ({submission.shape[0]} x {submission.shape[1]})")
print(f"\n  ** Final OOF R2 Score: {best_oof_r2:.4f}")
print(f"  ** Hackathon Score   : {max(0, 100 * best_oof_r2):.2f}")
