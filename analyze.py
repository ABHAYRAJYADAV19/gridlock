import pandas as pd, numpy as np

train = pd.read_csv('./work/train.csv')
test = pd.read_csv('./work/test.csv')

# Check day distribution
print('Train day value counts:')
print(train['day'].value_counts())

d48 = train[train['day']==48]
d49 = train[train['day']==49]
print(f'\nDay 48: {len(d48)} rows, demand mean={d48["demand"].mean():.6f}, std={d48["demand"].std():.6f}')
print(f'Day 49: {len(d49)} rows, demand mean={d49["demand"].mean():.6f}, std={d49["demand"].std():.6f}')

ts = train['timestamp'].str.split(':', expand=True)
train['hour'] = ts[0].astype(int)
train['minute'] = ts[1].astype(int)
print(f'\nHour range: {train["hour"].min()} - {train["hour"].max()}')
print(f'Minute range: {train["minute"].min()} - {train["minute"].max()}')

ts_t = test['timestamp'].str.split(':', expand=True)
test['hour'] = ts_t[0].astype(int)
test['minute'] = ts_t[1].astype(int)
print(f'\nTest hour range: {test["hour"].min()} - {test["hour"].max()}')
print(f'Test minute range: {test["minute"].min()} - {test["minute"].max()}')

print(f'\nUnique timestamps (train): {train["timestamp"].nunique()}')
print(f'Unique hour-minute combos (train): {train.groupby(["hour","minute"]).ngroups}')
print(f'Unique hour-minute combos (test): {test.groupby(["hour","minute"]).ngroups}')

train_ts = set(train['timestamp'])
test_ts = set(test['timestamp'])
print(f'Test timestamps in train: {len(test_ts & train_ts)} / {len(test_ts)}')

print(f'\nTest geohashes NOT in train: {len(set(test["geohash"]) - set(train["geohash"]))}')
unseen = set(test['geohash']) - set(train['geohash'])
print(f'Unseen geohashes: {unseen}')

import pygeohash as pgh
lat_lon = train['geohash'].apply(lambda gh: pd.Series(pgh.decode(gh), index=['lat','lon']))
train['lat'] = lat_lon['lat']
train['lon'] = lat_lon['lon']
numeric = ['day','hour','minute','NumberofLanes','Temperature','lat','lon']
for c in numeric:
    if c in train.columns:
        corr = train[c].corr(train['demand'])
        print(f'  Corr demand vs {c}: {corr:.4f}')

# Check demand distribution by day more carefully
print('\n--- Demand by Day x Hour (mean) ---')
pivot = train.groupby(['day','hour'])['demand'].mean().unstack(level=0)
print(pivot.head(24))

# Check if day 49 data in train overlaps with test hours
d49_train = train[train['day']==49]
d49_test_hours = set(test['hour'].unique())
d49_train_hours = set(d49_train['hour'].unique())
print(f'\nDay 49 train hours: {sorted(d49_train_hours)}')
print(f'Day 49 test hours: {sorted(d49_test_hours)}')
print(f'Overlap: {sorted(d49_train_hours & d49_test_hours)}')

# IQR clipping impact
Q1 = train['demand'].quantile(0.25)
Q3 = train['demand'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
clipped = ((train['demand'] < lower) | (train['demand'] > upper)).sum()
print(f'\nIQR bounds: [{lower:.6f}, {upper:.6f}]')
print(f'Values clipped: {clipped} ({100*clipped/len(train):.1f}%)')
print(f'Max demand: {train["demand"].max():.6f}')
