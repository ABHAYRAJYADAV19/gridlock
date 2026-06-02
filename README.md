# 🚦 GridLock — Traffic Demand Prediction

> **Flipkart GRiD 6.0 Hackathon** — Predict traffic demand across geohash-encoded locations using temporal, spatial, and environmental features.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-Gradient%20Boosting-green.svg)](https://lightgbm.readthedocs.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Boosting-orange.svg)](https://xgboost.readthedocs.io/)
[![CatBoost](https://img.shields.io/badge/CatBoost-Gradient%20Boosting-yellow.svg)](https://catboost.ai/)

---

## 📋 Problem Statement

Given a dataset of traffic observations across various geolocations, timestamps, road types, weather conditions, and other features — predict **`demand`** (a continuous value in `[0, 1]`) for unseen test data.

**Evaluation Metric:**  
```
Score = max(0, 100 × R²(actual, predicted))
```

---

## 🏗️ Repository Structure

```
gridlock/
├── pipeline.py            # v1 — Baseline: EDA + 3-model ensemble (LGBM/XGB/CatBoost)
├── pipeline_v2.py         # v2 — Log target, Optuna tuning, stacking meta-learner
├── pipeline_v3.py         # v3 — OOF target encoding, pseudo-labeling, temporal validation
├── pipeline_v4.py         # v4 — Calibrated lookup + aggressive pseudo-labeling (best)
├── analyze.py             # Quick data analysis & feature correlation script
├── work/
│   ├── train.csv          # Training data (not tracked in git)
│   ├── test.csv           # Test data (not tracked in git)
│   ├── approach.txt       # Approach summary notes
│   ├── submission.csv     # Latest submission file
│   ├── *.png              # EDA visualizations
│   └── pipeline_v*_log.txt # Training logs
├── .gitignore
└── README.md
```

---

## 🚀 Pipeline Evolution

### v1 — Baseline (`pipeline.py`)
- Full EDA with visualizations (distribution, correlation heatmap, feature importance)
- Feature engineering: geohash decoding, timestamp parsing, cyclical encoding, target aggregation
- IQR-based outlier clipping on target
- 5-fold CV with LightGBM, XGBoost, CatBoost
- Weighted ensemble (weights ∝ CV R² scores)

### v2 — Improved (`pipeline_v2.py`)
- **Removed IQR clipping** (was destroying 8.3% of high-demand signal)
- Log-transform on target for better regression
- Richer interaction features + target encoding with Bayesian smoothing
- **Optuna hyperparameter tuning** per model (15 trials each)
- **Stacking meta-learner** (Ridge) on top of base models
- Optuna ensemble weight search (200 trials)

### v3 — OOF + Pseudo-Labeling (`pipeline_v3.py`)
- **Out-of-Fold (OOF) target encoding** to fix CV leakage
- **Pseudo-labeling** (3 rounds) to learn test distribution
- **Day-49 weighted training** (test is day 49 → 3× upweight day 49 train rows)
- **Temporal validation** (day 48 → day 49 holdout for honest R² estimate)
- Dual training: both log and raw target, then ensemble both

### v4 — Calibrated Lookup ⭐ (`pipeline_v4.py`) — **Best Version**
- **Calibrated lookup predictions** — day 48 demand × per-geohash day 49/48 ratio as features
- Full target encoding (v2-style, outperforms OOF approach)
- Training on **raw target** (no log, empirically better)
- **Aggressive pseudo-labeling** (5 rounds with weight schedule: 0.3 → 0.9)
- Multi-seed ensemble (3 seeds × LightGBM + CatBoost per round)
- Final ensemble: weighted blend of base + rounds 3, 4, 5

---

## ⚙️ Feature Engineering

| Category | Features |
|----------|----------|
| **Spatial** | Geohash → lat/lon, geohash prefix (4/5 char), frequency encoding, lat × lon interaction |
| **Temporal** | Hour, minute, time_slot (15-min), day_of_week, cyclical sin/cos encodings |
| **Binary flags** | is_peak, is_morning, is_night, is_weekend, part_of_day |
| **Road/Traffic** | NumberOfLanes, LargeVehicles (binary), Landmarks (binary), RoadType (encoded) |
| **Weather** | Weather condition (encoded), Temperature, Temperature² |
| **Interactions** | peak × lanes, morning × lanes, night × lanes, large_vehicles × lanes, landmarks × lanes |
| **Target Encoding** | Per-geohash, per-hour, per-slot, geohash × hour, geohash × slot, RoadType × hour, Weather × hour |
| **Calibration** | Day-48 slot/hour lookup, calibration ratio, calibrated predictions, deviation features |

---

## 🧠 Models Used

| Model | Key Hyperparameters |
|-------|-------------------|
| **LightGBM** | `n_estimators=3000`, `lr=0.03`, `max_depth=7`, `num_leaves=63`, `subsample=0.8` |
| **XGBoost** | `n_estimators=3000`, `lr=0.03`, `max_depth=7`, `subsample=0.8`, `min_child_weight=10` |
| **CatBoost** | `iterations=3000`, `lr=0.03`, `depth=7`, `l2_leaf_reg=3` |

All models use **early stopping** (patience=100) and **sample weights** (3× for day-49 rows).

---

## 📊 Key Insights

1. **Day 49 demand is ~1.7× higher than day 48** at the same hours — the calibration ratio captures this shift
2. **No IQR clipping** — clipping destroyed 8.3% of signal in high-demand geohashes
3. **Raw target > log target** — despite skewed distribution, raw target performs better with calibration features
4. **Pseudo-labeling converges** — 5 rounds with increasing weights stabilize predictions
5. **Target encoding on full data > OOF** — when combined with calibration features, full TE outperforms OOF TE

---

## 🔧 Setup & Usage

### Prerequisites

```bash
pip install pandas numpy scikit-learn lightgbm xgboost catboost pygeohash optuna matplotlib seaborn
```

### Run

1. Place `train.csv` and `test.csv` in the `work/` directory
2. Run the desired pipeline:

```bash
# Quick analysis
python analyze.py

# Run best pipeline (v4)
python pipeline_v4.py
```

3. Output submissions will be saved to `work/submission*.csv`

---

## 📁 Output Files

| File | Description |
|------|-------------|
| `submission.csv` | Latest best submission (auto-overwritten) |
| `submission_v4.csv` | v4 final blend (base + pseudo rounds 3-5) |
| `submission_v4_base_only.csv` | v4 base ensemble only (no pseudo-labeling) |
| `submission_v4_pseudo_only.csv` | v4 last pseudo-labeling round only |

---

## 🛠️ Tech Stack

- **Python 3.8+**
- **pandas** / **numpy** — Data manipulation
- **scikit-learn** — KFold, metrics, preprocessing
- **LightGBM** / **XGBoost** / **CatBoost** — Gradient boosting models
- **Optuna** — Hyperparameter optimization & ensemble weight search
- **pygeohash** — Geohash decoding
- **matplotlib** / **seaborn** — EDA visualizations

---

## 👤 Author

**Abhay Raj Yadav**  
Flipkart GRiD 6.0 — Traffic Demand Prediction Challenge

---

## 📜 License

This project is for educational and competition purposes.
