# Run with: python3.12 model.py

import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import shap
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

print("Loading cleaned data...")
df = pd.read_csv("cleaned_data.csv")

# ── 1. Position groups ────────────────────────────────────────────────────────
POSITION_GROUP_MAP = {
    "Goalkeeper": "GK",
    "Defender":   "DEF",
    "Midfield":   "MID",
    "Attack":     "FWD",
}
df["position_group"] = df["position"].map(POSITION_GROUP_MAP).fillna("FWD")

# ── 2. Global preprocessing ───────────────────────────────────────────────────
df["foot"]                   = df["foot"].fillna("unknown")
df["sub_position"]           = df["sub_position"].fillna("Unknown")
df["height_in_cm"]           = df["height_in_cm"].fillna(df["height_in_cm"].median())
df["current_club_name"]      = df["current_club_name"].fillna("Unknown")
df["country_of_citizenship"] = df["country_of_citizenship"].fillna("Unknown")

# Contract: fill NaN (no data yet) with median so existing model runs unchanged.
# Once data_prep.py populates real values the median shifts and the feature fires.
CONTRACT_REF_DATE = pd.Timestamp("2025-09-01")
if "months_to_contract_end" not in df.columns or df["months_to_contract_end"].isna().all():
    df["months_to_contract_end"] = np.nan
contract_median = df["months_to_contract_end"].median()
if pd.isna(contract_median):
    contract_median = 24.0   # assume 2 years if entirely unknown
df["months_to_contract_end"] = df["months_to_contract_end"].fillna(contract_median)

# ── 3. Foot label encoding ────────────────────────────────────────────────────
foot_enc = LabelEncoder()
df["foot_enc"] = foot_enc.fit_transform(df["foot"].astype(str))

# ── 4. EU/EEA citizenship feature ─────────────────────────────────────────────
# Non-EU players require work permits for UK clubs, which affects marketability
EU_EEA_COUNTRIES = {
    "Austria", "Belgium", "Bulgaria", "Croatia", "Cyprus", "Czech Republic",
    "Denmark", "Estonia", "Finland", "France", "Germany", "Greece", "Hungary",
    "Ireland", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta",
    "Netherlands", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia",
    "Spain", "Sweden", "Iceland", "Liechtenstein", "Norway", "Switzerland",
}
df["is_eu_citizen"] = df["country_of_citizenship"].apply(
    lambda x: 1 if x in EU_EEA_COUNTRIES else 0
)

# ── 5. Target encodings (mean log-value per category) ────────────────────────
# Using full dataset for final model — CV loop below recomputes per fold
log_y           = np.log1p(df["market_value_in_eur"].values)
global_mean_log = float(np.mean(log_y))

subpos_enc_map = (
    df.groupby("sub_position")["market_value_in_eur"]
    .apply(lambda x: float(np.mean(np.log1p(x))))
    .to_dict()
)
df["sub_position_enc"] = df["sub_position"].map(subpos_enc_map).fillna(global_mean_log)

club_enc_map = (
    df.groupby("current_club_name")["market_value_in_eur"]
    .apply(lambda x: float(np.mean(np.log1p(x))))
    .to_dict()
)
df["club_enc"] = df["current_club_name"].map(club_enc_map).fillna(global_mean_log)

# ── 6. Derived features ───────────────────────────────────────────────────────
df["goal_contributions_per_90"] = (
    (df["goals"] + df["assists"]) / df["minutes_played"].clip(lower=1) * 90
)
# Age has a non-linear peak (~24-27 yrs), quadratic term captures the curve
df["age_squared"] = df["age"] ** 2
# Availability: fraction of a 38-game season the player appeared in
df["availability"] = (df["appearances"] / 38).clip(upper=1.0)

# FPL advanced stats — NaN where player not matched; XGBoost/LightGBM handle
# NaN natively so no imputation needed. Ensure float dtype.
# FPL stats (saves, clean_sheets, xG, xA) are fetched by fetch_fpl_data.py
# but excluded from the model: at 64% match rate the 36% imputed values add
# more noise than signal (xG ≈ goals at season level; saves require GK coverage).
# Re-enable once match rate exceeds ~85%.

# ── 7. Position-specific feature sets ─────────────────────────────────────────
COMMON = [
    "age", "age_squared", "height_in_cm",
    "appearances", "minutes_played", "minutes_per_game",
    "availability",
    "yellow_cards", "red_cards",
    "foot_enc", "sub_position_enc", "club_enc", "is_eu_citizen",
    "months_to_contract_end",
]
OUTFIELD = [
    "goals", "assists", "goals_per_90", "assists_per_90",
    "goal_contributions_per_90",
]

POSITION_FEATURES = {
    "GK":  COMMON,
    "DEF": COMMON + OUTFIELD,
    "MID": COMMON + OUTFIELD,
    "FWD": COMMON + OUTFIELD,
}

DISPLAY_NAMES = {
    "foot_enc":                  "foot",
    "sub_position_enc":          "sub_position",
    "club_enc":                  "club",
    "age_squared":               "age²",
    "goal_contributions_per_90": "g+a/90",
    "is_eu_citizen":             "EU citizen",
    "months_to_contract_end":    "contract months left",
}

# ── 8. Metric helpers ─────────────────────────────────────────────────────────
def mape(y_true_log, y_pred_log):
    """Standard MAPE — can be large for cheap players."""
    yt = np.expm1(y_true_log)
    yp = np.expm1(y_pred_log)
    return float(np.mean(np.abs((yt - yp) / yt)) * 100)


def wmape(y_true_log, y_pred_log):
    """Weighted MAPE: sum(|pred-actual|)/sum(actual)*100.
    Gives more weight to expensive players — better for skewed transfer values."""
    yt = np.expm1(y_true_log)
    yp = np.expm1(y_pred_log)
    return float(np.sum(np.abs(yt - yp)) / np.sum(yt) * 100)


def median_ape(y_true_log, y_pred_log):
    """Median absolute percentage error — robust to outliers."""
    yt = np.expm1(y_true_log)
    yp = np.expm1(y_pred_log)
    return float(np.median(np.abs((yt - yp) / yt)) * 100)


# ── 9. Optuna hyperparameter tuning (k-fold CV optimising wMAPE) ──────────────
def tune_xgb(X, y, n_trials):
    def objective(trial):
        params = dict(
            n_estimators     = trial.suggest_int("n_estimators", 200, 1000),
            learning_rate    = trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            max_depth        = trial.suggest_int("max_depth", 3, 7),
            subsample        = trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0),
            min_child_weight = trial.suggest_int("min_child_weight", 1, 10),
            reg_alpha        = trial.suggest_float("reg_alpha", 0.0, 2.0),
            reg_lambda       = trial.suggest_float("reg_lambda", 0.5, 5.0),
            random_state=42,
        )
        n_splits = 3 if len(X) < 60 else 5
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = []
        for tr_idx, val_idx in kf.split(X):
            m = XGBRegressor(**params)
            m.fit(X[tr_idx], y[tr_idx], verbose=False)
            scores.append(wmape(y[val_idx], m.predict(X[val_idx])))
        return float(np.mean(scores))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def tune_lgbm(X, y, n_trials):
    def objective(trial):
        params = dict(
            n_estimators      = trial.suggest_int("n_estimators", 200, 1000),
            learning_rate     = trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            max_depth         = trial.suggest_int("max_depth", 3, 8),
            num_leaves        = trial.suggest_int("num_leaves", 15, 63),
            subsample         = trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree  = trial.suggest_float("colsample_bytree", 0.5, 1.0),
            min_child_samples = trial.suggest_int("min_child_samples", 5, 30),
            reg_alpha         = trial.suggest_float("reg_alpha", 0.0, 2.0),
            reg_lambda        = trial.suggest_float("reg_lambda", 0.5, 5.0),
            random_state=42,
            verbose=-1,
        )
        n_splits = 3 if len(X) < 60 else 5
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = []
        for tr_idx, val_idx in kf.split(X):
            m = LGBMRegressor(**params)
            m.fit(X[tr_idx], y[tr_idx])
            scores.append(wmape(y[val_idx], m.predict(X[val_idx])))
        return float(np.mean(scores))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


# ── 10. Train one XGB + LGBM ensemble per position group ──────────────────────
models      = {}   # XGBoost — used for SHAP explanations
lgbm_models = {}   # LightGBM
group_data  = {}

for group, features in POSITION_FEATURES.items():
    print(f"\n── {group} {'─'*50}")
    df_g = df[df["position_group"] == group].copy().reset_index(drop=True)
    n = len(df_g)
    print(f"  Players: {n}")

    if n < 20:
        print(f"  ⚠ Too few players, skipping")
        continue

    X_g = df_g[features].values.astype(float)
    y_g = np.log1p(df_g["market_value_in_eur"].values)

    X_train, X_test, y_train, y_test = train_test_split(
        X_g, y_g, test_size=0.2, random_state=42
    )

    n_trials = 60 if n >= 150 else (40 if n >= 80 else 25)

    # ── Tune & train XGBoost ──────────────────────────────────────────────────
    print(f"  Tuning XGBoost ({n_trials} trials)…")
    xgb_params = tune_xgb(X_train, y_train, n_trials=n_trials)
    xgb_params["random_state"] = 42
    xgb_model = XGBRegressor(**xgb_params)
    xgb_model.fit(X_train, y_train, verbose=False)

    # ── Tune & train LightGBM ─────────────────────────────────────────────────
    print(f"  Tuning LightGBM ({n_trials} trials)…")
    lgbm_params = tune_lgbm(X_train, y_train, n_trials=n_trials)
    lgbm_params["random_state"] = 42
    lgbm_params["verbose"]      = -1
    lgbm_model = LGBMRegressor(**lgbm_params)
    lgbm_model.fit(X_train, y_train)

    # ── Ensemble = average of XGB + LGBM ─────────────────────────────────────
    y_pred_test = (xgb_model.predict(X_test) + lgbm_model.predict(X_test)) / 2
    y_pred_all  = (xgb_model.predict(X_g)   + lgbm_model.predict(X_g))   / 2

    mae    = mean_absolute_error(np.expm1(y_test), np.expm1(y_pred_test))
    rmse   = root_mean_squared_error(np.expm1(y_test), np.expm1(y_pred_test))
    r2     = r2_score(y_test, y_pred_test)
    wm     = wmape(y_test, y_pred_test)
    med_ap = median_ape(y_test, y_pred_test)
    print(f"  wMAPE: {wm:.1f}%  MedAPE: {med_ap:.1f}%  MAE: €{mae/1e6:.2f}m  RMSE: €{rmse/1e6:.2f}m  R²: {r2:.3f}")

    # ── SHAP from XGBoost ─────────────────────────────────────────────────────
    print(f"  Computing SHAP values…")
    explainer   = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_g)

    feat_display = [DISPLAY_NAMES.get(f, f) for f in features]

    models[group]      = xgb_model
    lgbm_models[group] = lgbm_model

    group_data[group] = {
        "X":             X_g,
        "y_true":        y_g,
        "y_pred":        y_pred_all,
        "market_values": df_g["market_value_in_eur"].values,
        "player_names":  df_g["name"].values,
        "shap_values":   shap_values,
        "feature_names": feat_display,
        "feature_cols":  features,
        "df":            df_g,
    }

# ── 11. Build combined df_all with ensemble predictions ───────────────────────
parts = []
for group, gd in group_data.items():
    gdf = gd["df"].copy()
    gdf["y_pred_log"]      = gd["y_pred"]
    gdf["predicted_value"] = np.expm1(gd["y_pred"])
    parts.append(gdf)
df_all = pd.concat(parts, ignore_index=True)

sub_positions_by_group = {
    grp: sorted(df[df["position_group"] == grp]["sub_position"].dropna().unique().tolist())
    for grp in POSITION_FEATURES if grp in group_data
}
clubs_sorted = sorted(club_enc_map.keys())

# ── 12. Save artifacts ────────────────────────────────────────────────────────
with open("model.pkl", "wb") as f:
    pickle.dump({
        "models":              models,
        "lgbm_models":         lgbm_models,
        "foot_enc":            foot_enc,
        "sub_pos_enc":         None,           # replaced by target encoding
        "subpos_enc_map":      subpos_enc_map,
        "club_enc_map":        club_enc_map,
        "global_mean_log":     global_mean_log,
        "eu_countries":        EU_EEA_COUNTRIES,
        "feature_cols":        POSITION_FEATURES,
        "position_group_map":  POSITION_GROUP_MAP,
        "contract_ref_date":   CONTRACT_REF_DATE,
        "contract_median_months": float(contract_median),
    }, f)

with open("model_data.pkl", "wb") as f:
    pickle.dump({
        "groups":                 group_data,
        "df_all":                 df_all,
        "position_group_map":     POSITION_GROUP_MAP,
        "sub_positions_by_group": sub_positions_by_group,
        "clubs_sorted":           clubs_sorted,
        "foot_classes":           foot_enc.classes_.tolist(),
    }, f)

print(f"\n✅ Done — {len(df_all)} players across {len(group_data)} position models")
print("   Run dashboard: python3.12 -m streamlit run app.py")
