# Run with: python3.12 model.py

import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from xgboost import XGBRegressor
import shap

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
df["foot"]         = df["foot"].fillna("unknown")
df["sub_position"] = df["sub_position"].fillna("Unknown")
df["height_in_cm"] = df["height_in_cm"].fillna(df["height_in_cm"].median())

foot_enc    = LabelEncoder()
sub_pos_enc = LabelEncoder()
df["foot_enc"]         = foot_enc.fit_transform(df["foot"].astype(str))
df["sub_position_enc"] = sub_pos_enc.fit_transform(df["sub_position"].astype(str))

# ── 3. Position-specific feature sets ─────────────────────────────────────────
# GKs: no goals/assists — they're always 0 and add noise
# DEF/MID/FWD: full outfield features
COMMON   = ["age", "height_in_cm", "appearances", "minutes_played",
            "minutes_per_game", "yellow_cards", "red_cards",
            "foot_enc", "sub_position_enc"]
OUTFIELD = ["goals", "assists", "goals_per_90", "assists_per_90"]

POSITION_FEATURES = {
    "GK":  COMMON,
    "DEF": COMMON + OUTFIELD,
    "MID": COMMON + OUTFIELD,
    "FWD": COMMON + OUTFIELD,
}

# Human-readable display names for encoded columns
DISPLAY_NAMES = {"foot_enc": "foot", "sub_position_enc": "sub_position"}

# ── 4. Train one model per position group ─────────────────────────────────────
models     = {}
group_data = {}

for group, features in POSITION_FEATURES.items():
    print(f"\n── {group} {'─'*50}")
    df_g = df[df["position_group"] == group].copy().reset_index(drop=True)
    n = len(df_g)
    print(f"  Players: {n}")

    if n < 20:
        print(f"  ⚠ Too few players, skipping")
        continue

    X_g = df_g[features].values
    y_g = np.log1p(df_g["market_value_in_eur"].values)

    X_train, X_test, y_train, y_test = train_test_split(
        X_g, y_g, test_size=0.2, random_state=42
    )

    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        random_state=42,
        early_stopping_rounds=20,
        eval_metric="rmse",
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_pred_test_log = model.predict(X_test)
    mae  = mean_absolute_error(np.expm1(y_test), np.expm1(y_pred_test_log))
    rmse = root_mean_squared_error(np.expm1(y_test), np.expm1(y_pred_test_log))
    r2   = r2_score(y_test, y_pred_test_log)
    print(f"  MAE: €{mae/1e6:.2f}m  RMSE: €{rmse/1e6:.2f}m  R²: {r2:.3f}")

    print(f"  Computing SHAP values...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_g)
    y_pred_all  = model.predict(X_g)

    feat_display = [DISPLAY_NAMES.get(f, f) for f in features]

    models[group] = model
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

# ── 5. Build combined df_all with predictions ─────────────────────────────────
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

# ── 6. Save artifacts ─────────────────────────────────────────────────────────
with open("model.pkl", "wb") as f:
    pickle.dump({
        "models":             models,
        "foot_enc":           foot_enc,
        "sub_pos_enc":        sub_pos_enc,
        "feature_cols":       POSITION_FEATURES,
        "position_group_map": POSITION_GROUP_MAP,
    }, f)

with open("model_data.pkl", "wb") as f:
    pickle.dump({
        "groups":                 group_data,
        "df_all":                 df_all,
        "position_group_map":     POSITION_GROUP_MAP,
        "sub_positions_by_group": sub_positions_by_group,
        "foot_classes":           foot_enc.classes_.tolist(),
    }, f)

print(f"\n✅ Done — {len(df_all)} players across {len(group_data)} position models")
print("   Run dashboard: python3.12 -m streamlit run app.py")
