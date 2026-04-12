# Run with:  python3.12 model.py
# Then start the dashboard with:  python3.12 -m streamlit run app.py

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

# ── 1. Preprocessing ──────────────────────────────────────────────────────────
df["foot"] = df["foot"].fillna("unknown")
df["height_in_cm"] = df["height_in_cm"].fillna(df["height_in_cm"].median())

cat_cols = ["position", "sub_position", "foot"]
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

feature_cols = [
    "age", "height_in_cm",
    "appearances", "goals", "assists", "minutes_played",
    "yellow_cards", "red_cards",
    "goals_per_90", "assists_per_90", "minutes_per_game",
    "position", "sub_position", "foot",
]

X = df[feature_cols].values
y = np.log1p(df["market_value_in_eur"].values)   # log-transform target

# ── 2. Train / test split ────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── 3. Train XGBoost ─────────────────────────────────────────────────────────
print("Training XGBoost...")
model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    early_stopping_rounds=20,
    eval_metric="rmse",
)
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False,
)

# ── 4. Evaluate on test set ──────────────────────────────────────────────────
y_pred_test_log = model.predict(X_test)
y_pred_test = np.expm1(y_pred_test_log)
y_true_test  = np.expm1(y_test)

mae  = mean_absolute_error(y_true_test, y_pred_test)
rmse = root_mean_squared_error(y_true_test, y_pred_test)
r2   = r2_score(y_test, y_pred_test_log)   # R² on log scale (more meaningful)

print(f"\n── Test-set performance ──────────────────────")
print(f"  MAE  : €{mae/1e6:.2f}m")
print(f"  RMSE : €{rmse/1e6:.2f}m")
print(f"  R²   : {r2:.3f}  (log scale)")

# ── 5. SHAP values on full dataset ───────────────────────────────────────────
print("\nComputing SHAP values (full dataset)...")
explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# ── 6. Full-dataset predictions (for dashboard scatter) ──────────────────────
y_pred_all = model.predict(X)

# ── 7. Save artifacts ────────────────────────────────────────────────────────
with open("model.pkl", "wb") as f:
    pickle.dump({"model": model, "encoders": encoders, "feature_cols": feature_cols}, f)

with open("model_data.pkl", "wb") as f:
    pickle.dump(
        {
            "X":             X,
            "y_true":        y,            # log scale
            "y_pred":        y_pred_all,   # log scale
            "market_values": df["market_value_in_eur"].values,
            "player_names":  df["name"].values,
            "positions":     df["position"].values,          # encoded ints
            "position_labels": encoders["position"].classes_,
            "feature_names": feature_cols,
            "shap_values":   shap_values,
            "df":            df,           # full cleaned dataframe (for dashboard)
        },
        f,
    )

print("\n✅ Artifacts saved: model.pkl, model_data.pkl")
print("   Run the dashboard with:  python3.12 -m streamlit run app.py")
