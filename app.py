# Run with:  python3.12 -m streamlit run app.py

import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Football Value Predictor",
    page_icon="⚽",
    layout="wide",
)

# ── Load artifacts ────────────────────────────────────────────────────────────
@st.cache_resource
def load_data():
    with open("model_data.pkl", "rb") as f:
        return pickle.load(f)

data = load_data()

X             = data["X"]
y_true_log    = data["y_true"]
y_pred_log    = data["y_pred"]
market_values = data["market_values"]          # original EUR
player_names  = data["player_names"]
position_labels = data["position_labels"]
positions_enc = data["positions"]              # encoded ints
feature_names = data["feature_names"]
shap_values   = data["shap_values"]
df_full       = data["df"]

# Decode position labels
positions_str = [position_labels[i] for i in positions_enc]

predicted_values = np.expm1(y_pred_log)

def fmt_eur(v):
    if v >= 1e6:
        return f"€{v/1e6:.1f}m"
    return f"€{v/1e3:.0f}k"

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("⚽ Football Value Predictor")
st.sidebar.markdown("Premier League · Seasons 2022+")

sorted_names = sorted(player_names)
selected_name = st.sidebar.selectbox("Select a player", sorted_names)

idx = int(np.where(player_names == selected_name)[0][0])

# ── Header ────────────────────────────────────────────────────────────────────
row = df_full.iloc[idx]
position_display = positions_str[idx]
club = row.get("current_club_name", "Unknown")

st.title(selected_name)
st.markdown(f"**{position_display}** · {club}")

# ── Metrics row ───────────────────────────────────────────────────────────────
actual    = market_values[idx]
predicted = predicted_values[idx]
error_pct = (predicted - actual) / actual * 100

col1, col2, col3, col4 = st.columns(4)
col1.metric("Age", f"{row['age']:.1f}")
col2.metric("Predicted Value", fmt_eur(predicted))
col3.metric("Actual Value", fmt_eur(actual))
col4.metric("Model Error", f"{error_pct:+.1f}%")

st.divider()

# ── SHAP Waterfall ────────────────────────────────────────────────────────────
st.subheader("Why this prediction? (SHAP)")

with open("model_data.pkl", "rb") as f:
    _data = pickle.load(f)

with open("model.pkl", "rb") as f:
    model_bundle = pickle.load(f)

model = model_bundle["model"]
explainer = shap.TreeExplainer(model)
ev = explainer.expected_value
if hasattr(ev, '__len__'):
    ev = ev[0]
expected_value = float(ev)

shap_exp = shap.Explanation(
    values=shap_values[idx],
    base_values=expected_value,
    data=X[idx],
    feature_names=feature_names,
)

fig_wf, ax_wf = plt.subplots(figsize=(9, 5))
shap.plots.waterfall(shap_exp, max_display=12, show=False)
st.pyplot(fig_wf, use_container_width=True)
plt.close(fig_wf)

st.caption(
    "Each bar shows how much a feature pushes the prediction up (red) or down (blue) "
    "from the model's average prediction. Values are in log-EUR space."
)

st.divider()

# ── Global Feature Importance ─────────────────────────────────────────────────
st.subheader("Global Feature Importance")

importance = pd.Series(
    np.abs(shap_values).mean(axis=0),
    index=feature_names,
).sort_values()

fig_imp = px.bar(
    importance,
    orientation="h",
    labels={"value": "Mean |SHAP value|", "index": "Feature"},
    color=importance.values,
    color_continuous_scale="Blues",
)
fig_imp.update_layout(
    coloraxis_showscale=False,
    height=420,
    margin=dict(l=10, r=10, t=10, b=10),
)
st.plotly_chart(fig_imp, use_container_width=True)

st.divider()

# ── Predicted vs Actual Scatter ───────────────────────────────────────────────
st.subheader("All Players: Predicted vs Actual Value")

scatter_df = pd.DataFrame({
    "Actual (€m)":    market_values / 1e6,
    "Predicted (€m)": predicted_values / 1e6,
    "Position":       positions_str,
    "Name":           player_names,
    "Selected":       ["★ " + n if n == selected_name else n for n in player_names],
})

fig_sc = px.scatter(
    scatter_df,
    x="Actual (€m)",
    y="Predicted (€m)",
    color="Position",
    hover_name="Name",
    log_x=True,
    log_y=True,
    opacity=0.65,
    height=480,
)

# Highlight selected player
fig_sc.add_trace(
    go.Scatter(
        x=[actual / 1e6],
        y=[predicted / 1e6],
        mode="markers",
        marker=dict(symbol="star", size=18, color="gold", line=dict(color="black", width=1)),
        name=selected_name,
        hovertext=selected_name,
    )
)

# Perfect-prediction line
max_val = max(market_values.max(), predicted_values.max()) / 1e6
fig_sc.add_shape(
    type="line",
    x0=0.1, y0=0.1, x1=max_val, y1=max_val,
    line=dict(dash="dash", color="gray", width=1),
)

fig_sc.update_layout(margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(fig_sc, use_container_width=True)
st.caption("Dashed line = perfect prediction. Star = selected player.")
