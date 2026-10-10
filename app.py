# Run with: python3.12 -m streamlit run app.py

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

st.set_page_config(
    page_title="Football Value Predictor",
    page_icon="⚽",
    layout="wide",
)

# ── Load artifacts ────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open("model_data.pkl", "rb") as f:
        data = pickle.load(f)
    with open("model.pkl", "rb") as f:
        bundle = pickle.load(f)
    explainers = {grp: shap.TreeExplainer(m) for grp, m in bundle["models"].items()}
    return data, bundle, explainers

data, bundle, explainers = load_artifacts()

groups           = data["groups"]
df_all           = data["df_all"]
sub_pos_by_group = data["sub_positions_by_group"]
clubs_sorted     = data.get("clubs_sorted", [])
models                  = bundle["models"]
lgbm_models             = bundle.get("lgbm_models", {})
foot_enc                = bundle["foot_enc"]
subpos_enc_map          = bundle.get("subpos_enc_map", {})
club_enc_map            = bundle.get("club_enc_map", {})
global_mean_log         = bundle.get("global_mean_log", 0.0)
eu_countries            = bundle.get("eu_countries", set())
feature_cols            = bundle["feature_cols"]
contract_ref_date       = bundle.get("contract_ref_date", pd.Timestamp("2025-09-01"))
contract_median_months  = bundle.get("contract_median_months", 24.0)


def fmt_eur(v):
    if v >= 1e6:
        return f"€{v/1e6:.1f}m"
    return f"€{v/1e3:.0f}k"


def get_player(name):
    row   = df_all[df_all["name"] == name].iloc[0]
    group = row["position_group"]
    gd    = groups[group]
    idx   = int(np.where(gd["player_names"] == name)[0][0])
    return row, group, gd, idx


def predict_ensemble(group, X):
    xgb_pred = models[group].predict(X)
    if group in lgbm_models:
        return (xgb_pred + lgbm_models[group].predict(X)) / 2
    return xgb_pred


# ── Sidebar nav ───────────────────────────────────────────────────────────────
st.sidebar.title("⚽ Football Value Predictor")
st.sidebar.markdown("Premier League · Seasons 2024+")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigate",
    ["🔍 Select a Player", "📊 Market Analysis", "⚖️ Compare Players", "🛠️ Build a Player"],
    label_visibility="collapsed",
)

all_names = sorted(df_all["name"].tolist())

if page == "🔍 Select a Player":
    st.sidebar.divider()
    selected_name = st.sidebar.selectbox("Select a player", all_names)
else:
    selected_name = all_names[0]

# ══════════════════════════════════════════════════════════════════════════════
# PAGE — Select a Player
# ══════════════════════════════════════════════════════════════════════════════
if page == "🔍 Select a Player":
    row, group, gd, idx = get_player(selected_name)
    actual    = float(row["market_value_in_eur"])
    predicted = float(row["predicted_value"])
    error_pct = (predicted - actual) / actual * 100

    st.title(selected_name)

    contract_months = row.get("months_to_contract_end", np.nan)
    has_contract    = pd.notna(contract_months) and contract_months != contract_median_months
    if has_contract:
        if contract_months <= 0:
            contract_badge = "🔴 Out of contract"
        elif contract_months <= 6:
            contract_badge = f"🔴 Contract expires in {contract_months:.0f} months"
        elif contract_months <= 12:
            contract_badge = f"🟠 Contract expires in {contract_months:.0f} months"
        elif contract_months <= 24:
            contract_badge = f"🟡 Contract expires in {contract_months:.0f} months"
        else:
            contract_badge = f"🟢 {contract_months:.0f} months on contract"
        st.markdown(
            f"**{row['sub_position']}** · {row.get('current_club_name', 'Unknown')} · "
            f"*{group}* · {contract_badge}"
        )
    else:
        st.markdown(f"**{row['sub_position']}** · {row.get('current_club_name', 'Unknown')} · *{group}*")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Age", f"{row['age']:.1f}")
    c2.metric("Predicted Value", fmt_eur(predicted))
    c3.metric("Actual Value", fmt_eur(actual))
    c4.metric("Model Error", f"{error_pct:+.1f}%")

    st.divider()

    st.subheader("Season Stats")
    s1, s2, s3, s4, s5, s6 = st.columns(6)
    s1.metric("Appearances", int(row["appearances"]))
    s2.metric("Minutes", f"{int(row['minutes_played']):,}")
    s3.metric("Goals", int(row["goals"]) if group != "GK" else "—")
    s4.metric("Assists", int(row["assists"]) if group != "GK" else "—")
    s5.metric("Yellow Cards", int(row["yellow_cards"]))
    s6.metric("Red Cards", int(row["red_cards"]))

    st.divider()

    st.subheader("Why this prediction? (SHAP)")
    exp = explainers[group]
    ev  = exp.expected_value
    if hasattr(ev, "__len__"):
        ev = ev[0]

    shap_exp = shap.Explanation(
        values=gd["shap_values"][idx],
        base_values=float(ev),
        data=gd["X"][idx],
        feature_names=gd["feature_names"],
    )
    fig_wf, _ = plt.subplots(figsize=(9, 5))
    shap.plots.waterfall(shap_exp, max_display=12, show=False)
    st.pyplot(fig_wf, use_container_width=True)
    plt.close(fig_wf)
    st.caption(
        "Red bars push the prediction up, blue bars push it down — relative to the "
        f"{group} model's average. Values are in log-EUR space."
    )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE — Market Analysis
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Market Analysis":
    st.subheader("Feature Importance by Position Group")
    grp_select = st.selectbox("Position group", list(groups.keys()), key="fi_grp")
    gd_fi = groups[grp_select]
    importance = pd.Series(
        np.abs(gd_fi["shap_values"]).mean(axis=0),
        index=gd_fi["feature_names"],
    ).sort_values()
    fig_imp = px.bar(
        importance, orientation="h",
        labels={"value": "Mean |SHAP value|", "index": "Feature"},
        color=importance.values,
        color_continuous_scale="Blues",
        height=380,
    )
    fig_imp.update_layout(coloraxis_showscale=False, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_imp, use_container_width=True)

    st.divider()

    st.subheader("All Players: Predicted vs Actual Value")
    row_sel, _, _, _ = get_player(selected_name)
    fig_sc = px.scatter(
        df_all,
        x=df_all["market_value_in_eur"] / 1e6,
        y=df_all["predicted_value"] / 1e6,
        color="position_group",
        hover_name="name",
        log_x=True, log_y=True,
        opacity=0.65, height=480,
        labels={"x": "Actual (€m)", "y": "Predicted (€m)", "color": "Group"},
    )
    fig_sc.add_trace(go.Scatter(
        x=[row_sel["market_value_in_eur"] / 1e6],
        y=[row_sel["predicted_value"] / 1e6],
        mode="markers",
        marker=dict(symbol="star", size=18, color="gold", line=dict(color="black", width=1)),
        name=selected_name,
    ))
    max_val = max(df_all["market_value_in_eur"].max(), df_all["predicted_value"].max()) / 1e6
    fig_sc.add_shape(type="line", x0=0.1, y0=0.1, x1=max_val, y1=max_val,
                     line=dict(dash="dash", color="gray", width=1))
    fig_sc.update_layout(margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_sc, use_container_width=True)
    st.caption("Dashed line = perfect prediction. Star = selected player.")

    st.divider()

    st.subheader("Most Undervalued & Overvalued Players")
    df_val = df_all[["name", "position_group", "current_club_name",
                      "market_value_in_eur", "predicted_value"]].copy()
    df_val["error_pct"] = (
        (df_val["predicted_value"] - df_val["market_value_in_eur"])
        / df_val["market_value_in_eur"] * 100
    )
    df_val["Actual"]    = df_val["market_value_in_eur"].apply(fmt_eur)
    df_val["Predicted"] = df_val["predicted_value"].apply(fmt_eur)
    df_val["Δ%"]        = df_val["error_pct"].apply(lambda x: f"{x:+.0f}%")

    uv_col, ov_col = st.columns(2)
    with uv_col:
        st.markdown("**Top 10 Undervalued** *(model thinks they're worth more)*")
        top_uv = (
            df_val.nlargest(10, "error_pct")
            [["name", "position_group", "current_club_name", "Actual", "Predicted", "Δ%"]]
            .rename(columns={"name": "Player", "position_group": "Group",
                              "current_club_name": "Club"})
        )
        st.dataframe(top_uv, hide_index=True, use_container_width=True)
    with ov_col:
        st.markdown("**Top 10 Overvalued** *(model thinks they're worth less)*")
        top_ov = (
            df_val.nsmallest(10, "error_pct")
            [["name", "position_group", "current_club_name", "Actual", "Predicted", "Δ%"]]
            .rename(columns={"name": "Player", "position_group": "Group",
                              "current_club_name": "Club"})
        )
        st.dataframe(top_ov, hide_index=True, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE — Compare Players
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚖️ Compare Players":
    st.subheader("Compare Two Players")
    cc1, cc2 = st.columns(2)
    with cc1:
        player_a = st.selectbox("Player A", all_names, key="cmp_a")
    with cc2:
        default_b = 1 if all_names[0] == player_a else 0
        player_b  = st.selectbox("Player B", all_names, index=default_b, key="cmp_b")

    if player_a == player_b:
        st.warning("Select two different players.")
    else:
        row_a = df_all[df_all["name"] == player_a].iloc[0]
        row_b = df_all[df_all["name"] == player_b].iloc[0]

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"### {player_a}")
            st.metric("Position",  row_a["sub_position"])
            st.metric("Club",      row_a.get("current_club_name", "Unknown"))
            st.metric("Age",       f"{row_a['age']:.1f}")
            st.metric("Predicted", fmt_eur(float(row_a["predicted_value"])))
            st.metric("Actual",    fmt_eur(float(row_a["market_value_in_eur"])))
            st.metric("Appearances", int(row_a["appearances"]))
            if row_a["position_group"] != "GK":
                st.metric("Goals",      int(row_a["goals"]))
                st.metric("Assists",    int(row_a["assists"]))
                st.metric("Goals/90",   f"{row_a['goals_per_90']:.2f}")
                st.metric("Assists/90", f"{row_a['assists_per_90']:.2f}")
        with col_b:
            st.markdown(f"### {player_b}")
            st.metric("Position",  row_b["sub_position"])
            st.metric("Club",      row_b.get("current_club_name", "Unknown"))
            st.metric("Age",       f"{row_b['age']:.1f}")
            st.metric("Predicted", fmt_eur(float(row_b["predicted_value"])))
            st.metric("Actual",    fmt_eur(float(row_b["market_value_in_eur"])))
            st.metric("Appearances", int(row_b["appearances"]))
            if row_b["position_group"] != "GK":
                st.metric("Goals",      int(row_b["goals"]))
                st.metric("Assists",    int(row_b["assists"]))
                st.metric("Goals/90",   f"{row_b['goals_per_90']:.2f}")
                st.metric("Assists/90", f"{row_b['assists_per_90']:.2f}")

        st.divider()

        st.subheader("Percentile Rank vs all PL players")
        stat_cols   = ["appearances", "minutes_played", "goals",
                       "assists", "goals_per_90", "assists_per_90"]
        stat_labels = ["Appearances", "Minutes", "Goals",
                       "Assists", "Goals/90", "Assists/90"]

        pct_a, pct_b = [], []
        for sc in stat_cols:
            vals = df_all[sc].fillna(0)
            pct_a.append(float((vals < row_a.get(sc, 0)).mean() * 100))
            pct_b.append(float((vals < row_b.get(sc, 0)).mean() * 100))

        fig_cmp = go.Figure([
            go.Bar(name=player_a, x=stat_labels, y=pct_a, marker_color="royalblue"),
            go.Bar(name=player_b, x=stat_labels, y=pct_b, marker_color="tomato"),
        ])
        fig_cmp.update_layout(
            barmode="group",
            yaxis_title="Percentile (vs all PL players in dataset)",
            yaxis=dict(range=[0, 100]),
            height=400,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_cmp, use_container_width=True)
        st.caption("Higher = better rank among all PL players in the dataset for that stat.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE — Build a Player
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🛠️ Build a Player":
    st.subheader("Build a Custom Player")
    st.markdown("Design a player with the sliders and see what the model predicts.")

    inp_col, out_col = st.columns([1, 1])

    with inp_col:
        custom_group  = st.selectbox("Position Group", ["GK", "DEF", "MID", "FWD"])
        custom_sub    = st.selectbox("Sub-position", sub_pos_by_group.get(custom_group, ["Unknown"]))
        custom_foot   = st.selectbox("Preferred Foot", ["right", "left", "both", "unknown"])

        club_options  = ["(Average Club)"] + clubs_sorted
        selected_club = st.selectbox("Club", club_options)
        is_eu         = st.checkbox("EU/EEA Citizen", value=True)

        import datetime
        default_expiry = (contract_ref_date + pd.DateOffset(months=24)).date()
        contract_expiry = st.date_input(
            "Contract expires",
            value=default_expiry,
            min_value=datetime.date(2024, 1, 1),
            max_value=datetime.date(2032, 6, 30),
            help="Date the player's current contract ends. Affects transfer value significantly.",
        )
        custom_contract_months = (
            pd.Timestamp(contract_expiry) - contract_ref_date
        ).days / 30.44

        custom_age    = st.slider("Age", 16, 40, 24)
        custom_height = st.slider("Height (cm)", 155, 210, 181)
        custom_apps   = st.slider("Appearances", 1, 50, 20)
        custom_mins   = st.slider("Minutes Played", 90, 4500, 1800)
        custom_yellow = st.slider("Yellow Cards", 0, 15, 2)
        custom_red    = st.slider("Red Cards", 0, 5, 0)
        if custom_group != "GK":
            custom_goals   = st.slider("Goals", 0, 40, 5)
            custom_assists = st.slider("Assists", 0, 25, 3)
        else:
            custom_goals, custom_assists = 0, 0

    # Derived features
    mins_safe = max(custom_mins, 1)
    apps_safe = max(custom_apps, 1)
    g90  = custom_goals / mins_safe * 90
    a90  = custom_assists / mins_safe * 90
    gc90 = (custom_goals + custom_assists) / mins_safe * 90
    mpg  = mins_safe / apps_safe
    age2 = custom_age ** 2

    # Encode inputs
    foot_val    = foot_enc.transform([custom_foot])[0] if custom_foot in foot_enc.classes_ else 0
    subpos_val  = subpos_enc_map.get(custom_sub, global_mean_log)
    club_val    = (
        club_enc_map.get(selected_club, global_mean_log)
        if selected_club != "(Average Club)"
        else global_mean_log
    )
    eu_val = 1 if is_eu else 0

    all_feature_values = {
        "age":                       custom_age,
        "age_squared":               age2,
        "height_in_cm":              custom_height,
        "appearances":               custom_apps,
        "minutes_played":            custom_mins,
        "minutes_per_game":          mpg,
        "availability":              min(custom_apps / 38, 1.0),
        "yellow_cards":              custom_yellow,
        "red_cards":                 custom_red,
        "foot_enc":                  foot_val,
        "sub_position_enc":          subpos_val,
        "club_enc":                  club_val,
        "is_eu_citizen":             eu_val,
        "months_to_contract_end":    custom_contract_months,
        "goals":                     custom_goals,
        "assists":                   custom_assists,
        "goals_per_90":              g90,
        "assists_per_90":            a90,
        "goal_contributions_per_90": gc90,
    }

    cols_for_group = feature_cols[custom_group]
    X_custom = np.array([[all_feature_values[f] for f in cols_for_group]], dtype=float)

    custom_pred_log = predict_ensemble(custom_group, X_custom)[0]
    custom_pred_eur = np.expm1(custom_pred_log)

    with out_col:
        st.markdown("### Predicted Market Value")
        st.metric("", fmt_eur(custom_pred_eur))

        if custom_contract_months <= 0:
            st.error("Out of contract — can be signed for free (Bosman)")
        elif custom_contract_months <= 6:
            st.error(f"Final {custom_contract_months:.0f} months — significant discount expected")
        elif custom_contract_months <= 12:
            st.warning(f"Last year of contract — clubs may sell at reduced fee")
        elif custom_contract_months <= 18:
            st.warning(f"Contract up in {custom_contract_months:.0f} months — mild discount")
        else:
            st.success(f"{custom_contract_months:.0f} months remaining — no contract pressure")

        st.markdown("#### What's driving this prediction?")
        exp_c = explainers[custom_group]
        ev_c  = exp_c.expected_value
        if hasattr(ev_c, "__len__"):
            ev_c = ev_c[0]

        shap_custom = exp_c.shap_values(X_custom)
        shap_exp_c  = shap.Explanation(
            values=shap_custom[0],
            base_values=float(ev_c),
            data=X_custom[0],
            feature_names=groups[custom_group]["feature_names"],
        )
        fig_c, _ = plt.subplots(figsize=(8, 4))
        shap.plots.waterfall(shap_exp_c, max_display=10, show=False)
        st.pyplot(fig_c, use_container_width=True)
        plt.close(fig_c)
