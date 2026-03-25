"""
Credit Decisioning — Analytical Dashboard
==========================================
Live data exploration, WoE analysis, full model analytics and comparison,
and interactive predictor.  Static theory / deployment / AI content lives at
alejandrohs.com/ow-quant-2025
"""

import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    f1_score,
)

from credit_model import (
    run_full_pipeline,
    CONTINUOUS_COLS, CATEGORICAL_COLS, ALL_FEATURES, TARGET,
    compute_roc_curve_data, compute_ks_curve_data,
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Model · Analytical Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Dark Theme CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="collapsedControl"] { display: none; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 0 !important; max-width: 100% !important; }

.stTabs [data-baseweb="tab-list"] {
    background: #09090b; gap: 0; padding: 0 1.5rem;
    border-radius: 0; border-bottom: 1px solid #27272a;
    overflow-x: auto; position: sticky; top: 0; z-index: 999;
}
.stTabs [data-baseweb="tab"] {
    color: #71717a !important; background: transparent !important;
    border: none !important; padding: 1rem 1.3rem !important;
    font-weight: 700 !important; font-size: 0.72rem !important;
    letter-spacing: 0.8px !important; text-transform: uppercase !important;
    border-bottom: 2px solid transparent !important; margin: 0 !important;
    border-radius: 0 !important; min-width: max-content;
}
.stTabs [aria-selected="true"] {
    color: #ffffff !important; border-bottom: 2px solid #3b82f6 !important;
    background: rgba(59,130,246,0.06) !important;
}
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] { display: none !important; }
.stTabs > div:first-child { margin-bottom: 0 !important; }

.stTabs [role="tabpanel"] p,
.stTabs [role="tabpanel"] li,
.stTabs [role="tabpanel"] span:not([class]),
.stTabs [role="tabpanel"] td,
.stTabs [role="tabpanel"] th { color: #a1a1aa; }

h1 { font-size: 2rem !important; font-weight: 800 !important; color: #ffffff !important; }
h2 { font-size: 1.35rem !important; font-weight: 700 !important; color: #ffffff !important; margin-top: 1.5rem !important; }
h3 { font-size: 1.05rem !important; font-weight: 700 !important; color: #fafafa !important; }

.ow-card {
    background: #18181b; border: 1px solid #27272a; border-radius: 12px;
    padding: 1.4rem 1.6rem; margin-bottom: 1rem; color: #e4e4e7 !important;
}
.ow-card code { background: rgba(59,130,246,0.15) !important; color: #93c5fd !important;
    padding: 1px 6px !important; border-radius: 4px !important; font-size: 0.85em !important; font-weight: 600 !important; }
.ow-card-accent {
    background: linear-gradient(135deg, rgba(59,130,246,0.08), rgba(99,102,241,0.08));
    border: 1px solid rgba(59,130,246,0.2); border-radius: 12px;
    padding: 1.4rem 1.6rem; color: #ffffff !important; margin-bottom: 1rem;
}
.ow-card-light {
    background: #18181b; border: 1px solid #27272a; border-radius: 12px;
    padding: 1.2rem 1.4rem; margin-bottom: 0.8rem; color: #e4e4e7 !important;
}

.ow-hero {
    background: linear-gradient(135deg, #09090b 0%, #111827 100%);
    padding: 2rem 2.5rem 1.8rem; margin: -1rem -1rem 2rem -1rem;
    border-bottom: 1px solid #27272a;
}
.ow-hero h1 { color: #ffffff !important; font-size: 1.9rem !important; margin: 0 !important; }
.ow-hero p  { color: #71717a !important; margin: 0.3rem 0 0; font-size: 0.9rem; }

[data-testid="metric-container"] {
    background: #18181b; border: 1px solid #27272a; border-radius: 10px;
    padding: 0.8rem 1rem;
}
[data-testid="stMetricLabel"]  { font-size: 0.72rem !important; font-weight: 700 !important;
    letter-spacing: 0.5px; color: #71717a !important; text-transform: uppercase; }
[data-testid="stMetricValue"]  { font-size: 1.6rem !important; font-weight: 800 !important; color: #ffffff !important; }
[data-testid="stMetricDelta"]  { font-size: 0.8rem !important; }

.tag { display:inline-block; background:rgba(59,130,246,0.12); color:#93c5fd; font-size:0.72rem;
    font-weight:700; padding:0.2rem 0.6rem; border-radius:20px; margin:0.1rem; }
.tag-green { background:rgba(74,222,128,0.12); color:#4ade80; }
.tag-red   { background:rgba(248,113,113,0.12); color:#f87171; }
.tag-amber { background:rgba(251,191,36,0.12); color:#fbbf24; }

.section-divider { border:none; border-top:1px solid #27272a; margin:1.5rem 0; }

details { background: #18181b !important; border: 1px solid #27272a !important; border-radius: 8px !important; }
.stForm { border: 1px solid #27272a !important; border-radius: 12px !important; }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:#09090b;padding:1.1rem 2.5rem;
            display:flex;align-items:center;justify-content:space-between;
            border-bottom:1px solid #27272a;">
  <div>
    <div style="font-size:0.6rem;color:#3b82f6;font-weight:700;letter-spacing:3px;
                text-transform:uppercase;margin-bottom:0.2rem;">OW Quant Challenge · Analytical Dashboard</div>
    <div style="font-size:1.3rem;font-weight:800;color:white;">Credit Decisioning Model</div>
    <div style="font-size:0.75rem;color:#71717a;margin-top:0.15rem;">
      WoE-IV Scorecard &nbsp;·&nbsp; Live Results from 10,000 Applications &nbsp;·&nbsp;
      6 Models Compared
    </div>
  </div>
  <div style="text-align:right;">
    <div style="font-size:0.6rem;color:#3b82f6;font-weight:700;letter-spacing:2px;text-transform:uppercase;">Scorecard AUC</div>
    <div style="font-size:1.5rem;font-weight:800;color:white;line-height:1.1;">0.9283</div>
    <div style="font-size:0.75rem;color:#a1a1aa;">Gini 0.857 · KS 0.699 · Brier 0.089</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Data loading ───────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Quant_Challenge_data_amended.csv")

@st.cache_resource(show_spinner="Training models on 10,000 applications… (cached after first run)")
def load_and_run():
    df = pd.read_csv(DATA_PATH, dtype=str)
    return run_full_pipeline(df)

results  = load_and_run()
sc       = results["scorecard"]
enc      = sc.woe_encoder
df_c     = results["df_clean"]
df_m     = results["df_model"]
X_train, X_test = results["X_train"], results["X_test"]
y_train, y_test = results["y_train"], results["y_test"]
metrics         = results["metrics"]
sc_proba        = results["sc_proba_test"]
bl_probas       = results["baseline_probas_test"]

# ── Color palette & Plotly template ───────────────────────────────────────────
OW = {"blue":"#3b82f6","green":"#4ade80","red":"#f87171","amber":"#fbbf24",
      "violet":"#a78bfa","teal":"#2dd4bf","zinc400":"#a1a1aa","zinc500":"#71717a",
      "zinc800":"#27272a","zinc900":"#18181b","bg":"#09090b"}
MODEL_PALETTE = {
    "WoE Scorecard":"#3b82f6","Vanilla LR":"#f87171","Decision Tree":"#fbbf24",
    "Random Forest":"#4ade80","Gradient Boosting":"#a78bfa","SVM (RBF kernel)":"#2dd4bf",
}

pio.templates["ow_dark"] = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor="#09090b",
        plot_bgcolor="#09090b",
        font=dict(color="#a1a1aa", family="Inter, system-ui, sans-serif", size=12),
        xaxis=dict(gridcolor="#27272a", zerolinecolor="#3f3f46", title_font_color="#a1a1aa"),
        yaxis=dict(gridcolor="#27272a", zerolinecolor="#3f3f46", title_font_color="#a1a1aa"),
        colorway=["#3b82f6","#f87171","#fbbf24","#4ade80","#a78bfa","#2dd4bf"],
        legend=dict(font=dict(color="#a1a1aa")),
        title=dict(font=dict(color="#fafafa")),
    )
)
TMPL = "ow_dark"

tabs = st.tabs(["📊 Data Explorer", "⚖ WoE Analysis", "🎯 Model Performance",
                "🔬 Model Deep Dive", "🔮 Live Predictor"])


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — DATA EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown("""
    <div class='ow-hero'>
        <h1>Data Explorer</h1>
        <p>Live statistics from the 10,000-application dataset · Distributions · Correlations · Missing values</p>
    </div>""", unsafe_allow_html=True)

    report = results["cleaning_report"]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Raw Rows",         f"{report['rows_before']:,}")
    c2.metric("Modellable Rows",  f"{report['rows_after']:,}")
    c3.metric("Missing (before)", f"{report['total_missing_before']:,}")
    c4.metric("Missing (after)",  f"{report['total_missing_after']:,}")
    c5.metric("Occup. Codes Fixed", f"{report['occup_recoded']}")

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### Class Balance")
        vc = df_m[TARGET].value_counts().reset_index()
        vc.columns = ["Class", "Count"]
        vc["Label"] = vc["Class"].map({0: "Non-default (0)", 1: "Default (1)"})
        fig_pie = px.pie(vc, values="Count", names="Label",
                         color_discrete_sequence=[OW["blue"], OW["red"]],
                         template=TMPL, height=260)
        fig_pie.update_layout(margin=dict(t=10, b=10),
                               legend=dict(orientation="h", y=-0.1))
        st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown("### Missing Rate by Feature")
        feat_cols = [c for c in ALL_FEATURES if c in report["missing_before"]]
        miss_df = pd.DataFrame({
            "Feature": feat_cols,
            "Missing %": [report["missing_after"].get(c, 0) / report["rows_before"] * 100
                          for c in feat_cols],
        }).sort_values("Missing %", ascending=False)
        fig_miss = px.bar(miss_df, x="Missing %", y="Feature", orientation="h",
                          color="Missing %", color_continuous_scale=["#18181b","#3b82f6"],
                          template=TMPL, height=280)
        fig_miss.update_layout(margin=dict(t=5, b=5), coloraxis_showscale=False,
                                yaxis_title="")
        st.plotly_chart(fig_miss, use_container_width=True)

    with col_b:
        st.markdown("### Feature Distribution by Outcome")
        feat_sel = st.selectbox("Select feature", ALL_FEATURES, key="dist_feat")
        if feat_sel in CONTINUOUS_COLS:
            fig_dist = go.Figure()
            for target_val, label, color in [(0, "Non-default", OW["blue"]),
                                              (1, "Default",     OW["red"])]:
                mask = df_m[TARGET] == target_val
                fig_dist.add_trace(go.Histogram(
                    x=df_m.loc[mask, feat_sel].dropna(),
                    name=label, marker_color=color, opacity=0.65, nbinsx=40,
                ))
            fig_dist.update_layout(barmode="overlay", template=TMPL,
                                   xaxis_title=feat_sel, yaxis_title="Count",
                                   legend=dict(orientation="h", y=1.1),
                                   margin=dict(t=10, b=20), height=280)
        else:
            cross = df_m.groupby([feat_sel, TARGET]).size().reset_index(name="Count")
            cross[TARGET] = cross[TARGET].map({0: "Non-default", 1: "Default"})
            fig_dist = px.bar(cross, x=feat_sel, y="Count", color=TARGET,
                              barmode="group", template=TMPL,
                              color_discrete_map={"Non-default": OW["blue"],
                                                   "Default": OW["red"]},
                              height=280)
            fig_dist.update_layout(margin=dict(t=10, b=20),
                                   legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig_dist, use_container_width=True)

        st.markdown("### Correlation Matrix (continuous features)")
        corr = df_m[CONTINUOUS_COLS].corr().round(2)
        fig_corr = px.imshow(corr, text_auto=True, aspect="auto",
                              color_continuous_scale=["#f87171", "#09090b", "#3b82f6"],
                              zmin=-1, zmax=1, template=TMPL, height=280)
        fig_corr.update_layout(margin=dict(t=5, b=5), coloraxis_showscale=False)
        fig_corr.update_traces(textfont_size=9, textfont_color="#a1a1aa")
        st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    st.markdown("### Outlier Summary (IQR method)")
    outlier_rows = []
    for col in CONTINUOUS_COLS:
        s = df_m[col].dropna()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        n_out = int(((s < q1 - 1.5*iqr) | (s > q3 + 1.5*iqr)).sum())
        outlier_rows.append({"Feature": col, "Min": round(s.min(),1), "Q1": round(q1,1),
                             "Median": round(s.median(),1), "Q3": round(q3,1),
                             "Max": round(s.max(),1), "IQR Outliers": n_out,
                             "% rows": f"{n_out/len(s):.1%}"})
    st.dataframe(pd.DataFrame(outlier_rows), use_container_width=True, hide_index=True)
    st.caption("Outliers are retained — WoE equal-frequency binning absorbs extreme values into the top/bottom bin.")

    if "obs_year" in df_c.columns:
        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        st.markdown("### Loan Volume Over Time")
        yr = df_c.dropna(subset=["obs_year"])
        yr_grp = yr.groupby(["obs_year", TARGET]).size().reset_index(name="Count")
        yr_grp = yr_grp.dropna(subset=[TARGET])
        yr_grp[TARGET] = yr_grp[TARGET].astype(int).map({0:"Non-default",1:"Default"})
        fig_yr = px.bar(yr_grp, x="obs_year", y="Count", color=TARGET,
                        barmode="stack", template=TMPL,
                        color_discrete_map={"Non-default":OW["blue"],"Default":OW["red"]},
                        height=260)
        fig_yr.update_layout(margin=dict(t=5,b=20), xaxis_title="Year",
                              legend=dict(orientation="h",y=1.1))
        st.plotly_chart(fig_yr, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — WoE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown("""
    <div class='ow-hero'>
        <h1>WoE Analysis</h1>
        <p>Per-feature bin profiles · Information Value from training data · Coefficient weights</p>
    </div>""", unsafe_allow_html=True)

    col_left, col_right = st.columns([1.1, 1])

    with col_left:
        feat_woe = st.selectbox("Select feature", list(enc.bin_stats.keys()), key="woe_feat")
        stats = enc.bin_stats[feat_woe].copy()

        is_mono, direction = enc.monotonic_check(feat_woe)
        if not is_mono:
            st.warning(f"⚠ WoE for **{feat_woe}** is non-monotonic ({direction}). "
                       "Review bin boundaries — may indicate insufficient data per bin.")

        stats_plot = stats[stats["bin"] != "Missing"].copy()
        fig_woe = go.Figure()
        fig_woe.add_bar(
            x=stats_plot["bin"].astype(str),
            y=stats_plot["woe"],
            name="WoE",
            marker_color=[OW["red"] if w > 0 else OW["blue"] for w in stats_plot["woe"]],
        )
        if len(stats_plot) > 0:
            fig_woe.add_scatter(
                x=stats_plot["bin"].astype(str),
                y=stats_plot["event_rate"],
                name="Event rate",
                mode="lines+markers",
                line=dict(color=OW["amber"], width=2, dash="dot"),
                yaxis="y2",
                marker=dict(size=6),
            )
        fig_woe.add_hline(y=0, line_dash="dash", line_color="#52525b", line_width=1)
        fig_woe.update_layout(
            template=TMPL, height=340, margin=dict(t=30, b=60),
            xaxis_title="Bin", yaxis_title="WoE",
            xaxis_tickangle=-30,
            legend=dict(orientation="h", y=1.12),
            yaxis2=dict(title="Event Rate", overlaying="y", side="right",
                        tickformat=".0%", showgrid=False,
                        title_font_color="#a1a1aa", tickfont_color="#a1a1aa"),
        )
        st.plotly_chart(fig_woe, use_container_width=True)

        miss_row = stats[stats["bin"] == "Missing"]
        if len(miss_row):
            mr = miss_row.iloc[0]
            signal = "⬆ Riskier" if mr["woe"] > 0.05 else ("⬇ Safer" if mr["woe"] < -0.05 else "≈ Neutral")
            st.markdown(f"""
            <div class='ow-card-light' style='font-size:0.85rem;'>
            <strong>Missing bin:</strong> &nbsp;
            {int(mr['count'])} rows ({mr['count']/len(df_m):.1%}) &nbsp;·&nbsp;
            Default rate: {mr['event_rate']:.1%} &nbsp;·&nbsp;
            WoE: {mr['woe']:+.3f} &nbsp;·&nbsp; <strong>{signal}</strong>
            </div>""", unsafe_allow_html=True)

        with st.expander("Full bin statistics table"):
            st.dataframe(
                stats[["bin","count","events","non_events","event_rate","woe","iv_contribution","smoothed"]]
                .style.format({"event_rate":"{:.1%}","woe":"{:.4f}","iv_contribution":"{:.4f}"}),
                use_container_width=True, hide_index=True
            )

    with col_right:
        st.markdown("### Information Value Table")
        iv_df = enc.get_iv_table()
        fig_iv = px.bar(iv_df, x="IV", y="Feature", orientation="h",
                        color="IV",
                        color_continuous_scale=["#18181b","#3b82f6","#60a5fa"],
                        template=TMPL, height=320)
        fig_iv.update_layout(margin=dict(t=5,b=5), coloraxis_showscale=False,
                              yaxis_title="", xaxis_title="Information Value (IV)")
        fig_iv.add_vline(x=0.3, line_dash="dot", line_color=OW["amber"],
                         annotation_text="Strong (0.3)", annotation_position="top right",
                         annotation_font_color="#fbbf24")
        st.plotly_chart(fig_iv, use_container_width=True)
        st.dataframe(iv_df.style.format({"IV":"{:.4f}"}),
                     use_container_width=True, hide_index=True)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    col_c, col_d = st.columns(2)

    with col_c:
        st.markdown("### Logistic Regression Coefficients (β)")
        coef_df = pd.DataFrame({
            "Feature": [f.replace("_woe","") for f in sc.feature_names_],
            "β": sc.lr.coef_[0].round(4),
        }).sort_values("β")
        fig_coef = go.Figure(go.Bar(
            x=coef_df["β"], y=coef_df["Feature"], orientation="h",
            marker_color=[OW["blue"] if c > 0 else OW["red"] for c in coef_df["β"]],
        ))
        fig_coef.add_vline(x=0, line_color="#52525b", line_width=1)
        fig_coef.update_layout(template=TMPL, height=300, margin=dict(t=5,b=5),
                                yaxis_title="", xaxis_title="Coefficient β")
        st.plotly_chart(fig_coef, use_container_width=True)
        st.caption("β > 0: higher WoE increases default log-odds. β < 0: decreases. "
                   "Score points = −Factor × β × WoE.")

    with col_d:
        st.markdown("### Score Distribution on Test Set")
        test_scores = sc.predict_score(X_test)
        fig_sdist = go.Figure()
        for tv, label, color in [(0,"Non-default",OW["blue"]),(1,"Default",OW["red"])]:
            fig_sdist.add_trace(go.Histogram(
                x=test_scores[y_test==tv], name=label,
                marker_color=color, opacity=0.65, nbinsx=40,
            ))
        fig_sdist.update_layout(
            barmode="overlay", template=TMPL,
            xaxis_title="Credit Score (0–1000)", yaxis_title="Count",
            legend=dict(orientation="h", y=1.1),
            margin=dict(t=10,b=20), height=300,
        )
        st.plotly_chart(fig_sdist, use_container_width=True)
        st.caption("Clear separation: defaulters cluster at lower scores. "
                   "Overlap region = review band for credit officer.")

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    with st.expander("📋 Full Scorecard Table (feature × bin → score points)"):
        sc_table = sc.get_scorecard_table()
        st.dataframe(
            sc_table.style.format({
                "Event Rate":"{:.1%}","WoE":"{:.4f}",
                "β (LR coef)":"{:.4f}","Score Points":"{:.1f}",
            }),
            use_container_width=True, hide_index=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown("""
    <div class='ow-hero'>
        <h1>Model Performance</h1>
        <p>All 6 models on held-out test set (2,000 applications) · ROC · PR · KS · Calibration</p>
    </div>""", unsafe_allow_html=True)

    all_probas = {"WoE Scorecard": sc_proba, **bl_probas}

    st.markdown("## Performance Comparison — Test Set")
    ap_scores = {name: round(average_precision_score(y_test, proba), 4)
                 for name, proba in all_probas.items()}
    metrics_df = pd.DataFrame(metrics).T.reset_index().rename(columns={"index":"Model"})
    metrics_df["Avg Precision"] = metrics_df["Model"].map(ap_scores)
    metrics_df = metrics_df[["Model","AUC","Gini","KS","Brier","Avg Precision"]]
    styled = (
        metrics_df.style
        .highlight_max(subset=["AUC","Gini","KS","Avg Precision"], color="rgba(74,222,128,0.15)")
        .highlight_min(subset=["Brier"], color="rgba(74,222,128,0.15)")
        .highlight_max(subset=["Brier"], color="rgba(248,113,113,0.15)")
        .format({"AUC":"{:.4f}","Gini":"{:.4f}","KS":"{:.4f}","Brier":"{:.4f}","Avg Precision":"{:.4f}"})
        .apply(lambda col: [
            "font-weight:800;color:#3b82f6;" if r == "WoE Scorecard" else ""
            for r in metrics_df["Model"]
        ] if col.name == "Model" else [""]*len(col), axis=0)
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)
    st.caption("Green = best · Red = worst · Bold blue = WoE Scorecard · Brier: lower = better · Avg Precision = area under PR curve")

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    col_roc, col_pr = st.columns(2)

    with col_roc:
        st.markdown("### ROC Curves — All Models")
        fig_roc = go.Figure()
        fig_roc.add_scatter(x=[0,1], y=[0,1], mode="lines", name="Random",
                            line=dict(dash="dash", color="#52525b", width=1))
        for name, proba in all_probas.items():
            fpr, tpr, _ = compute_roc_curve_data(y_test, proba)
            auc_val = metrics[name]["AUC"]
            fig_roc.add_scatter(
                x=fpr, y=tpr, mode="lines",
                name=f"{name} (AUC={auc_val:.4f})",
                line=dict(color=MODEL_PALETTE.get(name,"#71717a"),
                          width=3 if name=="WoE Scorecard" else 1.5),
            )
        fig_roc.update_layout(
            template=TMPL, xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            legend=dict(orientation="h", y=-0.45, font_size=10),
            margin=dict(t=10,b=110), height=430,
        )
        st.plotly_chart(fig_roc, use_container_width=True)
        st.caption("ROC: probability that the model ranks a random defaulter above a random non-defaulter. "
                   "Area = AUC. Gini = 2×AUC−1.")

    with col_pr:
        st.markdown("### Precision-Recall Curves — All Models")
        fig_pr = go.Figure()
        prevalence = float(y_test.mean())
        fig_pr.add_scatter(x=[0,1], y=[prevalence, prevalence], mode="lines",
                           name=f"Random ({prevalence:.1%})",
                           line=dict(dash="dash", color="#52525b", width=1))
        for name, proba in all_probas.items():
            prec, rec, _ = precision_recall_curve(y_test, proba)
            ap_val = ap_scores[name]
            fig_pr.add_scatter(
                x=rec, y=prec, mode="lines",
                name=f"{name} (AP={ap_val:.4f})",
                line=dict(color=MODEL_PALETTE.get(name,"#71717a"),
                          width=3 if name=="WoE Scorecard" else 1.5),
            )
        fig_pr.update_layout(
            template=TMPL, xaxis_title="Recall", yaxis_title="Precision",
            legend=dict(orientation="h", y=-0.45, font_size=10),
            margin=dict(t=10,b=110), height=430,
        )
        st.plotly_chart(fig_pr, use_container_width=True)
        st.caption("PR curve: more informative than ROC under class imbalance (~20% defaults). "
                   "AP = area under PR curve. High AP → model identifies defaulters precisely.")

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    col_ks, col_cal = st.columns(2)

    with col_ks:
        st.markdown("### KS Chart — WoE Scorecard")
        ks_df   = compute_ks_curve_data(y_test, sc_proba)
        ks_val  = metrics["WoE Scorecard"]["KS"]
        ks_idx  = ks_df["ks_diff"].idxmax()
        ks_pct  = float(ks_df.loc[ks_idx, "percentile"])
        fig_ks  = go.Figure()
        fig_ks.add_scatter(x=ks_df["percentile"], y=ks_df["cum_events"],
                           name="Defaulters", line=dict(color=OW["red"], width=2))
        fig_ks.add_scatter(x=ks_df["percentile"], y=ks_df["cum_non_events"],
                           name="Non-defaulters", line=dict(color=OW["blue"], width=2))
        fig_ks.add_vline(x=ks_pct, line_dash="dot", line_color="#71717a",
                         annotation_text=f"KS={ks_val:.3f} at {ks_pct:.0%}",
                         annotation_position="top right",
                         annotation_font_color="#a1a1aa")
        fig_ks.update_layout(
            template=TMPL,
            xaxis_title="Population % (highest risk → lowest)",
            yaxis_title="Cumulative %", yaxis_tickformat=".0%",
            legend=dict(orientation="h", y=1.1),
            margin=dict(t=10,b=20), height=350,
        )
        st.plotly_chart(fig_ks, use_container_width=True)
        st.caption(f"KS={ks_val:.3f}: maximum vertical gap between cumulative default and non-default "
                   "distributions. Used as a regulatory threshold in credit risk (KS > 0.4 = good).")

    with col_cal:
        st.markdown("### Calibration — Predicted PD vs Observed Default Rate")
        n_bins_cal = 10
        cal_df = pd.DataFrame({"prob": sc_proba, "y": y_test.values})
        cal_df["bin"] = pd.qcut(cal_df["prob"], q=n_bins_cal, duplicates="drop")
        cal_summary = cal_df.groupby("bin", observed=True).agg(
            mean_pred=("prob","mean"), obs_rate=("y","mean"), count=("y","size")
        ).reset_index()
        fig_cal = go.Figure()
        fig_cal.add_scatter(x=[0,1], y=[0,1], mode="lines", name="Perfect",
                            line=dict(dash="dash", color="#52525b"))
        fig_cal.add_scatter(x=cal_summary["mean_pred"], y=cal_summary["obs_rate"],
                            mode="lines+markers", name="WoE Scorecard",
                            line=dict(color=OW["blue"], width=2),
                            marker=dict(size=cal_summary["count"]/cal_summary["count"].max()*20+4))
        fig_cal.update_layout(
            template=TMPL, xaxis_title="Mean Predicted PD",
            yaxis_title="Observed Default Rate",
            legend=dict(orientation="h", y=1.1), height=350, margin=dict(t=10,b=20),
        )
        st.plotly_chart(fig_cal, use_container_width=True)
        st.caption("Dot size ∝ bin population. Proximity to the diagonal = calibration quality. "
                   "CalibratedClassifierCV (Platt scaling) corrects the distortion from class_weight='balanced'.")

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    col_gain, col_lift = st.columns(2)

    with col_gain:
        st.markdown("### Cumulative Gain Chart")
        fig_gain = go.Figure()
        fig_gain.add_scatter(x=[0,1], y=[0,1], mode="lines", name="Random",
                             line=dict(dash="dash", color="#52525b"))
        for name, proba in all_probas.items():
            df_g = pd.DataFrame({"y": y_test.values, "p": proba})
            df_g = df_g.sort_values("p", ascending=False).reset_index(drop=True)
            df_g["cum_default_pct"] = df_g["y"].cumsum() / df_g["y"].sum()
            df_g["pct_pop"] = (df_g.index + 1) / len(df_g)
            fig_gain.add_scatter(
                x=df_g["pct_pop"], y=df_g["cum_default_pct"], mode="lines",
                name=name,
                line=dict(color=MODEL_PALETTE.get(name,"#71717a"),
                          width=3 if name=="WoE Scorecard" else 1.5),
            )
        fig_gain.update_layout(
            template=TMPL, xaxis_title="% Population Contacted (by risk score)",
            yaxis_title="% Defaults Captured", yaxis_tickformat=".0%",
            xaxis_tickformat=".0%",
            legend=dict(orientation="h", y=-0.45, font_size=10),
            margin=dict(t=10,b=110), height=400,
        )
        st.plotly_chart(fig_gain, use_container_width=True)
        st.caption("Cumulative gain: if you target the top 20% highest-risk applicants, "
                   "what % of all defaults do you capture? Higher curve = better model.")

    with col_lift:
        st.markdown("### Lift Chart")
        fig_lift = go.Figure()
        fig_lift.add_hline(y=1.0, line_dash="dash", line_color="#52525b",
                           annotation_text="Random (lift=1)", annotation_position="right",
                           annotation_font_color="#71717a")
        for name, proba in all_probas.items():
            df_l = pd.DataFrame({"y": y_test.values, "p": proba})
            df_l = df_l.sort_values("p", ascending=False).reset_index(drop=True)
            baseline_rate = df_l["y"].mean()
            df_l["cum_rate"] = df_l["y"].cumsum() / (df_l.index + 1)
            df_l["lift"] = df_l["cum_rate"] / baseline_rate
            df_l["pct_pop"] = (df_l.index + 1) / len(df_l)
            fig_lift.add_scatter(
                x=df_l["pct_pop"], y=df_l["lift"], mode="lines",
                name=name,
                line=dict(color=MODEL_PALETTE.get(name,"#71717a"),
                          width=3 if name=="WoE Scorecard" else 1.5),
            )
        fig_lift.update_layout(
            template=TMPL, xaxis_title="% Population (by risk score)",
            yaxis_title="Lift", xaxis_tickformat=".0%",
            legend=dict(orientation="h", y=-0.45, font_size=10),
            margin=dict(t=10,b=110), height=400,
        )
        st.plotly_chart(fig_lift, use_container_width=True)
        st.caption("Lift = (default rate in top-k%) / (overall default rate). "
                   "A lift of 3× at 10% means the top-10% risky applicants default 3× more often than average.")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 4 — MODEL DEEP DIVE
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown("""
    <div class='ow-hero'>
        <h1>Model Deep Dive</h1>
        <p>Confusion matrices · Feature importance · Threshold analysis · Per-model descriptions</p>
    </div>""", unsafe_allow_html=True)

    st.markdown("## Model Descriptions")
    MODEL_DESCRIPTIONS = {
        "WoE Scorecard": {
            "algo": "Logistic Regression on WoE-transformed features",
            "math": "P(default) = σ(β₀ + Σᵢ βᵢ·WoEᵢ(xᵢ));  Score = Offset − Factor·log-odds",
            "strength": "Fully interpretable, Basel II/III auditable, per-feature score decomposition, "
                        "regulatory-grade calibration via Platt scaling",
            "weakness": "Assumes monotonic WoE-risk relationship; linear in WoE space; requires binning decisions",
            "missing": "WoE 'Missing' bin — missing data is its own bin with its own risk signal",
            "color": OW["blue"],
        },
        "Vanilla LR": {
            "algo": "Logistic Regression on raw features (median imputation + ordinal encoding)",
            "math": "P(default) = σ(β₀ + Σᵢ βᵢ·xᵢ);  same algorithm as scorecard, different inputs",
            "strength": "Interpretable coefficients, fast training, strong baseline",
            "weakness": "No WoE feature engineering; ordinal encoding imposes false ordinal structure on categories; "
                        "sensitive to outliers without binning",
            "missing": "Median imputation — loses the information that 'data is missing'",
            "color": OW["red"],
        },
        "Decision Tree": {
            "algo": "CART decision tree, max_depth=5, balanced class weights",
            "math": "Recursive binary splits maximising Gini impurity reduction: "
                    "Gini(t) = 1 − Σₖ p(k|t)²",
            "strength": "Human-readable rules, captures non-linearity and interactions, no scaling needed",
            "weakness": "High variance (unstable), limited capacity at depth 5, "
                        "poor calibration, tends to overfit leaf nodes",
            "missing": "Median imputation — can't natively handle NaN",
            "color": OW["amber"],
        },
        "Random Forest": {
            "algo": "100 decision trees with bagging and feature subsampling",
            "math": "P̂(default|x) = (1/B) Σᵦ P̂ᵦ(default|x);  "
                    "each tree trained on bootstrap sample with √p features per split",
            "strength": "Reduces variance via averaging, captures non-linear interactions, "
                        "robust to outliers, feature importance from permutation/impurity",
            "weakness": "Opaque (no single interpretable model), slow prediction at inference, "
                        "biased feature importance for high-cardinality features",
            "missing": "Median imputation before pipeline",
            "color": OW["green"],
        },
        "Gradient Boosting": {
            "algo": "HistGradientBoostingClassifier (sklearn) — histogram-binned boosted trees",
            "math": "Fₘ(x) = Fₘ₋₁(x) + η·hₘ(x);  hₘ fits negative gradient of log-loss: "
                    "−∂L/∂F = y − σ(F(x))",
            "strength": "Strongest off-the-shelf tabular model, handles missing values natively, "
                        "fast histogram-based splits, typically best AUC on structured data",
            "weakness": "Black-box by default (requires SHAP/LIME for interpretability), "
                        "many hyperparameters, risk of overfitting without tuning",
            "missing": "Native NaN handling — no imputation needed for numeric features",
            "color": OW["violet"],
        },
        "SVM (RBF kernel)": {
            "algo": "Support Vector Machine with radial basis function kernel + Platt scaling",
            "math": "K(x,x') = exp(−γ‖x−x'‖²);  maximise margin: min ½‖w‖² s.t. yᵢ(wᵀφ(xᵢ)+b) ≥ 1−ξᵢ",
            "strength": "Effective in high-dimensional spaces, memory-efficient (support vectors only), "
                        "powerful non-linear kernel trick",
            "weakness": "O(n²–n³) training complexity, requires StandardScaler, "
                        "probability calibration adds overhead (5-fold Platt scaling), "
                        "poor scalability on large datasets",
            "missing": "Median imputation — cannot handle NaN in kernel computation",
            "color": OW["teal"],
        },
    }

    desc_cols = st.columns(3)
    for i, (mname, d) in enumerate(MODEL_DESCRIPTIONS.items()):
        auc_v = metrics[mname]["AUC"]
        with desc_cols[i % 3]:
            st.markdown(f"""
            <div style="background:#18181b;border:1px solid #27272a;border-left:4px solid {d['color']};
                        border-radius:8px;padding:1rem 1.2rem;margin-bottom:1rem;">
              <div style="font-weight:800;color:{d['color']};font-size:0.9rem;margin-bottom:0.4rem;">
                {mname} &nbsp;<span style="background:{d['color']}1a;padding:2px 8px;border-radius:12px;
                font-size:0.75rem;border:1px solid {d['color']}33;">AUC {auc_v}</span>
              </div>
              <div style="font-size:0.78rem;color:#a1a1aa;margin-bottom:0.4rem;">
                <strong style="color:#e4e4e7;">Algorithm:</strong> {d['algo']}
              </div>
              <div style="font-size:0.75rem;color:#93c5fd;font-family:monospace;
                          background:#09090b;border:1px solid #27272a;
                          padding:4px 8px;border-radius:4px;margin-bottom:0.5rem;">
                {d['math']}
              </div>
              <div style="font-size:0.75rem;color:#4ade80;margin-bottom:0.2rem;">
                ✓ {d['strength']}
              </div>
              <div style="font-size:0.75rem;color:#f87171;margin-bottom:0.2rem;">
                ✗ {d['weakness']}
              </div>
              <div style="font-size:0.75rem;color:#fbbf24;">
                ∅ Missing: {d['missing']}
              </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    st.markdown("## Confusion Matrices at Optimal F1 Threshold")
    st.caption("Threshold per model chosen to maximise F1 score on test set. "
               "Shows the actual classification decision, not just ranking.")

    def _optimal_threshold(y_true, proba):
        prec, rec, thresholds = precision_recall_curve(y_true, proba)
        f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-9)
        best_idx = np.argmax(f1)
        return float(thresholds[best_idx]), float(prec[best_idx]), float(rec[best_idx]), float(f1[best_idx])

    cm_cols = st.columns(3)
    for i, (mname, proba) in enumerate(all_probas.items()):
        thresh, prec_v, rec_v, f1_v = _optimal_threshold(y_test, proba)
        y_pred = (proba >= thresh).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        color = MODEL_PALETTE.get(mname, "#71717a")

        with cm_cols[i % 3]:
            fig_cm = go.Figure(go.Heatmap(
                z=[[tn, fp], [fn, tp]],
                x=["Predicted: No Default", "Predicted: Default"],
                y=["Actual: No Default", "Actual: Default"],
                text=[[f"TN={tn:,}", f"FP={fp:,}"], [f"FN={fn:,}", f"TP={tp:,}"]],
                texttemplate="%{text}",
                textfont=dict(color="#fafafa"),
                colorscale=[[0,"#18181b"],[0.5,"rgb(80,80,80)"],[1,color]],
                showscale=False,
            ))
            fig_cm.update_layout(
                title=dict(text=f"{mname}<br><sup>thresh={thresh:.3f} · F1={f1_v:.3f} · "
                                f"Prec={prec_v:.3f} · Rec={rec_v:.3f}</sup>",
                           font_size=11, font_color="#fafafa"),
                template=TMPL, height=260, margin=dict(t=60,b=10,l=10,r=10),
            )
            st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    st.markdown("## Feature Importance — Tree-Based Models")
    feat_imp_cols = st.columns(2)

    feat_names = CONTINUOUS_COLS + CATEGORICAL_COLS

    for j, mname in enumerate(["Random Forest", "Gradient Boosting"]):
        pipe = results["baselines"].pipelines[mname]
        clf  = pipe.named_steps["clf"]
        # Unwrap CalibratedClassifierCV if present
        if hasattr(clf, "feature_importances_"):
            importances = clf.feature_importances_
        elif hasattr(clf, "estimator") and hasattr(clf.estimator, "feature_importances_"):
            importances = clf.estimator.feature_importances_
        elif hasattr(clf, "calibrated_classifiers_"):
            importances = clf.calibrated_classifiers_[0].estimator.feature_importances_
        else:
            continue
        imp_df = pd.DataFrame({
            "Feature": feat_names[:len(importances)],
            "Importance": importances,
        }).sort_values("Importance", ascending=True)

        color = MODEL_PALETTE[mname]
        fig_imp = go.Figure(go.Bar(
            x=imp_df["Importance"], y=imp_df["Feature"], orientation="h",
            marker_color=color, marker_opacity=0.85,
        ))
        fig_imp.update_layout(
            title=dict(text=f"{mname} — Feature Importance (MDI)", font_size=13),
            template=TMPL, height=320, margin=dict(t=40,b=10),
            xaxis_title="Mean Decrease in Impurity", yaxis_title="",
        )
        with feat_imp_cols[j]:
            st.plotly_chart(fig_imp, use_container_width=True)

    st.caption("MDI (Mean Decrease in Impurity): fraction of total splits weighted by node impurity "
               "reduction. Note: biased toward high-cardinality continuous features. "
               "Compare with WoE IV table in Tab 2 for the interpretable scorecard perspective.")

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    st.markdown("## Threshold Analysis — Precision / Recall / F1 Trade-off")
    thresh_model = st.selectbox("Select model", list(all_probas.keys()), key="thresh_model")
    proba_sel = all_probas[thresh_model]

    prec_arr, rec_arr, thresh_arr = precision_recall_curve(y_test, proba_sel)
    f1_arr = 2 * prec_arr[:-1] * rec_arr[:-1] / (prec_arr[:-1] + rec_arr[:-1] + 1e-9)
    best_t_idx = np.argmax(f1_arr)

    fig_thresh = go.Figure()
    fig_thresh.add_scatter(x=thresh_arr, y=prec_arr[:-1], mode="lines",
                           name="Precision", line=dict(color=OW["blue"], width=2))
    fig_thresh.add_scatter(x=thresh_arr, y=rec_arr[:-1], mode="lines",
                           name="Recall", line=dict(color=OW["red"], width=2))
    fig_thresh.add_scatter(x=thresh_arr, y=f1_arr, mode="lines",
                           name="F1", line=dict(color=OW["green"], width=2.5))
    fig_thresh.add_vline(x=thresh_arr[best_t_idx], line_dash="dot", line_color="#71717a",
                         annotation_text=f"Best F1={f1_arr[best_t_idx]:.3f} @ t={thresh_arr[best_t_idx]:.3f}",
                         annotation_position="top right", annotation_font_color="#a1a1aa")
    fig_thresh.add_vline(x=0.5, line_dash="dot", line_color=OW["amber"],
                         annotation_text="t=0.5", annotation_position="bottom left",
                         annotation_font_color="#fbbf24")
    fig_thresh.update_layout(
        template=TMPL, xaxis_title="Decision Threshold",
        yaxis_title="Score", yaxis=dict(range=[0,1.02]),
        legend=dict(orientation="h", y=1.1),
        margin=dict(t=20,b=20), height=350,
    )
    st.plotly_chart(fig_thresh, use_container_width=True)
    st.caption("Move the threshold left (lower) to catch more defaults (higher recall, lower precision). "
               "Move right (higher) to be more conservative (higher precision, lower recall). "
               "F1 balances both. In credit risk, recall = % defaults caught; "
               "precision = % of flagged applicants who actually default.")

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    st.markdown("## Predicted Probability Distribution by Class — All Models")
    dist_cols = st.columns(3)
    for k, (mname, proba) in enumerate(all_probas.items()):
        fig_vio = go.Figure()
        for tv, label, lcolor in [(0, "Non-default", OW["blue"]), (1, "Default", OW["red"])]:
            mask = y_test.values == tv
            fig_vio.add_trace(go.Violin(
                y=proba[mask], name=label,
                box_visible=True, meanline_visible=True,
                fillcolor=lcolor + "55", line_color=lcolor,
            ))
        fig_vio.update_layout(
            title=dict(text=mname, font_size=11),
            template=TMPL, height=280, margin=dict(t=40,b=10),
            yaxis_title="P(default)", showlegend=(k == 0),
            violingap=0.3,
        )
        with dist_cols[k % 3]:
            st.plotly_chart(fig_vio, use_container_width=True)
    st.caption("Violin plots: ideal model pushes the two distributions fully apart. "
               "Overlap region = uncertainty zone where a decision threshold must be placed.")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 5 — LIVE PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown("""
    <div class='ow-hero'>
        <h1>Live Predictor</h1>
        <p>Score any applicant in real time · Waterfall decomposition · Sensitivity analysis</p>
    </div>""", unsafe_allow_html=True)

    FEATURE_PLAIN = {
        "income":"annual income","loan_amount":"loan amount","term_length":"loan term",
        "install_to_inc":"instalment-to-income ratio","schufa":"SCHUFA score",
        "num_applic":"number of co-applicants","occup":"occupation","marital":"marital status",
    }

    with st.form("predictor_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            income         = st.number_input("Annual Income (€)",     5000.0, 150000.0, float(df_m["income"].median()),        500.0)
            loan_amount    = st.number_input("Loan Amount (€)",        5000.0,  60000.0, float(df_m["loan_amount"].median()),   500.0)
            term_length    = st.number_input("Term (months)",            12.0,    240.0, float(df_m["term_length"].median()),     6.0)
        with c2:
            install_to_inc = st.number_input("Instalment / Income",    0.001,     0.80, round(float(df_m["install_to_inc"].median()),3), 0.005, format="%.3f")
            schufa         = st.number_input("SCHUFA Score",          3000.0, 20000.0, float(df_m["schufa"].median()),         100.0)
            num_applic     = st.number_input("No. of Applicants",         1.0,      5.0,  1.0,                                    1.0)
        with c3:
            occup   = st.selectbox("Occupation",     ["Worker","Employee","Student","Missing"])
            marital = st.selectbox("Marital Status", ["Single","Married","Divorced","Separated","Living together","Missing"])
            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button("▶ Assess Applicant", use_container_width=True, type="primary")

    if submitted:
        row = pd.DataFrame([{
            "income":         income,
            "loan_amount":    loan_amount,
            "term_length":    term_length,
            "install_to_inc": install_to_inc,
            "schufa":         schufa,
            "num_applic":     num_applic,
            "occup":          np.nan if occup  == "Missing" else occup,
            "marital":        np.nan if marital == "Missing" else marital,
        }])

        pd_prob     = float(sc.predict_proba(row)[0])
        score       = int(sc.predict_score(row)[0])
        contrib_df, base_sc = sc.get_feature_contributions(row)
        contrib_sorted = contrib_df.sort_values("Score Points", key=abs, ascending=False)

        all_pds = {"WoE Scorecard": pd_prob}
        for mname, pipe in results["baselines"].pipelines.items():
            all_pds[mname] = float(pipe.predict_proba(row)[0, 1])

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Credit Score",           str(score),          delta=f"{score-600:+d} vs base")
        k2.metric("Probability of Default", f"{pd_prob:.1%}")
        risk_label = "Low Risk" if score >= 700 else ("Medium Risk" if score >= 500 else "High Risk")
        k3.metric("Risk Band", risk_label)
        decision   = "APPROVE" if score >= 650 else ("REVIEW" if score >= 500 else "DECLINE")
        k4.metric("Indicative Decision", decision)

        if   score >= 700: st.success(f"**{risk_label}** — Score {score} ≥ 700. Indicative approval.")
        elif score >= 500: st.warning(f"**{risk_label}** — Score {score} in review band [500–700).")
        else:              st.error(  f"**{risk_label}** — Score {score} < 500. Indicative decline.")

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        col_wf, col_right_pred = st.columns([1.1, 1])

        with col_wf:
            st.markdown("### Score Decomposition")
            features_wf = contrib_sorted["Feature"].tolist()
            points_wf   = contrib_sorted["Score Points"].tolist()
            final = base_sc + sum(points_wf)

            fig_wf = go.Figure(go.Waterfall(
                orientation="v",
                measure=["absolute"] + ["relative"]*len(features_wf) + ["total"],
                x=["Base"] + features_wf + ["Final"],
                y=[base_sc] + points_wf + [0],
                text=[f"{base_sc:.0f}"] + [f"{p:+.0f}" for p in points_wf] + [f"{final:.0f}"],
                textposition="outside",
                textfont=dict(color="#a1a1aa"),
                connector={"line": {"color":"#27272a","width":1}},
                increasing={"marker":{"color":OW["green"]}},
                decreasing={"marker":{"color":OW["red"]}},
                totals={"marker":{"color":OW["blue"]}},
            ))
            fig_wf.add_hline(y=650, line_dash="dot", line_color=OW["amber"],
                             annotation_text="Approval threshold (650)",
                             annotation_position="bottom right",
                             annotation_font_color="#fbbf24")
            fig_wf.update_layout(template=TMPL, height=400, margin=dict(t=20,b=60),
                                  showlegend=False, xaxis_tickangle=-25, yaxis_title="Score")
            st.plotly_chart(fig_wf, use_container_width=True)
            st.caption("Green bars raise the score · Red bars lower it")
            st.dataframe(contrib_sorted.style.format({"WoE":"{:.4f}","Coefficient (β)":"{:.4f}","Score Points":"{:.1f}"}),
                         use_container_width=True, hide_index=True)

        with col_right_pred:
            st.markdown("### All-Model PD for This Applicant")
            pd_df = pd.DataFrame([
                {"Model": m, "PD": v, "Color": MODEL_PALETTE.get(m,"#71717a")}
                for m, v in sorted(all_pds.items(), key=lambda x: x[1])
            ])
            fig_pd_bar = go.Figure(go.Bar(
                x=pd_df["PD"], y=pd_df["Model"], orientation="h",
                marker_color=pd_df["Color"].tolist(),
                text=[f"{v:.1%}" for v in pd_df["PD"]],
                textposition="outside",
                textfont=dict(color="#a1a1aa"),
            ))
            fig_pd_bar.add_vline(x=0.2, line_dash="dot", line_color=OW["amber"],
                                 annotation_text="~20% base rate",
                                 annotation_position="top right",
                                 annotation_font_color="#fbbf24")
            fig_pd_bar.update_layout(
                template=TMPL, height=280,
                xaxis_title="P(default)", xaxis_tickformat=".0%",
                margin=dict(t=10,b=20,r=80),
                yaxis_title="",
            )
            st.plotly_chart(fig_pd_bar, use_container_width=True)
            st.caption("All 6 models scoring the same applicant. "
                       "Agreement across models → higher confidence in the decision.")

            st.markdown("### Client Communication")
            top_neg = contrib_sorted[contrib_sorted["Score Points"] < 0].head(3)
            top_pos = contrib_sorted[contrib_sorted["Score Points"] > 0].head(2)
            neg_reasons = [FEATURE_PLAIN.get(r,r) for r in top_neg["Feature"]]
            pos_reasons = [FEATURE_PLAIN.get(r,r) for r in top_pos["Feature"]]
            st.markdown(f"""
            <div class='ow-card' style='font-family:Georgia,serif;font-size:0.87rem;line-height:1.7;'>
            <strong style="color:#fafafa;">MORTGAGE APPLICATION — ASSESSMENT NOTICE</strong><br><br>
            Dear Applicant,<br><br>
            Your application has received a credit score of
            <strong style="color:#ffffff;">{score} / 1000</strong>.<br><br>
            {"Your application <strong style='color:#4ade80;'>meets</strong> our automated approval criteria."
              if score >= 650
              else "Your application <strong style='color:#f87171;'>does not currently meet</strong> our automated approval criteria."}
            <br><br>
            {"<strong style='color:#fafafa;'>Key negative factors:</strong><br>• " + "<br>• ".join(neg_reasons) if neg_reasons else ""}
            {"<br><br><strong style='color:#fafafa;'>Positive factors:</strong><br>• " + "<br>• ".join(pos_reasons) if pos_reasons else ""}
            <br><br>
            You have the right to request a review by a credit officer.
            This is an automated assessment and does not constitute a final decision.<br><br>
            <em style="color:#71717a;">Credit Risk Department</em>
            </div>""", unsafe_allow_html=True)

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        st.markdown("### Sensitivity Analysis — What Could Improve This Score?")
        sens_rows = []
        for col in CONTINUOUS_COLS:
            base_val = float(row[col].iloc[0]) if not pd.isna(row[col].iloc[0]) else None
            if not base_val:
                continue
            for delta_pct in [10, 20, -10, -20]:
                new_val = base_val * (1 + delta_pct/100)
                if new_val <= 0:
                    continue
                new_row = row.copy()
                new_row[col] = new_val
                new_score = int(sc.predict_score(new_row)[0])
                delta = new_score - score
                if abs(delta) >= 1:
                    sens_rows.append({
                        "Feature":   FEATURE_PLAIN.get(col,col),
                        "Change":    f"{delta_pct:+d}%",
                        "New Value": f"{new_val:,.1f}",
                        "Score Δ":   f"{delta:+d}",
                        "New Score": new_score,
                        "_delta":    delta,
                    })

        if sens_rows:
            sens_df     = pd.DataFrame(sens_rows)
            improvements = sens_df[sens_df["_delta"] > 0].sort_values("_delta", ascending=False).head(8)
            if len(improvements):
                st.dataframe(improvements[["Feature","Change","New Value","Score Δ","New Score"]],
                             use_container_width=True, hide_index=True)
                best = improvements.iloc[0]
                st.success(f"Largest gain: change **{best['Feature']}** by {best['Change']} → "
                           f"score rises **{best['Score Δ']} pts** to {best['New Score']}.")
            else:
                st.info("No single-feature change produces a significant improvement.")
    else:
        st.info("Fill in the form above and click **▶ Assess Applicant**.")
