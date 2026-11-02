"""
Employee Attrition Risk — Streamlit App
AIML Group 14

A multi-page Streamlit dashboard for predicting and exploring
employee attrition risk using XGBoost, SHAP, Platt calibration,
and K-Means persona clustering.

Run:  streamlit run app.py
"""
import os
import numpy as np
import pandas as pd
import joblib
import shap
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─────────────────────────  CONFIG  ─────────────────────────
ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
DROP = ["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"]

st.set_page_config(
    page_title="Employee Attrition Risk — AIML Group 14",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────  CUSTOM CSS  ─────────────────────────
st.markdown("""
<style>
    /* Main metrics cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #667eea11, #764ba211);
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    div[data-testid="stMetric"] label {
        font-size: 0.85rem !important;
        color: #666 !important;
    }

    /* Risk badge styles */
    .risk-high { color: #e74c3c; font-weight: 700; font-size: 1.5rem; }
    .risk-low  { color: #27ae60; font-weight: 700; font-size: 1.5rem; }

    /* Page header */
    .page-header {
        background: linear-gradient(135deg, #1a1a2e, #16213e, #0f3460);
        padding: 24px 32px;
        border-radius: 16px;
        margin-bottom: 24px;
        color: white;
    }
    .page-header h1 { color: white; margin: 0; font-size: 1.8rem; }
    .page-header p  { color: #b0b0d0; margin: 4px 0 0 0; font-size: 0.95rem; }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8f9fc, #eef1f8);
    }

    /* Cards */
    .info-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #e8e8e8;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────  LOAD ARTIFACTS  ─────────────────────────
@st.cache_resource
def load_artifacts():
    """Load all serialised model artefacts from the outputs directory."""
    model = joblib.load(f"{ART}/model1_xgboost_attrition.joblib")
    return {
        "model":     model,
        "platt":     joblib.load(f"{ART}/model2_confidence_engine.joblib"),
        "kmeans":    joblib.load(f"{ART}/model4_persona_clusterer.joblib"),
        "scaler":    joblib.load(f"{ART}/scaler.joblib"),
        "columns":   joblib.load(f"{ART}/model_columns.joblib"),
        "config":    joblib.load(f"{ART}/config.joblib"),
        "explainer": shap.TreeExplainer(model),
    }


# ─────────────────────────  HELPERS  ─────────────────────────
def prepare(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Replicate the training preprocessing pipeline."""
    df = df.copy()
    if "OverTime" in df:
        df["OverTime"] = df["OverTime"].map({"Yes": 1, "No": 0})
    df = df.drop(columns=[c for c in DROP + ["Attrition"] if c in df.columns])
    df = pd.get_dummies(df)
    return df.reindex(columns=columns, fill_value=0).astype(float)


def sigmoid_logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def risk_color(prob, threshold):
    """Return a hex colour on a green→amber→red gradient."""
    if prob >= threshold * 1.3:
        return "#e74c3c"
    elif prob >= threshold:
        return "#e67e22"
    else:
        return "#27ae60"


def make_gauge(value, title, color):
    """Create a Plotly gauge / indicator chart."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        number={"suffix": "%", "font": {"size": 36}},
        title={"text": title, "font": {"size": 14}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#ddd"},
            "bar": {"color": color},
            "bgcolor": "#f7f7f7",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 35], "color": "#e8f5e9"},
                {"range": [35, 65], "color": "#fff3e0"},
                {"range": [65, 100], "color": "#ffebee"},
            ],
        },
    ))
    fig.update_layout(height=220, margin=dict(l=20, r=20, t=40, b=10))
    return fig


# ─────────────────────────  SIDEBAR  ─────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/conference-call.png", width=64)
    st.title("HR Attrition Risk")
    st.caption("AIML Group 14 · NCP Semester 3")
    st.divider()

    page = st.radio(
        "Navigate",
        ["🏠 Dashboard", "👤 Employee Deep-Dive", "📊 Model Performance",
         "⚖️ Fairness Audit", "📈 Workforce Analytics"],
        label_visibility="collapsed",
    )
    st.divider()

    file = st.file_uploader(
        "📂 Upload IBM HR CSV",
        type="csv",
        help="Upload *WA_Fn-UseC_-HR-Employee-Attrition.csv*",
    )

if file is None:
    # ─── Landing page ───
    st.markdown("""
    <div class="page-header">
        <h1>🏢 Employee Attrition Risk Platform</h1>
        <p>Predict which employees are likely to leave, understand why, and plan retention strategies.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="info-card">
            <h3>🤖 4-Model Pipeline</h3>
            <p>XGBoost predictor → Platt calibrator → SHAP explainer → K-Means personas.
            Each employee gets a risk score, calibrated probability, explanation, and persona.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="info-card">
            <h3>🔍 Explainable AI</h3>
            <p>SHAP values reveal the top factors pushing each employee toward attrition.
            No black box — every prediction is transparent.</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="info-card">
            <h3>⚖️ Fairness Aware</h3>
            <p>Built-in fairness audit across gender, marital status, and age bands.
            Helps HR teams spot and mitigate bias.</p>
        </div>
        """, unsafe_allow_html=True)

    st.info("👈 **Upload the IBM HR attrition CSV in the sidebar to begin.**")
    st.stop()

# ─────────────────────────  DATA LOADING  ─────────────────────────
art = load_artifacts()
threshold = art["config"]["threshold"]
persona_names = art["config"]["persona_names"]

raw_df = pd.read_csv(file)
X = prepare(raw_df, art["columns"])
X_scaled = art["scaler"].transform(X)
raw_probs = art["model"].predict_proba(X_scaled)[:, 1]

# Pre-compute calibrated probs and personas for all employees
cal_probs = art["platt"].predict_proba(sigmoid_logit(raw_probs))[:, 1]
flagged_mask = raw_probs >= threshold
clusters = art["kmeans"].predict(X_scaled)
personas = [persona_names[c] for c in clusters]

raw_df["_risk_score"] = raw_probs
raw_df["_calibrated_prob"] = cal_probs
raw_df["_flagged"] = flagged_mask
raw_df["_persona"] = personas


# ═══════════════════════════════════════════════════════════
#  PAGE 1 — DASHBOARD
# ═══════════════════════════════════════════════════════════
if page == "🏠 Dashboard":
    st.markdown("""
    <div class="page-header">
        <h1>🏠 Workforce Risk Dashboard</h1>
        <p>Bird's-eye view of attrition risk across your workforce.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Top-level KPIs ──
    n_total = len(raw_df)
    n_flagged = int(flagged_mask.sum())
    avg_risk = float(raw_probs.mean())
    avg_cal = float(cal_probs[flagged_mask].mean()) if n_flagged > 0 else 0.0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Employees", f"{n_total:,}")
    k2.metric("⚠️ Flagged At-Risk", f"{n_flagged:,}",
              delta=f"{n_flagged/n_total*100:.1f}%", delta_color="inverse")
    k3.metric("Avg Risk Score", f"{avg_risk*100:.1f}%")
    k4.metric("Avg Calibrated Prob (flagged)", f"{avg_cal*100:.1f}%")

    st.divider()

    # ── Row 2: Distribution + Personas ──
    col_dist, col_persona = st.columns([3, 2])

    with col_dist:
        st.subheader("Risk Score Distribution")
        fig_hist = px.histogram(
            raw_df, x="_risk_score", nbins=40,
            color="_flagged",
            color_discrete_map={True: "#e74c3c", False: "#27ae60"},
            labels={"_risk_score": "Risk Score", "_flagged": "Flagged"},
            opacity=0.8,
        )
        fig_hist.add_vline(
            x=threshold, line_dash="dash", line_color="#333",
            annotation_text=f"Threshold = {threshold:.2f}",
            annotation_position="top right",
        )
        fig_hist.update_layout(
            bargap=0.05, height=380,
            xaxis_title="Risk Score", yaxis_title="Employees",
            legend_title="At Risk",
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_persona:
        st.subheader("Risk Personas (flagged)")
        flagged_df = raw_df[raw_df["_flagged"]]
        if len(flagged_df) > 0:
            persona_counts = flagged_df["_persona"].value_counts().reset_index()
            persona_counts.columns = ["Persona", "Count"]
            fig_pie = px.pie(
                persona_counts, names="Persona", values="Count",
                hole=0.45,
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig_pie.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=10))
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.success("No employees flagged at the current threshold!")

    # ── Row 3: Department breakdown ──
    if "Department" in raw_df.columns:
        st.subheader("Risk by Department")
        dept_risk = raw_df.groupby("Department").agg(
            avg_risk=("_risk_score", "mean"),
            n_flagged=("_flagged", "sum"),
            total=("_flagged", "count"),
        ).reset_index()
        dept_risk["flag_rate"] = dept_risk["n_flagged"] / dept_risk["total"] * 100
        dept_risk = dept_risk.sort_values("flag_rate", ascending=True)

        fig_dept = px.bar(
            dept_risk, y="Department", x="flag_rate",
            orientation="h",
            text=dept_risk["flag_rate"].apply(lambda x: f"{x:.1f}%"),
            color="flag_rate",
            color_continuous_scale=["#27ae60", "#e67e22", "#e74c3c"],
            labels={"flag_rate": "Flag Rate (%)"},
        )
        fig_dept.update_layout(
            height=280, showlegend=False, coloraxis_showscale=False,
            xaxis_title="Flag Rate (%)", yaxis_title="",
        )
        fig_dept.update_traces(textposition="outside")
        st.plotly_chart(fig_dept, use_container_width=True)

    # ── Row 4: Flagged employees table ──
    st.subheader("🚨 Flagged Employees")
    if n_flagged > 0:
        show_cols = [c for c in ["EmployeeNumber", "Age", "Department", "JobRole",
                                  "MonthlyIncome", "OverTime", "YearsAtCompany"]
                     if c in raw_df.columns]
        show_cols += ["_risk_score", "_calibrated_prob", "_persona"]
        display_df = flagged_df[show_cols].sort_values("_risk_score", ascending=False)
        display_df.columns = [c.lstrip("_").replace("_", " ").title() for c in display_df.columns]
        st.dataframe(
            display_df.style.format({
                "Risk Score": "{:.1%}",
                "Calibrated Prob": "{:.1%}",
            }).background_gradient(subset=["Risk Score"], cmap="RdYlGn_r"),
            use_container_width=True,
            height=400,
        )
    else:
        st.success("No employees are flagged at the current threshold.")


# ═══════════════════════════════════════════════════════════
#  PAGE 2 — EMPLOYEE DEEP-DIVE
# ═══════════════════════════════════════════════════════════
elif page == "👤 Employee Deep-Dive":
    st.markdown("""
    <div class="page-header">
        <h1>👤 Employee Deep-Dive</h1>
        <p>Inspect an individual employee's risk profile, SHAP explanation, and persona.</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar controls
    with st.sidebar:
        only_flagged = st.checkbox("Show only flagged employees", value=True)
        candidates = np.where(flagged_mask)[0] if only_flagged else np.arange(len(X))

        if len(candidates) == 0:
            st.warning("No employees match the filter.")
            st.stop()

        idx = st.selectbox(
            "Choose an employee (row #)",
            candidates,
            format_func=lambda i: f"Row {i}  —  risk {raw_probs[i]*100:.0f}%",
        )

    raw_prob = float(raw_probs[idx])
    cal_prob = float(cal_probs[idx])
    is_flagged = bool(flagged_mask[idx])
    confidence = cal_prob if is_flagged else 1 - cal_prob
    persona = personas[idx]

    # ── Gauge Row ──
    g1, g2, g3 = st.columns(3)
    with g1:
        st.plotly_chart(
            make_gauge(raw_prob, "Model Risk Score", risk_color(raw_prob, threshold)),
            use_container_width=True,
        )
    with g2:
        st.plotly_chart(
            make_gauge(cal_prob, "Calibrated Probability", risk_color(cal_prob, threshold)),
            use_container_width=True,
        )
    with g3:
        st.plotly_chart(
            make_gauge(confidence, "Model Confidence", "#3498db"),
            use_container_width=True,
        )

    # ── Status banner ──
    if is_flagged:
        st.error(f"⚠️ **AT RISK** — Risk Persona: **{persona}** — Threshold: {threshold*100:.0f}%")
    else:
        st.success(f"✅ **LOW RISK** — Persona: **{persona}** — Below the {threshold*100:.0f}% threshold")

    st.divider()

    # ── SHAP explanation ──
    st.subheader("🔬 Why is this employee at risk?")
    sv = art["explainer"].shap_values(
        pd.DataFrame(X_scaled[idx:idx + 1], columns=art["columns"])
    )
    sv = np.asarray(sv[1] if isinstance(sv, list) else sv)[0]

    # Top N factors
    n_top = st.slider("Number of top factors to show", 3, 15, 5)
    top_pos = np.argsort(sv)[::-1][:n_top]

    shap_df = pd.DataFrame({
        "Feature": [art["columns"][j] for j in top_pos],
        "Value": [X.iloc[idx][art["columns"][j]] for j in top_pos],
        "SHAP Impact": [sv[j] for j in top_pos],
    })

    fig_shap = px.bar(
        shap_df, y="Feature", x="SHAP Impact",
        orientation="h",
        color="SHAP Impact",
        color_continuous_scale=["#27ae60", "#f1c40f", "#e74c3c"],
        text=shap_df["Value"].apply(lambda v: f"val={v:g}"),
    )
    fig_shap.update_layout(
        height=max(250, n_top * 45), yaxis={"autorange": "reversed"},
        coloraxis_showscale=False,
        xaxis_title="SHAP Value (positive = increases risk)",
        yaxis_title="",
    )
    fig_shap.update_traces(textposition="outside")
    st.plotly_chart(fig_shap, use_container_width=True)

    # ── Full employee record ──
    with st.expander("📋 Full Employee Record"):
        rec = raw_df.iloc[[idx]].drop(columns=[c for c in raw_df.columns if c.startswith("_")])
        st.dataframe(rec.T.rename(columns={rec.index[0]: "Value"}), use_container_width=True)


# ═══════════════════════════════════════════════════════════
#  PAGE 3 — MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════
elif page == "📊 Model Performance":
    st.markdown("""
    <div class="page-header">
        <h1>📊 Model Performance</h1>
        <p>Baseline comparisons, threshold analysis, and explainability plots from training.</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📋 Baselines", "🎯 Threshold Analysis", "🔍 Explainability"])

    # ── Tab 1: Baselines ──
    with tab1:
        st.subheader("Cross-Validation Results (5-Fold)")
        cv_df = pd.read_csv(f"{ART}/baseline_cv.csv")
        st.dataframe(cv_df.style.highlight_max(axis=0, color="#d4edda"), use_container_width=True)

        st.subheader("Hold-Out Test Results")
        test_df = pd.read_csv(f"{ART}/baseline_test.csv")
        st.dataframe(test_df.style.highlight_max(axis=0, color="#d4edda"), use_container_width=True)

        # Visual comparison
        fig_compare = make_subplots(rows=1, cols=2, subplot_titles=["ROC-AUC", "PR-AUC"])
        for i, metric in enumerate(["roc_auc", "pr_auc"]):
            fig_compare.add_trace(
                go.Bar(
                    x=test_df["model"], y=test_df[metric],
                    marker_color=["#3498db", "#2ecc71", "#e74c3c"],
                    text=test_df[metric].apply(lambda v: f"{v:.3f}"),
                    textposition="outside",
                    showlegend=False,
                ),
                row=1, col=i + 1,
            )
        fig_compare.update_layout(height=350, margin=dict(t=40, b=20))
        st.plotly_chart(fig_compare, use_container_width=True)

    # ── Tab 2: Threshold ──
    with tab2:
        st.subheader("Precision / Recall Trade-off")
        thresh_df = pd.read_csv(f"{ART}/threshold_analysis.csv")
        fig_thresh = go.Figure()
        for col, color, name in [
            ("precision", "#e67e22", "Precision"),
            ("recall", "#3498db", "Recall"),
            ("f1", "#27ae60", "F1"),
        ]:
            fig_thresh.add_trace(go.Scatter(
                x=thresh_df["threshold"], y=thresh_df[col],
                mode="lines+markers", name=name,
                line=dict(color=color, width=2),
            ))
        fig_thresh.add_vline(
            x=threshold, line_dash="dash", line_color="#333",
            annotation_text=f"Chosen = {threshold}",
        )
        fig_thresh.update_layout(
            height=400,
            xaxis_title="Decision Threshold", yaxis_title="Score",
            legend=dict(orientation="h", y=1.12),
        )
        st.plotly_chart(fig_thresh, use_container_width=True)

        st.subheader("Flagged % by Threshold")
        fig_flag = px.area(
            thresh_df, x="threshold", y="flagged_%",
            labels={"flagged_%": "Flagged %", "threshold": "Threshold"},
            color_discrete_sequence=["#e74c3c"],
        )
        fig_flag.update_layout(height=300)
        st.plotly_chart(fig_flag, use_container_width=True)

    # ── Tab 3: Explainability ──
    with tab3:
        st.subheader("Confusion Matrix (Threshold = 0.35)")
        st.image(f"{ART}/confusion_matrix.png", width=500)

        st.subheader("SHAP Beeswarm — Top 10 Global Drivers")
        st.image(f"{ART}/shap_summary_beeswarm.png", use_container_width=True)

        st.subheader("SHAP + Persona Dashboard")
        st.image(f"{ART}/dashboard_shap_personas.png", use_container_width=True)


# ═══════════════════════════════════════════════════════════
#  PAGE 4 — FAIRNESS AUDIT
# ═══════════════════════════════════════════════════════════
elif page == "⚖️ Fairness Audit":
    st.markdown("""
    <div class="page-header">
        <h1>⚖️ Fairness Audit</h1>
        <p>Evaluate model behaviour across protected attributes: Gender, Marital Status, Age.</p>
    </div>
    """, unsafe_allow_html=True)

    fair_df = pd.read_csv(f"{ART}/fairness_check.csv")

    for attr in fair_df["attribute"].unique():
        st.subheader(f"📌 {attr}")
        subset = fair_df[fair_df["attribute"] == attr]

        col_chart, col_table = st.columns([3, 2])

        with col_chart:
            fig_fair = go.Figure()
            fig_fair.add_trace(go.Bar(
                x=subset["group"], y=subset["actual_leave_%"],
                name="Actual Leave %", marker_color="#3498db",
            ))
            fig_fair.add_trace(go.Bar(
                x=subset["group"], y=subset["flag_rate_%"],
                name="Flag Rate %", marker_color="#e74c3c",
            ))
            fig_fair.update_layout(
                barmode="group", height=320,
                yaxis_title="Percentage (%)",
                legend=dict(orientation="h", y=1.12),
            )
            st.plotly_chart(fig_fair, use_container_width=True)

        with col_table:
            st.dataframe(
                subset.drop(columns=["attribute"]).style.format({
                    "actual_leave_%": "{:.1f}%",
                    "flag_rate_%": "{:.1f}%",
                    "recall": "{:.3f}",
                    "precision": "{:.3f}",
                }),
                use_container_width=True,
                height=300,
            )

    st.divider()
    st.markdown("""
    > **Interpreting this audit:**
    > - **Flag Rate** should roughly track **Actual Leave %** across groups for equitable treatment.
    > - Large gaps between flag rate and actual leave rate may indicate systematic over- or under-flagging.
    > - **Recall** shows how well the model catches actual leavers within each group.
    > - This is a *descriptive* audit on the IBM fictional dataset — real deployments need deeper analysis.
    """)


# ═══════════════════════════════════════════════════════════
#  PAGE 5 — WORKFORCE ANALYTICS
# ═══════════════════════════════════════════════════════════
elif page == "📈 Workforce Analytics":
    st.markdown("""
    <div class="page-header">
        <h1>📈 Workforce Analytics</h1>
        <p>Explore patterns in the uploaded workforce data through interactive charts.</p>
    </div>
    """, unsafe_allow_html=True)

    analysis_df = raw_df.drop(columns=[c for c in raw_df.columns if c.startswith("_")])

    tab_a, tab_b, tab_c = st.tabs(["Demographics", "Compensation", "Risk Factors"])

    with tab_a:
        c1, c2 = st.columns(2)
        with c1:
            if "Age" in analysis_df.columns:
                st.subheader("Age Distribution")
                fig_age = px.histogram(
                    raw_df, x="Age", color="_flagged",
                    color_discrete_map={True: "#e74c3c", False: "#27ae60"},
                    nbins=30, barmode="overlay", opacity=0.7,
                    labels={"_flagged": "At Risk"},
                )
                fig_age.update_layout(height=350)
                st.plotly_chart(fig_age, use_container_width=True)
        with c2:
            if "Department" in analysis_df.columns and "Gender" in analysis_df.columns:
                st.subheader("Department × Gender")
                fig_dg = px.histogram(
                    analysis_df, x="Department", color="Gender",
                    barmode="group",
                    color_discrete_sequence=px.colors.qualitative.Set2,
                )
                fig_dg.update_layout(height=350)
                st.plotly_chart(fig_dg, use_container_width=True)

        if "YearsAtCompany" in analysis_df.columns:
            st.subheader("Tenure Distribution")
            fig_tenure = px.histogram(
                raw_df, x="YearsAtCompany", color="_flagged",
                color_discrete_map={True: "#e74c3c", False: "#27ae60"},
                nbins=20, barmode="overlay", opacity=0.7,
                labels={"_flagged": "At Risk", "YearsAtCompany": "Years at Company"},
            )
            fig_tenure.update_layout(height=300)
            st.plotly_chart(fig_tenure, use_container_width=True)

    with tab_b:
        c1, c2 = st.columns(2)
        with c1:
            if "MonthlyIncome" in analysis_df.columns:
                st.subheader("Monthly Income Distribution")
                fig_inc = px.box(
                    raw_df, x="_flagged", y="MonthlyIncome",
                    color="_flagged",
                    color_discrete_map={True: "#e74c3c", False: "#27ae60"},
                    labels={"_flagged": "At Risk", "MonthlyIncome": "Monthly Income"},
                )
                fig_inc.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig_inc, use_container_width=True)
        with c2:
            if "MonthlyIncome" in analysis_df.columns and "JobRole" in analysis_df.columns:
                st.subheader("Income by Job Role")
                fig_role_inc = px.box(
                    analysis_df, y="JobRole", x="MonthlyIncome",
                    orientation="h",
                    color_discrete_sequence=["#3498db"],
                )
                fig_role_inc.update_layout(height=400)
                st.plotly_chart(fig_role_inc, use_container_width=True)

    with tab_c:
        st.subheader("Risk Score vs Key Features")
        numeric_cols = [c for c in analysis_df.select_dtypes(include=[np.number]).columns
                        if c not in DROP and c != "EmployeeNumber"]
        sel_x = st.selectbox("X-axis feature", numeric_cols,
                             index=numeric_cols.index("MonthlyIncome") if "MonthlyIncome" in numeric_cols else 0)
        sel_y = st.selectbox("Y-axis feature", numeric_cols,
                             index=numeric_cols.index("Age") if "Age" in numeric_cols else 1)

        fig_scatter = px.scatter(
            raw_df, x=sel_x, y=sel_y,
            color="_risk_score",
            color_continuous_scale=["#27ae60", "#f1c40f", "#e74c3c"],
            size="_risk_score", size_max=12,
            opacity=0.6,
            labels={"_risk_score": "Risk Score"},
            hover_data=["_persona"],
        )
        fig_scatter.update_layout(height=500)
        st.plotly_chart(fig_scatter, use_container_width=True)

# ─────────────────────────  FOOTER  ─────────────────────────
st.divider()
st.caption(
    "🏢 Employee Attrition Risk Platform · AIML Group 14 · NCP Semester 3 · "
    "Built on the fictional IBM HR dataset — prototype only, not a validated HR tool."
)
