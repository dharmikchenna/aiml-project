"""Streamlit demo: Employee Attrition Risk (AIML Group 14).

Run:  streamlit run app.py
Needs the 'outputs/' folder produced by the notebook (same folder as this file)
and the IBM HR CSV (upload it in the sidebar).
"""
import numpy as np
import pandas as pd
import joblib
import shap
import streamlit as st

ART = "outputs"
DROP = ["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"]

st.set_page_config(page_title="Employee Attrition Risk", layout="wide")


@st.cache_resource
def load_artifacts():
    model = joblib.load(f"{ART}/model1_xgboost_attrition.joblib")
    return {
        "model": model,
        "platt": joblib.load(f"{ART}/model2_confidence_engine.joblib"),
        "kmeans": joblib.load(f"{ART}/model4_persona_clusterer.joblib"),
        "scaler": joblib.load(f"{ART}/scaler.joblib"),
        "columns": joblib.load(f"{ART}/model_columns.joblib"),
        "config": joblib.load(f"{ART}/config.joblib"),
        "explainer": shap.TreeExplainer(model),     # rebuilt from model 1
    }


def prepare(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Same preprocessing as training. get_dummies WITHOUT drop_first, then keep only the trained columns,
    so the result is correct even if the uploaded file lacks some categories."""
    df = df.copy()
    if "OverTime" in df:
        df["OverTime"] = df["OverTime"].map({"Yes": 1, "No": 0})
    df = df.drop(columns=[c for c in DROP + ["Attrition"] if c in df.columns])
    df = pd.get_dummies(df)
    return df.reindex(columns=columns, fill_value=0).astype(float)


def sigmoid_logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


st.title("Employee Attrition Risk — HR view")
st.caption("Flags employees likely to leave within 12 months, how sure the model is, and why. "
           "Trained on the fictional IBM HR dataset: a prototype, not a validated HR tool.")

art = load_artifacts()
threshold = art["config"]["threshold"]

file = st.sidebar.file_uploader("Upload the IBM HR CSV (WA_Fn-UseC_-HR-Employee-Attrition.csv)", type="csv")
if file is None:
    st.info("Upload the IBM HR attrition CSV in the sidebar to begin.")
    st.stop()

raw_df = pd.read_csv(file)
X = prepare(raw_df, art["columns"])
X_scaled = art["scaler"].transform(X)
raw_probs = art["model"].predict_proba(X_scaled)[:, 1]

st.sidebar.metric("Employees in file", len(X))
st.sidebar.metric("Flagged at-risk", int((raw_probs >= threshold).sum()))
only_flagged = st.sidebar.checkbox("Show only flagged employees", value=True)

candidates = np.where(raw_probs >= threshold)[0] if only_flagged else np.arange(len(X))
if len(candidates) == 0:
    st.warning("No employees to show.")
    st.stop()

idx = st.sidebar.selectbox("Choose an employee (row number)", candidates,
                           format_func=lambda i: f"Row {i}  (risk score {raw_probs[i]*100:.0f}%)")

raw_prob = float(raw_probs[idx])
cal_prob = float(art["platt"].predict_proba(sigmoid_logit([raw_prob]))[0, 1])
flagged = raw_prob >= threshold
confidence = cal_prob if flagged else 1 - cal_prob
cluster = int(art["kmeans"].predict(X_scaled[idx:idx + 1])[0])
persona = art["config"]["persona_names"][cluster]

sv = art["explainer"].shap_values(pd.DataFrame(X_scaled[idx:idx + 1], columns=art["columns"]))
sv = np.asarray(sv[1] if isinstance(sv, list) else sv)[0]
top = np.argsort(sv)[::-1][:3]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Status", "⚠️ At risk" if flagged else "✅ Low risk")
c2.metric("Model risk score", f"{raw_prob*100:.1f}%", help=f"Flag threshold = {threshold*100:.0f}%")
c3.metric("Calibrated chance of leaving", f"{cal_prob*100:.1f}%")
c4.metric("Confidence", f"{confidence*100:.1f}%", help="Chance the model's call is correct")

st.subheader(f"Risk persona: {persona}")
st.subheader("Top 3 factors increasing this employee's risk")
for rank, j in enumerate(top, 1):
    name = art["columns"][j]
    st.write(f"**{rank}. {name}** = {X.iloc[idx][name]:g}   (SHAP {sv[j]:+.2f})")

with st.expander("Employee record"):
    st.dataframe(raw_df.iloc[[idx]].T)
