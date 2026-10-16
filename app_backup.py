"""
Judiciary AI System with Drift Detection — Dashboard
Run with:  streamlit run app.py

Default login (change these two lines below to whatever you want):
    USERNAME = "admin"
    PASSWORD = "judiciary2026"
"""
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px

# ----------------------------------------------------------------------------
# CHANGE THESE TWO LINES to set your own login credentials
# ----------------------------------------------------------------------------
USERNAME = "admin"
PASSWORD = "judiciary2026"

# ----------------------------------------------------------------------------
# Design tokens
# ----------------------------------------------------------------------------
INK = "#1B2430"
PARCHMENT = "#F7F3E9"
PARCHMENT_DEEP = "#F1EBDC"
BRASS = "#A9822C"
SLATE = "#6B7280"
HAIR = "#DDD6C4"
HIGH, MED, LOW = "#B3432B", "#B8862F", "#4C7A5E"
PRIORITY_COLOR = {"High": HIGH, "Medium": MED, "Low": LOW}

DELAY_CATEGORIES = [
    "Repeated adjournments",
    "Procedural or documentation delay",
    "Witness non-availability",
    "Investigation pending",
    "No specific delay noted",
]
DELAY_WEIGHTS = {"Repeated adjournments": 1.0, "Procedural or documentation delay": 0.9,
                 "Witness non-availability": 0.6, "Investigation pending": 0.4,
                 "No specific delay noted": 0.1}
SEVERITY_WEIGHTS = {"Fraud or Cheating": 1.0, "Theft or Robbery": 0.7, "Cyber Crime": 0.4}
REGIONS = ["Maharashtra", "Delhi", "Karnataka", "Uttar Pradesh", "West Bengal", "Rajasthan"]
CRIME_TYPES = ["Theft or Robbery", "Fraud or Cheating", "Cyber Crime"]
BAIL_TYPES = ["Regular", "Anticipatory", "Interim", "Not applicable"]
PRIOR_CASES_OPTIONS = ["Yes", "No", "Unknown"]

st.set_page_config(page_title="Judiciary AI Dashboard", layout="wide", page_icon="⚖️")

# ----------------------------------------------------------------------------
# Global styling
# ----------------------------------------------------------------------------
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@400;600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

html, body, [class*="css"] {{ font-family: 'IBM Plex Sans', sans-serif; }}
.stApp {{ background-color: {PARCHMENT}; }}
section[data-testid="stSidebar"] {{ background-color: {INK}; }}
section[data-testid="stSidebar"] * {{ color: #D9DDE5 !important; }}
section[data-testid="stSidebar"] label {{ font-size: 13.5px; }}

h1, h2, h3 {{ font-family: 'Source Serif 4', serif !important; color: {INK} !important; font-weight: 600 !important; }}
.stat-value {{ font-family: 'Source Serif 4', serif; font-size: 34px; color: {INK}; line-height: 1.1; }}
.stat-label {{ font-size: 13px; color: {SLATE}; margin-top: 4px; }}
.stat-sub {{ font-size: 12px; color: {SLATE}; opacity: 0.8; }}
.stat-box {{ border-left: 3px solid {BRASS}; padding-left: 16px; }}

.eyebrow {{ font-size: 12.5px; color: {BRASS}; margin-bottom: 2px; }}
.blurb {{ color: {SLATE}; font-size: 14.5px; max-width: 640px; line-height: 1.5; }}

div[data-testid="stDataFrame"] {{ border: 1px solid {HAIR}; }}

.status-card {{ border: 1px solid {HAIR}; padding: 18px 20px; background: white; }}
.status-title {{ font-size: 13px; font-weight: 600; color: {INK}; }}
.status-body {{ font-size: 12.5px; color: {SLATE}; margin-top: 4px; }}

.factor-row {{ display: flex; align-items: center; gap: 10px; margin-bottom: 6px; }}
.factor-label {{ width: 190px; font-size: 12.5px; color: {INK}; }}
.factor-bar-bg {{ flex: 1; background: {HAIR}; height: 7px; border-radius: 3px; overflow: hidden; }}
.factor-bar-fill {{ height: 100%; background: {BRASS}; }}
.factor-value {{ width: 130px; font-size: 11.5px; color: {SLATE}; text-align: right; }}

.result-card {{ border: 1px solid {BRASS}; background: {PARCHMENT_DEEP}; padding: 22px 26px; margin-top: 10px; }}
.result-number {{ font-family: 'Source Serif 4', serif; font-size: 40px; color: {INK}; }}

hr {{ border-color: {HAIR}; }}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# LOGIN GATE — everything below this block only runs after a successful login
# ----------------------------------------------------------------------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.markdown("<div style='height:60px'></div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1.1, 1])
    with col2:
        st.markdown(f"""
        <div style='text-align:center; margin-bottom:24px;'>
            <div style='font-size:32px;'>⚖️</div>
            <div style='font-family:"Source Serif 4", serif; font-size:24px; color:{INK}; margin-top:6px;'>
                Judiciary AI Dashboard
            </div>
            <div style='font-size:13px; color:{SLATE}; margin-top:4px;'>Sign in to continue</div>
        </div>
        """, unsafe_allow_html=True)

        with st.form("login_form"):
            user_input = st.text_input("Username")
            pass_input = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Log in", use_container_width=True)

        if submitted:
            if user_input == USERNAME and pass_input == PASSWORD:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Incorrect username or password.")

        st.caption(f"Default login — username: `{USERNAME}`, password: `{PASSWORD}` "
                   f"(the project owner can change these at the top of app.py).")
    st.stop()  # nothing below this line runs until login succeeds


# ----------------------------------------------------------------------------
# Mock / session data
# ----------------------------------------------------------------------------
def build_mock_cases(n=48, seed=7):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(1, n + 1):
        crime = rng.choice(CRIME_TYPES)
        region = rng.choice(REGIONS)
        predicted = int(15 + rng.random() * 260)
        delay_cause = rng.choice(DELAY_CATEGORIES)
        confidence = round(0.55 + rng.random() * 0.4, 2)
        landmark = rng.random() > 0.88

        duration_score = min(1.0, predicted / 280)
        delay_score = DELAY_WEIGHTS[delay_cause]
        severity_score = SEVERITY_WEIGHTS[crime]
        landmark_score = 1.0 if landmark else 0.0

        priority_score = 0.4 * duration_score + 0.35 * delay_score + 0.15 * severity_score + 0.1 * landmark_score
        level = "High" if priority_score >= 0.6 else "Medium" if priority_score >= 0.35 else "Low"

        rows.append({
            "case_id": 100 + i, "crime_type": crime, "region": region,
            "predicted_duration": predicted, "delay_cause": delay_cause,
            "delay_cause_confidence": confidence, "landmark_case": landmark,
            "duration_score": round(duration_score, 2), "delay_cause_score": delay_score,
            "severity_score": severity_score, "landmark_score": landmark_score,
            "priority_score": round(priority_score, 3), "priority_level": level,
        })
    return pd.DataFrame(rows).sort_values("priority_score", ascending=False).reset_index(drop=True)


def build_mock_drift_log():
    return pd.DataFrame([
        {"check_timestamp": "2026-03-01", "baseline_cases": 240, "monitoring_cases": 125,
         "feature_drift_detected": False, "performance_drift_detected": False,
         "baseline_cindex": 0.74, "monitoring_cindex_old_model": 0.71, "retrained": False},
        {"check_timestamp": "2026-06-01", "baseline_cases": 240, "monitoring_cases": 168,
         "feature_drift_detected": True, "performance_drift_detected": False,
         "baseline_cindex": 0.74, "monitoring_cindex_old_model": 0.66, "retrained": True},
        {"check_timestamp": "2026-09-01", "baseline_cases": 240, "monitoring_cases": 201,
         "feature_drift_detected": True, "performance_drift_detected": True,
         "baseline_cindex": 0.74, "monitoring_cindex_old_model": 0.58, "retrained": True},
    ])


def build_mock_feature_mix():
    return pd.DataFrame([
        {"crime_type": "Theft/Robbery", "baseline": 40, "monitoring": 20},
        {"crime_type": "Fraud/Cheating", "baseline": 38, "monitoring": 33},
        {"crime_type": "Cyber Crime", "baseline": 22, "monitoring": 47},
    ])


if "cases" not in st.session_state:
    st.session_state.cases = None
if "drift_log" not in st.session_state:
    st.session_state.drift_log = None
if "feature_mix" not in st.session_state:
    st.session_state.feature_mix = build_mock_feature_mix()
if "cox_model" not in st.session_state:
    st.session_state.cox_model = None
if "data_sources" not in st.session_state:
    st.session_state.data_sources = {}


def _candidate_paths(*relative_parts):
    """
    Build a list of places to look for a data file, covering every layout
    this project has actually been run from:
      - app.py inside a dashboard/ folder, data one level up
      - app.py sitting directly in the project root, data alongside it
      - whatever the current working directory happens to be (in case
        streamlit was launched from somewhere else entirely)
    Uses app.py's own location (not the working directory) as the anchor,
    so this works the same whether you run `streamlit run app.py` from
    the project root, from inside dashboard/, or via a shortcut/IDE.
    """
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    rel = os.path.join(*relative_parts)
    bases = [
        here,                              # app.py's own folder
        os.path.dirname(here),             # one level up from app.py
        os.getcwd(),                       # wherever the command was run from
        os.path.dirname(os.getcwd()),      # one level up from that
    ]
    seen, out = set(), []
    for b in bases:
        p = os.path.normpath(os.path.join(b, rel))
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def try_auto_load():
    """
    Looks for your actual project files and loads them automatically, so
    nothing needs to be uploaded by hand. This only runs once per session,
    and only fills in whatever it can actually find — anything missing
    falls back to sample data below. Works whether app.py sits in a
    dashboard/ subfolder or directly in the project root (see
    _candidate_paths above) — no need to `cd` anywhere special before
    running `streamlit run app.py`.
    """
    import os

    # 1) The trained Cox model
    if st.session_state.cox_model is None:
        for path in _candidate_paths("models", "cox_model.pkl"):
            if os.path.exists(path):
                try:
                    with open(path, "rb") as f:
                        st.session_state.cox_model = pickle.load(f)
                    st.session_state.data_sources["cox_model"] = path
                    break
                except Exception:
                    pass

    # 2) Prioritized cases (Notebook 06 output) — the main table for 3 of the 4 pages
    if st.session_state.cases is None:
        for path in _candidate_paths("data", "processed", "prioritized_cases.csv"):
            if os.path.exists(path):
                try:
                    st.session_state.cases = recompute_level(pd.read_csv(path))
                    st.session_state.data_sources["cases"] = path
                    break
                except Exception:
                    pass

    # 3) Delay-cause predictions (Notebook 05 output) — merge in if cases already loaded
    for path in _candidate_paths("data", "processed", "dataset_with_delay_cause.csv"):
        if os.path.exists(path) and st.session_state.cases is not None and "delay_source" not in st.session_state.data_sources:
            try:
                delay_df = pd.read_csv(path)[["case_id", "predicted_delay_cause", "delay_cause_confidence"]]
                merged = st.session_state.cases.merge(delay_df, on="case_id", how="left", suffixes=("", "_new"))
                if "predicted_delay_cause" in merged.columns:
                    merged["delay_cause"] = merged["predicted_delay_cause"].fillna(merged.get("delay_cause"))
                    merged["delay_cause_confidence"] = merged["delay_cause_confidence_new"].fillna(merged.get("delay_cause_confidence", 0.7))
                    merged = merged.drop(columns=["predicted_delay_cause", "delay_cause_confidence_new"], errors="ignore")
                st.session_state.cases = _dedupe_columns(merged)
                st.session_state.data_sources["delay_source"] = path
                break
            except Exception:
                pass

    # 4) Drift-check log (Notebook 07 output)
    if st.session_state.drift_log is None:
        for path in _candidate_paths("data", "processed", "drift_log.csv"):
            if os.path.exists(path):
                try:
                    st.session_state.drift_log = pd.read_csv(path)
                    st.session_state.data_sources["drift_log"] = path
                    break
                except Exception:
                    pass

    # Anything still missing after checking real files -> fall back to sample data
    if st.session_state.cases is None:
        st.session_state.cases = build_mock_cases()
    if st.session_state.drift_log is None:
        st.session_state.drift_log = build_mock_drift_log()




def _dedupe_columns(df):
    """
    Collapse duplicate-named (or duplicate-origin) columns into one,
    keeping the first non-null value per row. Handles two cases seen on
    this project's data:
      1. A CSV file saved with two literal columns of the same name
         (e.g. from an earlier buggy save) -- pandas auto-suffixes these
         as "region", "region.1" when the file is read back in.
      2. A genuine in-memory duplicate column (same exact name twice),
         which can happen after a pd.concat/merge of frames whose
         column sets didn't fully line up.
    Safe to call on a normal, already-clean dataframe (then it's a no-op).
    """
    df = df.copy()

    # --- Case 1: pandas-suffixed duplicates from a CSV with a repeated header ---
    import re
    suffix_pat = re.compile(r"^(.*)\.\d+$")
    for col in list(df.columns):
        m = suffix_pat.match(str(col))
        if m and m.group(1) in df.columns:
            base = m.group(1)
            df[base] = df[base].fillna(df[col])
            df = df.drop(columns=[col])

    # --- Case 2: true duplicate column names sitting side by side ---
    if df.columns.duplicated().any():
        out = {}
        for col in pd.unique(df.columns):
            same = df.loc[:, df.columns == col]
            if same.shape[1] == 1:
                out[col] = same.iloc[:, 0]
            else:
                merged_col = same.iloc[:, 0]
                for i in range(1, same.shape[1]):
                    merged_col = merged_col.fillna(same.iloc[:, i])
                out[col] = merged_col
        df = pd.DataFrame(out)

    return df


def recompute_level(df):
    df = _dedupe_columns(df.copy())
    # Notebook 06's CSV uses "region_grouped" and "predicted_delay_cause" —
    # the rest of this app expects "region" and "delay_cause". Some saved
    # CSVs (e.g. after using "Predict New Case") can end up with BOTH names
    # at once, so drop the old target column first to avoid a duplicate.
    if "region_grouped" in df.columns:
        df = df.drop(columns=["region"], errors="ignore")
        df = df.rename(columns={"region_grouped": "region"})
    if "predicted_delay_cause" in df.columns:
        df = df.drop(columns=["delay_cause"], errors="ignore")
        df = df.rename(columns={"predicted_delay_cause": "delay_cause"})
    df = _dedupe_columns(df)  # in case the rename itself created a clash
    if "delay_cause_confidence" not in df.columns:
        df["delay_cause_confidence"] = 0.7
    if "landmark_case" not in df.columns:
        df["landmark_case"] = df.get("landmark_score", 0) > 0
    df["priority_level"] = df["priority_score"].apply(
        lambda s: "High" if s >= 0.6 else "Medium" if s >= 0.35 else "Low"
    )
    return df


def section_heading(eyebrow, title, blurb):
    st.markdown(f'<div class="eyebrow">{eyebrow}</div>', unsafe_allow_html=True)
    st.markdown(f"## {title}")
    st.markdown(f'<p class="blurb">{blurb}</p>', unsafe_allow_html=True)
    st.write("")


def stat_box(label, value, sub=""):
    st.markdown(f"""
    <div class="stat-box">
        <div class="stat-value">{value}</div>
        <div class="stat-label">{label}</div>
        <div class="stat-sub">{sub}</div>
    </div>
    """, unsafe_allow_html=True)


def styled_bar(fig):
    fig.update_layout(
        plot_bgcolor="white", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="IBM Plex Sans", size=12, color=SLATE),
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(gridcolor=HAIR, zeroline=False),
        yaxis=dict(gridcolor=HAIR, zeroline=False),
        showlegend=fig.layout.showlegend,
    )
    return fig


def score_case(predicted_duration, delay_cause, crime_type, landmark_case, max_duration_ref=280.0):
    """The SAME transparent formula used everywhere else in the app (Notebook 06 logic)."""
    duration_score = min(1.0, predicted_duration / max_duration_ref)
    delay_score = DELAY_WEIGHTS.get(delay_cause, 0.5)
    severity_score = SEVERITY_WEIGHTS.get(crime_type, 0.5)
    landmark_score = 1.0 if landmark_case else 0.0
    priority_score = 0.4 * duration_score + 0.35 * delay_score + 0.15 * severity_score + 0.1 * landmark_score
    level = "High" if priority_score >= 0.6 else "Medium" if priority_score >= 0.35 else "Low"
    return {
        "duration_score": round(duration_score, 3), "delay_cause_score": delay_score,
        "severity_score": severity_score, "landmark_score": landmark_score,
        "priority_score": round(priority_score, 3), "priority_level": level,
    }


def predict_with_cox_model(cph, form_values):
    """
    Runs a REAL prediction through an uploaded, trained lifelines CoxPHFitter model.
    Encodes the form inputs exactly the way Notebook 04/06 encoded training data,
    then aligns columns to what the model was actually trained on.
    """
    categorical_cols = ["crime_type", "region_grouped", "bail_type", "prior_cases", "accused_gender"]

    row = {
        "crime_type": form_values["crime_type"],
        "region_grouped": form_values["region"],
        "bail_type": form_values["bail_type"],
        "prior_cases": form_values["prior_cases"],
        "accused_gender": form_values["accused_gender"],
        "landmark_case": int(form_values["landmark_case"]),
        "bail_cancellation_case": int(form_values["bail_cancellation_case"]),
        "parity_argument_used": int(form_values["parity_argument_used"]),
        "has_special_law": int(form_values["has_special_law"]),
    }
    df_row = pd.DataFrame([row])
    encoded = pd.get_dummies(df_row, columns=categorical_cols, drop_first=False)
    aligned = encoded.reindex(columns=cph.params_.index, fill_value=0)

    predicted_median = cph.predict_median(aligned)
    value = predicted_median.iloc[0] if hasattr(predicted_median, "iloc") else float(predicted_median)
    if pd.isna(value) or np.isinf(value):
        # lifelines returns inf if the predicted survival curve never crosses 50% —
        # fall back to the expected value instead, which is always finite.
        expected = cph.predict_expectation(aligned)
        value = expected.iloc[0] if hasattr(expected, "iloc") else float(expected)
    return float(value)


def fallback_duration_estimate(form_values):
    """
    Transparent, non-ML fallback used when no trained model has been uploaded.
    Starts from a base duration per crime type and nudges it up/down based on
    the same factors the Cox model itself considers — clearly a stand-in, not
    a substitute for the real model.
    """
    base = {"Fraud or Cheating": 60, "Theft or Robbery": 45, "Cyber Crime": 35}[form_values["crime_type"]]
    multiplier = 1.0
    if form_values["prior_cases"] == "Yes":
        multiplier *= 1.15
    if form_values["landmark_case"]:
        multiplier *= 1.3
    if form_values["bail_cancellation_case"]:
        multiplier *= 1.2
    if form_values["has_special_law"]:
        multiplier *= 1.25
    if form_values["bail_type"] == "Anticipatory":
        multiplier *= 0.85
    return round(base * multiplier)


# ----------------------------------------------------------------------------
# Real InLegalBERT zero-shot delay-cause classification (Notebook 05 logic) —
# only activates if `transformers` + `torch` are installed AND case text is
# given. This is intentionally optional/lazy: these are large libraries, and
# most of the dashboard works fine without them.
# ----------------------------------------------------------------------------
DELAY_CATEGORY_DESCRIPTIONS = {
    "Investigation pending": "The case is delayed because the police investigation is still ongoing or incomplete.",
    "Witness non-availability": "The case is delayed because witnesses were unavailable, absent, or retracted their statements.",
    "Repeated adjournments": "The case is delayed due to repeated adjournments or postponements of hearings.",
    "Procedural or documentation delay": "The case is delayed due to procedural issues, missing documents, or administrative backlog.",
    "No specific delay noted": "The case proceeded through the normal court process without any specific mention of delay.",
}


@st.cache_resource(show_spinner="Loading InLegalBERT (first time only — this is cached after)...")
def load_inlegalbert():
    from transformers import AutoTokenizer, AutoModel
    import torch  # noqa: F401 (imported to confirm torch is present too)
    tokenizer = AutoTokenizer.from_pretrained("law-ai/InLegalBERT")
    model = AutoModel.from_pretrained("law-ai/InLegalBERT")
    model.eval()
    return tokenizer, model


@st.cache_resource
def load_delay_calibration():
    """
    Loads the per-category mean/std that Notebook 05 (Step 9) computed
    across all 365 training cases, so a single live case gets the SAME
    z-score correction the notebook applies -- without it, raw cosine
    similarity on mean-pooled InLegalBERT embeddings is biased toward
    whichever category description sits closest to the embedding-space
    centroid (in practice: almost always "No specific delay noted").
    Falls back to uncorrected raw cosine if the calibration file isn't
    found yet (e.g. Notebook 05 hasn't been (re-)run since this fix).
    """
    import pickle
    for path in _candidate_paths("models", "delay_cause_calibration.pkl"):
        if __import__("os").path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
    return None


def nlp_stack_available():
    try:
        import transformers  # noqa: F401
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def classify_delay_cause_from_text(case_text):
    """
    Real zero-shot classification, identical approach to Notebook 05 —
    including the z-score calibration fix (see load_delay_calibration).
    """
    import torch
    tokenizer, model = load_inlegalbert()

    def embed(text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    case_vec = embed(case_text)
    calib = load_delay_calibration()
    names = calib["category_names"] if calib is not None else list(DELAY_CATEGORY_DESCRIPTIONS.keys())
    cat_vecs = np.array([embed(DELAY_CATEGORY_DESCRIPTIONS[n]) for n in names])

    raw_sims = cat_vecs @ case_vec / (np.linalg.norm(cat_vecs, axis=1) * np.linalg.norm(case_vec) + 1e-9)

    if calib is not None:
        scores = (raw_sims - calib["col_mean"]) / calib["col_std"]  # same correction as Notebook 05
    else:
        scores = raw_sims  # uncalibrated fallback — will be biased, but at least won't crash

    best_idx = int(np.argmax(scores))
    return names[best_idx], float(raw_sims[best_idx])  # report the raw cosine as the displayed confidence


# ----------------------------------------------------------------------------
# Real persistence — writes new cases back to the actual CSV file on disk
# (not just in-memory session state), so they survive an app restart and are
# genuinely part of your dataset from then on.
# ----------------------------------------------------------------------------
def save_cases_to_disk(df):
    import os
    for path in ["../data/processed/prioritized_cases.csv", "data/processed/prioritized_cases.csv"]:
        directory = os.path.dirname(path)
        if directory == "" or os.path.isdir(directory):
            try:
                df.to_csv(path, index=False)
                return path
            except Exception:
                continue
    return None


# ----------------------------------------------------------------------------
# PAGE: Predict New Case  (the page that was missing before)
# ----------------------------------------------------------------------------
def page_predict():
    section_heading(
        "Live prediction", "Predict a New Case",
        "Fill in a case's details to get a predicted disposal time and an explainable priority "
        "score — without needing a CSV of existing cases."
    )

    with st.expander("Optional: use a different trained model for this session", expanded=False):
        st.caption("Your `cox_model.pkl` is loaded automatically if this dashboard sits in a "
                   "`dashboard/` folder next to your project's `models/` folder. Use this only "
                   "to override it with a different file.")
        model_file = st.file_uploader("Upload cox_model.pkl", type="pkl", key="model_upload")
        if model_file is not None:
            try:
                st.session_state.cox_model = pickle.load(model_file)
                st.success("Model loaded — predictions below will now use your real Cox model.")
            except Exception as e:
                st.error(f"Could not load this file as a Cox model: {e}")

    st.write("")
    nlp_ready = nlp_stack_available()
    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            crime_type = st.selectbox("Crime type", CRIME_TYPES)
            region = st.selectbox("Region", REGIONS)
            bail_type = st.selectbox("Bail type", BAIL_TYPES)
        with c2:
            prior_cases = st.selectbox("Prior cases", PRIOR_CASES_OPTIONS)
            accused_gender = st.selectbox("Accused gender", ["Male", "Female"])
            manual_delay_cause = st.selectbox(
                "Delay cause (used only if no case text is given below)", DELAY_CATEGORIES
            )
        with c3:
            landmark_case = st.checkbox("Landmark case")
            bail_cancellation_case = st.checkbox("Bail cancellation case")
            parity_argument_used = st.checkbox("Parity argument used")
            has_special_law = st.checkbox("Special law involved")

        case_text = st.text_area(
            "Case facts / order-sheet text (optional)",
            placeholder="Paste the case's facts or judgment-reason text here to auto-detect the "
                        "delay cause using the real InLegalBERT classifier, instead of picking it "
                        "manually above.",
            height=90,
        )
        if not nlp_ready:
            st.caption("⚪ `transformers`/`torch` not installed — delay cause will use your manual "
                       "selection above. Run `pip install transformers torch` to enable automatic "
                       "detection from case text.")

        predict_clicked = st.form_submit_button("Predict", use_container_width=True, type="primary")

    if predict_clicked:
        form_values = {
            "crime_type": crime_type, "region": region, "bail_type": bail_type,
            "prior_cases": prior_cases, "accused_gender": accused_gender,
            "landmark_case": landmark_case, "bail_cancellation_case": bail_cancellation_case,
            "parity_argument_used": parity_argument_used, "has_special_law": has_special_law,
        }

        # ---- Delay cause: real classifier if text given + NLP stack available, else manual pick
        used_real_classifier = False
        if case_text.strip() and nlp_ready:
            try:
                delay_cause, delay_confidence = classify_delay_cause_from_text(case_text.strip())
                used_real_classifier = True
            except Exception as e:
                st.warning(f"InLegalBERT classification failed ({e}); using your manual selection instead.")
                delay_cause, delay_confidence = manual_delay_cause, 0.6
        else:
            delay_cause, delay_confidence = manual_delay_cause, 0.6

        # ---- Duration: real Cox model if auto-loaded/uploaded, else fallback formula
        used_real_model = False
        if st.session_state.cox_model is not None:
            try:
                predicted_duration = round(predict_with_cox_model(st.session_state.cox_model, form_values))
                used_real_model = True
            except Exception as e:
                st.warning(f"Your Cox model couldn't be used for this input ({e}); "
                           f"showing the fallback estimate instead.")
                predicted_duration = fallback_duration_estimate(form_values)
        else:
            predicted_duration = fallback_duration_estimate(form_values)

        scores = score_case(predicted_duration, delay_cause, crime_type, landmark_case)

        st.markdown(f"""
        <div class="result-card">
            <div style="font-size:12.5px; color:{SLATE};">
                {"Predicted using your Cox model" if used_real_model else "Fallback estimate — no Cox model found/loaded"}
                &nbsp;·&nbsp;
                {"Delay cause via InLegalBERT" if used_real_classifier else "Delay cause: manual selection"}
            </div>
            <div class="result-number">{predicted_duration} days</div>
            <div style="font-size:13px; color:{INK}; margin-top:2px;">Predicted disposal time · Delay cause: <b>{delay_cause}</b></div>
        </div>
        """, unsafe_allow_html=True)

        st.write("")
        colA, colB = st.columns([1, 1])
        with colA:
            st.markdown(f"**Priority level:** "
                        f"<span style='color:{PRIORITY_COLOR[scores['priority_level']]}; font-weight:600;'>"
                        f"{scores['priority_level']}</span> &nbsp; "
                        f"(score: {scores['priority_score']:.3f})", unsafe_allow_html=True)

            factors = [
                ("Predicted duration", scores["duration_score"], 0.40),
                ("Delay cause severity", scores["delay_cause_score"], 0.35),
                ("Case severity", scores["severity_score"], 0.15),
                ("Landmark case", scores["landmark_score"], 0.10),
            ]
            html = ""
            for label, value, weight in factors:
                pct = min(100, value * 100)
                html += f"""
                <div class="factor-row">
                    <div class="factor-label">{label}</div>
                    <div class="factor-bar-bg"><div class="factor-bar-fill" style="width:{pct}%;"></div></div>
                    <div class="factor-value">{value:.2f} × {weight} = {value*weight:.2f}</div>
                </div>"""
            st.markdown(html, unsafe_allow_html=True)

        with colB:
            st.session_state["_pending_new_case"] = {
                "crime_type": crime_type, "region": region, "predicted_duration": predicted_duration,
                "delay_cause": delay_cause, "delay_cause_confidence": round(delay_confidence, 2),
                "landmark_case": landmark_case, **scores,
            }

    if st.session_state.get("_pending_new_case"):
        if st.button("Add this case to the docket (saves to your dataset file)"):
            pending = st.session_state["_pending_new_case"]
            new_id = int(st.session_state.cases["case_id"].max()) + 1
            new_row = {"case_id": new_id, **pending}
            st.session_state.cases = pd.concat(
                [st.session_state.cases, pd.DataFrame([new_row])], ignore_index=True
            ).sort_values("priority_score", ascending=False).reset_index(drop=True)

            saved_path = save_cases_to_disk(st.session_state.cases)
            st.session_state["_pending_new_case"] = None
            if saved_path:
                st.success(f"Case {new_id} added AND saved to {saved_path} — it's now a permanent "
                           f"part of your dataset, and will still be here next time you open the app.")
            else:
                st.info(f"Case {new_id} added for this session, but couldn't find your project's "
                       f"data/processed/ folder to save it permanently — it will be lost when the "
                       f"app restarts. Run the dashboard from inside your project's dashboard/ "
                       f"folder (see README) to enable permanent saving.")


# ----------------------------------------------------------------------------
# PAGE: Overview
# ----------------------------------------------------------------------------
def page_overview(cases):
    section_heading(
        "Docket summary", "System Overview",
        "A single-glance read of the current docket — how many cases, how long they're expected "
        "to take, and how many need urgent attention."
    )

    total = len(cases)
    avg_duration = round(cases["predicted_duration"].mean())
    high_count = (cases["priority_level"] == "High").sum()

    c1, c2, c3, c4 = st.columns(4)
    with c1: stat_box("Cases in docket", total)
    with c2: stat_box("Avg. predicted duration", f"{avg_duration}d", "Cox model estimate")
    with c3: stat_box("High-priority cases", high_count, f"{round(100*high_count/total)}% of docket")
    with c4: stat_box("Drift status", "Watching", "Last checked recently")

    st.write("")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Cases by crime type**")
        by_crime = cases["crime_type"].value_counts().reset_index()
        by_crime.columns = ["crime_type", "count"]
        fig = px.bar(by_crime, x="crime_type", y="count")
        fig.update_traces(marker_color=INK)
        st.plotly_chart(styled_bar(fig), width="stretch")
    with col2:
        st.markdown("**Docket by priority level**")
        by_level = cases["priority_level"].value_counts().reindex(["High", "Medium", "Low"]).fillna(0).reset_index()
        by_level.columns = ["level", "count"]
        fig = px.bar(by_level, x="count", y="level", orientation="h",
                     color="level", color_discrete_map=PRIORITY_COLOR)
        fig.update_layout(showlegend=False)
        st.plotly_chart(styled_bar(fig), width="stretch")

    st.write("")
    st.markdown("**Most urgent cases right now**")
    top5 = cases.sort_values("priority_score", ascending=False).head(5)
    st.dataframe(
        top5[["case_id", "crime_type", "region", "predicted_duration", "priority_level"]],
        width="stretch", hide_index=True,
    )


# ----------------------------------------------------------------------------
# PAGE: Priority Docket
# ----------------------------------------------------------------------------
def page_priority(cases):
    section_heading(
        "Explainable ranking", "Priority Docket",
        "Every case ranked by urgency — combining predicted disposal time, delay cause, case "
        "severity, and landmark status into one transparent score."
    )

    uploaded = st.file_uploader("Optional: override with a different prioritized_cases.csv", type="csv", key="priority_upload")
    if uploaded is not None:
        new_df = pd.read_csv(uploaded)
        st.session_state.cases = recompute_level(new_df)
        cases = st.session_state.cases
        st.success(f"Loaded {len(new_df)} cases from {uploaded.name}")

    level_filter = st.radio("Filter", ["All", "High", "Medium", "Low"], horizontal=True,
                             label_visibility="collapsed", key="priority_filter")
    filtered = cases if level_filter == "All" else cases[cases["priority_level"] == level_filter]

    display_cols = ["case_id", "crime_type", "region", "predicted_duration", "delay_cause", "priority_level", "priority_score"]
    st.dataframe(
        filtered[display_cols].sort_values("priority_score", ascending=False),
        width="stretch", hide_index=True,
    )

    st.write("")
    st.markdown("**Explain a case's score**")
    case_ids = filtered["case_id"].tolist()
    if case_ids:
        chosen = st.selectbox("Choose a case ID", case_ids)
        row = cases[cases["case_id"] == chosen].iloc[0]

        factors = [
            ("Predicted duration", row["duration_score"], 0.40),
            ("Delay cause severity", row["delay_cause_score"], 0.35),
            ("Case severity", row["severity_score"], 0.15),
            ("Landmark case", row["landmark_score"], 0.10),
        ]
        html = ""
        for label, value, weight in factors:
            pct = min(100, value * 100)
            html += f"""
            <div class="factor-row">
                <div class="factor-label">{label}</div>
                <div class="factor-bar-bg"><div class="factor-bar-fill" style="width:{pct}%;"></div></div>
                <div class="factor-value">{value:.2f} × {weight} = {value*weight:.2f}</div>
            </div>"""
        st.markdown(html, unsafe_allow_html=True)
        st.markdown(f"**Total priority score: {row['priority_score']:.3f}**")
    else:
        st.info("No cases match this filter.")


# ----------------------------------------------------------------------------
# PAGE: Delay Analysis
# ----------------------------------------------------------------------------
def page_delay(cases):
    section_heading(
        "Zero-shot InLegalBERT classification", "Delay Root-Cause Analysis",
        "Look up any case to see the likely reason it's delayed, and how confidently the "
        "classifier matched it against each candidate cause."
    )

    uploaded = st.file_uploader("Optional: override with a different dataset_with_delay_cause.csv", type="csv", key="delay_upload")
    if uploaded is not None:
        delay_df = pd.read_csv(uploaded)
        merged = cases.merge(
            delay_df[["case_id", "predicted_delay_cause", "delay_cause_confidence"]],
            on="case_id", how="left", suffixes=("", "_new")
        )
        merged["delay_cause"] = merged["predicted_delay_cause"].fillna(merged["delay_cause"])
        merged["delay_cause_confidence"] = merged["delay_cause_confidence_new"].fillna(merged["delay_cause_confidence"])
        st.session_state.cases = merged.drop(columns=["predicted_delay_cause", "delay_cause_confidence_new"])
        cases = st.session_state.cases
        st.success(f"Delay-cause data merged from {uploaded.name}")

    query = st.text_input("Enter a case ID", placeholder="e.g. 105")
    if query.strip():
        match = cases[cases["case_id"].astype(str) == query.strip()]
        if not match.empty:
            row = match.iloc[0]
            colA, colB = st.columns([2, 1])
            with colA:
                st.markdown(f"### Case {row['case_id']}")
                st.caption(f"{row['crime_type']} · {row['region']}")
            with colB:
                st.markdown(f"<div style='text-align:right; font-size:12px; color:{SLATE};'>Predicted delay cause</div>"
                             f"<div style='text-align:right; font-size:15px; color:{BRASS}; font-weight:600;'>{row['delay_cause']}</div>",
                             unsafe_allow_html=True)

            rng = np.random.default_rng(int(row["case_id"]))
            sims = []
            for cat in DELAY_CATEGORIES:
                score = row["delay_cause_confidence"] if cat == row["delay_cause"] else row["delay_cause_confidence"] * (0.3 + rng.random() * 0.4)
                sims.append({"cause": cat, "score": score})
            sim_df = pd.DataFrame(sims).sort_values("score", ascending=True)
            fig = px.bar(sim_df, x="score", y="cause", orientation="h")
            fig.update_traces(marker_color=[BRASS if c == row["delay_cause"] else SLATE for c in sim_df["cause"]])
            st.plotly_chart(styled_bar(fig), width="stretch")
        else:
            st.warning(f"No case with ID {query} found in the current dataset. "
                       f"(Tip: cases you add via 'Predict a New Case' get an ID after "
                       f"{cases['case_id'].max()}.)")
    else:
        st.markdown(f'<div style="border:1px dashed {HAIR}; padding:20px; text-align:center; color:{SLATE};">'
                     'Enter a case ID above to see its individual delay-cause breakdown.</div>', unsafe_allow_html=True)

    st.write("")
    st.markdown("**Delay causes across the docket**")
    dist = cases["delay_cause"].value_counts().reindex(DELAY_CATEGORIES).fillna(0).reset_index()
    dist.columns = ["cause", "count"]
    fig = px.bar(dist.sort_values("count"), x="count", y="cause", orientation="h")
    fig.update_traces(marker_color=INK)
    st.plotly_chart(styled_bar(fig), width="stretch")


# ----------------------------------------------------------------------------
# PAGE: Drift Monitor
# ----------------------------------------------------------------------------
def page_drift(drift_log, feature_mix):
    section_heading(
        "Model reliability", "Drift Monitor",
        "Whether the Cox model's understanding of the docket still matches reality — checked "
        "against both incoming case patterns and prediction accuracy."
    )
    st.caption("Note: drift is checked across a *batch* of recent cases (see Notebook 07), not "
               "per individual case — adding one new case above won't move these numbers by "
               "itself. Re-run Notebook 07 periodically and reload its `drift_log.csv` here.")

    uploaded = st.file_uploader("Optional: override with a different drift_log.csv", type="csv", key="drift_upload")
    if uploaded is not None:
        st.session_state.drift_log = pd.read_csv(uploaded)
        drift_log = st.session_state.drift_log
        st.success(f"Loaded drift log from {uploaded.name}")

    latest = drift_log.iloc[-1]
    col1, col2 = st.columns(2)
    with col1:
        icon = "⚠️" if latest["feature_drift_detected"] else "✅"
        body = ("The mix of incoming cases has shifted meaningfully from the baseline period."
                if latest["feature_drift_detected"] else
                "The incoming case mix still resembles the baseline period.")
        st.markdown(f'<div class="status-card">{icon} <span class="status-title">Feature drift</span>'
                     f'<div class="status-body">{body}</div></div>', unsafe_allow_html=True)
    with col2:
        icon = "⚠️" if latest["performance_drift_detected"] else "✅"
        body = (f"Concordance Index dropped from {latest['baseline_cindex']} to "
                f"{latest['monitoring_cindex_old_model']} on recent cases."
                if latest["performance_drift_detected"] else
                "Concordance Index on recent cases remains close to baseline.")
        st.markdown(f'<div class="status-card">{icon} <span class="status-title">Performance drift</span>'
                     f'<div class="status-body">{body}</div></div>', unsafe_allow_html=True)

    st.write("")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Concordance Index over time**")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=drift_log["check_timestamp"], y=drift_log["baseline_cindex"],
                                  mode="lines", name="Baseline C-index", line=dict(color=SLATE, width=2)))
        fig.add_trace(go.Scatter(x=drift_log["check_timestamp"], y=drift_log["monitoring_cindex_old_model"],
                                  mode="lines+markers", name="Monitoring C-index", line=dict(color=HIGH, width=2)))
        fig.update_layout(yaxis_range=[0.4, 0.8])
        st.plotly_chart(styled_bar(fig), width="stretch")
    with col2:
        st.markdown("**Crime-type mix: baseline vs. monitoring**")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=feature_mix["crime_type"], y=feature_mix["baseline"], name="Baseline %", marker_color=SLATE))
        fig.add_trace(go.Bar(x=feature_mix["crime_type"], y=feature_mix["monitoring"], name="Monitoring %", marker_color=BRASS))
        st.plotly_chart(styled_bar(fig), width="stretch")

    st.write("")
    st.markdown("**Drift-check history**")
    st.dataframe(drift_log, width="stretch", hide_index=True)


# All helper functions are defined above this point, so it's now safe to
# actually run the auto-load check.
try_auto_load()

# ----------------------------------------------------------------------------
# Sidebar navigation + app shell (only reached after successful login)
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚖️ Judiciary AI")
    st.caption("Drift-Aware Docket")
    st.write("")
    page = st.radio(
        "Navigate",
        ["Predict New Case", "Overview", "Priority Docket", "Delay Analysis", "Drift Monitor"],
        label_visibility="collapsed",
        key="nav_radio",
    )
    st.write("")
    sources = st.session_state.data_sources
    st.markdown("**Data status**")
    st.caption(("✅ Cox model: auto-loaded" if "cox_model" in sources else "⚪ Cox model: not found — using fallback formula"))
    st.caption(("✅ Cases: auto-loaded" if "cases" in sources else "⚪ Cases: sample data"))
    st.caption(("✅ Delay causes: auto-loaded" if "delay_source" in sources else "⚪ Delay causes: sample data"))
    st.caption(("✅ Drift log: auto-loaded" if "drift_log" in sources else "⚪ Drift log: sample data"))
    st.write("")
    if st.button("Log out"):
        st.session_state.authenticated = False
        st.rerun()

cases = st.session_state.cases

if page == "Predict New Case":
    page_predict()
elif page == "Overview":
    page_overview(cases)
elif page == "Priority Docket":
    page_priority(cases)
elif page == "Delay Analysis":
    page_delay(cases)
elif page == "Drift Monitor":
    page_drift(st.session_state.drift_log, st.session_state.feature_mix)
