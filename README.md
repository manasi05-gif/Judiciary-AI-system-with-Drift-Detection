# ⚖️ Judicial AI Intelligence System with Concept Drift Detection

An executive-grade Artificial Intelligence and Decision Support Platform tailored for the Indian Judicial System. The system addresses judicial pendency in bail applications by providing:
1. **Survival Analysis & Disposal Forecasting** using a trained Cox Proportional Hazards Model (`lifelines`).
2. **Dynamic Survival Probability Curves $S(t)$** mapping case resolution probability across time.
3. **Semantic Delay Root-Cause Diagnosis** via zero-shot `law-ai/InLegalBERT` with z-score anisotropy calibration.
4. **Transparent, Multi-Factor Priority Docketing** (40% Duration, 35% Delay, 15% Severity, 10% Landmark).
5. **Real-Time Concept & Performance Drift Monitoring** with automated Chi-Square, Kolmogorov-Smirnov, and C-index tracking and triggered retraining.

---

## 🚀 Quick Start Guide (Kaise Run Karein)

### 1. Terminal / PowerShell open karein:
```powershell
cd "e:\Projects\Judicial AI system with drift detection"
```

### 2. Dependencies install karein (agar pehle nahi kiya hai):
```powershell
.venv\Scripts\pip install -r requirements.txt
```

### 3. Dashboard start karein:
```powershell
.venv\Scripts\streamlit run app.py
```

Dashboard automatically browser mein khul jayega: **`http://localhost:8501`**

---

## 🔐 Login Credentials (Portal Access)

Default administrative credentials:
- **Username:** `admin`
- **Password:** `judiciary2026`

*(Aap in credentials ko `app.py` ki shuruat mein kabhi bhi change kar sakte hain).*

---

## 🏛️ Platform Modules & Capabilities

### 1. 🏛️ Executive Judicial Overview
- **Macro Caseload KPIs:** Active caseload, average predicted disposal (days), critical urgency backlog count, and concept drift status.
- **Crime & Pendency Distributions:** Grouped analysis of bail applications by crime type (Fraud, Theft, Cyber Crime) and average disposal duration.
- **Jurisdictional Load:** Case inflow across High Courts and States.
- **Urgent Priority Queue:** Instant ticker of top priority cases requiring expedited hearings.

### 2. 📋 Explainable Priority Docket & Case Explorer
- **Multi-Faceted Search & Filter:** Filter by Case ID, Case Title, Presiding Bench, Priority Level (High/Med/Low), Crime Type, and State.
- **Explainable AI Inspector:** Select any case to inspect its 4-factor scoring breakdown (waterfall chart).
- **Interactive Survival Curve $S(t)$:** Inspect the exact survival probability curve for any selected case.
- **Judicial Dossier:** Read Judge, Court, IPC Sections, Special Acts (PMLA, NDPS, POCSO), Case Facts, and Judgment Reasoning.
- **CSV Export:** 1-click export of the filtered prioritized docket.

### 3. ⚡ Live Case Pendency & Delay Predictor
- **Comprehensive Case Intake:** Covers all 17 supported judicial regions, crime types, bail categories, and special legal flags (landmark precedent, parity arguments, bail cancellation).
- **InLegalBERT Zero-Shot NLP Engine:** Paste case order sheets or facts text to auto-diagnose the primary delay root cause with calibrated confidence scores.
- **Real-Time Cox Survival Prediction:** Computes expected disposal days and renders an interactive survival curve.
- **Permanent Save:** Commit newly evaluated cases directly to the persistent judicial database.

### 4. 🔍 Delay Root-Cause Analysis
- **Systemic Delay Diagnostics:** Cross-tabulation of delay causes against crime types and state jurisdictions.
- **Policy Recommendations:** Actionable procedural interventions targeting repeated adjournments, witness non-availability, and pending investigations.

### 5. 📡 Concept & Performance Drift Monitor
- **Feature Drift Detection:** Chi-Square contingency testing on categorical crime distribution shifts (Baseline 102 cases vs Monitoring 263 cases; p < 0.001 detected for cyber crime surge).
- **Numeric Duration Drift:** Kolmogorov-Smirnov 2-sample test on disposal duration distribution.
- **Performance Drift:** Concordance Index (C-Index) tracking over time.
- **Custom Cohort Drift Testing:** Upload custom monitoring datasets (CSV/Excel) to test real-time drift against the baseline benchmark.
- **Interactive Drift Simulator:** Court administrators can simulate caseload shifts and trigger model retraining dynamically.

### 6. 📂 Custom Dataset Ingestion & Batch AI Hub
- **Universal File Ingestion:** Drag-and-drop CSV or Excel (`.xlsx`, `.xls`) case files directly into the platform.
- **Smart Column Auto-Detection:** Automatically reconciles arbitrary field headers (`CNR`, `Party_Names`, `Offence`, `State`, `Facts`, etc.) with customizable manual overrides.
- **Automated AI Batch Pipeline:** Runs Cox Proportional Hazards survival forecasting, InLegalBERT calibrated semantic delay classification, and 4-factor priority score calculation for all rows in seconds.
- **Merge or Replace:** Ingest cases by merging with the active docket or replacing it entirely, with 1-click persistent saving to disk.
- **Ready-to-Use Templates:** Download standard CSV templates (`judicial_cases_template.csv`) and export AI-enriched datasets.

---

## 📁 Project Architecture & Directory Layout

```
Judicial AI system with drift detection/
├── app.py                      # Main Streamlit Executive Dashboard
├── requirements.txt            # Production Python dependencies
├── README.md                   # Complete system documentation
│
├── models/
│   ├── cox_model.pkl           # Trained lifelines CoxPHFitter survival model
│   ├── delay_cause_calibration.pkl  # InLegalBERT calibrated mean/std embeddings
│   └── generate_calibration.py # Calibration generator script
│
├── data/
│   └── processed/
│       ├── structured_dataset.csv          # 365 cases with full judicial metadata
│       ├── prioritized_cases.csv           # Calibrated priority-ranked docket
│       ├── dataset_with_delay_cause.csv    # Cases enriched with InLegalBERT delay labels
│       └── drift_log.csv                   # Historical concept drift audit log
│
└── Notebook/                   # 7 Research & Training Notebooks (01 to 07)
```

---

## 🛠️ Verification & Testing

To execute the automated validation suite:
```powershell
.venv\Scripts\python tests/test_system.py
```
