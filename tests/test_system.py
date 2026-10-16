import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if sys.path[0] != PROJECT_ROOT:
    sys.path.insert(0, PROJECT_ROOT)

import pickle
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency

# Fix UTF-8 encoding on Windows consoles
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

print("=" * 65)
print("  ⚖️  JUDICIAL AI SYSTEM WITH DRIFT DETECTION — VERIFICATION SUITE")
print("=" * 65)

# 1. Models & Calibration Artifacts
print("\n[1] Verifying Models & Calibration Assets:")
cox_path = os.path.join("models", "cox_model.pkl")
calib_path = os.path.join("models", "delay_cause_calibration.pkl")

assert os.path.exists(cox_path), f"Missing {cox_path}"
with open(cox_path, "rb") as f:
    cph = pickle.load(f)
print(f"  ✓ Cox model loaded successfully. Features count: {len(cph.params_)}")

assert os.path.exists(calib_path), f"Missing {calib_path}"
with open(calib_path, "rb") as f:
    calib = pickle.load(f)
print(f"  ✓ Calibration loaded. Categories: {calib['category_names']}")
print(f"  ✓ Calibration col_mean: {calib['col_mean'].round(3)}")

# 2. Processed Datasets
print("\n[2] Verifying Processed Datasets:")
prio_path = os.path.join("data", "processed", "prioritized_cases.csv")
struct_path = os.path.join("data", "processed", "structured_dataset.csv")
delay_path = os.path.join("data", "processed", "dataset_with_delay_cause.csv")
drift_path = os.path.join("data", "processed", "drift_log.csv")

for p in [prio_path, struct_path, delay_path, drift_path]:
    assert os.path.exists(p), f"Missing {p}"
    df = pd.read_csv(p)
    print(f"  ✓ {p} -> {df.shape[0]} rows, {df.shape[1]} cols")

# 3. Dynamic Survival Function S(t)
print("\n[3] Testing Dynamic Survival Function S(t):")
sample_form = {
    "crime_type": "Fraud or Cheating",
    "region": "Maharashtra",
    "bail_type": "Regular",
    "prior_cases": "Yes",
    "accused_gender": "Male",
    "landmark_case": True,
    "bail_cancellation_case": 0,
    "parity_argument_used": 0,
    "has_special_law": 1,
}
row = {
    "crime_type": sample_form["crime_type"],
    "region_grouped": sample_form["region"],
    "bail_type": sample_form["bail_type"],
    "prior_cases": sample_form["prior_cases"],
    "accused_gender": sample_form["accused_gender"],
    "landmark_case": int(sample_form["landmark_case"]),
    "bail_cancellation_case": int(sample_form["bail_cancellation_case"]),
    "parity_argument_used": int(sample_form["parity_argument_used"]),
    "has_special_law": int(sample_form["has_special_law"]),
}
df_row = pd.DataFrame([row])
categorical_cols = ["crime_type", "region_grouped", "bail_type", "prior_cases", "accused_gender"]
encoded = pd.get_dummies(df_row, columns=categorical_cols, drop_first=False)
aligned = encoded.reindex(columns=cph.params_.index, fill_value=0)

pred_median = cph.predict_median(aligned)
val = pred_median.iloc[0] if hasattr(pred_median, "iloc") else float(pred_median)
sf = cph.predict_survival_function(aligned)
print(f"  ✓ Predicted median disposal time: {val} days")
print(f"  ✓ Survival function S(t) generated: {sf.shape[0]} discrete timeline points")
assert sf.shape[0] > 10, "Survival curve should have multiple points"

# 4. Statistical Concept Drift Tests
print("\n[4] Testing Concept Drift Statistical Tests:")
s_df = pd.read_csv(struct_path)
b_df = s_df[s_df["period"] == "baseline"]
m_df = s_df[s_df["period"] == "monitoring"]

ks_stat, ks_pval = ks_2samp(b_df["duration_days"].dropna(), m_df["duration_days"].dropna())
ct = pd.crosstab(s_df["crime_type"], s_df["period"])
chi2_stat, chi2_pval, _, _ = chi2_contingency(ct)

print(f"  ✓ KS 2-Sample Test (duration): stat={ks_stat:.3f}, p-value={ks_pval:.4f}")
print(f"  ✓ Chi-Square Test (crime_type): stat={chi2_stat:.3f}, p-value={chi2_pval:.4f}")
assert chi2_pval < 0.05, "Feature drift on crime type should be statistically significant"

# 5. Delay Cause Classifier Output Distribution
print("\n[5] Testing Calibrated Delay Cause Distribution:")
del_df = pd.read_csv(delay_path)
dist = del_df["predicted_delay_cause"].value_counts()
print(f"  ✓ Distribution of calibrated delay causes across all 365 cases:\n{dist}")
assert len(dist) == 5, "All 5 delay causes should be represented in the dataset"
assert dist.min() > 30, "No category should be starved or trivialized due to anisotropy"

# 6. Test Custom Dataset Ingestion & Auto-Mapping
print("\n[6] Testing Custom Dataset Ingestion & Auto-Mapping Pipeline:")
template_path = os.path.join("data", "sample_cases_template.csv")
assert os.path.exists(template_path), f"Missing {template_path}"
custom_raw = pd.read_csv(template_path)
print(f"  ✓ Sample template loaded: {custom_raw.shape[0]} rows, columns: {list(custom_raw.columns)}")

from app import auto_detect_columns, process_uploaded_custom_dataset
det_map = auto_detect_columns(custom_raw.columns)
print(f"  ✓ Auto-detected columns: {det_map}")
assert "case_id" in det_map and "crime_type" in det_map and "region" in det_map

ingested = process_uploaded_custom_dataset(
    raw_df=custom_raw,
    column_mapping=det_map,
    run_cox_batch=True,
    run_nlp_batch=False,
    cph=cph,
    calibration=calib,
    current_max_id=2000
)
print(f"  ✓ Successfully processed {len(ingested)} custom cases.")
print(f"  ✓ Ingested columns: {list(ingested.columns[:8])}")
print(f"  ✓ Predicted durations: {ingested['predicted_duration'].tolist()}")
print(f"  ✓ Priority levels: {ingested['priority_level'].value_counts().to_dict()}")

assert len(ingested) == len(custom_raw)
assert "predicted_duration" in ingested.columns
assert "priority_score" in ingested.columns
assert "priority_level" in ingested.columns
assert ingested["priority_score"].between(0, 1).all()

# 7. Test InLegalBERT Plain Language Explainer & Delay Extractor
print("\n[7] Testing InLegalBERT Plain Language Explainer & Delay Root-Cause Extractor:")
from app import decode_ipc_sections, extract_specific_delay_reason, IPC_DICTIONARY

test_ipc = "Accused booked under Sections 420, 468, 471 and 120B IPC"
decoded = decode_ipc_sections(test_ipc)
print(f"  ✓ Decoded IPC sections: {[d['code'] + ' (' + d['en_title'] + ')' for d in decoded]}")
assert len(decoded) == 4, f"Expected 4 sections, got {len(decoded)}"
assert any(d["code"] == "420" for d in decoded)
assert any(d["code"] == "468" for d in decoded)

sample_facts = "Investigation is held up as FSL forensic report is awaited by the cyber cell. Accused sought repeated adjournments."
delay_res = extract_specific_delay_reason(sample_facts, "", "Investigation pending", "Cyber Crime")
print(f"  ✓ Extracted Delay Reason: {delay_res['specific_fact']}")
print(f"  ✓ English Remedy: {delay_res['remedy_en'][:70]}...")
print(f"  ✓ Hindi Remedy: {delay_res['remedy_hi'][:70]}...")
assert "FSL" in delay_res["specific_fact"] or "Investigation" in delay_res["specific_fact"]
assert "167(2)" in delay_res["remedy_en"]
assert "167(2)" in delay_res["remedy_hi"]

# 8. Test Live Streaming Inflow Generator & Drift Retraining Engine
print("\n[8] Testing Live Streaming Inflow Generator & Drift Retraining Engine:")
from app import generate_synthetic_inflow_batch, retrain_cox_sliding_window
from scipy.stats import ks_2samp, chi2_contingency

# A. Test Cyber Surge Generation
surge_batch = generate_synthetic_inflow_batch(scenario="cyber_surge", n_cases=30, batch_idx=1)
assert len(surge_batch) == 30, f"Expected 30 cases, got {len(surge_batch)}"
assert "crime_type" in surge_batch.columns
assert "duration_days" in surge_batch.columns
cyber_pct = (surge_batch["crime_type"] == "Cyber Crime").mean()
print(f"  ✓ Generated 30 cyber surge cases. Cyber Crime share: {cyber_pct:.1%}")
assert cyber_pct >= 0.60, f"Expected cyber share >= 60%, got {cyber_pct}"

# B. Chi-Square Drift Test on Cyber Surge vs Baseline
b_df = s_df[s_df["period"] == "baseline"]
b_counts = b_df["crime_type"].value_counts()
m_counts = surge_batch["crime_type"].value_counts()
crimes = sorted(list(set(b_counts.index).union(set(m_counts.index))))
ct = pd.DataFrame({
    "baseline": [b_counts.get(c, 0) for c in crimes],
    "monitoring": [m_counts.get(c, 0) for c in crimes]
}, index=crimes)
chi2_s, chi2_p, _, _ = chi2_contingency(ct)
print(f"  ✓ Chi-Square Contingency Test on Cyber Surge: p-val = {chi2_p:.6f} -> Drift: {chi2_p < 0.05}")
assert chi2_p < 0.05, f"Expected Chi2 drift p < 0.05, got {chi2_p}"

# C. Test Strike Delay Generation & KS Test
strike_batch = generate_synthetic_inflow_batch(scenario="strike_delay", n_cases=35, batch_idx=2)
ks_s, ks_p = ks_2samp(b_df["duration_days"].dropna(), strike_batch["duration_days"])
print(f"  ✓ KS 2-Sample Test on Strike Backlog: p-val = {ks_p:.6f} -> Duration Drift: {ks_p < 0.05}")
assert ks_p < 0.05, f"Expected KS drift p < 0.05, got {ks_p}"

# D. Test Sliding-Window Cox Retraining
combined = pd.concat([b_df, surge_batch], ignore_index=True)
retrained_cph, new_cidx = retrain_cox_sliding_window(combined)
print(f"  ✓ Retrained CoxPH Model on Sliding Window: C-index = {new_cidx:.3f}")
assert new_cidx > 0.60, f"Expected C-index > 0.60, got {new_cidx}"

print("\n" + "=" * 65)
print("  🎉 ALL 8 SYSTEM CHECKS & VALIDATION ASSERTIONS PASSED SUCCESSFULLY!")
print("=" * 65)


