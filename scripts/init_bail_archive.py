import os
import pandas as pd
import datetime

archive_dir = os.path.join("data", "disposed_bail")
os.makedirs(archive_dir, exist_ok=True)
archive_file = os.path.join(archive_dir, "bail_granted_archive.csv")

# Load structured dataset to extract historical granted cases
s_df = pd.read_csv(os.path.join("data", "processed", "structured_dataset.csv"))
s_df["case_id"] = s_df["case_id"].astype(str)

granted_df = s_df[s_df["bail_outcome"].astype(str).str.lower() == "granted"].copy()
print(f"Total historical cases with bail granted: {len(granted_df)}")

# Build archive records
archive_records = []
now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

for _, r in granted_df.iterrows():
    cid = str(r.get("case_id", ""))
    title = str(r.get("case_title", ""))
    if not title or title.lower() == "nan":
        title = f"State vs. Accused (#{cid})"
    court = str(r.get("court", r.get("region", "High Court")))
    if not court or court.lower() == "nan":
        court = f"High Court of {r.get('region', 'India')}"

    archive_records.append({
        "case_id": cid,
        "case_title": title,
        "court": court,
        "crime_type": str(r.get("crime_type", "Fraud or Cheating")),
        "region": str(r.get("region", "National")),
        "bail_outcome": "Granted",
        "bail_type": str(r.get("bail_type", "Regular")),
        "duration_days": int(r.get("duration_days", 90)) if pd.notna(r.get("duration_days")) else 90,
        "delay_cause": str(r.get("delay_cause", "No specific delay noted")),
        "disposed_timestamp": now_str,
        "granted_by": str(r.get("judge", "Hon'ble Justice Division Bench")),
        "surety_amount": 50000,
        "order_summary": "Regular bail granted subject to personal bond of Rs 50,000 and two local sureties."
    })

arch_df = pd.DataFrame(archive_records)
arch_df.to_csv(archive_file, index=False)
print(f"Successfully initialized {archive_file} with {len(arch_df)} archived records!")
