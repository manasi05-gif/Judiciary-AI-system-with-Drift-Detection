"""
stream_incoming_cases.py — Live Judicial Inflow Streaming Daemon / CLI Generator
Simulates real-time eCourts CIS filings arriving as batch CSV files for Concept Drift monitoring.

Usage:
    python scripts/stream_incoming_cases.py --scenario cyber_surge --cases 15 --interval 2 --batches 3
    python scripts/stream_incoming_cases.py --scenario normal --cases 20 --batches 1
    python scripts/stream_incoming_cases.py --scenario strike_delay --cases 25 --batches 2
"""

import os
import sys
import time
import argparse
from datetime import datetime
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

COURTS = [
    ("Delhi High Court", "Delhi"),
    ("Bombay High Court", "Maharashtra"),
    ("High Court of Judicature at Allahabad", "Uttar Pradesh"),
    ("High Court of Karnataka", "Karnataka"),
    ("High Court of Punjab and Haryana", "Punjab"),
    ("High Court of Gujarat", "Gujarat"),
    ("High Court of Madhya Pradesh", "Madhya Pradesh"),
    ("Calcutta High Court", "West Bengal"),
    ("High Court of Kerala", "Kerala"),
]

JUDGES = [
    "Justice A. K. Sharma", "Justice R. V. Patel", "Justice S. Mukherjee",
    "Justice M. K. Rao", "Justice P. N. Deshmukh", "Justice N. K. Singh",
    "Justice V. Sundaram", "Justice H. C. Verma"
]

CRIME_DETAILS = {
    "Cyber Crime": {
        "ipc": "['IT Act 66D', '420', '120B']",
        "facts": "Allegations of unauthorized access, phishing of banking OTP credentials, and siphoning funds via counterfeit payment gateway accounts.",
        "legal_issues": "['Whether custodial interrogation is required for cold-wallet transaction traceability', 'Application of Section 66D IT Act with Section 420 IPC']",
        "judgment_reason": "Investigation into digital audit trail and mule accounts ongoing; device forensic analysis awaiting FSL confirmation.",
    },
    "Fraud or Cheating": {
        "ipc": "['420', '406', '467', '471']",
        "facts": "Applicant accused of multi-layered corporate embezzlement, misrepresenting audited balance sheets, and creating shell entities.",
        "legal_issues": "['Whether commercial dispute has been given a criminal color', 'Parity with co-signatories who were granted pre-arrest bail']",
        "judgment_reason": "Documentary trail submitted by the Serious Fraud Investigation Office; disputed records impounded by trial magistrate.",
    },
    "Theft or Robbery": {
        "ipc": "['379', '411', '457']",
        "facts": "Nocturnal housebreaking and recovery of stolen commercial consignment from godown premises; identification parade pending.",
        "legal_issues": "['Recovery under Section 27 Indian Evidence Act without independent panch witnesses', 'Previous antecedents of similar offenses']",
        "judgment_reason": "Material exhibits recovered; charge-sheet submitted under Section 173 CrPC; trial committal underway.",
    }
}

def generate_stream_batch(scenario="cyber_surge", n_cases=15, batch_idx=1):
    rng = np.random.default_rng()
    crimes = ["Fraud or Cheating", "Theft or Robbery", "Cyber Crime"]

    if scenario == "cyber_surge":
        p_crimes = [0.12, 0.10, 0.78]
        durations = rng.integers(25, 95, size=n_cases)
        delay_pool = ["Investigation pending", "Procedural or documentation delay"]
    elif scenario == "strike_delay":
        p_crimes = [0.44, 0.36, 0.20]
        durations = rng.integers(140, 310, size=n_cases)
        delay_pool = ["Repeated adjournments", "Procedural or documentation delay"]
    elif scenario == "compound":
        p_crimes = [0.15, 0.15, 0.70]
        durations = rng.integers(135, 290, size=n_cases)
        delay_pool = ["Repeated adjournments", "Investigation pending"]
    else:  # normal
        p_crimes = [0.44, 0.38, 0.18]
        durations = rng.integers(20, 85, size=n_cases)
        delay_pool = [
            "Witness non-availability", "Investigation pending",
            "Repeated adjournments", "Procedural or documentation delay", "No specific delay noted"
        ]

    sampled_crimes = rng.choice(crimes, p=p_crimes, size=n_cases)
    ts_now = datetime.now()
    records = []

    for i in range(n_cases):
        c_type = sampled_crimes[i]
        dur = int(durations[i])
        cause = rng.choice(delay_pool)
        court_info = COURTS[rng.integers(0, len(COURTS))]
        judge = JUDGES[rng.integers(0, len(JUDGES))]
        details = CRIME_DETAILS[c_type]
        uid = f"STREAM-{ts_now.strftime('%y%m%d')}-{batch_idx:02d}{i+1:03d}"

        priority_score = min(0.95, max(0.15, round(0.40 * (dur / 150) + 0.35 * (1.0 if c_type == 'Fraud or Cheating' else 0.5) + rng.uniform(-0.05, 0.05), 3)))
        if priority_score >= 0.70 or dur > 150:
            prio_level = "High"
        elif priority_score >= 0.45 or dur > 60:
            prio_level = "Medium"
        else:
            prio_level = "Low"

        records.append({
            "case_id": uid,
            "case_title": f"State vs. Accused {uid[-5:]}",
            "court": court_info[0],
            "region": court_info[1],
            "date": ts_now.strftime("%Y-%m-%d"),
            "judge": judge,
            "ipc_sections": details["ipc"],
            "bail_type": rng.choice(["Regular", "Anticipatory", "Interim"]),
            "accused_name": f"Petitioner_{uid[-4:]}",
            "accused_gender": rng.choice(["Male", "Female"], p=[0.85, 0.15]),
            "prior_cases": rng.choice(["No", "Yes", "Unknown"], p=[0.65, 0.20, 0.15]),
            "bail_outcome": "Pending",
            "crime_type": c_type,
            "facts": details["facts"],
            "legal_issues": details["legal_issues"],
            "judgment_reason": details["judgment_reason"],
            "disposal_days": dur,
            "duration_days": dur,
            "predicted_duration": dur,
            "delay_cause": cause,
            "priority_level": prio_level,
            "priority_score": priority_score,
            "period": "monitoring",
            "is_disposed": 0,
            "ingested_timestamp": ts_now.isoformat()
        })

    return pd.DataFrame(records)

def main():
    parser = argparse.ArgumentParser(description="Live Inflow Case Streamer for Judicial AI System")
    parser.add_argument("--scenario", choices=["normal", "cyber_surge", "strike_delay", "compound"], default="cyber_surge", help="Drift injection scenario")
    parser.add_argument("--cases", type=int, default=15, help="Number of cases per batch")
    parser.add_argument("--batches", type=int, default=1, help="Total batches to generate (-1 for infinite stream)")
    parser.add_argument("--interval", type=float, default=2.0, help="Interval between batches in seconds")
    parser.add_argument("--out_dir", type=str, default="data/live_incoming", help="Output directory for batch CSV files")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print("=" * 60)
    print("⚖️  JUDICIAL AI SYSTEM — LIVE INCOMING FILING STREAMER")
    print("=" * 60)
    print(f"Scenario  : {args.scenario}")
    print(f"Cases/Batch: {args.cases}")
    print(f"Batches   : {'Infinite' if args.batches == -1 else args.batches}")
    print(f"Interval  : {args.interval}s")
    print(f"Target Dir: {os.path.abspath(args.out_dir)}")
    print("-" * 60)

    batch_idx = 1
    try:
        while True:
            df_batch = generate_stream_batch(scenario=args.scenario, n_cases=args.cases, batch_idx=batch_idx)
            ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"filing_batch_{ts_str}_{args.scenario}.csv"
            filepath = os.path.join(args.out_dir, filename)
            df_batch.to_csv(filepath, index=False)

            crime_dist = df_batch["crime_type"].value_counts().to_dict()
            avg_dur = df_batch["duration_days"].mean()
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Ingested Batch #{batch_idx}: {len(df_batch)} cases -> {filename}")
            print(f"   Distribution: {crime_dist} | Avg Duration: {avg_dur:.1f} days")

            if args.batches != -1 and batch_idx >= args.batches:
                print("-" * 60)
                print(f"✅ Generated {batch_idx} batch(es) successfully in {args.out_dir}.")
                break

            batch_idx += 1
            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\n🛑 Inflow streaming stopped by user.")

if __name__ == "__main__":
    main()
