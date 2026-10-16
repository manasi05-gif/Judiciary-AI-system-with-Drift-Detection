"""
Generates delay_cause_calibration.pkl using InLegalBERT and the 365 case texts.
Applies mean-pooling and z-score calibration as specified in Notebook 05.
Saves models/delay_cause_calibration.pkl and updates datasets with calibrated predictions.
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

# Set working directory to project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

DELAY_CATEGORIES = {
    "Investigation pending":
        "The case is delayed because the police investigation is still ongoing or incomplete.",
    "Witness non-availability":
        "The case is delayed because witnesses were unavailable, absent, or retracted their statements.",
    "Repeated adjournments":
        "The case is delayed due to repeated adjournments or postponements of hearings.",
    "Procedural or documentation delay":
        "The case is delayed due to procedural issues, missing documents, or administrative backlog.",
    "No specific delay noted":
        "The case proceeded through the normal court process without any specific mention of delay.",
}

category_names = list(DELAY_CATEGORIES.keys())

def main():
    print("Loading InLegalBERT...")
    tokenizer = AutoTokenizer.from_pretrained("law-ai/InLegalBERT")
    model = AutoModel.from_pretrained("law-ai/InLegalBERT")
    model.eval()

    # Category embeddings
    print("Computing category embeddings...")
    cat_embeddings = []
    with torch.no_grad():
        for name in category_names:
            desc = DELAY_CATEGORIES[name]
            inputs = tokenizer(desc, return_tensors="pt", truncation=True, max_length=128, padding=True)
            out = model(**inputs)
            emb = out.last_hidden_state.mean(dim=1).squeeze().numpy()
            cat_embeddings.append(emb)
    cat_embeddings = np.array(cat_embeddings)
    print("Category embeddings shape:", cat_embeddings.shape)

    # Load dataset
    data_path = os.path.join(DATA_DIR, "dataset_with_delay_cause.csv")
    if not os.path.exists(data_path):
        data_path = os.path.join(DATA_DIR, "structured_dataset.csv")
    df = pd.read_csv(data_path)
    if "case_text" not in df.columns:
        df["case_text"] = df["facts"].fillna("") + ". " + df["judgment_reason"].fillna("")

    print(f"Embedding {len(df)} case texts in batches...")
    batch_size = 32
    case_embeddings = []
    with torch.no_grad():
        for i in range(0, len(df), batch_size):
            batch_texts = df["case_text"].iloc[i:i+batch_size].tolist()
            # Clean / ensure string
            batch_texts = [str(t) if pd.notna(t) and str(t).strip() else "No details" for t in batch_texts]
            inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True, max_length=256, padding=True)
            out = model(**inputs)
            # mean pooling over token dimension
            mask = inputs["attention_mask"].unsqueeze(-1)
            sum_emb = (out.last_hidden_state * mask).sum(dim=1)
            count = mask.sum(dim=1).clamp(min=1e-9)
            mean_emb = (sum_emb / count).numpy()
            case_embeddings.append(mean_emb)
            print(f"Processed {min(i+batch_size, len(df))} / {len(df)} cases...")

    case_embeddings = np.vstack(case_embeddings)
    print("Case embeddings computed:", case_embeddings.shape)

    # Cosine similarities
    raw_similarities = cosine_similarity(case_embeddings, cat_embeddings)

    # Z-score calibration across all cases
    col_mean = raw_similarities.mean(axis=0)
    col_std = raw_similarities.std(axis=0)
    col_std[col_std == 0] = 1.0

    z_scores = (raw_similarities - col_mean) / col_std
    predicted_indices = z_scores.argmax(axis=1)
    predicted_categories = [category_names[idx] for idx in predicted_indices]
    confidence_scores = raw_similarities[np.arange(len(df)), predicted_indices]

    df["predicted_delay_cause"] = predicted_categories
    df["delay_cause_confidence"] = confidence_scores

    print("\nCalibrated Delay-Cause Distribution across all cases:")
    print(df["predicted_delay_cause"].value_counts())

    # Save calibration artifact
    calib = {
        "category_names": category_names,
        "col_mean": col_mean,
        "col_std": col_std,
        "category_embeddings": cat_embeddings,
    }
    calib_out = os.path.join(MODELS_DIR, "delay_cause_calibration.pkl")
    with open(calib_out, "wb") as f:
        pickle.dump(calib, f)
    print(f"\nSaved calibration object to {calib_out}")

    # Save updated dataset_with_delay_cause.csv
    df.to_csv(os.path.join(DATA_DIR, "dataset_with_delay_cause.csv"), index=False)
    print(f"Updated {os.path.join(DATA_DIR, 'dataset_with_delay_cause.csv')}")

    # Also update prioritized_cases.csv with calibrated delay cause and recalculate priority scores
    prio_path = os.path.join(DATA_DIR, "prioritized_cases.csv")
    if os.path.exists(prio_path):
        prio_df = pd.read_csv(prio_path)
        # Update delay cause and confidence
        id_to_cause = dict(zip(df["case_id"], df["predicted_delay_cause"]))
        id_to_conf = dict(zip(df["case_id"], df["delay_cause_confidence"]))
        prio_df["predicted_delay_cause"] = prio_df["case_id"].map(id_to_cause).fillna(prio_df.get("predicted_delay_cause"))
        prio_df["delay_cause_confidence"] = prio_df["case_id"].map(id_to_conf).fillna(prio_df.get("delay_cause_confidence", 0.7))

        delay_weights = {
            "Repeated adjournments": 1.0,
            "Procedural or documentation delay": 0.9,
            "Witness non-availability": 0.6,
            "Investigation pending": 0.4,
            "No specific delay noted": 0.1
        }
        severity_weights = {"Fraud or Cheating": 1.0, "Theft or Robbery": 0.7, "Cyber Crime": 0.4}

        prio_df["delay_cause_score"] = prio_df["predicted_delay_cause"].map(lambda x: delay_weights.get(x, 0.5))
        prio_df["severity_score"] = prio_df["crime_type"].map(lambda x: severity_weights.get(x, 0.5))
        prio_df["landmark_score"] = prio_df["landmark_case"].apply(lambda x: 1.0 if x is True or x == 1 else 0.0)
        prio_df["duration_score"] = (prio_df["predicted_duration"] / 280.0).clip(upper=1.0)

        prio_df["priority_score"] = (
            0.40 * prio_df["duration_score"] +
            0.35 * prio_df["delay_cause_score"] +
            0.15 * prio_df["severity_score"] +
            0.10 * prio_df["landmark_score"]
        ).round(3)

        prio_df["priority_level"] = prio_df["priority_score"].apply(
            lambda s: "High" if s >= 0.6 else "Medium" if s >= 0.35 else "Low"
        )
        prio_df = prio_df.sort_values("priority_score", ascending=False).reset_index(drop=True)
        prio_df.to_csv(prio_path, index=False)
        print(f"Updated {prio_path} with calibrated scores.")
        print("\nNew Priority Level Distribution:")
        print(prio_df["priority_level"].value_counts())

if __name__ == "__main__":
    main()
