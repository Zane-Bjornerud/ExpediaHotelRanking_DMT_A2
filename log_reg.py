"""
=============================================================================
Expedia Hotel Ranking — Logistic Regression Baseline (with Hyperparameter Tuning)
VU Data Mining Techniques 2026 — Assignment 2
=============================================================================
Pipeline:
  Load train/test data
  Time-based split: train → train_fit + validation
  Feature engineering
  Hyperparameter tuning (grid search over C, penalty)
  Evaluate best model on validation set (NDCG@5)
  Retrain best config on full train set
  Predict & create Kaggle submission

Usage:
  pip install pandas numpy scikit-learn tqdm
  python baseline_logistic_regression.py

Expects train.csv and test.csv in the same directory.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ndcg_score
from tqdm import tqdm
from itertools import product as iterproduct
import warnings
warnings.filterwarnings("ignore")

#PATHS
TRAIN_PATH = "training_set_VU_DM.csv"
TEST_PATH = "test_set_VU_DM.csv"
SUBMISSION_PATH = "submission_logreg_baseline.csv"

#LOAD DATA
print("STEP 1: Loading data...")

dtypes = {
    "srch_id": "int32",
    "site_id": "int8",
    "visitor_location_country_id": "int16",
    "prop_country_id": "int16",
    "prop_id": "int32",
    "prop_starrating": "int8",
    "prop_brand_bool": "int8",
    "promotion_flag": "int8",
    "srch_destination_id": "int32",
    "srch_length_of_stay": "int8",
    "srch_booking_window": "int16",
    "srch_adults_count": "int8",
    "srch_children_count": "int8",
    "srch_room_count": "int8",
    "srch_saturday_night_bool": "int8",
    "random_bool": "int8",
}

train_full = pd.read_csv(TRAIN_PATH, dtype=dtypes)
test = pd.read_csv(TEST_PATH, dtype=dtypes)

print(f"  Train: {train_full.shape[0]:,} rows, {train_full.shape[1]} columns")
print(f"  Test:  {test.shape[0]:,} rows, {test.shape[1]} columns")


#TIME-BASED TRAIN/VALIDATION SPLIT
print("STEP 2: Time-based train/validation split...")

train_full["date_time"] = pd.to_datetime(train_full["date_time"])
cutoff = train_full["date_time"].quantile(0.8)

# Split by search ID to keep all hotels in a search together
search_times = train_full.groupby("srch_id")["date_time"].min()
val_search_ids = set(search_times[search_times >= cutoff].index)

val_mask = train_full["srch_id"].isin(val_search_ids)
train_mask = ~val_mask

train_df = train_full[train_mask].reset_index(drop=True)
val_df = train_full[val_mask].reset_index(drop=True)

print(f"  Cutoff date: {cutoff}")
print(f"  Train split: {len(train_df):,} rows ({train_df['srch_id'].nunique():,} searches)")
print(f"  Val split:   {len(val_df):,} rows ({val_df['srch_id'].nunique():,} searches)")
print(f"  Train booking rate: {train_df['booking_bool'].mean():.4f}")
print(f"  Val booking rate:   {val_df['booking_bool'].mean():.4f}")


#FEATURE ENGINEERING
print("Feature engineering...")

def engineer_features(df):
    """Create features from raw data. Works on train, val, and test."""
    f = pd.DataFrame(index=df.index)

    #Raw features
    raw_cols = [
        "prop_starrating", "prop_review_score", "prop_brand_bool",
        "prop_location_score1", "prop_location_score2",
        "prop_log_historical_price", "price_usd", "promotion_flag",
        "srch_length_of_stay", "srch_booking_window",
        "srch_adults_count", "srch_children_count", "srch_room_count",
        "srch_saturday_night_bool", "orig_destination_distance",
        "srch_query_affinity_score",
    ]
    for col in raw_cols:
        if col in df.columns:
            f[col] = df[col].astype("float32")

    #Missing value indicators
    f["missing_visitor_hist"] = df["visitor_hist_starrating"].isna().astype("int8")
    f["missing_review_score"] = df["prop_review_score"].isna().astype("int8")
    f["missing_distance"] = df["orig_destination_distance"].isna().astype("int8")
    f["missing_affinity"] = df["srch_query_affinity_score"].isna().astype("int8")

    #Visitor history
    f["visitor_hist_starrating"] = df["visitor_hist_starrating"].fillna(0).astype("float32")
    f["visitor_hist_adr_usd"] = df["visitor_hist_adr_usd"].fillna(0).astype("float32")

    # Does this hotel match the visitor's historical preferences?
    f["star_diff"] = (df["prop_starrating"] - df["visitor_hist_starrating"].fillna(df["prop_starrating"])).astype("float32")
    f["price_diff_hist"] = (df["price_usd"] - df["visitor_hist_adr_usd"].fillna(df["price_usd"])).astype("float32")

    #Price relative to search group
    grp = df.groupby("srch_id")["price_usd"]
    search_mean_price = grp.transform("mean")
    search_min_price = grp.transform("min")

    f["price_vs_mean"] = (df["price_usd"] / search_mean_price.replace(0, 1)).astype("float32")
    f["price_rank_in_search"] = grp.rank(method="min").astype("float32")
    f["is_cheapest"] = (df["price_usd"] == search_min_price).astype("int8")

    #Star & review ranking within search
    f["star_rank_in_search"] = df.groupby("srch_id")["prop_starrating"].rank(method="min", ascending=False).astype("float32")
    review_filled = df["prop_review_score"].fillna(0)
    f["review_rank_in_search"] = review_filled.groupby(df["srch_id"]).rank(method="min", ascending=False).astype("float32")

    #Location score combined
    f["location_combined"] = (df["prop_location_score1"].fillna(0) + df["prop_location_score2"].fillna(0)).astype("float32")

    #Competitor aggregates
    comp_rate_cols = [f"comp{i}_rate" for i in range(1, 9)]
    comp_inv_cols = [f"comp{i}_inv" for i in range(1, 9)]
    existing_rate_cols = [c for c in comp_rate_cols if c in df.columns]
    existing_inv_cols = [c for c in comp_inv_cols if c in df.columns]

    if existing_rate_cols:
        comp_rates = df[existing_rate_cols]
        f["comp_rate_cheaper_count"] = (comp_rates == 1).sum(axis=1).astype("int8")
        f["comp_rate_expensive_count"] = (comp_rates == -1).sum(axis=1).astype("int8")
        f["comp_rate_available_count"] = comp_rates.notna().sum(axis=1).astype("int8")
    if existing_inv_cols:
        comp_inv = df[existing_inv_cols]
        f["comp_no_inventory_count"] = (comp_inv == 1).sum(axis=1).astype("int8")

    #Derived
    total_guests = (df["srch_adults_count"] + df["srch_children_count"]).replace(0, 1)
    f["price_per_person"] = (df["price_usd"] / total_guests).astype("float32")
    f["total_cost"] = (df["price_usd"] * df["srch_length_of_stay"]).astype("float32")

    #Fill remaining NaNs
    f = f.fillna(0)
    return f


train_features = engineer_features(train_df)
val_features = engineer_features(val_df)
test_features = engineer_features(test)

feature_cols = train_features.columns.tolist()
print(f"  Created {len(feature_cols)} features")

y_train = train_df["booking_bool"].values
y_val = val_df["booking_bool"].values


#4. SCALE FEATUREs
print("Scaling features...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(train_features)
X_val_scaled = scaler.transform(val_features)

print("  Done.")


#NDCG@5 EVALUATION FUNCTION

def compute_ndcg5(srch_ids, preds, click_bools, booking_bools):
    """Compute mean NDCG@5 using competition relevance grades (booking=5, click=1)."""
    relevance = np.zeros(len(preds))
    relevance[click_bools == 1] = 1
    relevance[booking_bools == 1] = 5

    df = pd.DataFrame({
        "srch_id": srch_ids,
        "pred": preds,
        "relevance": relevance,
    })

    scores = []
    for _, group in df.groupby("srch_id"):
        if group["relevance"].sum() == 0:
            continue
        true_rel = group["relevance"].values.reshape(1, -1)
        pred_scores = group["pred"].values.reshape(1, -1)
        try:
            scores.append(ndcg_score(true_rel, pred_scores, k=5))
        except:
            continue

    return np.mean(scores), len(scores)


#HYPERPARAMETER TUNING
print("Hyperparameter tuning...")

param_grid = {
    "C": [0.001, 0.01, 0.1, 1.0, 10.0],
    "penalty": ["l1", "l2"],
}

param_combos = list(iterproduct(param_grid["C"], param_grid["penalty"]))
print(f"  Testing {len(param_combos)} combinations...\n")

best_ndcg = -1
best_params = None
all_results = []

for i, (C, penalty) in enumerate(param_combos):
    print(f"  [{i+1}/{len(param_combos)}] C={C:<6}, penalty={penalty}", end=" ... ")

    model = LogisticRegression(
        C=C,
        penalty=penalty,
        solver="saga",
        max_iter=200,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train_scaled, y_train)

    val_preds = model.predict_proba(X_val_scaled)[:, 1]

    ndcg, n_searches = compute_ndcg5(
        val_df["srch_id"].values, val_preds,
        val_df["click_bool"].values, val_df["booking_bool"].values,
    )

    result = {"C": C, "penalty": penalty, "ndcg5": ndcg, "n_searches": n_searches}
    all_results.append(result)

    marker = " *** BEST ***" if ndcg > best_ndcg else ""
    print(f"NDCG@5 = {ndcg:.5f}{marker}")

    if ndcg > best_ndcg:
        best_ndcg = ndcg
        best_params = result

print(f"BEST: C={best_params['C']}, penalty={best_params['penalty']}")
print(f"NDCG@5 = {best_ndcg:.5f}")

# Save tuning results
results_df = pd.DataFrame(all_results).sort_values("ndcg5", ascending=False)
results_df.to_csv("tuning_results.csv", index=False)
print("\n  Tuning results saved to tuning_results.csv")


#RETRAIN BEST MODEL ON FULL TRAINING DATA
print("Retraining best model on full training set...")

full_train_features = engineer_features(train_full)

scaler_final = StandardScaler()
X_full_scaled = scaler_final.fit_transform(full_train_features)
y_full = train_full["booking_bool"].values

final_model = LogisticRegression(
    C=best_params["C"],
    penalty=best_params["penalty"],
    solver="saga",
    max_iter=200,
    random_state=42,
    n_jobs=-1,
    verbose=1,
)
final_model.fit(X_full_scaled, y_full)

# Feature coefficients
print("\n  Feature coefficients (top 15 by absolute value):")
coef_df = pd.DataFrame({
    "feature": feature_cols,
    "coefficient": final_model.coef_[0]
}).sort_values("coefficient", key=abs, ascending=False)
for _, row in coef_df.head(15).iterrows():
    print(f"    {row['feature']:35s} {row['coefficient']:+.4f}")


#PREDICT TEST SET & CREATE SUBMISSION
print("Predicting on test set & creating submission...")

X_test_scaled = scaler_final.transform(test_features)
test_preds = final_model.predict_proba(X_test_scaled)[:, 1]

submission_df = pd.DataFrame({
    "srch_id": test["srch_id"],
    "prop_id": test["prop_id"],
    "pred": test_preds,
})

submission_df = submission_df.sort_values(
    by=["srch_id", "pred"],
    ascending=[True, False],
)

submission_df[["srch_id", "prop_id"]].to_csv(SUBMISSION_PATH, index=False)

print(f"  Submission saved to: {SUBMISSION_PATH}")
print(f"  Total rows: {len(submission_df):,}")
print(f"  Unique searches: {submission_df['srch_id'].nunique():,}")


# SAVE STATS FOR REPORt
print("Saving stats for report...")

stats = {
    "dataset": {
        "total_train_rows": len(train_full),
        "total_test_rows": len(test),
        "train_searches": int(train_full["srch_id"].nunique()),
        "test_searches": int(test["srch_id"].nunique()),
        "booking_rate": float(train_full["booking_bool"].mean()),
        "click_rate": float(train_full["click_bool"].mean()),
        "avg_hotels_per_search": float(train_full.groupby("srch_id").size().mean()),
        "unique_hotels": int(train_full["prop_id"].nunique()),
        "unique_destinations": int(train_full["srch_destination_id"].nunique()),
    },
    "missing_pct": {
        col: round(float(train_full[col].isna().mean()), 4)
        for col in train_full.columns if train_full[col].isna().any()
    },
    "validation": {
        "split_cutoff": str(cutoff),
        "train_rows": int(train_mask.sum()),
        "val_rows": int(val_mask.sum()),
        "best_ndcg5": float(best_ndcg),
    },
    "best_hyperparams": {
        "C": best_params["C"],
        "penalty": best_params["penalty"],
    },
    "all_tuning_results": all_results,
    "features_used": feature_cols,
    "top_features": coef_df.head(10).to_dict("records"),
}

with open("eda_stats.json", "w") as fout:
    json.dump(stats, fout, indent=2, default=str)

print("  Saved to eda_stats.json")

print(f"  Best validation NDCG@5: {best_ndcg:.5f}")
print(f"  Best params: C={best_params['C']}, penalty={best_params['penalty']}")