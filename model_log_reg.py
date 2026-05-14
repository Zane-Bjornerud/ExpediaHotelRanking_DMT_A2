import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ndcg_score
from itertools import product as iterproduct
import json
import warnings
warnings.filterwarnings("ignore")

# LOAD DATA
print("Loading data")
TRAIN_PATH = "training_set_VU_DM.csv"
TEST_PATH = "test_set_VU_DM.csv"

dfTrain = pd.read_csv(TRAIN_PATH, low_memory=False)
dfTest = pd.read_csv(TEST_PATH, low_memory=False)
print(f"  Train: {dfTrain.shape[0]:,} rows")
print(f"  Test:  {dfTest.shape[0]:,} rows")


# TRAIN/VALIDATION SPLIT (time-based, by search ID)
# Done BEFORE feature engineering so we can compute per-property stats
# on the training split only (avoids data leakage)
print("\nSplitting train/validation by time...")
dfTrain["date_time"] = pd.to_datetime(dfTrain["date_time"])
cutoff = dfTrain["date_time"].quantile(0.8)

search_times = dfTrain.groupby("srch_id")["date_time"].min()
val_search_ids = set(search_times[search_times >= cutoff].index)

val_mask = dfTrain["srch_id"].isin(val_search_ids)
train_mask = ~val_mask

print(f"  Train: {train_mask.sum():,} rows ({dfTrain.loc[train_mask, 'srch_id'].nunique():,} searches)")
print(f"  Val:   {val_mask.sum():,} rows ({dfTrain.loc[val_mask, 'srch_id'].nunique():,} searches)")
print(f"  Cutoff: {cutoff}")


# COMPUTE PER-PROPERTY HISTORICAL STATS (from training split only)
# Some hotels are just better than others — capture that signal
print("\nComputing per-property and per-destination stats...")

train_split = dfTrain.loc[train_mask]

# Per-property stats
prop_stats = train_split.groupby("prop_id").agg(
    prop_click_rate=("click_bool", "mean"),
    prop_book_rate=("booking_bool", "mean"),
    prop_count=("booking_bool", "count"),
).reset_index()

# Per-destination stats
dest_stats = train_split.groupby("srch_destination_id").agg(
    dest_click_rate=("click_bool", "mean"),
    dest_book_rate=("booking_bool", "mean"),
    dest_count=("booking_bool", "count"),
).reset_index()

# Global averages for filling unknown hotels/destinations in val/test
global_click_rate = train_split["click_bool"].mean()
global_book_rate = train_split["booking_bool"].mean()

print(f"  {len(prop_stats):,} unique properties with stats")
print(f"  {len(dest_stats):,} unique destinations with stats")


# FEATURE ENGINEERING
print("\nEngineering features")

def engineer_features(df, prop_stats, dest_stats, global_click_rate, global_book_rate):
    f = pd.DataFrame(index=df.index)

    # Raw features (ones with signal from EDA correlation analysis)
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

    # Missing value indicators
    f["missing_visitor_hist"] = df["visitor_hist_starrating"].isna().astype("int8")
    f["missing_review_score"] = df["prop_review_score"].isna().astype("int8")
    f["missing_distance"] = df["orig_destination_distance"].isna().astype("int8")
    f["missing_affinity"] = df["srch_query_affinity_score"].isna().astype("int8")

    # Visitor history
    f["visitor_hist_starrating"] = df["visitor_hist_starrating"].fillna(0).astype("float32")
    f["visitor_hist_adr_usd"] = df["visitor_hist_adr_usd"].fillna(0).astype("float32")

    # Does this hotel match what the user usually books?
    f["star_diff"] = (df["prop_starrating"] - df["visitor_hist_starrating"].fillna(df["prop_starrating"])).astype("float32")
    f["price_diff_hist"] = (df["price_usd"] - df["visitor_hist_adr_usd"].fillna(df["price_usd"])).astype("float32")

    # Price relative to search group
    grp = df.groupby("srch_id")["price_usd"]
    search_mean_price = grp.transform("mean")
    search_min_price = grp.transform("min")

    f["price_vs_mean"] = (df["price_usd"] / search_mean_price.replace(0, 1)).astype("float32")
    f["price_rank_in_search"] = grp.rank(method="min").astype("float32")
    f["is_cheapest"] = (df["price_usd"] == search_min_price).astype("int8")

    # Star & review ranking within search
    f["star_rank_in_search"] = df.groupby("srch_id")["prop_starrating"].rank(method="min", ascending=False).astype("float32")
    review_filled = df["prop_review_score"].fillna(0)
    f["review_rank_in_search"] = review_filled.groupby(df["srch_id"]).rank(method="min", ascending=False).astype("float32")

    # Location score combined
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

    #Per-property historical stats
    # Some hotels consistently get booked more — this captures hotel quality
    # beyond what star rating and review score tell us
    merged_prop = df[["prop_id"]].merge(prop_stats, on="prop_id", how="left")
    f["prop_click_rate"] = merged_prop["prop_click_rate"].fillna(global_click_rate).astype("float32").values
    f["prop_book_rate"] = merged_prop["prop_book_rate"].fillna(global_book_rate).astype("float32").values
    f["prop_count"] = merged_prop["prop_count"].fillna(0).astype("float32").values

    #Per-destination historical stats
    # Some destinations have higher booking rates
    merged_dest = df[["srch_destination_id"]].merge(dest_stats, on="srch_destination_id", how="left")
    f["dest_click_rate"] = merged_dest["dest_click_rate"].fillna(global_click_rate).astype("float32").values
    f["dest_book_rate"] = merged_dest["dest_book_rate"].fillna(global_book_rate).astype("float32").values
    f["dest_count"] = merged_dest["dest_count"].fillna(0).astype("float32").values

    f = f.fillna(0)
    return f


train_features = engineer_features(dfTrain, prop_stats, dest_stats, global_click_rate, global_book_rate)
test_features = engineer_features(dfTest, prop_stats, dest_stats, global_click_rate, global_book_rate)
feature_cols = train_features.columns.tolist()
print(f"  {len(feature_cols)} features created")


# PREPARE TRAIN/VAL ARRAYS
X_train = train_features.loc[train_mask]
X_val = train_features.loc[val_mask]
y_train_book = dfTrain.loc[train_mask, "booking_bool"].values
y_val_book = dfTrain.loc[val_mask, "booking_bool"].values
y_train_click = dfTrain.loc[train_mask, "click_bool"].values
y_val_click = dfTrain.loc[val_mask, "click_bool"].values


# SCALE FEATURES
print("\nScaling features")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)


# NDCG@5 EVALUATION
def compute_ndcg5(srch_ids, preds, click_bools, booking_bools):
    """NDCG@5 with competition relevance: booking=5, click=1, neither=0."""
    relevance = np.zeros(len(preds))
    relevance[click_bools == 1] = 1
    relevance[booking_bools == 1] = 5

    df = pd.DataFrame({"srch_id": srch_ids, "pred": preds, "relevance": relevance})

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


# HYPERPARAMETER TUNING (using combined click+booking score)
print("\nTuning hyperparameters")

param_grid = {
    "C": [0.001, 0.01, 0.1, 1.0, 10.0],
    "penalty": ["l1", "l2"],
}

param_combos = list(iterproduct(param_grid["C"], param_grid["penalty"]))
print(f"  Testing {len(param_combos)} combinations:\n")

best_ndcg = -1
best_params = None
all_results = []

val_srch_ids = dfTrain.loc[val_mask, "srch_id"].values
val_clicks = dfTrain.loc[val_mask, "click_bool"].values
val_books = dfTrain.loc[val_mask, "booking_bool"].values

for i, (C, penalty) in enumerate(param_combos):
    print(f"  [{i+1}/{len(param_combos)}] C={C:<6}, penalty={penalty}", end=" ... ")

    # Train booking model
    book_model = LogisticRegression(
        C=C, penalty=penalty, solver="saga",
        max_iter=200, random_state=42, n_jobs=-1,
    )
    book_model.fit(X_train_scaled, y_train_book)
    book_preds = book_model.predict_proba(X_val_scaled)[:, 1]

    # Train click model with same hyperparameters
    click_model = LogisticRegression(
        C=C, penalty=penalty, solver="saga",
        max_iter=200, random_state=42, n_jobs=-1,
    )
    click_model.fit(X_train_scaled, y_train_click)
    click_preds = click_model.predict_proba(X_val_scaled)[:, 1]

    # Combine: booking is worth 5x click in NDCG scoring
    combined_preds = 5 * book_preds + 1 * click_preds

    ndcg, n = compute_ndcg5(val_srch_ids, combined_preds, val_clicks, val_books)

    result = {"C": C, "penalty": penalty, "ndcg5": ndcg}
    all_results.append(result)
    marker = " *** BEST ***" if ndcg > best_ndcg else ""
    print(f"NDCG@5 = {ndcg:.5f}{marker}")

    if ndcg > best_ndcg:
        best_ndcg = ndcg
        best_params = result

print(f"\n  BEST: C={best_params['C']}, penalty={best_params['penalty']} -> NDCG@5 = {best_ndcg:.5f}")

pd.DataFrame(all_results).sort_values("ndcg5", ascending=False).to_csv("tuning_results.csv", index=False)


#RETRAIN ON FULL TRAINING SET
print("\nRetraining best models on full training data")

# Recompute property/destination stats on FULL training set for final model
prop_stats_full = dfTrain.groupby("prop_id").agg(
    prop_click_rate=("click_bool", "mean"),
    prop_book_rate=("booking_bool", "mean"),
    prop_count=("booking_bool", "count"),
).reset_index()

dest_stats_full = dfTrain.groupby("srch_destination_id").agg(
    dest_click_rate=("click_bool", "mean"),
    dest_book_rate=("booking_bool", "mean"),
    dest_count=("booking_bool", "count"),
).reset_index()

global_click_full = dfTrain["click_bool"].mean()
global_book_full = dfTrain["booking_bool"].mean()

full_features = engineer_features(dfTrain, prop_stats_full, dest_stats_full, global_click_full, global_book_full)
test_features_final = engineer_features(dfTest, prop_stats_full, dest_stats_full, global_click_full, global_book_full)

scaler_final = StandardScaler()
X_full_scaled = scaler_final.fit_transform(full_features)

# Final booking model
final_book_model = LogisticRegression(
    C=best_params["C"], penalty=best_params["penalty"], solver="saga",
    max_iter=200, random_state=42, n_jobs=-1,
)
final_book_model.fit(X_full_scaled, dfTrain["booking_bool"].values)

# Final click model
final_click_model = LogisticRegression(
    C=best_params["C"], penalty=best_params["penalty"], solver="saga",
    max_iter=200, random_state=42, n_jobs=-1,
)
final_click_model.fit(X_full_scaled, dfTrain["click_bool"].values)

# Print feature importances (booking model)
coef_df = pd.DataFrame({
    "feature": feature_cols, "coefficient": final_book_model.coef_[0]
}).sort_values("coefficient", key=abs, ascending=False)
print("\n  Top 15 features (booking model):")
for _, row in coef_df.head(15).iterrows():
    print(f"    {row['feature']:35s} {row['coefficient']:+.4f}")


# PREDICT & SUBMIT

X_test_scaled = scaler_final.transform(test_features_final)
test_book_preds = final_book_model.predict_proba(X_test_scaled)[:, 1]
test_click_preds = final_click_model.predict_proba(X_test_scaled)[:, 1]
test_combined = 5 * test_book_preds + 1 * test_click_preds

submission = pd.DataFrame({
    "srch_id": dfTest["srch_id"],
    "prop_id": dfTest["prop_id"],
    "pred": test_combined,
}).sort_values(["srch_id", "pred"], ascending=[True, False])

submission[["srch_id", "prop_id"]].to_csv("submission_logreg_v2.csv", index=False)
print(f"  Saved submission_logreg_v2.csv ({len(submission):,} rows)")

# Save stats
stats = {
    "best_params": best_params,
    "all_tuning_results": all_results,
    "top_features": coef_df.head(10).to_dict("records"),
    "features": feature_cols,
}
with open("model_stats_v2.json", "w") as f:
    json.dump(stats, f, indent=2, default=str)

print(f"\nBest validation NDCG@5: {best_ndcg:.5f}")