"""Data quality checks for MP204 Cold Storage Spoilage.
Author: Kunsh Kakkar (Data Lead) - S24CSEU1029

Checks for the 7 potential data issues listed in the Student Handbook:
1. Leaking column
2. Unit mixup
3. Sentinel values
4. Duplicate rows
5. Stuck sensor
6. Target typos
7. Spelling variants
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from src.mp204.utils import load_params


def run_all_checks(train_path: str = "data/raw/train.csv",
                   later_path: str = "data/raw/later.csv",
                   holdout_path: str = "data/raw/holdout_features.csv"):
    """Execute all 7 data checks on the dataset."""
    print("=" * 60)
    print("MP204 DATA QUALITY AUDIT - KOCHI COLD CHAIN COOPERATIVE")
    print("=" * 60)

    train = pd.read_csv(train_path)
    later = pd.read_csv(later_path)
    holdout = pd.read_csv(holdout_path)

    findings = {}

    # 1. Leaking column
    leaking_cols = [c for c in train.columns if c not in holdout.columns and c != "spoilage_pct"]
    print("\n[Check 1] Leaking Column:")
    if leaking_cols:
        col = leaking_cols[0]
        corr = train[[col, "spoilage_pct"]].dropna().corr().iloc[0, 1]
        mean_diff = (train[col] - train["spoilage_pct"]).abs().mean()
        print(f"  -> PRESENT: Column '{col}' is present in train but missing from holdout.")
        print(f"     Correlation with target: {corr:.4f}, Mean absolute difference: {mean_diff:.4f}")
        findings["leaking_column"] = {"present": True, "column": col}
    else:
        print("  -> ABSENT: No unexpected columns found.")
        findings["leaking_column"] = {"present": False, "column": None}

    # 2. Unit mixup
    print("\n[Check 2] Unit Mixup:")
    candidate_cols = ["storage_days", "temp_set_c", "humidity_pct", "power_cuts_hours", "bag_weight_kg", "door_openings_per_day"]
    unit_found = False
    for col in candidate_cols:
        group_means = train.groupby("chamber")[col].mean()
        max_mean = group_means.max()
        min_mean = group_means.min()
        ratio = max_mean / (min_mean + 1e-9)
        if ratio > 10:  # significant unit discrepancy
            group_val = group_means.idxmax()
            other_mean = group_means[group_means.index != group_val].mean()
            est_factor = round(max_mean / other_mean)
            print(f"  -> PRESENT: Column '{col}' in '{group_val}' has mean {max_mean:.1f} vs other chambers {other_mean:.1f}.")
            print(f"     Factor: {est_factor} (recorded in minutes instead of hours, 60 min = 1 hr).")
            findings["unit_mixup"] = {"present": True, "column": col, "group_value": group_val, "factor": est_factor}
            unit_found = True
            break
    if not unit_found:
        print("  -> ABSENT: No unit discrepancies across groups.")
        findings["unit_mixup"] = {"present": False, "column": None, "group_value": None, "factor": None}

    # 3. Sentinel values
    print("\n[Check 3] Sentinel Value:")
    sentinel_found = False
    for col in candidate_cols:
        vc = train[col].value_counts()
        sentinels = vc[vc.index.isin([-999, -99, -1, 999, 9999])]
        if len(sentinels) > 0:
            val = sentinels.index[0]
            cnt = int(sentinels.iloc[0])
            print(f"  -> PRESENT: Column '{col}' contains sentinel value {val} repeated {cnt} times.")
            findings["sentinel_value"] = {"present": True, "column": col, "value": float(val), "count": cnt}
            sentinel_found = True
            break
    if not sentinel_found:
        print("  -> ABSENT: No sentinel values detected.")
        findings["sentinel_value"] = {"present": False, "column": None, "value": None, "count": None}

    # 4. Duplicate rows
    print("\n[Check 4] Duplicate Rows:")
    dup_count = int(train.duplicated().sum())
    if dup_count > 0:
        print(f"  -> PRESENT: {dup_count} duplicate rows found.")
        findings["duplicate_rows"] = {"present": True, "count": dup_count}
    else:
        print("  -> ABSENT: 0 extra duplicate rows found.")
        findings["duplicate_rows"] = {"present": False, "count": None}

    # 5. Stuck sensor
    print("\n[Check 5] Stuck Sensor:")
    train_sorted = train.sort_values("date").reset_index(drop=True)
    stuck_found = False
    for col in ["door_openings_per_day", "temp_set_c", "humidity_pct", "power_cuts_hours", "bag_weight_kg"]:
        date_groups = train_sorted.groupby("date")[col].unique()
        # Look for continuous dates with single constant value
        constant_dates = [d for d, vals in date_groups.items() if len(vals) == 1]
        # Find consecutive streak of dates
        if len(constant_dates) >= 10:
            streak = []
            max_streak = []
            for d in constant_dates:
                val = date_groups[d][0]
                if not streak or (d_prev_val == val):
                    streak.append((d, val))
                else:
                    if len(streak) > len(max_streak):
                        max_streak = streak
                    streak = [(d, val)]
                d_prev_val = val
            if len(streak) > len(max_streak):
                max_streak = streak

            if len(max_streak) >= 10:
                start_date = max_streak[0][0]
                end_date = max_streak[-1][0]
                stuck_val = max_streak[0][1]
                print(f"  -> PRESENT: Column '{col}' stuck at {stuck_val} from {start_date} to {end_date} ({len(max_streak)} days).")
                findings["stuck_sensor"] = {
                    "present": True,
                    "column": col,
                    "start_date": str(start_date),
                    "end_date": str(end_date)
                }
                stuck_found = True
                break
    if not stuck_found:
        print("  -> ABSENT: No stuck sensor found.")
        findings["stuck_sensor"] = {"present": False, "column": None, "start_date": None, "end_date": None}

    # 6. Target typos
    print("\n[Check 6] Target Typos:")
    target_high = train[train["spoilage_pct"] > 100]
    if len(target_high) > 0:
        print(f"  -> PRESENT: {len(target_high)} targets typed > 100%.")
        findings["target_typos"] = {"present": True, "count": len(target_high)}
    else:
        print("  -> ABSENT: Target values are well bounded (min: {:.2f}, max: {:.2f}).".format(
            train["spoilage_pct"].min(), train["spoilage_pct"].max()
        ))
        findings["target_typos"] = {"present": False, "count": None}

    # 7. Spelling variants
    print("\n[Check 7] Spelling Variants:")
    cat_cols = ["crop", "chamber"]
    spelling_found = False
    for col in cat_cols:
        train_cats = set(train[col].dropna().unique())
        later_cats = set(later[col].dropna().unique())
        # Check if stripping or case normalization changes count
        normalized = set(train[col].dropna().astype(str).str.strip().str.lower().unique())
        if len(train_cats) > len(normalized) or (train_cats != later_cats):
            print(f"  -> PRESENT: Categorical '{col}' has variants.")
            findings["spelling_variants"] = {"present": True, "column": col, "true_levels": len(later_cats)}
            spelling_found = True
            break
    if not spelling_found:
        print("  -> ABSENT: Categorical columns ('crop', 'chamber') match clean categories exactly.")
        findings["spelling_variants"] = {"present": False, "column": None, "true_levels": None}

    print("\nAudit complete.")
    return findings


if __name__ == "__main__":
    params = load_params()
    raw_train = params["data"]["raw_train_path"]
    raw_later = params["data"]["raw_later_path"]
    raw_holdout = params["data"]["raw_holdout_path"]
    run_all_checks(raw_train, raw_later, raw_holdout)
