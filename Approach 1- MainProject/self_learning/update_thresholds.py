# self_learning/update_thresholds.py

import pandas as pd
import json
import numpy as np

DEFAULT_THRESHOLDS = {
    "lap_good": 120.0,
    "lap_recover": 60.0,
    "orb_good": 800,
    "orb_recover": 300,
    "haze_bad": 0.35,
    "haze_recover": 0.30,
    "clipped_bad": 0.10
}

def update_thresholds_from_log(log_csv_path, out_json_path):
    """
    Reads pipeline_log.csv and adjusts thresholds using simple quantiles.
    """
    df = pd.read_csv(log_csv_path)

    if "true_label" not in df.columns:
        print("No true_label column found. Cannot adapt thresholds reliably.")
        json.dump(DEFAULT_THRESHOLDS, open(out_json_path, "w"))
        return

    th = DEFAULT_THRESHOLDS.copy()

    # For GOOD images, look at distributions
    good = df[df["true_label"] == "GOOD"]
    if len(good) > 20:
        th["lap_good"] = float(good["lap_var"].quantile(0.3))  # 30% quantile
        th["orb_good"] = int(good["orb_kpts"].quantile(0.3))
        th["haze_recover"] = float(good["haze_score"].quantile(0.8))

    # For BAD images, update "bad" thresholds
    bad = df[df["true_label"] == "BAD"]
    if len(bad) > 20:
        th["lap_recover"] = float(bad["lap_var"].quantile(0.7))  # 70% quantile
        th["orb_recover"] = int(bad["orb_kpts"].quantile(0.7))
        th["haze_bad"] = float(bad["haze_score"].quantile(0.3))

    with open(out_json_path, "w") as f:
        json.dump(th, f, indent=2)

    print("Updated thresholds written to", out_json_path)
    return th
