#!/usr/bin/env python3
"""Non-learned anchors for the main table: persistence (last observation-year
cost, capped at $1M) and the capped training-label mean.  Writes
<section>/model_gen/{persistence,trainmean}/cost_predictions.csv."""

import os
import argparse
import numpy as np
import pandas as pd

CAP = 1_000_000.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline_train", required=True,
                   help="<baseline_dir>/train/preprocessed_input.parquet")
    p.add_argument("--section_dirs", nargs="+", required=True)
    p.add_argument("--history_length", default="1year")
    p.add_argument("--token_feature_type", default="count")
    a = p.parse_args()

    train = pd.read_parquet(a.baseline_train, columns=["next_annual_cost"])
    mean_cost = float(train.next_annual_cost.clip(upper=CAP).mean())
    print(f"train mean (capped): {mean_cost:.2f} over {len(train)} rows")
    for sec in a.section_dirs:
        feats = pd.read_parquet(os.path.join(
            sec, f"baseline_features_{a.history_length}_{a.token_feature_type}.parquet"),
            columns=["enrollee_id", "prev_cost"])
        for name, vals in [("persistence", np.minimum(feats.prev_cost.values, CAP)),
                           ("trainmean", np.full(len(feats), mean_cost))]:
            out_dir = os.path.join(sec, "model_gen", name)
            os.makedirs(out_dir, exist_ok=True)
            pd.DataFrame({"enrollee_id": feats.enrollee_id,
                          "predicted_next_annual_cost": vals}).to_csv(
                os.path.join(out_dir, "cost_predictions.csv"), index=False)
            print(f"  {sec}: {name} ({len(feats)} rows)")


if __name__ == "__main__":
    main()
