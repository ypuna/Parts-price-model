"""
Score the parts no competitor lists.

Fits the final model on every part that has a competitor price, predicts a market price
for each part that has none, and proposes a list price with the same rule used for
benchmarked parts: 1% below the market estimate, never below a 30% gross margin
(net price = 50% of list). Parts in categories the model prices poorly in
cross-validation are flagged for manual review instead.
"""
import json
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, "src")
from features import FeatureBuilder
from train_evaluate import LGB_PARAMS, load

UNDERCUT = 0.01
NET_SHARE = 0.50        # shops pay 50% of list
MIN_MARGIN = 0.30


def main():
    df = load()
    lab = df[df["n_competitors"] > 0].reset_index(drop=True)
    new = df[df["n_competitors"] == 0].reset_index(drop=True)

    fb = FeatureBuilder()
    Xa, Xb = fb.fit_transform(lab), fb.transform(new)
    y = np.log(lab["market_price"].values)
    model = lgb.LGBMRegressor(**LGB_PARAMS).fit(Xa, y)

    # categories with high held-out error (from train_evaluate.py) go to manual review
    res = json.load(open("reports/results.json"))
    review_cats = {r["category"] for r in res["by_category"]
                   if r["median_abs_pct_error"] > res["review_threshold"]}
    small_cats = set(new["category"]) - {r["category"] for r in res["by_category"]}  # too few parts to judge

    new["market_estimate"] = np.exp(model.predict(Xb))
    margin_floor = new["unit_cost"] / (NET_SHARE * (1 - MIN_MARGIN))
    new["proposed_list"] = np.maximum(new["market_estimate"] * (1 - UNDERCUT), margin_floor).round(0)
    new["margin_floor_applied"] = (margin_floor > new["market_estimate"] * (1 - UNDERCUT)).astype(int)
    new["review"] = new["category"].isin(review_cats | small_cats).astype(int)

    cur = new[new["current_list"] > 0]
    gap = cur["current_list"] / cur["market_estimate"] - 1
    summary = dict(
        parts_without_competitor_price=int(len(new)),
        auto_priced=int((new["review"] == 0).sum()), sent_to_review=int(new["review"].sum()),
        review_categories=sorted(review_cats), small_categories_reviewed=sorted(small_cats),
        margin_floor_applied=int(new["margin_floor_applied"].sum()),
        current_list_vs_market_estimate=dict(
            median_pct=float(gap.median()),
            share_more_than_20pct_above=float((gap > 0.20).mean()),
            share_more_than_20pct_below=float((gap < -0.20).mean()),
            share_within_20pct=float((gap.abs() <= 0.20).mean())),
    )
    json.dump(summary, open("reports/unbenchmarked_summary.json", "w"), indent=2)
    cols = ["description", "category", "unit_cost", "current_list", "market_estimate",
            "proposed_list", "margin_floor_applied", "review"]
    new[cols].round(2).to_csv("reports/unbenchmarked_proposed_prices.csv", index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
