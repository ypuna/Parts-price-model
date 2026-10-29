"""
Estimate a part's market price (median competitor list price) from its catalog
description and unit cost, for parts no competitor lists.

Validation: 5-fold GroupKFold. Left/right and certified/non-certified versions of the
same part share a group, so the model is never scored on a near-copy of a training row.

Compares three approaches:
  1. cost-plus     : cost x the category's median market-to-cost ratio (the usual fallback)
  2. ridge         : linear model on the same features
  3. lightgbm      : gradient-boosted trees on the same features

Also reports error by category, so categories the model prices poorly can go to
manual review instead of being priced automatically.
"""
import json
import os
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

sys.path.insert(0, "src")
from features import FeatureBuilder, parse

SEED = 7
LGB_PARAMS = dict(n_estimators=1500, learning_rate=0.03, num_leaves=63, min_child_samples=20,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0,
                  random_state=SEED, verbose=-1)
COMP = ["comp_a", "comp_b", "comp_c", "comp_d"]
REVIEW_THRESHOLD = 0.25   # categories with median error above 25% are priced by a person


def load(path=None):
    df = pd.read_csv(path or os.environ.get("PARTS_CSV", "data/parts.csv"))
    df["market_price"] = df[COMP].median(axis=1)
    return df


def metrics(y_true, y_pred):
    ape = np.abs(y_pred / y_true - 1)
    return dict(median_abs_pct_error=float(np.median(ape)), within_10pct=float(np.mean(ape <= 0.10)),
                within_20pct=float(np.mean(ape <= 0.20)), n=int(len(y_true)))


def cross_validate(df, groups):
    y = np.log(df["market_price"].values)
    oof = {k: np.zeros(len(df)) for k in ["cost_plus", "ridge", "lightgbm"]}
    folds = np.zeros(len(df), dtype=int)
    for k, (tr, te) in enumerate(GroupKFold(n_splits=5).split(df, y, groups)):
        folds[te] = k
        a, b = df.iloc[tr], df.iloc[te]
        # 1. cost-plus by category
        ratio = (np.log(a["market_price"]) - np.log(a["unit_cost"])).groupby(a["category"]).median()
        overall = float(np.median(np.log(a["market_price"]) - np.log(a["unit_cost"])))
        oof["cost_plus"][te] = np.log(b["unit_cost"]) + b["category"].map(ratio).fillna(overall).values
        # 2-3. models on the same features
        fb = FeatureBuilder()
        Xa, Xb = fb.fit_transform(a), fb.transform(b)
        oof["ridge"][te] = Ridge(alpha=1.0).fit(Xa, y[tr]).predict(Xb)
        oof["lightgbm"][te] = lgb.LGBMRegressor(**LGB_PARAMS).fit(Xa, y[tr]).predict(Xb)
        print(f"fold {k + 1}/5 done")
    return y, oof, folds


def competitor_agreement(df):
    """How closely do competitors agree? Score each competitor against the median of the others
    (parts with 3+ competitors). This is a rough floor for any model of 'the market price'."""
    m = df[df[COMP].notna().sum(axis=1) >= 3]
    errs = []
    for c in COMP:
        x = m[m[c].notna()]
        others = x[[o for o in COMP if o != c]].median(axis=1)
        errs.append(np.abs(others / x[c] - 1))
    e = np.concatenate(errs)
    return dict(median_abs_pct_error=float(np.median(e)), within_10pct=float(np.mean(e <= 0.10)),
                within_20pct=float(np.mean(e <= 0.20)), n=int(len(e)))


def main():
    df = load()
    lab = df[df["n_competitors"] > 0].reset_index(drop=True)
    groups = parse(lab)["family"].values
    print(f"{len(lab):,} parts with a competitor price, {len(set(groups)):,} part families")

    y, oof, folds = cross_validate(lab, groups)
    results = {name: metrics(np.exp(y), np.exp(p)) for name, p in oof.items()}
    # same comparison restricted to parts with 2+ competitors (a steadier market price)
    multi = (lab["n_competitors"] >= 2).values
    results_multi = {name: metrics(np.exp(y[multi]), np.exp(p[multi])) for name, p in oof.items()}

    # per-category accuracy for the main model
    ape = np.abs(np.exp(oof["lightgbm"] - y) - 1)
    by_cat = (pd.DataFrame({"category": lab["category"], "ape": ape})
              .groupby("category")["ape"].agg(parts="size", median_abs_pct_error="median",
                                               within_10pct=lambda s: float(np.mean(s <= 0.10)))
              .query("parts >= 200").sort_values("parts", ascending=False))

    agreement = competitor_agreement(lab)
    out = dict(n_parts=int(len(lab)), n_families=int(len(set(groups))), all_parts=results,
               competitor_vs_other_competitors=agreement,
               parts_with_2plus_competitors=results_multi,
               review_threshold=REVIEW_THRESHOLD,
               by_category=by_cat.reset_index().to_dict(orient="records"))
    json.dump(out, open("reports/results.json", "w"), indent=2)
    pd.DataFrame({"fold": folds, "y_log": y, **{f"pred_{k}": v for k, v in oof.items()}}).to_csv("reports/oof_predictions.csv", index=False)

    # figures
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = {"cost_plus": "Cost-plus by category", "ridge": "Ridge regression", "lightgbm": "LightGBM"}
    for name in ["cost_plus", "ridge", "lightgbm"]:
        e = np.sort(np.abs(np.exp(oof[name] - y) - 1))
        ax.plot(e, np.arange(1, len(e) + 1) / len(e), label=labels[name], lw=2)
    ax.axhline(0.5, color="grey", lw=0.5)
    ax.axvline(agreement["median_abs_pct_error"], color="black", ls="--", lw=1,
               label=f"Competitors vs. each other (median {agreement['median_abs_pct_error']:.0%})")
    ax.set_xlim(0, 0.6); ax.set_xlabel("Absolute % error vs. market price"); ax.set_ylabel("Share of parts")
    ax.axvline(0.10, color="grey", ls=":", lw=1); ax.legend(frameon=False)
    ax.set_title("Held-out error, 5-fold grouped CV")
    fig.tight_layout(); fig.savefig("reports/figures/error_cdf.png", dpi=150)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bc = by_cat.sort_values("median_abs_pct_error")
    colors = ["#c0392b" if v > REVIEW_THRESHOLD else "#2e86c1" for v in bc["median_abs_pct_error"]]
    ax.barh(bc.index, bc["median_abs_pct_error"] * 100, color=colors)
    ax.axvline(REVIEW_THRESHOLD * 100, color="black", ls="--", lw=1, label="Review threshold")
    ax.set_xlabel("Median absolute % error (held out)")
    ax.set_title("LightGBM error by category (red = send to manual review)")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig("reports/figures/error_by_category.png", dpi=150)

    print(json.dumps({k: out[k] for k in ["all_parts", "parts_with_2plus_competitors", "competitor_vs_other_competitors"]}, indent=1))
    print(by_cat.round(3).to_string())


if __name__ == "__main__":
    main()
