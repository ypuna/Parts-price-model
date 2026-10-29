"""
Quote-to-sale conversion vs. price gap to the lowest competitor.

Input: Texas_Competitor.xlsx (sheet "Pricing"), one row per part with
Qty Quoted, Qty Sold, Retail (UCC list price), Low List (lowest competitor list) and Category.

Model: binomial GLM (logistic regression on sold/quoted counts) with category
fixed effects; standard errors clustered by category.
Result on the 9/15/2026 file: coef on gap = -0.92 (95% CI -1.12 to -0.72), i.e.
each +10% above the lowest competitor -> odds ratio 0.91 (about 9% lower odds), n = 42,455 parts.
This is a cross-sectional association, not a causal estimate.
"""
import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm

path = sys.argv[1] if len(sys.argv) > 1 else "Texas_Competitor.xlsx"
c = pd.read_excel(path, sheet_name="Pricing")

q = c[(c["Low List"] > 0) & (c["Retail"] > 0) & (c["Qty Quoted"] > 0)].copy()
q["sold"] = np.minimum(q["Qty Sold"].clip(lower=0), q["Qty Quoted"])
q["gap"] = q["Retail"] / q["Low List"] - 1          # 0.10 = priced 10% above lowest competitor
q = q[(q["gap"] > -0.5) & (q["gap"] < 1)]            # drop extreme mismatches

# Conversion by price-gap bucket
q["bucket"] = pd.cut(q["gap"], [-0.5, -0.1, 0, 0.1, 0.25, 0.5, 1])
t = q.groupby("bucket", observed=True).agg(parts=("sold", "size"), sold=("sold", "sum"), quoted=("Qty Quoted", "sum"))
t["conversion"] = t["sold"] / t["quoted"]
print(t, "\n")

# Logistic regression on counts with category fixed effects
X = pd.get_dummies(q["Category"], drop_first=True, prefix="cat").astype(float)
X["gap"] = q["gap"].values
X = sm.add_constant(X)
y = np.column_stack([q["sold"], q["Qty Quoted"] - q["sold"]]).astype(float)
m = sm.GLM(y, X, family=sm.families.Binomial()).fit(
    cov_type="cluster", cov_kwds={"groups": pd.factorize(q["Category"])[0]})
b, (lo, hi) = m.params["gap"], m.conf_int().loc["gap"]
print(f"gap coefficient {b:.3f} (95% CI {lo:.3f} to {hi:.3f}), n = {len(q):,}")
print(f"odds ratio per +10% above lowest competitor: {np.exp(0.1*b):.3f} (95% CI {np.exp(0.1*lo):.3f}-{np.exp(0.1*hi):.3f})")
