"""Make a small synthetic file with the same columns as data/parts.csv so the code runs
without the private data. Descriptions are real catalog text (public fitment wording);
every cost and price is simulated, so nothing in it reflects real costs or margins."""
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
src = pd.read_csv("data/parts.csv").sample(1500, random_state=1).reset_index(drop=True)
cat_markup = {c: rng.uniform(2.2, 4.0) for c in src["category"].unique()}
cost = np.round(np.exp(rng.normal(3.4, 0.9, len(src))), 2)
market = cost * src["category"].map(cat_markup).values * np.exp(rng.normal(0, 0.25, len(src)))
out = src[["description", "category", "make_code", "certified"]].copy()
out["unit_cost"] = cost
for c in ["comp_a", "comp_b", "comp_c", "comp_d"]:
    listed = rng.random(len(src)) < 0.6
    out[c] = np.where(listed, np.round(market * np.exp(rng.normal(0, 0.12, len(src))), 0), np.nan)
out.loc[rng.random(len(src)) < 0.15, ["comp_a", "comp_b", "comp_c", "comp_d"]] = np.nan
out["n_competitors"] = out[["comp_a", "comp_b", "comp_c", "comp_d"]].notna().sum(axis=1)
out["lowest_competitor"] = out[["comp_a", "comp_b", "comp_c", "comp_d"]].min(axis=1)
out["current_list"] = np.round(market * np.exp(rng.normal(0.05, 0.2, len(src))), 0)
out.to_csv("data/sample_parts.csv", index=False)
print(f"wrote data/sample_parts.csv: {len(out)} synthetic rows")
