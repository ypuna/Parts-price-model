"""
Build the anonymized modeling file from the private pricing export.

Private input (never committed): the competitor pricing workbook (sheet "Pricing").
Output: data/parts.csv with
  - no internal item, supplier or OEM numbers
  - competitor names replaced with comp_a..comp_d (real names never appear in the code)
  - every dollar value multiplied by one undisclosed constant (PRICE_SCALE env var),
    so all results are reported as percentages, which the scaling does not change
  - no sales or quote volumes
"""
import os
import sys
import numpy as np
import pandas as pd

MAKES = {  # item-number prefix -> make, used only to split make from model text
    "AC": "ACURA", "AU": "AUDI", "BM": "BMW", "CH": "CHRYSLER GROUP", "FO": "FORD", "GM": "GM",
    "HO": "HONDA", "HY": "HYUNDAI", "IN": "INFINITI", "KI": "KIA", "LX": "LEXUS", "MA": "MAZDA",
    "MB": "MERCEDES-BENZ", "MI": "MITSUBISHI", "NI": "NISSAN", "SU": "SUBARU", "TO": "TOYOTA",
    "VW": "VOLKSWAGEN", "VO": "VOLVO", "MC": "MINI", "SC": "SCION", "SZ": "SUZUKI", "FI": "FIAT",
    "IZ": "ISUZU", "RO": "LAND ROVER", "JA": "JAGUAR", "PO": "PORSCHE",
}
# Real competitor column names are passed in privately, in a fixed order, e.g.
#   COMPETITOR_COLUMNS="name1,name2,name3,name4" python src/prepare_data.py export.xlsx
ALIASES = ["comp_a", "comp_b", "comp_c", "comp_d"]


def main(src, dst="data/parts.csv"):
    scale = float(os.environ.get("PRICE_SCALE", "1.0"))
    raw = pd.read_excel(src, sheet_name="Pricing")
    raw = raw[raw["Avg Cost"] > 0].copy()

    out = pd.DataFrame({
        "description": raw["Description"].astype(str).str.strip(),
        "category": raw["Category"].astype(str),
        "make_code": raw["Item Number"].astype(str).str[:2].map(MAKES).fillna("OTHER"),
        "certified": (raw["Item Number"].astype(str).str.endswith("-C")
                 | raw["Description"].astype(str).str.contains(r"\(CAPA\)", regex=True)).astype(int),
        "unit_cost": (raw["Avg Cost"] * scale).round(2),
    })
    names = os.environ["COMPETITOR_COLUMNS"].split(",")
    for name, alias in zip(names, ALIASES):
        out[alias] = (raw[name.strip()] * scale).round(2)
    comp_cols = ALIASES
    out["n_competitors"] = out[comp_cols].notna().sum(axis=1)
    out["lowest_competitor"] = out[comp_cols].min(axis=1)
    out["current_list"] = (raw["Retail"] * scale).round(2)
    out = out.sample(frac=1.0, random_state=7).reset_index(drop=True)   # break any ordering by item number
    out.to_csv(dst, index=False)
    print(f"wrote {dst}: {len(out):,} parts, {int((out.n_competitors > 0).sum()):,} with a competitor price")


if __name__ == "__main__":
    main(*sys.argv[1:])
