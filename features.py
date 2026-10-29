"""Turn a catalog description into model features.

Descriptions follow a fixed pattern, e.g.
  'TOYOTA CAMRY 18-23 - LT FRONT BUMPER COVER RETAINER (CAPA)'
  -> make/model, model years 2018-2023, side = LT, part text, option text.
"""
import re
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder

PAT = re.compile(r"^(?P<vehicle>.+?)\s+(?P<y1>\d{2})-(?P<y2>\d{2})\s*-\s*(?P<rest>.*)$")
SIDE = re.compile(r"^(LT|RT|LH|RH)\b\s*", re.I)


def _year(two_digit):
    y = int(two_digit)
    return 2000 + y if y <= 30 else 1900 + y


def parse(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for d in df["description"]:
        m = PAT.match(d)
        if not m:
            rows.append(dict(vehicle="UNKNOWN", y1=np.nan, y2=np.nan, side="NONE", part=d.lower(), options=""))
            continue
        rest = m["rest"]
        side = "NONE"
        s = SIDE.match(rest)
        if s:
            side = {"LH": "LT", "RH": "RT"}.get(s.group(1).upper(), s.group(1).upper())
            rest = rest[s.end():]
        part, _, options = rest.partition("(")
        rows.append(dict(vehicle=m["vehicle"].upper(), y1=_year(m["y1"]), y2=_year(m["y2"]),
                         side=side, part=part.strip().lower(), options=options.lower()))
    p = pd.DataFrame(rows, index=df.index)
    p["n_years"] = p["y2"] - p["y1"] + 1
    p["years_since_launch"] = 2026 - p["y1"]
    p["front"] = p["part"].str.contains(r"\bfront\b").astype(int)
    p["rear"] = p["part"].str.contains(r"\brear\b").astype(int)
    # parts that differ only by side or certification share a family; CV keeps a family together
    p["family"] = (df["make_code"] + "|" + p["vehicle"] + "|" + p["y1"].astype(str) + "|"
                   + p["part"].str.replace(r"\bcapa\b", "", regex=True).str.strip())
    p["family"] = p["family"].fillna("UNPARSED|" + df["description"])
    return p


class FeatureBuilder:
    def __init__(self):
        self.part_vec = TfidfVectorizer(ngram_range=(1, 2), min_df=5, sublinear_tf=True)
        self.opt_vec = TfidfVectorizer(ngram_range=(1, 1), min_df=10, sublinear_tf=True)
        self.onehot = OneHotEncoder(handle_unknown="ignore", min_frequency=5)

    def _num(self, df, p):
        return sp.csr_matrix(np.column_stack([
            np.log(df["unit_cost"].values), df["certified"].values,
            p["y1"].fillna(2015).values, p["y2"].fillna(2015).values,
            p["n_years"].fillna(1).values, p["years_since_launch"].fillna(10).values,
            p["front"].values, p["rear"].values,
        ]).astype(float))

    def fit_transform(self, df):
        p = parse(df)
        cats = pd.concat([df[["make_code", "category"]], p[["vehicle", "side"]]], axis=1).astype(str)
        return sp.hstack([self._num(df, p), self.onehot.fit_transform(cats),
                          self.part_vec.fit_transform(p["part"]),
                          self.opt_vec.fit_transform(p["options"])]).tocsr()

    def transform(self, df):
        p = parse(df)
        cats = pd.concat([df[["make_code", "category"]], p[["vehicle", "side"]]], axis=1).astype(str)
        return sp.hstack([self._num(df, p), self.onehot.transform(cats),
                          self.part_vec.transform(p["part"]),
                          self.opt_vec.transform(p["options"])]).tocsr()
