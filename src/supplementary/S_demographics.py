#!/usr/bin/env python
"""
S_demographics.py  (manuscript Table 1)

Baseline demographics of the analytic cohort (n=141), overall and by 10-year
conversion status, from the raw workbook.

Input :  data/Bipolar_Data.xlsx   (or set BIPOLAR_XLSX); sheet "Data from Boys & Girls Studies"
Output:  results/cohort_demographics.csv
"""
import os, numpy as np, pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
XLSX = os.environ.get("BIPOLAR_XLSX", os.path.join(ROOT, "data", "Bipolar_Data.xlsx"))
raw = pd.read_excel(XLSX, sheet_name="Data from Boys & Girls Studies")
inc = raw[(raw["mdd_baseline"] == 3) & (raw["bpdp_baseline"] == 1)].copy()
assert len(inc) == 141, f"expected 141, got {len(inc)}"

age = pd.to_numeric(inc["age_baseline"], errors="coerce")
sex = pd.to_numeric(inc["sex"], errors="coerce")           # 1=male, 2=female
out = pd.to_numeric(inc["bpdp_10years"], errors="coerce")  # 1=None,2=Sub,3=Full
conv = out == 3

def block(mask, label):
    a, s = age[mask], sex[mask]; nn = int(mask.sum())
    return dict(group=label, n=nn,
                age_mean=round(a.mean(), 1), age_sd=round(a.std(ddof=1), 1),
                age_min=int(a.min()), age_max=int(a.max()),
                female=int((s == 2).sum()), male=int((s == 1).sum()),
                pct_female=round(100*(s == 2).mean(), 1))

rows = [block(pd.Series(True, index=inc.index), "All"),
        block(conv, "Full converters"), block(~conv, "Non-converters")]
demo = pd.DataFrame(rows)
demo.to_csv(os.path.join(ROOT, "results", "cohort_demographics.csv"), index=False)

t, pa = stats.ttest_ind(age[conv].dropna(), age[~conv].dropna())
chi2, ps, _, _ = stats.chi2_contingency(pd.crosstab(conv, sex))
print(demo.to_string(index=False))
print(f"\nAge converters vs non: t={t:.2f}, p={pa:.3f}")
print(f"Sex vs converter status: chi2={chi2:.2f}, p={ps:.3f}")
print(f"Families: {inc['famid'].nunique()}")
