"""
01_build_cohort.py
Build the analytic cohort from the raw Excel workbook.

Inclusion: baseline full MDD (mdd_baseline==3) AND no baseline bipolar features
(bpdp_baseline==1). Outcome: bpdp_10years on a 3-level scale (1=None, 2=Subthreshold,
3=Full). Binary evaluation target: Full (3) vs None/Subthreshold (1,2).

Features: all *_baseline columns EXCEPT the target-adjacent bpdp_baseline and the
inclusion column mdd_baseline; *_4years / *_10years excluded to avoid temporal leakage.
Columns/subjects with >30% missing are dropped.

Output: data/cohort.parquet  (features + label3 + label_bin + famid)
"""
import pandas as pd, numpy as np, json, os

DATA_XLSX = os.environ.get("BIPOLAR_XLSX", "data/Bipolar_Data.xlsx")
SHEET = "Data from Boys & Girls Studies"

def main():
    df = pd.read_excel(DATA_XLSX, sheet_name=SHEET)
    # Inclusion
    inc = df[(df["mdd_baseline"] == 3) & (df["bpdp_baseline"] == 1)].reset_index(drop=True)
    # Outcome
    lab3 = inc["bpdp_10years"].astype(int).values          # 1/2/3
    lab_bin = (lab3 == 3).astype(int)                        # Full vs rest
    # Candidate baseline features
    drop_cols = {"bpdp_baseline", "mdd_baseline", "bpdp_10years"}
    base_cols = [c for c in inc.columns
                 if c.endswith("_baseline")
                 and not c.endswith("_4years") and not c.endswith("_10years")
                 and c not in drop_cols]
    # >30% missing filter applied to FEATURES only (drop high-missing / zero-variance
    # columns); no subject-level drop — the inclusion set (n=141) is retained in full,
    # with remaining missingness handled by in-fold median imputation downstream.
    keep = [c for c in base_cols
            if inc[c].isna().mean() <= 0.30 and inc[c].nunique(dropna=True) > 1]

    out = inc[keep].copy()
    out["famid"] = inc["famid"].values
    out["sex"] = inc["sex"].values if "sex" in inc else np.nan
    out["label3"] = lab3
    out["label_bin"] = lab_bin
    os.makedirs("data", exist_ok=True)
    out.to_csv("data/cohort.csv", index=False)
    json.dump({"features": keep}, open("data/features.json", "w"))
    print(f"cohort n={len(out)}  features={len(keep)}  "
          f"None={int((lab3==1).sum())} Sub={int((lab3==2).sum())} Full={int((lab3==3).sum())} "
          f"({lab_bin.mean():.1%} positive)")

if __name__ == "__main__":
    main()
