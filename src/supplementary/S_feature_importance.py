#!/usr/bin/env python
"""
S_feature_importance.py  (manuscript Figure 2)

Standardized L2 logistic-regression coefficients over the 64 baseline features,
with a FAMILY-CLUSTERED bootstrap (resample whole families, preserving sibling
structure) for 95% CIs and two-sided p-values. Matches the paper: 2,000 resamples.

Input :  results/ml_matrix_n141.npz   (keys: X, ybin, y3, feat_names, famid, fold_train, fold_test)
         data/feat_desc.json          (optional: {feature: human-readable description})
Output:  results/feature_importance.csv
         results/figures/Figure2_baseline_feature_importance.png
"""
import os, json
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from imblearn.combine import SMOTEENN

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
RES  = os.path.join(ROOT, "results"); FIG = os.path.join(RES, "figures")
os.makedirs(FIG, exist_ok=True)
B = 2000

d = np.load(os.path.join(RES, "ml_matrix_n141.npz"), allow_pickle=True)
Xall, ybin = d["X"], d["ybin"]
feat_names = list(d["feat_names"]); famid = d["famid"]
n, pfeat = Xall.shape
fp = os.path.join(ROOT, "data", "feat_desc.json")
feat_desc = json.load(open(fp)) if os.path.exists(fp) else {}

Xs = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(Xall))

def fit_beta(Xz, y, seed):
    try:
        Xr, yr = SMOTEENN(random_state=seed).fit_resample(Xz, y)
    except Exception:
        Xr, yr = Xz, y
    return LogisticRegression(penalty="l2", C=1.0, max_iter=5000).fit(Xr, yr).coef_[0]

beta_point = fit_beta(Xs, ybin, 42)
fams = np.unique(famid); rng = np.random.default_rng(42)
boot = np.zeros((B, pfeat))
for b in range(B):
    sf = rng.choice(fams, len(fams), replace=True)
    idx = np.concatenate([np.where(famid == f)[0] for f in sf])
    boot[b] = fit_beta(Xs[idx], ybin[idx], 1000 + b)
lo, hi = np.percentile(boot, 2.5, 0), np.percentile(boot, 97.5, 0)
p_boot = np.clip(2 * np.minimum((boot > 0).mean(0), (boot < 0).mean(0)), 1 / B, 1)

fi = pd.DataFrame({"feature": feat_names,
                   "description": [feat_desc.get(f, f) for f in feat_names],
                   "beta_std": beta_point, "beta_lo": lo, "beta_hi": hi,
                   "p_bootstrap": p_boot, "OR": np.exp(beta_point), "sig": p_boot < 0.05})
fi = fi.sort_values("beta_std", key=lambda s: s.abs(), ascending=False)
fi.to_csv(os.path.join(RES, "feature_importance.csv"), index=False)

top = fi.head(12).iloc[::-1].reset_index(drop=True)
fig, ax = plt.subplots(figsize=(7.6, 5.2))
for i, r in top.iterrows():
    c = "#1b6ca8" if r.beta_std > 0 else "#c0504d"
    ax.plot([r.beta_lo, r.beta_hi], [i, i], color=c, lw=2)
    ax.plot(r.beta_std, i, "o", color=c, ms=6)
    if r.sig:
        ax.text(max(r.beta_hi, 0) + 0.03, i, "*", va="center", fontsize=13)
ax.axvline(0, ls="--", color="#888888", lw=1)
ax.set_yticks(range(len(top))); ax.set_yticklabels(top["description"], fontsize=8)
ax.set_xlabel("Standardized logistic-regression coefficient (\u03b2)  \u00b7  \u2192 higher bipolar-conversion risk")
ax.set_title(f"Direction and strength of baseline predictors (n={n})\n"
             f"\u03b2 with {B}\u00d7 family-clustered bootstrap 95% CI; * p<0.05", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "Figure2_baseline_feature_importance.png"), dpi=300, bbox_inches="tight")
print("wrote feature_importance.csv and Figure2 (significant:",
      list(fi[fi.sig].feature), ")")
