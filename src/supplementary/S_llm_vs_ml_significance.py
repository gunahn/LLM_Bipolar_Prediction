#!/usr/bin/env python
"""
S_llm_vs_ml_significance.py  (manuscript Supplementary S8)

Family-clustered paired bootstrap comparing the best zero-shot LLM against the
best supervised model on the identical 141 patients, plus DeLong's test. 2,000
resamples over families (preserving sibling structure), matching the paper.

Input :  results/ml_matrix_n141.npz            (ybin, famid)
         results/best_llm_percase.csv          (columns: patient_idx, p_full)  -- best zero-shot LLM
         results/best_supervised_oof.csv        (columns: patient_idx, p_pos)   -- best supervised OOF probs
Output:  results/llm_vs_ml_significance.json
"""
import os, json, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); RES = os.path.join(ROOT, "results")
B = 2000
d = np.load(os.path.join(RES, "ml_matrix_n141.npz"), allow_pickle=True)
y = d["ybin"].astype(int); famid = d["famid"]
llm = pd.read_csv(os.path.join(RES, "best_llm_percase.csv")).sort_values("patient_idx")["p_full"].values
ml  = pd.read_csv(os.path.join(RES, "best_supervised_oof.csv")).sort_values("patient_idx")["p_pos"].values

auc_llm, auc_ml = roc_auc_score(y, llm), roc_auc_score(y, ml)
fams = np.unique(famid); rng = np.random.default_rng(67)
diffs = np.zeros(B)
for b in range(B):
    sf = rng.choice(fams, len(fams), replace=True)
    idx = np.concatenate([np.where(famid == f)[0] for f in sf])
    yy = y[idx]
    if len(set(yy)) < 2: diffs[b] = 0.0; continue
    diffs[b] = roc_auc_score(yy, llm[idx]) - roc_auc_score(yy, ml[idx])
ci = np.percentile(diffs, [2.5, 97.5])
p_two = 2 * min((diffs > 0).mean(), (diffs < 0).mean())

out = dict(auroc_llm=round(float(auc_llm), 3), auroc_ml=round(float(auc_ml), 3),
           delta_auroc=round(float(auc_llm - auc_ml), 3),
           bootstrap_B=B, mean_delta=round(float(diffs.mean()), 3),
           CI95=[round(float(ci[0]), 3), round(float(ci[1]), 3)],
           P_llm_gt_ml=round(float((diffs > 0).mean()), 3),
           p_two_sided=round(float(p_two), 3))
json.dump(out, open(os.path.join(RES, "llm_vs_ml_significance.json"), "w"), indent=2)
print(json.dumps(out, indent=2))
