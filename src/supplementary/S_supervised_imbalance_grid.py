#!/usr/bin/env python
"""
S_supervised_imbalance_grid.py  (manuscript Supplementary S11, Table S4)

Full grid of four supervised classifiers x three imbalance-resampling methods,
under the SAME fixed family-grouped 5-fold split, resampling applied strictly
within each training fold. Out-of-fold probabilities pooled for evaluation.

Input :  results/ml_matrix_n141.npz  (X, ybin, feat_names, fold_train, fold_test)
Output:  results/supervised_imbalance_grid.csv   (AUROC / sens / spec / Brier / ECE per cell)
"""
import os, numpy as np, pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss, confusion_matrix
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTEENN
from imblearn.ensemble import BalancedRandomForestClassifier
import xgboost as xgb

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "results")
d = np.load(os.path.join(RES, "ml_matrix_n141.npz"), allow_pickle=True)
X, y = d["X"], d["ybin"]
folds = list(zip(d["fold_train"], d["fold_test"]))

def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1])
        if m.sum(): e += m.mean() * abs(y[m].mean() - p[m].mean())
    return e

RESAMPLERS = {"smote": SMOTE, "smote_enn": SMOTEENN, "adasyn": ADASYN}
def make_clf(name):
    if name == "LR_L1": return LogisticRegression(penalty="l1", solver="liblinear", C=1.0, max_iter=5000)
    if name == "LR_L2": return LogisticRegression(penalty="l2", C=1.0, max_iter=5000)
    if name == "XGBoost": return xgb.XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.1,
                                                   eval_metric="logloss", verbosity=0)
    if name == "BalancedRF": return BalancedRandomForestClassifier(n_estimators=400, random_state=0)

rows = []
for clf_name in ["LR_L1", "LR_L2", "XGBoost", "BalancedRF"]:
    for rs_name, RS in RESAMPLERS.items():
        oof = np.full(len(y), np.nan)
        for tr, te in folds:
            imp = SimpleImputer(strategy="median").fit(X[tr]); sc = StandardScaler().fit(imp.transform(X[tr]))
            Xtr, Xte = sc.transform(imp.transform(X[tr])), sc.transform(imp.transform(X[te]))
            try:
                Xr, yr = RS(random_state=0).fit_resample(Xtr, y[tr])
            except Exception:
                Xr, yr = Xtr, y[tr]
            m = make_clf(clf_name).fit(Xr, yr)
            oof[te] = m.predict_proba(Xte)[:, list(getattr(m, "classes_", [0, 1])).index(1)]
        auroc = roc_auc_score(y, oof)
        fpr, tpr, thr = roc_curve(y, oof); j = np.argmax(tpr - fpr); t = thr[j]
        tn, fp, fn, tp = confusion_matrix(y, (oof >= t).astype(int), labels=[0, 1]).ravel()
        rows.append(dict(model=clf_name, resampler=rs_name, AUROC=round(auroc, 3),
                         Sensitivity=round(tp/(tp+fn), 3), Specificity=round(tn/(tn+fp), 3),
                         Brier=round(brier_score_loss(y, oof), 3), ECE=round(ece(y, oof), 3)))
tab = pd.DataFrame(rows).sort_values("AUROC", ascending=False)
tab.to_csv(os.path.join(RES, "supervised_imbalance_grid.csv"), index=False)
print(tab.to_string(index=False))
print(f"\nAUROC range: {tab.AUROC.min():.2f} to {tab.AUROC.max():.2f}")
