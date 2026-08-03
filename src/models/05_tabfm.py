#!/usr/bin/env python
"""
Matched TabFM re-run for the bipolar-conversion baseline.

Purpose: reproduce the TabFM baseline under EXACTLY the same setting as every
other model in the paper, so the comparison is fold-for-fold matched:
  * 64 features (same matched feature matrix; NOT the earlier 80)
  * single fixed family-grouped 5-fold split (same folds; NOT 10x repeated CV)
  * out-of-fold pooled predictions -> one metric set per resampler variant
Outputs tabfm_matched_metrics.json in the SAME schema as the previous run.

The ONLY thing you may need to touch is predict_tabfm() below -- point it at the
exact TabFM model/call your working ORCD setup already uses.
"""
import json, numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

RNG = np.random.default_rng(42)
BOOT = 2000

# ------------------------------------------------------------------ TabFM call
def predict_tabfm(X_train, y_train, X_test):
    """Return P(class==1) for X_test, fit on (X_train, y_train).

    >>> ALIGN THIS with the exact TabFM you already run on ORCD. <<<
    The block below is the common tabpfn-style API; replace the model line with
    your working call if it differs. Everything else in this script is fixed.
    """
    from tabpfn import TabPFNClassifier            # <-- swap to your TabFM import if different
    clf = TabPFNClassifier()                        # <-- swap to your model construction
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)
    # locate the column for the positive class robustly
    classes = list(getattr(clf, "classes_", [0, 1]))
    pos = classes.index(1) if 1 in classes else 1
    return np.asarray(proba)[:, pos]

# ------------------------------------------------------------------ resamplers
def make_resampler(kind):
    if kind == "none":  return None
    if kind == "smote": return SMOTE(random_state=42)
    if kind == "ros":   return RandomOverSampler(random_state=42)
    if kind == "rus":   return RandomUnderSampler(random_state=42)
    raise ValueError(kind)

# ------------------------------------------------------------------ metrics
def wilson(k, n, z=1.96):
    if n == 0: return [float("nan"), float("nan")]
    p = k / n; d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return [round(c-h, 4), round(c+h, 4)]

def ece_score(y, p, bins=10):
    edges = np.linspace(0, 1, bins+1); e = 0.0; n = len(y)
    for i in range(bins):
        m = (p > edges[i]) & (p <= edges[i+1]) if i > 0 else (p >= edges[i]) & (p <= edges[i+1])
        if m.sum() == 0: continue
        e += (m.sum()/n) * abs(y[m].mean() - p[m].mean())
    return float(e)

def boot_ci(y, p, fn):
    vals = []
    idx = np.arange(len(y))
    for _ in range(BOOT):
        b = RNG.choice(idx, size=len(idx), replace=True)
        if len(np.unique(y[b])) < 2: continue
        try: vals.append(fn(y[b], p[b]))
        except Exception: pass
    if not vals: return [float("nan"), float("nan")]
    return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)]

def metric_block(y, p):
    y = np.asarray(y); p = np.asarray(p)
    auc = roc_auc_score(y, p)
    fpr, tpr, thr = roc_curve(y, p)
    j = np.argmax(tpr - fpr); t = thr[j]
    pred = (p >= t).astype(int)
    tp = int(((pred==1)&(y==1)).sum()); fn = int(((pred==0)&(y==1)).sum())
    tn = int(((pred==0)&(y==0)).sum()); fp = int(((pred==1)&(y==0)).sum())
    sens = tp/(tp+fn) if (tp+fn) else float("nan")
    spec = tn/(tn+fp) if (tn+fp) else float("nan")
    acc  = (tp+tn)/len(y)
    bacc = np.nanmean([sens, spec])
    return {
        "n": int(len(y)), "n_pos": int(y.sum()),
        "roc_auc": round(float(auc), 4),
        "roc_auc_95ci": boot_ci(y, p, roc_auc_score),
        "youden_threshold": round(float(t), 4),
        "sensitivity": round(float(sens), 4), "sensitivity_95ci": wilson(tp, tp+fn),
        "specificity": round(float(spec), 4), "specificity_95ci": wilson(tn, tn+fp),
        "accuracy": round(float(acc), 4), "accuracy_95ci": wilson(tp+tn, len(y)),
        "balanced_accuracy": round(float(bacc), 4),
        "brier": round(float(brier_score_loss(y, p)), 4),
        "brier_95ci": boot_ci(y, p, brier_score_loss),
        "ece": round(ece_score(y, p), 4),
        "ece_95ci": boot_ci(y, p, ece_score),
        "imbalance": round(float(y.mean()), 4),
    }

# ------------------------------------------------------------------ CV driver
def run_variant(X, y, folds, kind):
    """Out-of-fold pooled predictions under the SINGLE stored 5-fold split.
    Resampling (if any) is applied INSIDE each training fold only."""
    oof = np.full(len(y), np.nan)
    rs = make_resampler(kind)
    for f in folds:
        tr = np.array(f["train"]); te = np.array(f["test"])
        Xtr, ytr = X[tr], y[tr]
        if rs is not None:
            Xtr, ytr = rs.fit_resample(Xtr, ytr)
        oof[te] = predict_tabfm(Xtr, ytr, X[te])
    assert not np.isnan(oof).any(), "some test rows never predicted"
    return metric_block(y, oof), oof

def main():
    d = json.load(open("tabfm_matched_data.json"))
    X = np.asarray(d["X"], float); y = np.asarray(d["ybin"], int)
    folds = d["folds"]
    assert X.shape == (141, 64), X.shape
    print(f"n={X.shape[0]} feats={X.shape[1]} pos={int(y.sum())} folds={len(folds)}")

    binary = {}; oof_all = {}
    for kind in ["none", "smote", "ros", "rus"]:
        print(f"  variant: {kind}")
        binary[kind], oof = run_variant(X, y, folds, kind)
        oof_all[kind] = oof.tolist()
        print(f"    AUROC={binary[kind]['roc_auc']}  sens={binary[kind]['sensitivity']}  spec={binary[kind]['specificity']}")

    out = {
        "cohort_n": 141, "n_features": 64, "model": "tabfm",
        "cv": "single fixed StratifiedGroupKFold(k=5, seed=42) by famid  [MATCHED to other models]",
        "features": d["feat_names"],
        "binary": binary,
    }
    json.dump(out, open("tabfm_matched_metrics.json", "w"), indent=2)
    # per-case out-of-fold predictions (the 'none' variant is the reported baseline)
    import csv
    with open("tabfm_matched_percase.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["patient_idx", "label"] + [f"p_full_{k}" for k in oof_all])
        for i in range(len(y)):
            w.writerow([i, int(y[i])] + [round(oof_all[k][i], 6) for k in oof_all])
    print("wrote tabfm_matched_metrics.json + tabfm_matched_percase.csv")

if __name__ == "__main__":
    main()
