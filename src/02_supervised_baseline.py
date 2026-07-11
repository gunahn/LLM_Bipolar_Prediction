"""
02_supervised_baseline.py
Supervised ML baselines for Full-vs-rest under family-grouped 5-fold CV.

Models: Logistic Regression, Random Forest, XGBoost (all full-data, all features),
plus TabNet and TabPFN if their packages are installed. Also computes the k-shot
supervised curve (LogReg trained on prevalence-sampled k-shot subsets, k=0..64).

Metrics pooled across held-out folds. Writes results/baselines.csv and
results/ml_kshot_curve.csv.
"""
import pandas as pd, numpy as np, json, math, random, os
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

SEED = 42
def load():
    df = pd.read_csv("data/cohort.csv")
    feats = json.load(open("data/features.json"))["features"]
    X = df[feats].values
    y = df["label_bin"].values.astype(int)
    lab3 = df["label3"].values.astype(int)
    groups = df["famid"].values
    return df, feats, X, y, lab3, groups

def make_folds(y3, groups):
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    return [(tr, te) for tr, te in sgkf.split(np.zeros(len(y3)), y3, groups)]

def oof(fit_predict, X, y, lab3, folds):
    p = np.full(len(y), np.nan)
    for tr, te in folds:
        imp = SimpleImputer(strategy="median").fit(X[tr])
        p[te] = fit_predict(imp.transform(X[tr]), (lab3[tr] == 3).astype(int), imp.transform(X[te]))
    return p

def lr_fp(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=5000, class_weight="balanced").fit(sc.transform(Xtr), ytr)
    return clf.predict_proba(sc.transform(Xte))[:, 1]

def rf_fp(Xtr, ytr, Xte):
    clf = RandomForestClassifier(n_estimators=400, class_weight="balanced",
                                 random_state=SEED).fit(Xtr, ytr)
    return clf.predict_proba(Xte)[:, 1]

def xgb_fp(Xtr, ytr, Xte):
    from xgboost import XGBClassifier
    spw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
    clf = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
                        eval_metric="logloss", random_state=SEED).fit(Xtr, ytr)
    return clf.predict_proba(Xte)[:, 1]

def kshot_curve(X, lab3, folds, prev, Kvals=(0,2,4,8,16,32,64), n_seeds=5):
    def prevalence_shots(train_idx, K, seed):
        if K == 0: return []
        rng = random.Random(1000 + seed); by = {1:[],2:[],3:[]}
        for i in train_idx: by[int(lab3[i])].append(i)
        for c in by: rng.shuffle(by[c])
        raw = {c: K*prev[c] for c in [1,2,3]}
        base = {c: int(math.floor(raw[c])) for c in [1,2,3]}; rem = K - sum(base.values())
        for c in sorted([1,2,3], key=lambda c: raw[c]-base[c], reverse=True)[:rem]: base[c]+=1
        picks = []
        for c in [1,2,3]: picks += by[c][:min(base[c], len(by[c]))]
        rng.shuffle(picks); return picks
    rows = []
    for K in Kvals:
        vals = []
        for s in range(n_seeds):
            ys, ps = [], []
            for tr, te in folds:
                shots = prevalence_shots(tr, K, s)
                ytr = np.array([1 if lab3[i]==3 else 0 for i in shots])
                if K == 0 or len(set(ytr)) < 2:
                    pr = prev[3]
                    ys += [1 if lab3[i]==3 else 0 for i in te]; ps += [pr]*len(te)
                    continue
                imp = SimpleImputer(strategy="median").fit(X[shots])
                sc = StandardScaler().fit(imp.transform(X[shots]))
                clf = LogisticRegression(max_iter=5000, class_weight="balanced").fit(
                    sc.transform(imp.transform(X[shots])), ytr)
                pp = clf.predict_proba(sc.transform(imp.transform(X[te])))[:, 1]
                ys += [1 if lab3[i]==3 else 0 for i in te]; ps += list(pp)
            vals.append(roc_auc_score(ys, ps))
        rows.append({"k": K, "AUROC_mean": np.mean(vals), "AUROC_sd": np.std(vals, ddof=1)})
    return pd.DataFrame(rows)

def main():
    df, feats, X, y, lab3, groups = load()
    folds = make_folds(lab3, groups)
    prev = {c: (lab3 == c).mean() for c in [1,2,3]}
    res = {}
    for name, fp in [("Logistic Regression", lr_fp), ("Random Forest", rf_fp), ("XGBoost", xgb_fp)]:
        try:
            p = oof(fp, X, y, lab3, folds)
            res[name] = (roc_auc_score(y, p), average_precision_score(y, p))
        except Exception as e:
            print(f"skip {name}: {e}")
    # Optional TabNet / TabPFN (only if installed)
    try:
        from pytorch_tabnet.tab_model import TabNetClassifier
        def tabnet_fp(Xtr, ytr, Xte):
            sc = StandardScaler().fit(Xtr)
            w = {0: 1.0, 1: float((ytr==0).sum()/max((ytr==1).sum(),1))}
            clf = TabNetClassifier(n_d=8, n_a=8, n_steps=3, seed=SEED, verbose=0)
            clf.fit(sc.transform(Xtr).astype("float32"), ytr, max_epochs=120,
                    batch_size=64, weights=w)
            return clf.predict_proba(sc.transform(Xte).astype("float32"))[:, 1]
        p = oof(tabnet_fp, X, y, lab3, folds)
        res["TabNet"] = (roc_auc_score(y, p), average_precision_score(y, p))
    except Exception as e:
        print(f"skip TabNet: {e}")
    out = pd.DataFrame([{"Model": k, "AUROC": round(v[0], 3), "AUPRC": round(v[1], 3)}
                        for k, v in res.items()]).sort_values("AUROC", ascending=False)
    os.makedirs("results", exist_ok=True)
    out.to_csv("results/baselines.csv", index=False)
    kshot_curve(X, lab3, folds, prev).to_csv("results/ml_kshot_curve.csv", index=False)
    print(out.to_string(index=False))

if __name__ == "__main__":
    main()
