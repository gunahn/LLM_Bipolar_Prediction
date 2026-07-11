"""
04_analyze_and_figure.py
Aggregate few-shot results across seeds, build the k-sweep table + operating points,
and render the 3-panel main figure (AUROC vs k; ROC; PR).

Expects results/fewshot_<model>_all_s<seed>.json for each seed and
results/baselines.csv + results/ml_kshot_curve.csv from 02.
"""
import pandas as pd, numpy as np, json, glob, os
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             roc_curve, precision_recall_curve, confusion_matrix)
import matplotlib.pyplot as plt

K_VALUES = [0, 2, 4, 8, 16, 32, 64]
C_SON, C_HAI, C_ML, C_SUP = "#C44E52", "#DD8452", "#4C72B0", "#55A868"

def load(tag, mode="all"):
    seeds = {}
    for fn in sorted(glob.glob(f"results/fewshot_{tag}_{mode}_s*.json")):
        s = int(fn.split("_s")[-1].split(".")[0]); seeds[s] = json.load(open(fn))
    return seeds

def agg(seeds, metric="auroc"):
    out = {}
    Ks = sorted({r["K"] for recs in seeds.values() for r in recs})
    for K in Ks:
        vals = []
        for recs in seeds.values():
            rr = [r for r in recs if r["K"] == K and r["p_full"] is not None]
            if not rr: continue
            y = np.array([r["label_bin"] for r in rr]); p = np.array([r["p_full"] for r in rr])
            vals.append(roc_auc_score(y, p) if metric == "auroc" else average_precision_score(y, p))
        if vals: out[K] = (np.mean(vals), np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
    return out

def pooled(seeds, K):
    ys, ps = [], []
    for recs in seeds.values():
        for r in recs:
            if r["K"] == K and r["p_full"] is not None:
                ys.append(r["label_bin"]); ps.append(r["p_full"])
    return np.array(ys), np.array(ps)

def youden(y, p):
    best, bt = -1, 0.5
    for t in np.linspace(0.05, 0.8, 76):
        pred = (p >= t).astype(int); tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
        j = (tp/(tp+fn) if tp+fn else 0) + (tn/(tn+fp) if tn+fp else 0) - 1
        if j > best: best, bt = j, t
    pred = (p >= bt).astype(int); tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
    return round(bt,2), round(tp/(tp+fn),2), round(tn/(tn+fp),2)

def main():
    son = load("sonnet"); hai = load("haiku")
    son_auc, son_ap = agg(son), agg(son, "auprc")
    hai_auc = agg(hai)
    base = pd.read_csv("results/baselines.csv"); best_sup = base["AUROC"].max()
    mlc = pd.read_csv("results/ml_kshot_curve.csv").set_index("k")

    # k-sweep table
    rows = []
    for K in K_VALUES:
        r = {"k": K}
        if K in son_auc:
            se_t, se, sp = youden(*pooled(son, K))
            r["Sonnet_AUROC"] = f"{son_auc[K][0]:.3f}±{son_auc[K][1]:.3f}"
            r["Sonnet_AUPRC"] = f"{son_ap[K][0]:.3f}"
            r["Sonnet_SensSpec"] = f"{se:.2f}/{sp:.2f}"
        if K in hai_auc:
            _, hse, hsp = youden(*pooled(hai, K))
            r["Haiku_AUROC"] = f"{hai_auc[K][0]:.3f}±{hai_auc[K][1]:.3f}"
            r["Haiku_SensSpec"] = f"{hse:.2f}/{hsp:.2f}"
        rows.append(r)
    pd.DataFrame(rows).to_csv("results/fewshot_ksweep.csv", index=False)

    # figure
    try:
        from figure_style import apply_figure_style, panel_letter, META_GREY
        apply_figure_style()
    except Exception:
        panel_letter = lambda ax, l: ax.text(-0.1, 1.05, l, transform=ax.transAxes,
                                              fontweight="bold", fontsize=14)
        META_GREY = "#888888"
    fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.2))
    xs = list(range(len(K_VALUES)))
    son_m = [son_auc[k][0] for k in K_VALUES]; son_s = [son_auc[k][1] for k in K_VALUES]
    hK = [k for k in K_VALUES if k in hai_auc]
    hai_m = [hai_auc[k][0] for k in hK]; hai_s = [hai_auc[k][1] for k in hK]
    ml_m = [mlc.loc[k, "AUROC_mean"] for k in K_VALUES]; ml_s = [mlc.loc[k, "AUROC_sd"] for k in K_VALUES]
    ax[0].errorbar(xs, son_m, yerr=son_s, fmt="-o", color=C_SON, lw=2.4, ms=7, capsize=3, label="Claude Sonnet (few-shot)")
    ax[0].errorbar([K_VALUES.index(k) for k in hK], hai_m, yerr=hai_s, fmt="-o", color=C_HAI, lw=2.4, ms=7, capsize=3, label="Claude Haiku (few-shot)")
    ax[0].errorbar(xs, ml_m, yerr=ml_s, fmt="-s", color=C_ML, lw=2.2, ms=6, capsize=3, label="Supervised ML, k-shot (LogReg)")
    ax[0].axhline(best_sup, ls="--", color=C_SUP, lw=2, label=f"Best supervised ML, full data: {best_sup:.2f}")
    ax[0].axhline(0.5, color=META_GREY, lw=1, alpha=0.6)
    ax[0].set_xticks(xs); ax[0].set_xticklabels(K_VALUES)
    ax[0].set_xlabel("k  (labeled examples)"); ax[0].set_ylabel("AUROC")
    ax[0].set_title("Few-shot LLMs need far fewer examples than supervised ML")
    ax[0].set_ylim(0.45, 0.80); ax[0].legend(loc="lower right", fontsize=7); panel_letter(ax[0], "a")

    y0, p0 = pooled(son, 0); y2, p2 = pooled(son, 2)
    for a, curvef, xl, yl, ttl, letter in [
        (ax[1], roc_curve, "1 − Specificity", "Sensitivity", "ROC: sensitivity–specificity trade-off", "b"),
        (ax[2], precision_recall_curve, "Recall (sensitivity)", "Precision (PPV)", "Precision–Recall", "c")]:
        for y, p, c, lab in [(y0, p0, "#7E2F32", "Sonnet zero-shot"), (y2, p2, C_SON, "Sonnet k=2")]:
            if a is ax[1]:
                fpr, tpr, _ = roc_curve(y, p); a.plot(fpr, tpr, color=c, lw=2.2, label=lab)
            else:
                pr, rc, _ = precision_recall_curve(y, p); a.plot(rc, pr, color=c, lw=2.2, label=lab)
        if a is ax[1]: a.plot([0,1],[0,1], ls=":", color=META_GREY, lw=1)
        else: a.axhline(y0.mean(), ls=":", color=META_GREY, lw=1.2, label=f"Prevalence ({y0.mean():.2f})"); a.set_ylim(0,1.02)
        a.set_xlabel(xl); a.set_ylabel(yl); a.set_title(ttl); a.legend(fontsize=7.5); panel_letter(a, letter)
    fig.tight_layout()
    fig.savefig("results/fig1_main.png", dpi=200, bbox_inches="tight")
    print("wrote results/fewshot_ksweep.csv and results/fig1_main.png")

if __name__ == "__main__":
    main()
