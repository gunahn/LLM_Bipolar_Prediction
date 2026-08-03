#!/usr/bin/env python
"""
S_clinician_benchmark.py  (manuscript Supplementary S10, Figure S2)

Human expert benchmark. An experienced academic child psychiatrist predicted
10-year conversion for 24 patients, each summarized by the top-10 baseline
features from the supervised L2 logistic-regression model, in three successive
rounds of eight (0-shot, then 8-shot, then 16-shot, with true outcomes revealed
between rounds).

Input :  data/clinician_baseline_predictions.csv
             columns: round, shots, case_id, actual_converter,
                      confidence_pct, p_converter, predicted_converter, correct
Output:  results/clinician_baseline_metrics.csv        (per-round + pooled)
         results/figures/FigureS2_clinician_baseline.png
"""
import os, numpy as np, pandas as pd, math
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, brier_score_loss

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
RES = os.path.join(ROOT, "results"); FIG = os.path.join(RES, "figures"); os.makedirs(FIG, exist_ok=True)
pr = pd.read_csv(os.path.join(ROOT, "data", "clinician_baseline_predictions.csv"))

def metrics(df):
    y = df.actual_converter.values.astype(int)
    p = df.p_converter.values.astype(float)
    pred = df.predicted_converter.values.astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    auroc = roc_auc_score(y, p) if len(set(y)) > 1 else float("nan")
    return dict(n=len(df), TP=tp, FN=fn, FP=fp, TN=tn, AUROC=round(auroc, 3),
                Accuracy=round((tp + tn) / len(df), 3),
                Sensitivity=round(tp / (tp + fn), 3) if (tp + fn) else float("nan"),
                Specificity=round(tn / (tn + fp), 3) if (tn + fp) else float("nan"),
                Brier=round(brier_score_loss(y, p), 3))

rows = [dict(level="Pooled", **metrics(pr))]
for s in sorted(pr.shots.unique()):
    rows.append(dict(level=f"{s}-shot", **metrics(pr[pr.shots == s])))
out = pd.DataFrame(rows)
out.to_csv(os.path.join(RES, "clinician_baseline_metrics.csv"), index=False)
print(out.to_string(index=False))

# Figure S2: (a) pooled confusion matrix  (b) full pooled ROC + native (thr 0.5) point
y = pr.actual_converter.values.astype(int); p = pr.p_converter.values.astype(float)
pred = pr.predicted_converter.values.astype(int)
tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
cm = np.array([[tp, fn], [fp, tn]])
fpr, tpr, _ = roc_curve(y, p); auroc = roc_auc_score(y, p)
op_sens, op_spec = tp / (tp + fn), tn / (tn + fp)
auroc_lbl = f"{math.floor(auroc*100)/100:.2f}"          # truncate: 0.745 -> 0.74

fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0))
ax = axes[0]; ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max())
for (r, c), v in np.ndenumerate(cm):
    ax.text(c, r, int(v), ha="center", va="center", fontsize=15,
            color="white" if v > cm.max()*0.55 else "#12263a", fontweight="bold")
ax.set_xticks([0, 1]); ax.set_xticklabels(["Converter", "Non-conv."])
ax.set_yticks([0, 1]); ax.set_yticklabels(["Converter", "Non-conv."])
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_title(f"Clinician pooled decisions (n={len(y)}, {tp+fn} converters)")
ax.text(-0.20, 1.02, "a", transform=ax.transAxes, fontweight="bold", fontsize=10)
ax = axes[1]
ax.plot(fpr, tpr, color="#c0392b", lw=2.3, label=f"Clinician ROC (AUROC {auroc_lbl})")
ax.plot([0, 1], [0, 1], ls=":", color="#888888", lw=1)
ax.scatter([1-op_spec], [op_sens], s=95, color="#12263a", zorder=6, edgecolor="white", linewidth=0.9,
           label=f"Native decision (thr 0.5):\nsens {op_sens:.2f}, spec {op_spec:.2f}")
ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
ax.set_xlabel("1 \u2212 Specificity"); ax.set_ylabel("Sensitivity")
ax.set_title("Clinician pooled ROC and operating point")
ax.legend(loc="lower right", fontsize=6.8)
ax.text(-0.18, 1.02, "b", transform=ax.transAxes, fontweight="bold", fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "FigureS2_clinician_baseline.png"), dpi=300, bbox_inches="tight")
print("wrote FigureS2 (pooled AUROC shown as", auroc_lbl + ")")
