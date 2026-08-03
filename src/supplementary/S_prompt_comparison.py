#!/usr/bin/env python
"""
S_prompt_comparison.py  (manuscript main Figure 3, Supplementary S9)

Minimal vs clinician-informed zero-shot prompt on the SAME model
(claude-sonnet-5), same narratives, same 141 patients, 5 seeds, temperature 1.0.
Computes per-seed and seed-averaged AUROC, the 5-seed paired t-test (the headline
significance test), and each prompt's native categorical decision. Builds the
3-panel Figure 3.

The two prompt conditions are produced by src/03_fewshot_llm.py at K=0:
  * clinician-informed = primary prevalence-free system prompt + 3-class tool
  * minimal            = no system prompt + Yes/No tool  (run with --prompt minimal)

Input :  results/promptAB_percase.csv
             columns: prompt(minimal|clinician), seed, patient_idx, label_full,
                      p_full, native_pred(0/1)
Output:  results/prompt_ab_per_seed_metrics.csv
         results/prompt_ab_significance.json
         results/figures/Figure3_prompt_comparison.png
"""
import os, json, numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "results"); FIG = os.path.join(RES, "figures"); os.makedirs(FIG, exist_ok=True)
pc = pd.read_csv(os.path.join(RES, "promptAB_percase.csv"))
SEEDS = sorted(pc.seed.unique())

def seed_auc(prompt, seed):
    s = pc[(pc.prompt == prompt) & (pc.seed == seed)].sort_values("patient_idx")
    return roc_auc_score(s.label_full, s.p_full)

rows = []
for prompt in ["minimal", "clinician"]:
    for seed in SEEDS:
        rows.append(dict(prompt=prompt, seed=seed, AUROC=round(seed_auc(prompt, seed), 4)))
per = pd.DataFrame(rows); per.to_csv(os.path.join(RES, "prompt_ab_per_seed_metrics.csv"), index=False)

mn = per[per.prompt == "minimal"].sort_values("seed").AUROC.values
cl = per[per.prompt == "clinician"].sort_values("seed").AUROC.values
t, p_t = stats.ttest_rel(cl, mn); w, p_w = stats.wilcoxon(cl, mn)

def seedavg(prompt):
    s = pc[pc.prompt == prompt].groupby("patient_idx").agg(label_full=("label_full", "first"),
                                                           p=("p_full", "mean")).reset_index()
    return s.label_full.values, s.p.values, roc_auc_score(s.label_full, s.p)
ym, pm, auc_m = seedavg("minimal"); yc, pc_, auc_c = seedavg("clinician")

def native(prompt):
    s = pc[pc.prompt == prompt]
    # majority native decision per patient across seeds
    g = s.groupby("patient_idx").agg(y=("label_full", "first"), flag=("native_pred", "mean")).reset_index()
    pred = (g.flag >= 0.5).astype(int); y = g.y.values
    tp = int(((y == 1) & (pred == 1)).sum()); fn = int(((y == 1) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum()); tn = int(((y == 0) & (pred == 0)).sum())
    return dict(sens=round(tp/(tp+fn), 3), spec=round(tn/(tn+fp), 3), caught=f"{tp}/{tp+fn}")

sig = dict(seeds=list(map(int, SEEDS)),
           minimal_auroc_mean=round(float(mn.mean()), 4), clinician_auroc_mean=round(float(cl.mean()), 4),
           paired_t_p=round(float(p_t), 4), wilcoxon_p=round(float(p_w), 4),
           seedavg_auroc_minimal=round(float(auc_m), 4), seedavg_auroc_clinician=round(float(auc_c), 4),
           native_minimal=native("minimal"), native_clinician=native("clinician"))
json.dump(sig, open(os.path.join(RES, "prompt_ab_significance.json"), "w"), indent=2)
print(json.dumps(sig, indent=2))

# 3-panel Figure 3
C_MIN, C_CLIN = "#8c8c8c", "#c0392b"
fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0))
ax = axes[0]
for M, c, lab in [(pm, C_MIN, "Minimal prompt"), (pc_, C_CLIN, "Clinician-informed prompt")]:
    fpr, tpr, _ = roc_curve(ym, M); ax.plot(fpr, tpr, color=c, lw=2.3, label=f"{lab} (AUROC {roc_auc_score(ym, M):.2f})")
ax.plot([0, 1], [0, 1], ls="--", color="#888888", lw=1)
ax.set_xlabel("False positive rate (1 \u2212 specificity)"); ax.set_ylabel("True positive rate (sensitivity)")
ax.set_title("Discrimination of full conversion"); ax.legend(loc="lower right", fontsize=7)
ax.text(-0.18, 1.02, "a", transform=ax.transAxes, fontweight="bold", fontsize=10)
# (b) threshold sweep
ax = axes[1]; grid = np.linspace(0, 1, 101)
def sweep(y, p):
    return [np.mean(p[y == 1] >= t) for t in grid], [np.mean(p[y == 0] < t) for t in grid]
sm, spm = sweep(ym, pm); sc, spc = sweep(yc, pc_)
ax.plot(grid, sc, color=C_CLIN, lw=2.2, label="Sensitivity (clinician)")
ax.plot(grid, spc, color=C_CLIN, lw=2.2, ls="--", label="Specificity (clinician)")
ax.plot(grid, sm, color=C_MIN, lw=1.8, label="Sensitivity (minimal)")
ax.plot(grid, spm, color=C_MIN, lw=1.8, ls="--", label="Specificity (minimal)")
ax.set_xlim(0, 0.6); ax.set_ylim(0, 1.02); ax.set_xlabel("Decision threshold on P(conversion)"); ax.set_ylabel("Rate")
ax.set_title("Sensitivity\u2013specificity trade-off vs threshold"); ax.legend(loc="center right", fontsize=6.4)
ax.text(-0.18, 1.02, "b", transform=ax.transAxes, fontweight="bold", fontsize=10)
# (c) 5-seed paired reproducibility
ax = axes[2]
for s, c in zip(mn, cl): ax.plot([0, 1], [s, c], color="#888888", lw=0.9, alpha=0.6, zorder=1)
ax.scatter([0]*len(mn), mn, s=70, color=C_MIN, zorder=3, edgecolor="white", linewidth=0.6)
ax.scatter([1]*len(cl), cl, s=70, color=C_CLIN, zorder=3, edgecolor="white", linewidth=0.6)
ax.plot([-0.16, 0.16], [mn.mean()]*2, color="black", lw=3, zorder=4)
ax.plot([0.84, 1.16], [cl.mean()]*2, color="black", lw=3, zorder=4)
ax.set_xlim(-0.5, 1.5); ax.set_xticks([0, 1]); ax.set_xticklabels(["Minimal", "Clinician"]); ax.set_ylabel("AUROC")
ax.set_title("Reproducibility across %d seeds" % len(SEEDS))
ax.annotate(f"+{cl.mean()-mn.mean():.2f} AUROC\npaired t-test p = {p_t:.3f}",
            xy=(0.5, 0.97), xycoords="axes fraction", ha="center", va="top", fontsize=8)
ax.text(-0.18, 1.02, "c", transform=ax.transAxes, fontweight="bold", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "Figure3_prompt_comparison.png"), dpi=300, bbox_inches="tight")
print("wrote Figure3 (paired t p = %.3f)" % p_t)
