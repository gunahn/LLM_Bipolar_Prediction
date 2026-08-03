# MentalRoBERTa bipolar-conversion baseline — MIT ORCD job

Fine-tunes `mental/mental-roberta-base` on 141 serialized clinical narratives to
predict 10-year full bipolar conversion, under the same family-grouped 5-fold CV
as the paper's supervised-ML and few-shot-LLM baselines.

## Files
- `data.json` — narratives, binary labels, and the 5 fold index splits
- `train_mentalroberta.py` — training + OOF evaluation (class-weighted for imbalance)
- `prefetch_weights.py` — downloads model weights (run on login node)
- `submit.sbatch` — SLURM script (partition `ou_bcs_normal`, 1 GPU, 32G, 2h)
- `requirements.txt` — torch, transformers, scikit-learn, numpy

## Run sequence (ORCD, user guna23)
```bash
# 1. SSH in (Duo 2FA — keep this session authenticated; needed for submission)
ssh guna23@orcd-login.mit.edu

# 2. Copy this bundle over (or scp the tarball and extract)
cd /path/to/mentalroberta_job

# 3. Pre-fetch weights on the LOGIN node (has internet; compute nodes do not)
export HF_HOME=$PWD/hf_cache
python prefetch_weights.py        # needs torch+transformers available here

# 4. Submit
sbatch submit.sbatch
squeue -u guna23                  # watch; scheduled within minutes on ou_bcs_normal
```

## Outputs (harvest these back)
- `mentalroberta_oof.npz` — per-patient OOF probabilities + fold ids
- `mentalroberta_metrics.json` — AUROC (bootstrap CI), sens/spec@Youden, Brier, ECE
- `mentalroberta_percase.csv` — patient_idx, fold, label, p_full
- `mentalroberta_<jobid>.out` — training log

These slot directly into manuscript §2.7 / Table 1 / Figure 1 as the neural-text
baseline row, matched to the n=141 cohort and folds.

## Notes
- **Duo 2FA**: every SSH login needs password + Duo. Non-interactive submission
  requires an already-authenticated session; a script cannot complete the prompt.
- **Offline compute nodes**: the job runs with `HF_HUB_OFFLINE=1`; weights MUST be
  pre-fetched (step 3) or the job dies on a network call.
- **Partition**: `ou_bcs_normal` (A100/H100 80GB). For a 4h higher-priority slot
  use `ou_bcs_high`; general fallback `mit_normal_gpu`.
