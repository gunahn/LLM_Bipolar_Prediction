# MentalLLaMA zero-shot baseline — bipolar-conversion prediction (ORCD)

A second open, mental-health-specialized LLM baseline for the manuscript. Zero-shot:
prompts MentalLLaMA once per patient and reads the converter probability from the
Yes/No first-token logits. Same 141 patients and family-grouped folds as the
MentalRoBERTa / TabFM baselines; outputs the same schema so it drops straight into the
baseline table.

Expected result: AUROC ~0.60–0.70 (a specialized open LLM), below the Claude models
(0.73–0.74). The point of including it is to show that *general-purpose* frontier LLMs
beat *mental-health-specialized* open models on this longitudinal-prediction task.

## Files
- `run_mentalllama.py`  — inference + metrics (AUROC+CI, Youden sens/spec, Brier, ECE)
- `prefetch_weights.py` — download weights on the **login node** (has internet)
- `submit.sbatch`       — SLURM job for `ou_bcs_normal` (1 GPU, 2 h)
- `data.json`           — 141 narratives + labels + folds (identical to MentalRoBERTa job)
- `requirements.txt`    — torch / transformers / etc.

## Run (from Claude Code on your Mac, via SSH to ORCD)
```bash
# 1. copy the bundle up
scp -r mentalllama_job guna23@orcd-login.mit.edu:~/

# 2. ssh in (answer Duo), set up env if not already present
ssh guna23@orcd-login.mit.edu
cd ~/mentalllama_job
python -m venv ~/envs/mental && source ~/envs/mental/bin/activate   # if you don't already have one
pip install -r requirements.txt

# 3. prefetch weights ON THE LOGIN NODE (compute nodes have no internet)
export HF_HOME=$HOME/hf_cache
python prefetch_weights.py --model klyang/MentaLLaMA-chat-7B     # ~27 GB, 15-40 min

# 4. submit the GPU job
sbatch submit.sbatch
squeue -u guna23        # watch it; scheduled within minutes on ou_bcs_normal

# 5. when done, copy the 3 result files back and upload them here
#    mentalllama_metrics.json  mentalllama_oof.npz  mentalllama_percase.csv
```

## Notes
- `klyang/MentaLLaMA-chat-7B` is MIT-licensed and openly downloadable (no LLaMA-2
  gating), fine-tuned from Meta LLaMA2-chat-7B. The HF repo tree lists ~27 GB total
  (weights stored as a single full-precision pytorch_model.bin — a fp16 7B checkpoint
  would be ~13 GB, so the 27 GB reflects fp32 storage). Allow disk/download time
  accordingly; the script loads in fp16 on GPU regardless of the stored precision.
- The 13B variant `klyang/MentaLLaMA-chat-13B` uses the same code (pass `--model`) if
  you want the larger model.
- The Yes/No logit approach gives a proper probability without sampling, so the metric
  is directly comparable to the other baselines (no threshold tuning needed beyond Youden).
