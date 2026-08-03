#!/usr/bin/env python
"""
Zero-shot MentalLLaMA baseline for bipolar-conversion prediction.

Prompts MentalLLaMA (klyang/MentaLLaMA-chat-7B or 13B) once per patient with the
serialized baseline clinical narrative and asks for a Yes/No judgement on whether the
adolescent will develop full bipolar disorder over 10-year follow-up. The converter
probability is read from the normalized Yes-vs-No first-token logits (no generation
sampling needed for the score), so the metric is a proper probability, directly
comparable to the other baselines.

Outputs (same schema as the MentalRoBERTa job so it drops into the baseline table):
  mentalllama_oof.npz      -> p_full (141,), y (141,)
  mentalllama_metrics.json -> AUROC[+95% CI], Youden sens/spec/acc, Brier, ECE
  mentalllama_percase.csv  -> patient_idx, fold, label, p_full

Design notes:
  * Zero-shot: no fold-specific training, so folds only tag provenance; every patient
    is scored under the identical prompt. (Kept for parity with the other baselines.)
  * Full-data evaluation: pooled over all 141 patients (matches how the LLM few-shot,
    MentalRoBERTa, and TabFM results were pooled).
  * Compute nodes have no internet: weights must be prefetched on the login node with
    HF_HOME set, and this script run with HF_HUB_OFFLINE=1.
"""
import os, json, argparse, numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, brier_score_loss

SYSTEM = (
    "You are a clinical expert in child and adolescent psychiatry. You are shown a "
    "structured clinical summary of an adolescent who currently has full major "
    "depressive disorder and NO bipolar features at baseline. Your task is to judge "
    "whether this adolescent will develop FULL bipolar disorder over the next 10 years."
)
QUESTION = (
    "\n\nBased on this baseline profile, will this adolescent develop full bipolar "
    "disorder within 10 years? Answer with a single word: Yes or No."
)

def build_prompt(tok, narrative):
    user = narrative.strip() + QUESTION
    # Use the model's chat template when available; fall back to a plain instruction format.
    try:
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": user}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        return f"[INST] <<SYS>>\n{SYSTEM}\n<</SYS>>\n\n{user} [/INST]"

def yes_no_prob(model, tok, prompt, yes_ids, no_ids, device):
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=3072).to(device)
    with torch.no_grad():
        logits = model(**enc).logits[0, -1, :].float()  # next-token logits
    # aggregate over all case variants of Yes / No tokens
    y = torch.logsumexp(logits[yes_ids], dim=0)
    n = torch.logsumexp(logits[no_ids], dim=0)
    # normalized P(Yes) over the Yes/No two-way contrast
    p = torch.softmax(torch.stack([n, y]), dim=0)[1].item()
    return p

def token_ids(tok, words):
    ids = set()
    for w in words:
        for variant in (w, " " + w):
            t = tok(variant, add_special_tokens=False)["input_ids"]
            if t:
                ids.add(t[0])
    return sorted(ids)

def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= edges[i + 1])
        if m.sum():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="klyang/MentaLLaMA-chat-7B")
    ap.add_argument("--data", default="data.json")
    ap.add_argument("--tag", default="mentalllama")
    args = ap.parse_args()

    data = json.load(open(args.data))
    narratives = data["narratives"]
    y = np.array(data["labels_bin"], dtype=int)
    n = len(narratives)
    fold_of = np.full(n, -1)
    for fi, f in enumerate(data["folds"]):
        for idx in f["test"]:
            fold_of[idx] = fi

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading {args.model} on {device} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map={"": 0} if device == "cuda" else None)
    model.eval()

    yes_ids = torch.tensor(token_ids(tok, ["Yes", "yes", "YES"]), device=device)
    no_ids = torch.tensor(token_ids(tok, ["No", "no", "NO"]), device=device)
    print("yes_ids", yes_ids.tolist(), "no_ids", no_ids.tolist(), flush=True)

    p = np.zeros(n)
    for i, nar in enumerate(narratives):
        prompt = build_prompt(tok, nar)
        p[i] = yes_no_prob(model, tok, prompt, yes_ids, no_ids, device)
        if (i + 1) % 20 == 0:
            print(f"  scored {i+1}/{n}", flush=True)

    auroc = roc_auc_score(y, p)
    # bootstrap AUROC 95% CI (patient-level, 2000 reps, seed 42)
    rng = np.random.default_rng(42)
    boots = []
    idx = np.arange(n)
    for _ in range(2000):
        b = rng.choice(idx, n, replace=True)
        if 0 < y[b].sum() < len(b):
            boots.append(roc_auc_score(y[b], p[b]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    fpr, tpr, thr = roc_curve(y, p)
    j = int(np.argmax(tpr - fpr)); t = float(thr[j])
    pred = (p >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    metrics = {
        "model": args.model, "n": int(n), "positives": int(y.sum()),
        "approach": "zero-shot Yes/No logit probability, pooled over all patients",
        "AUROC": float(auroc), "AUROC_lo": float(lo), "AUROC_hi": float(hi),
        "youden_thr": t,
        "Sens": tp / (tp + fn), "Spec": tn / (tn + fp),
        "Acc": (tp + tn) / n,
        "Brier": float(brier_score_loss(y, p)), "ECE": ece(y, p),
    }
    json.dump(metrics, open(f"{args.tag}_metrics.json", "w"), indent=2)
    np.savez(f"{args.tag}_oof.npz", p_full=p, y=y)
    import csv
    with open(f"{args.tag}_percase.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["patient_idx", "fold", "label", "p_full"])
        for i in range(n):
            w.writerow([i, int(fold_of[i]), int(y[i]), f"{p[i]:.6f}"])
    print(json.dumps(metrics, indent=2), flush=True)

if __name__ == "__main__":
    main()
