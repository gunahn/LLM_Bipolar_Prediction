#!/usr/bin/env python
"""
Fine-tune MentalRoBERTa (mental/mental-roberta-base) on serialized clinical
narratives to predict 10-year full bipolar conversion (binary Full-vs-rest),
under the SAME family-grouped 5-fold CV used for the supervised ML and few-shot
LLM paradigms. Produces out-of-fold predicted probabilities and pooled metrics
directly comparable to those baselines.

Inputs  : data.json  {narratives:[141], labels_bin:[141], folds:[5x{train,test}], n}
Outputs : mentalroberta_oof.npz      (p_full[141], ybin[141], fold_id[141])
          mentalroberta_metrics.json (pooled AUROC + bootstrap CI, sens/spec at
                                      Youden, Brier, ECE)
          mentalroberta_percase.csv  (patient_idx, fold, label, p_full)

Class imbalance handled via class-weighted cross-entropy (pos_weight from the
training fold), matching the "imbalance-corrected" spirit of the ML baseline
without touching held-out patients.
"""
import json, numpy as np, argparse, os
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, brier_score_loss

def ece(y, p, nb=10):
    bins = np.linspace(0, 1, nb + 1); e = 0.0
    for i in range(nb):
        m = (p >= bins[i]) & (p < bins[i+1]) if i < nb-1 else (p >= bins[i]) & (p <= bins[i+1])
        if m.sum():
            e += abs(p[m].mean() - y[m].mean()) * m.sum() / len(y)
    return float(e)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data.json")
    ap.add_argument("--model", default="mental/mental-roberta-base")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--maxlen", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=".")
    args = ap.parse_args()

    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                              get_linear_schedule_with_warmup)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[env] device={dev} torch={torch.__version__} "
          f"cuda_avail={torch.cuda.is_available()}", flush=True)
    if dev == "cuda":
        print(f"[env] gpu={torch.cuda.get_device_name(0)}", flush=True)

    D = json.load(open(args.data))
    narr = D["narratives"]; y = np.array(D["labels_bin"]); folds = D["folds"]; n = D["n"]
    assert len(narr) == n == len(y)
    tok = AutoTokenizer.from_pretrained(args.model)

    class DS(Dataset):
        def __init__(self, idx):
            self.idx = idx
        def __len__(self):
            return len(self.idx)
        def __getitem__(self, i):
            j = self.idx[i]
            enc = tok(narr[j], truncation=True, max_length=args.maxlen,
                      padding="max_length", return_tensors="pt")
            return {k: v.squeeze(0) for k, v in enc.items()}, int(y[j]), j

    p_oof = np.full(n, np.nan); fold_id = np.full(n, -1)
    for fi, fold in enumerate(folds):
        tr, te = fold["train"], fold["test"]
        print(f"\n[fold {fi}] train={len(tr)} test={len(te)} "
              f"pos_train={int(y[tr].sum())}", flush=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model, num_labels=2).to(dev)
        # class weight from training fold only
        pos = max(int(y[tr].sum()), 1); neg = len(tr) - pos
        w = torch.tensor([1.0, neg / pos], dtype=torch.float).to(dev)
        lossf = torch.nn.CrossEntropyLoss(weight=w)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
        dl_tr = DataLoader(DS(tr), batch_size=args.batch, shuffle=True)
        steps = len(dl_tr) * args.epochs
        sch = get_linear_schedule_with_warmup(opt, int(0.1*steps), steps)
        model.train()
        for ep in range(args.epochs):
            tot = 0.0
            for enc, lab, _ in dl_tr:
                enc = {k: v.to(dev) for k, v in enc.items()}
                lab = torch.tensor(lab).to(dev)
                opt.zero_grad()
                out = model(**enc).logits
                loss = lossf(out, lab)
                loss.backward(); opt.step(); sch.step()
                tot += loss.item()
            print(f"  ep{ep} loss={tot/len(dl_tr):.4f}", flush=True)
        # predict held-out
        model.eval()
        dl_te = DataLoader(DS(te), batch_size=args.batch)
        with torch.no_grad():
            for enc, lab, jj in dl_te:
                enc = {k: v.to(dev) for k, v in enc.items()}
                prob = torch.softmax(model(**enc).logits, dim=1)[:, 1].cpu().numpy()
                for k, j in enumerate(jj.numpy()):
                    p_oof[j] = prob[k]; fold_id[j] = fi
        del model
        if dev == "cuda":
            torch.cuda.empty_cache()

    assert not np.isnan(p_oof).any(), "some patients never predicted"
    # pooled metrics
    rng = np.random.default_rng(args.seed)
    boots = []
    idx = np.arange(n)
    for _ in range(2000):
        b = rng.choice(idx, n, replace=True)
        if 0 < y[b].sum() < n:
            boots.append(roc_auc_score(y[b], p_oof[b]))
    auroc = float(roc_auc_score(y, p_oof))
    fpr, tpr, thr = roc_curve(y, p_oof); ji = np.argmax(tpr - fpr); t = float(thr[ji])
    pred = (p_oof >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    metrics = dict(
        model=args.model, n=n, positives=int(y.sum()),
        AUROC=auroc, AUROC_lo=float(np.percentile(boots, 2.5)),
        AUROC_hi=float(np.percentile(boots, 97.5)),
        youden_thr=t, Sens=float(tp/(tp+fn)), Spec=float(tn/(tn+fp)),
        Acc=float((tp+tn)/n), Brier=float(brier_score_loss(y, p_oof)),
        ECE=ece(y, p_oof),
        epochs=args.epochs, lr=args.lr, batch=args.batch, maxlen=args.maxlen)
    os.makedirs(args.out, exist_ok=True)
    np.savez(os.path.join(args.out, "mentalroberta_oof.npz"),
             p_full=p_oof, ybin=y, fold_id=fold_id)
    json.dump(metrics, open(os.path.join(args.out, "mentalroberta_metrics.json"), "w"),
              indent=2)
    with open(os.path.join(args.out, "mentalroberta_percase.csv"), "w") as f:
        f.write("patient_idx,fold,label,p_full\n")
        for j in range(n):
            f.write(f"{j},{int(fold_id[j])},{int(y[j])},{p_oof[j]:.6f}\n")
    print("\n[RESULT]", json.dumps(metrics, indent=2), flush=True)

if __name__ == "__main__":
    main()
