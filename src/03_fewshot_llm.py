"""
03_fewshot_llm.py
Few-shot LLM experiment: classify each held-out patient into None/Subthreshold/Full
with k in-context exemplars sampled at the true 3-class prevalence, under the same
family-grouped 5-fold CV. Evaluated binary as Full-vs-rest.

Two feature representations:
  --mode all   : serialize all baseline features (primary; no label info used)
  --mode top10 : serialize only the in-fold top-10 features (leakage-controlled
                 robustness check; ranking fit on TRAIN patients only)

Model ids (Anthropic):
  Sonnet : claude-sonnet-5
  Haiku  : claude-haiku-4-5-20251001

This reference implementation uses the Anthropic Messages API directly (set
ANTHROPIC_API_KEY). Prompt caching is applied to the shared exemplar prefix.
Writes results/fewshot_<model>_<mode>_s<seed>.json (one record per patient x K).
"""
import pandas as pd, numpy as np, json, math, random, os, argparse
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import f_classif
from serialize import build_desc, serialize

SEED_CV = 42
DICT_XLSX = os.environ.get("BIPOLAR_XLSX", "data/Bipolar_Data.xlsx")
K_VALUES = [0, 2, 4, 8, 16, 32, 64]

SYSTEM = (
"You are a clinical expert in child and adolescent psychiatry assessing the 10-year prognosis of "
"depressed adolescents. Every patient described has FULL major depressive disorder and NO bipolar "
"features at baseline. Your task: predict each patient's most likely bipolar status 10 years later, "
"as one of three ordered categories:\n"
"  - None: no bipolar features develop\n"
"  - Subthreshold: subthreshold bipolar features develop\n"
"  - Full: full bipolar disorder develops\n\n"
"In this depressed-adolescent population the base rates are approximately: None 72%, Subthreshold 9%, "
"Full 19%. Roughly one in five converts to full bipolar disorder.\n\n"
"Clinical reference for interpreting the narratives:\n"
"- CBCL problem T-scores: T>=65 borderline, T>=70 clinical range (higher = more pathology).\n"
"- CBCL competence T-scores: T<=35 borderline-low, T<=30 clinical deficit (lower = worse).\n"
"- WISC cognitive scaled scores: mean=10, SD=3; <=7 below average, >=13 above average.\n"
"- SAICA items rated mild/moderate/severe (higher = more impairment).\n"
"- Lifetime diagnoses reported as subthreshold or full.\n"
"- Only ELEVATED or PRESENT findings are listed; anything not mentioned is normal/absent.\n\n"
"Weigh known bipolar-risk indicators (early-onset severe mood/behavior problems, thought problems, "
"externalizing and attention problems, emotional dysregulation, family history of bipolar disorder). "
"Because missing a future converter is clinically dangerous (a depressed youth wrongly judged low-risk "
"may receive antidepressant monotherapy and switch to mania), do not be overly conservative: when "
"risk indicators are present, lean toward flagging elevated risk. "
"Return your prediction using the classify_patient tool.")

TOOL = [{"name": "classify_patient",
         "description": "Record the 10-year bipolar prognosis prediction.",
         "input_schema": {"type": "object", "properties": {
             "brief_reasoning": {"type": "string"},
             "p_none": {"type": "number"}, "p_subthreshold": {"type": "number"},
             "p_full": {"type": "number"},
             "predicted_class": {"type": "string", "enum": ["None","Subthreshold","Full"]}},
             "required": ["p_none","p_subthreshold","p_full","predicted_class"]}}]

def prevalence_shots(train_idx, lab3, prev, K, seed):
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

def build_messages(narr_by_i, fold, shots_idx, test_i, lab3):
    ex = ""
    for j, i in enumerate(shots_idx, 1):
        lab = {1:"None",2:"Subthreshold",3:"Full"}[lab3[i]]
        ex += f"\n[Example {j}]\n{narr_by_i[fold][i]}\nCorrect 10-year outcome: {lab}\n"
    prefix = (f"Here are {len(shots_idx)} labeled example patients from the same population:\n{ex}\n\n"
              "Now assess the following new patient." if shots_idx else "Assess the following patient.")
    return [{"role": "user", "content": [
        {"type": "text", "text": prefix, "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": f"\nPATIENT:\n{narr_by_i[fold][test_i]}\n\nCall classify_patient with your prediction."}]}]

def extract(resp):
    tu = None
    for b in resp.content:
        if b.type == "tool_use": tu = b.input; break
    if tu is None: return None
    pn, ps, pf = tu.get("p_none"), tu.get("p_subthreshold"), tu.get("p_full")
    s = (pn or 0) + (ps or 0) + (pf or 0)
    if s <= 0: return None
    pn, ps, pf = pn/s, ps/s, pf/s
    order = ["None","Subthreshold","Full"]
    pc = tu.get("predicted_class") or order[int(np.argmax([pn, ps, pf]))]
    return pf, pn, ps, pc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="claude-sonnet-5 or claude-haiku-4-5-20251001")
    ap.add_argument("--mode", choices=["all", "top10"], default="all")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    import anthropic
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY

    df = pd.read_csv("data/cohort.csv")
    feats = json.load(open("data/features.json"))["features"]
    lab3 = df["label3"].values.astype(int)
    groups = df["famid"].values
    desc = build_desc(DICT_XLSX)
    prev = {c: (lab3 == c).mean() for c in [1,2,3]}

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED_CV)
    folds = [{"train": tr.tolist(), "test": te.tolist()}
             for tr, te in sgkf.split(np.zeros(len(lab3)), lab3, groups)]

    # per-fold feature list + narratives
    narr = {}
    for fi, f in enumerate(folds):
        if args.mode == "top10":
            X = df[feats].values
            imp = SimpleImputer(strategy="median").fit(X[f["train"]])
            F, _ = f_classif(imp.transform(X[f["train"]]), (lab3[f["train"]] == 3).astype(int))
            order = np.argsort(np.nan_to_num(F, nan=-1))[::-1][:10]
            fl = [feats[j] for j in order]; only_elev = False
        else:
            fl = feats; only_elev = True
        narr[fi] = {i: serialize(df.iloc[i], fl, desc, only_elevated=only_elev)
                    for i in range(len(df))}

    recs = []
    for fi, f in enumerate(folds):
        for K in K_VALUES:
            shots = prevalence_shots(f["train"], lab3, prev, K, args.seed)
            for i in f["test"]:
                msg = build_messages(narr, fi, shots, i, lab3)
                resp = client.messages.create(model=args.model, max_tokens=500,
                          system=SYSTEM, tools=TOOL,
                          tool_choice={"type": "tool", "name": "classify_patient"},
                          messages=msg)
                e = extract(resp)
                row = {"idx": i, "fold": fi, "K": K, "seed": args.seed,
                       "label3": int(lab3[i]), "label_bin": int(lab3[i] == 3)}
                if e: pf, pn, ps, pc = e; row.update(p_full=pf, p_none=pn, p_sub=ps, pred=pc)
                else: row.update(p_full=None, p_none=None, p_sub=None, pred=None)
                recs.append(row)
    os.makedirs("results", exist_ok=True)
    tag = "sonnet" if "sonnet" in args.model else "haiku"
    fn = f"results/fewshot_{tag}_{args.mode}_s{args.seed}.json"
    json.dump(recs, open(fn, "w"))
    print(f"wrote {fn}: {sum(1 for r in recs if r['p_full'] is not None)}/{len(recs)} parsed")

if __name__ == "__main__":
    main()
