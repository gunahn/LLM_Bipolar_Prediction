# Few-Shot LLMs for Predicting Bipolar Conversion in Depressed Adolescents

Reproducible code for the study comparing few-shot large language models (LLMs),
supervised machine learning, and clinicians at predicting 10-year conversion to full
bipolar disorder among adolescents presenting with major depression.

**Clinical motivation.** Bipolar disorder in youth often begins as depression. Because
antidepressant monotherapy can precipitate a manic switch in latent bipolarity, a
*missed* future converter (false negative) is the most harmful error — so the analysis
prioritizes sensitivity, always reported jointly with specificity.

## Cohort

- Inclusion: baseline full MDD (`mdd_baseline==3`) **and** no baseline bipolar features
  (`bpdp_baseline==1`).
- Outcome: `bpdp_10years` — 1=None, 2=Subthreshold, 3=Full. Models train on the 3-class
  label; evaluation is the binary **Full vs None/Subthreshold** contrast.
- Features: all `*_baseline` columns except the target-adjacent `bpdp_baseline` and the
  inclusion column `mdd_baseline`; `*_4years`/`*_10years` excluded (temporal leakage).
  Columns/subjects with >30% missing dropped.
- Final analytic sample: **n = 141, 64 features, 27 converters (19%)**.
- Evaluation: family-grouped 5-fold cross-validation (siblings never split), metrics
  pooled across held-out folds.

## Pipeline

```bash
# 0. install
pip install -r requirements.txt

# 1. build cohort from the raw workbook (place it at data/Bipolar_Data.xlsx or set BIPOLAR_XLSX)
python src/01_build_cohort.py

# 2. supervised baselines + ML k-shot curve
python src/02_supervised_baseline.py

# 3. few-shot LLM sweep  (needs ANTHROPIC_API_KEY)
#    --mode all   : full clinical narrative (primary; no label info used)
#    --mode top10 : in-fold top-10 features (leakage-controlled robustness check)
export ANTHROPIC_API_KEY=sk-...
for s in 0 1 2; do python src/03_fewshot_llm.py --model claude-sonnet-5 --mode all --seed $s; done
for s in 0 1;   do python src/03_fewshot_llm.py --model claude-haiku-4-5-20251001 --mode all --seed $s; done

# 4. aggregate + main figure
python src/04_analyze_and_figure.py
```

## Models

| Role | API identifier |
|---|---|
| Claude Sonnet (primary few-shot predictor) | `claude-sonnet-5` |
| Claude Haiku (secondary few-shot predictor) | `claude-haiku-4-5-20251001` |

Anthropic; accessed 2026. Cite with the access date, as `claude-sonnet-5` is a rolling
alias; the exact dated snapshot is returned in each API response's `model` field.

## Headline results

- Best supervised ML (full data): **AUROC 0.67** (TabNet / XGBoost).
- Few-shot Claude Sonnet: **0.72 zero-shot, 0.74 at k=2** (range 0.70–0.74 across k).
- Few-shot Claude Haiku: **0.67–0.74**.
- Supervised ML trained on the same k examples stays near chance (0.50 at k≤2 → 0.60 at k=64).
- Restricting the LLM to the in-fold top-10 features erases the advantage (0.62–0.67),
  showing the edge comes from reasoning over the full clinical narrative.

## Layout

```
src/01_build_cohort.py          cohort construction
src/serialize.py                feature-row -> clinical narrative (shared module)
src/02_supervised_baseline.py   LR/RF/XGBoost/TabNet + ML k-shot curve
src/03_fewshot_llm.py           few-shot LLM sweep (all / top10 modes)
src/04_analyze_and_figure.py    aggregation, k-sweep table, 3-panel figure
data/                           cohort.csv, features.json (generated)
results/                        baselines.csv, fewshot_*.json, figure (generated)
```

## Notes / caveats

- Single-site sample enriched for ADHD family history; retrospective; prospective
  validation required before any clinical use. **Research code — not a medical device.**
- Clinician-panel and MentalBERT text-model baselines are part of the study design but
  are collected/added separately.
- The raw clinical data are not included in this repository (human-subjects data).

## Prompt & exemplar specification

The exact system prompt, structured-output tool schema, exemplar format, and
prevalence-based sampling are documented verbatim in
[APPENDIX_prompt_and_exemplars.md](APPENDIX_prompt_and_exemplars.md).
