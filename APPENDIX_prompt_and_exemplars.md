# Appendix: LLM Prompt and Exemplar Specification

This appendix reproduces the exact system prompts, structured-output tool, exemplar
format, and serialization used for the large language model (LLM) experiments,
verbatim from the manuscript's Supplementary Appendix (S1–S6). It is the
authoritative prompt reference for the code in `src/03_fewshot_llm.py`,
`src/models/08_gemini_zeroshot.py`, and `src/supplementary/S_prompt_comparison.py`.

> **Which prompt is primary?** All main-text LLM results use the **prevalence-free**
> primary prompt (A.1), which states *no* outcome base rate and instructs the model
> to infer the conversion rate itself. The prevalence-stated prompt (A.2) is used
> **only** for the base-rate ablation (Supplementary S7). A separate *minimal* prompt
> (A.3) is used only for the prompt-specification comparison (main Figure 3 /
> Supplementary S9).

## A.1 Primary system prompt (prevalence-free)

The primary prompt for all main-text LLM results. It describes the study design and
cohort but states **no outcome base rate**. Sent identically with every request, for
both models (`claude-sonnet-5` and `claude-haiku-4-5-20251001`, Anthropic) and — with
the identical text — for the cross-vendor Gemini run.

```text
You are a clinical expert in child and adolescent psychiatry assessing the 10-year prognosis of depressed adolescents. Your task: predict each patient's most likely bipolar status 10 years later, as one of three ordered categories:
  - None: no bipolar features develop
  - Subthreshold: subthreshold bipolar features develop
  - Full: full bipolar disorder develops

Study and sample context:
- These patients come from longitudinal case-control family studies of attention-deficit/hyperactivity disorder (ADHD) in youth (companion 'boys' and 'girls' cohorts), in which children and adolescents were assessed in detail and followed prospectively into adulthood.
- The studies ascertained pediatric probands (with and without ADHD) together with their siblings, so the overall sample is enriched for ADHD and for family psychiatric history, and siblings from the same family are present. Most patients carry one or more positive family-history flags.
- The present analytic sample is restricted to adolescents who, at the baseline assessment, met criteria for FULL major depressive disorder and had NO bipolar features. You are estimating which of these depressed adolescents will later develop bipolar illness.
- Baseline data include the Child Behavior Checklist (CBCL), structured diagnostic interviews (lifetime diagnoses), the Social Adjustment Inventory for Children and Adolescents (SAICA), Wechsler (WISC) cognitive testing, and family psychiatric history. Infer the likely conversion rate yourself from the clinical literature on depressed youth and from the sample described.

Clinical reference for interpreting the narratives:
- CBCL problem T-scores: T>=65 borderline, T>=70 clinical range (higher = more pathology).
- CBCL competence T-scores: T<=35 borderline-low, T<=30 clinical deficit (lower = worse).
- WISC cognitive scaled scores: mean=10, SD=3; <=7 below average, >=13 above average.
- SAICA items rated mild/moderate/severe (higher = more impairment).
- Lifetime diagnoses reported as subthreshold or full.
- Only ELEVATED or PRESENT findings are listed; anything not mentioned is normal/absent.

Weigh known bipolar-risk indicators (early-onset severe mood/behavior problems, thought problems, externalizing and attention problems, family history of bipolar disorder, emotional dysregulation). Because missing a future converter is clinically dangerous (a depressed youth wrongly judged low-risk may receive antidepressant monotherapy and switch to mania), do not be overly conservative: when risk indicators are present, lean toward flagging elevated risk. Return your prediction using the classify_patient tool.
```

## A.2 Base-rate ablation system prompt (prevalence stated)

Identical to A.1 except that it explicitly states the outcome base rates. Used **only**
for the prompt ablation reported in Supplementary S7 (`results` comparing V1 vs V2).

```text
You are a clinical expert in child and adolescent psychiatry assessing the 10-year prognosis of depressed adolescents. Every patient described has FULL major depressive disorder and NO bipolar features at baseline. Your task: predict each patient's most likely bipolar status 10 years later, as one of three ordered categories:
  - None: no bipolar features develop
  - Subthreshold: subthreshold bipolar features develop
  - Full: full bipolar disorder develops

In this depressed-adolescent population the base rates are approximately: None 72%, Subthreshold 9%, Full 19%. Roughly one in five converts to full bipolar disorder.

Clinical reference for interpreting the narratives:
- CBCL problem T-scores: T>=65 borderline, T>=70 clinical range (higher = more pathology).
- CBCL competence T-scores: T<=35 borderline-low, T<=30 clinical deficit (lower = worse).
- WISC cognitive scaled scores: mean=10, SD=3; <=7 below average, >=13 above average.
- SAICA items rated mild/moderate/severe (higher = more impairment).
- Lifetime diagnoses reported as subthreshold or full.
- Only ELEVATED or PRESENT findings are listed; anything not mentioned is normal/absent.

Weigh known bipolar-risk indicators (early-onset severe mood/behavior problems, thought problems, externalizing and attention problems, family history of bipolar disorder, emotional dysregulation). Because missing a future converter is clinically dangerous (a depressed youth wrongly judged low-risk may receive antidepressant monotherapy and switch to mania), do not be overly conservative: when risk indicators are present, lean toward flagging elevated risk. Return your prediction using the classify_patient tool.
```

## A.3 Minimal prompt (prompt-specification comparison)

Used only for the minimal-vs-clinician-informed comparison (main Figure 3,
Supplementary S9). No system prompt and no clinical framing; the entire user message is:

```text
Assess the following patient. PATIENT: <patient clinical narrative>. Call classify_patient with your prediction whether they will develop bipolar in 10 years. Give me yes or no.
```

with a minimal two-field tool returning a categorical `Yes`/`No` and a probability
(in place of the three-class tool in A.4).

## A.4 Structured-output tool

Predictions were elicited through a forced tool call (`tool_choice` set to this tool),
so every response returned a parseable probability for each class. The probability of
the `Full` class (renormalized so `p_none + p_subthreshold + p_full = 1`) was used as
the conversion score for AUROC/AUPRC.

```json
[
  {
    "name": "classify_patient",
    "description": "Record the 10-year bipolar prognosis prediction.",
    "input_schema": {
      "type": "object",
      "properties": {
        "brief_reasoning": {
          "type": "string"
        },
        "p_none": {
          "type": "number"
        },
        "p_subthreshold": {
          "type": "number"
        },
        "p_full": {
          "type": "number"
        },
        "predicted_class": {
          "type": "string",
          "enum": [
            "None",
            "Subthreshold",
            "Full"
          ]
        }
      },
      "required": [
        "p_none",
        "p_subthreshold",
        "p_full",
        "predicted_class"
      ]
    }
  }
]
```

## A.5 Exemplar (few-shot example) format

For *k* > 0, the *k* labeled exemplars were concatenated ahead of the target patient.
Each exemplar used the same natural-language clinical narrative as the target, followed
by its correct outcome label:

```text
[Example j]
<patient clinical narrative>
Correct 10-year outcome: <None|Subthreshold|Full>
```

The framing sentence before the target patient (*k* > 0) was:

```text
Here are <k> labeled example patients from the same population:
<exemplar blocks>

Now assess the following new patient.
```

For *k* = 0 (zero-shot) the framing was simply `Assess the following patient.` The
target block was:

```text
PATIENT:
<patient clinical narrative>

Call classify_patient with your prediction.
```

A rendered two-exemplar message (truncated) is shown below:

```json
[{"role": "user", "content": [{"type": "text", "text": "Here are 2 labeled example patients from the same population:\n\n[Example 1]\nBaseline presentation: 15-year-old female. All features below are from the baseline assessment; the patient has full major depressive disorder and no bipolar features at baseline. Socioeconomic status score: 1. SAICA problems: spare time activities (mild); activities with peers (mild); problems with peers (mild); relationship with siblings (mild); relationship with father (mild). Family history: alacohol abuse; social phobia. Cognitive (WISC scaled, mean=10): WISC-III Block Design high (15); WAIS-III/WISC-IV Digit Symbol high (19); WAIS-III/WISC-IV Arithmetic high (13).\nCorrect 10-year outcome: None\n\n[Example 2]\nBaseline presentation: 11-year-old male. All features below are from the baseline assessment; the patient has full major depressive disorder and no bipolar features at baseline. Socioeconomic status score: 3. Elevated CBCL problem scales: CBCL Internalizing t-score T=72 (clinical range); CBCL Total t-score T=75 (clinical range); CBCL Attention Problems t-score T=86 (clinical range); CBCL Thought Problems t-score T=70 (clinical range);
```

## A.6 Exemplar sampling at true prevalence

Exemplars were drawn from the training partition of each cross-validation fold only
(never the test patient's fold) and sampled to reflect the true three-class composition
(None 72% / Subthreshold 9% / Full 19%) via largest-remainder allocation with a fixed
per-seed random seed. The class allocation by *k* (shown for fold 0, seed 0; other
folds/seeds differ only in which specific training patients are drawn, not the class
counts):

| k | None | Subthreshold | Full | total |
|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0 |
| 2 | 2 | 0 | 0 | 2 |
| 4 | 3 | 0 | 1 | 4 |
| 8 | 6 | 1 | 1 | 8 |
| 16 | 11 | 2 | 3 | 16 |
| 32 | 23 | 3 | 6 | 32 |
| 64 | 46 | 6 | 12 | 64 |

## A.7 Patient clinical narrative (serialization)

Each patient's baseline feature row was rendered into a natural-language narrative
reporting only clinically notable findings, using the following thresholds (full
serialization code in `src/serialize.py`):

- CBCL problem T-scores: T ≥ 65 borderline, T ≥ 70 clinical range
- CBCL competence T-scores: T ≤ 35 borderline-low, T ≤ 30 clinical deficit
- WISC cognitive scaled scores: ≤ 7 below average, ≥ 13 above average (mean 10, SD 3)
- SAICA items: reported when ≥ 2 (mild / moderate / severe)
- Lifetime diagnoses: reported when subthreshold or full
- Family-history flags: reported when present

The serializer walks each baseline variable, applies the thresholds above, and emits a
short phrase **only** when the finding is elevated or present; normal, absent, or
missing variables generate no text. Two complete examples from the cohort follow (the
10-year outcome label shown here was used only for exemplars, never revealed for the
patient being predicted).

**Example 1 - 10-year outcome: FULL bipolar disorder (converter).**

```text
Baseline presentation: 8-year-old male. All features below are from the baseline assessment; the patient has full major depressive disorder and no bipolar features at baseline. Socioeconomic status score: 4. Elevated CBCL problem scales: CBCL Externalizing t-score T=80 (clinical range); CBCL Internalizing t-score T=77 (clinical range); CBCL Total t-score T=80 (clinical range); CBCL Aggressive t-score T=82 (clinical range); CBCL Rule Breaking t-score T=81 (clinical range); CBCL Attention Problems t-score T=75 (clinical range); CBCL Thought Problems t-score T=70 (clinical range); CBCL Social Problems t-score T=80 (clinical range); CBCL Anxious/Depressed t-score T=79 (clinical range); CBCL Somatic t-score T=72 (clinical range); CBCL Withdrawn t-score T=73 (clinical range). Low CBCL competence: CBCL School Competence t-score T=35 (borderline low); CBCL Social Competence t-score T=25 (clinical deficit). SAICA problems: school behavior problems (severe); spare time problems (severe); activities with peers (moderate); problems with peers (severe); problems with siblings (mild); relationship with father (severe); problems with parents (moderate). Lifetime diagnoses/history: placed in a special class (full); received extra help in school (full); panic disorder (subthreshold); overanxious disorder (subthreshold); attention-deficit/hyperactivity disorder (full). Family history: receiving extra help in school; major depressive disorder; alcohol dependence; substance dependence; agoraphobia; simple phobia; overanxious disorder; bipolar disorder; attention-deficit/hyperactivity disorder. Cognitive (WISC scaled, mean=10): WISC-III Block Design high (13); WAIS-III/WISC-IV Digit Span low (7); WASI Vocabulary high (14).
```

This 8-year-old presents with pervasive clinical-range CBCL elevations (externalizing, thought problems, aggression, attention), severe SAICA behavioral and peer problems, lifetime ADHD, and - critically - a family history that includes bipolar disorder. The co-occurrence of early severe dysregulation, thought problems, and familial bipolarity is the high-risk pattern the model is asked to recognize.

**Example 2 - 10-year outcome: NONE (non-converter).**

```text
Baseline presentation: 9-year-old male. All features below are from the baseline assessment; the patient has full major depressive disorder and no bipolar features at baseline. Socioeconomic status score: 1. Elevated CBCL problem scales: CBCL Externalizing t-score T=75 (clinical range); CBCL Internalizing t-score T=80 (clinical range); CBCL Total t-score T=74 (clinical range); CBCL Aggressive t-score T=82 (clinical range); CBCL Rule Breaking t-score T=70 (clinical range); CBCL Social Problems t-score T=73 (clinical range); CBCL Anxious/Depressed t-score T=86 (clinical range); CBCL Withdrawn t-score T=86 (clinical range). Low CBCL competence: CBCL Social Competence t-score T=30 (clinical deficit). SAICA problems: school behavior problems (severe); spare time activities (mild); spare time problems (moderate); activities with peers (severe); problems with peers (moderate); relationship with siblings (moderate); problems with siblings (severe); relationship with father (mild); problems with parents (moderate). Lifetime diagnoses/history: received extra help in school (full); simple phobia (subthreshold); separation anxiety disorder (full); attention-deficit/hyperactivity disorder (subthreshold). Family history: being placed in a special class; receiving extra help in school; major depressive disorder; overanxious disorder; attention-deficit/hyperactivity disorder. Cognitive (WISC scaled, mean=10): WISC-III Block Design high (14); WAIS-III/WISC-IV Digit Span low (7); WAIS-III/WISC-IV Digit Symbol high (15); WASI Vocabulary high (14).
```

This 9-year-old also shows marked baseline psychopathology — clinical-range internalizing, anxious/depressed, and withdrawn scores, severe SAICA problems, and lifetime separation anxiety — illustrating why the task is hard: superficially this profile looks as severe as Example 1. The distinguishing features are the ABSENCE of a bipolar family history and of clinical-range thought problems, and this adolescent did not convert. Discriminating the two requires weighing the specific bipolar-risk indicators, not overall symptom load.
