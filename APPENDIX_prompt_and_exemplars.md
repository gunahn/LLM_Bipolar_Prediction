# Appendix: Few-Shot LLM Prompt and Exemplar Specification

This appendix documents the exact prompt, tool schema, and exemplar-construction procedure used for the few-shot large language model (LLM) experiments, so the protocol is fully reproducible.

## A.1 System prompt

The identical system prompt was sent with every request, for both models (`claude-sonnet-5`, `claude-haiku-4-5-20251001`) and at every value of *k*:

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

## A.2 Structured-output tool

Predictions were elicited through a forced tool call (`tool_choice` set to this tool), so every response returned a parseable probability distribution over the three ordered outcome classes plus a categorical prediction:

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

The binary conversion probability used for AUROC/AUPRC is `p_full` (renormalized so `p_none + p_subthreshold + p_full = 1`).

## A.3 Exemplar (few-shot example) format

For *k* > 0, the *k* labeled exemplars were concatenated ahead of the target patient. Each exemplar used the same natural-language clinical narrative as the target patient (see A.5), followed by its true 10-year outcome label. The exemplar block and target patient were sent as follows (this prefix block is marked for prompt caching so the shared exemplars are billed once per batch):

**Exemplar block (repeated for each of the *k* examples):**

```text
[Example j]
<patient clinical narrative>
Correct 10-year outcome: <None|Subthreshold|Full>
```

**Framing sentence before the target** (k>0):

```text
Here are <k> labeled example patients from the same population:
<exemplar blocks>

Now assess the following new patient.
```

For *k* = 0 (zero-shot) the framing sentence was simply `Assess the following patient.`

**Target patient block:**

```text
PATIENT:
<patient clinical narrative>

Call classify_patient with your prediction.
```

## A.4 Exemplar sampling at true prevalence

Exemplars were drawn from the training partition of each cross-validation fold only (never the test patient's fold), and sampled to reflect the true 3-class prevalence (None 72% / Subthreshold 9% / Full 19%) via largest-remainder allocation, with a fixed per-seed random seed. The resulting class composition at each *k*:

| k | None | Subthreshold | Full | total |
|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0 |
| 2 | 2 | 0 | 0 | 2 |
| 4 | 3 | 0 | 1 | 4 |
| 8 | 6 | 1 | 1 | 8 |
| 16 | 11 | 2 | 3 | 16 |
| 32 | 23 | 3 | 6 | 32 |
| 64 | 46 | 6 | 12 | 64 |

(Allocation shown for fold 0, seed 0; other folds/seeds differ only in which specific training patients are drawn, not the class counts.)

## A.5 Patient clinical narrative (serialization)

Each patient's baseline feature row was rendered into a natural-language narrative reporting only clinically notable findings (all-64-feature primary mode). Thresholds:

- CBCL problem T-scores: T ≥ 65 borderline, T ≥ 70 clinical range
- CBCL competence T-scores: T ≤ 35 borderline-low, T ≤ 30 clinical deficit
- WISC cognitive scaled scores: ≤ 7 below average, ≥ 13 above average (mean 10, SD 3)
- SAICA items: reported when ≥ 2 (mild/moderate/severe)
- Lifetime diagnoses: reported when subthreshold or full
- Family history flags: reported when present

The full serialization code is in `src/serialize.py`. Example rendered narrative:

```text
Baseline presentation: 8-year-old male. All features below are from the baseline assessment; the patient has full major depressive disorder and no bipolar features at baseline. Socioeconomic status score: 2. SAICA problems: school behavior problems (moderate); spare time activities (mild); spare time problems (mild); problems with peers (mild); problems with siblings (mild); problems with parents (mild). Lifetime diagnoses/history: social phobia (full); attention-deficit/hyperactivity disorder (full). Family history: repeating a grade; simple phobia.
```
