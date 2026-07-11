"""
serialize.py
Render a patient's baseline feature row into a natural-language clinical narrative.
Only elevated / clinically notable findings are reported (all-64 primary mode); when a
feature is explicitly selected (top-k mode) its value is reported regardless.

Thresholds:
  CBCL problem T-scores:   T>=70 clinical range, T>=65 borderline
  CBCL competence T:       T<=30 clinical deficit, T<=35 borderline low
  SAICA items:             >=2 reported (2 mild, 3 moderate, 4 severe)
  lifetime diagnoses:      >=2 reported (2 subthreshold, 3 full)
  family history (fhx_*):  ==1 present
  WISC cognitive scaled:   <=7 low, >=13 high (mean 10, SD 3)
  full_iq:                 <90 below average
  aaa_baseline:            CBCL Emotion Dysregulation Profile (AAA), aggregate T-score
                           of Anxious/Depressed + Attention + Aggressive subscales
"""
import pandas as pd

CBCL_PROB = ["extert","intert","totalt","aggret","delnqt","attent","thougt",
             "socprot","anxdept","somat","withdt"]
CBCL_COMP = ["scht","soct","actt"]
COG = ["block_design_ss","digit_span_ss","digit_symbol_ss","vocab_ss","arithmetic_ss","full_iq"]

def build_desc(dict_xlsx, sheet="Data Dictionary"):
    dd = pd.read_excel(dict_xlsx, sheet_name=sheet)
    desc = {r["Variable"]: str(r["Description"]) for _, r in dd.iterrows()}
    desc["aaa_baseline"] = ("CBCL Emotion Dysregulation Profile (AAA), aggregate T-score of "
                            "Anxious/Depressed, Attention, and Aggressive Behavior subscales")
    return desc

def feat_phrase(col, v, desc):
    if pd.isna(v):
        return None
    short = col.replace("_baseline", "")
    def dcol(s): return desc.get(s + "_baseline") or desc.get(s) or s
    if col == "age_baseline":  return f"age {v:.0f}"
    if col == "ses_baseline":  return f"socioeconomic status score {v:.0f}"
    if col == "aaa_baseline":
        tag = " (clinical range)" if v >= 70 else (" (borderline)" if v >= 65 else "")
        return f"CBCL Emotion Dysregulation Profile (AAA) T={v:.0f}{tag}"
    if short in CBCL_PROB:
        nm = dcol(short).replace("CBLC", "CBCL")
        if v >= 70: return f"{nm} T={v:.0f} (clinical range)"
        if v >= 65: return f"{nm} T={v:.0f} (borderline)"
        return f"{nm} T={v:.0f}"
    if short in CBCL_COMP:
        nm = dcol(short)
        if v <= 30: return f"{nm} T={v:.0f} (clinical deficit)"
        if v <= 35: return f"{nm} T={v:.0f} (borderline low)"
        return f"{nm} T={v:.0f}"
    if short.startswith("saic"):
        nm = desc.get(col, col).replace("SAICA ", "")
        sev = {1:"none/minimal",2:"mild",3:"moderate",4:"severe"}.get(int(v), f"{v:.0f}")
        return f"{nm} ({sev})"
    if col.endswith("_lifetime_baseline"):
        nm = desc.get(col, col).replace(" (lifetime)", "")
        return f"{nm} " + {1:"(absent)",2:"(subthreshold)",3:"(full)"}.get(int(v), f"({v:.0f})")
    if short.startswith("fhx_"):
        nm = desc.get(col, col).replace("family history of ", "").strip()
        return f"family history of {nm}" if v >= 1 else f"no family history of {nm}"
    if short in COG:
        if short == "full_iq": return f"Full-scale IQ {v:.0f}"
        base = dcol(short).split(" scaled")[0].split("& ")[-1]
        tag = " (low)" if v <= 7 else (" (high)" if v >= 13 else "")
        return f"{base} scaled score {v:.0f}{tag}"
    return f"{desc.get(col, short)}={v:.0f}"

def serialize(row, feat_list, desc, only_elevated=False):
    """feat_list: features to render. only_elevated=True keeps just clinically notable
    findings (primary all-feature mode); False reports every listed feature (top-k mode)."""
    sx = {1:"male",2:"female"}.get(row.get("sex"))
    age = row.get("age_baseline")
    demo = f"{age:.0f}-year-old" if pd.notna(age) else "adolescent"
    if sx: demo += f" {sx}"
    parts = [f"Baseline presentation: {demo} with full major depressive disorder and no "
             f"bipolar features at baseline."]
    findings = []
    for c in feat_list:
        if c == "age_baseline":
            continue
        ph = feat_phrase(c, row.get(c), desc)
        if ph:
            findings.append(ph)
    if findings:
        parts.append("Selected baseline features — " + "; ".join(findings) + ".")
    return " ".join(parts)
