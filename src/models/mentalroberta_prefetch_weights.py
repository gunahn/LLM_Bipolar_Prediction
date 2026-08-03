#!/usr/bin/env python
"""Run this ON THE LOGIN NODE (which has internet) before submitting the job.
Downloads mental/mental-roberta-base into $HF_HOME so the offline compute node
can load it. Mirrors the TabFM/TabPFN prefetch pattern used on this cluster."""
import os
os.environ.setdefault("HF_HOME", os.path.join(os.getcwd(), "hf_cache"))
from transformers import AutoTokenizer, AutoModelForSequenceClassification
m = "mental/mental-roberta-base"
print(f"[prefetch] HF_HOME={os.environ['HF_HOME']}  model={m}", flush=True)
AutoTokenizer.from_pretrained(m)
AutoModelForSequenceClassification.from_pretrained(m, num_labels=2)
print("[prefetch] done — weights cached, safe to submit with HF_HUB_OFFLINE=1", flush=True)
