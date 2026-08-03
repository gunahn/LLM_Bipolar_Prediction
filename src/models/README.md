# Additional model baselines

Standalone scripts for the non-Claude model baselines in the paper. Each reads the
generated cohort matrix / narratives and writes a metrics file into `results/`.

| Script | Model | Notes |
|---|---|---|
| `05_tabfm.py` | TabFM tabular foundation model | Same fixed family-grouped 5-fold split, same 64 features as the other supervised models. Point `predict_tabfm()` at your installed TabFM/TabPFN call. |
| `06_mentalroberta.py` | MentalRoBERTa (fine-tuned text encoder) | Fine-tunes on the serialized narratives under the same folds; run `mentalroberta_prefetch_weights.py` first on a GPU node. See `README_mentalroberta.md`. |
| `07_mentallama.py` | MentaLLaMA-7B (specialized open LLM, zero-shot) | Zero-shot classification over narratives; run `mentallama_prefetch_weights.py` first. ~27 GB weights; GPU required. See `README_mentallama.md`. |
| `08_gemini_zeroshot.py` | Google gemini-2.5-flash (cross-vendor zero-shot) | Same clinician-informed prevalence-free prompt + 3-class tool as Claude. Needs `GEMINI_API_KEY`; free tier has a daily cap. |

All model API/weights details (identifiers, hardware, prefetch) are in the per-model
README files. No key or weight is committed.
