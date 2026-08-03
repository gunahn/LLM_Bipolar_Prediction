#!/usr/bin/env python
"""
Run this ON THE LOGIN NODE (which has internet). Downloads MentalLLaMA weights into
HF_HOME so the compute node — which has NO internet — can load them offline with
HF_HUB_OFFLINE=1.

  export HF_HOME=$HOME/hf_cache
  python prefetch_weights.py --model klyang/MentaLLaMA-chat-7B

7B repo ~27 GB (fp32 stored), 13B larger. Use the 7B unless you specifically want the 13B.
"""
import argparse
from huggingface_hub import snapshot_download

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="klyang/MentaLLaMA-chat-7B")
    args = ap.parse_args()
    path = snapshot_download(repo_id=args.model)
    print("downloaded to:", path)
