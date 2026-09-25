#!/usr/bin/env python3
"""
Entry point.

  # quick score estimate on a train holdout (know your F_0.5 before uploading)
  python run.py dev --sample 20000

  # full pipeline -> writes output/matching_results.tsv + output/candidate_pairs.tsv
  python run.py full --train-sample 200000

Env toggles (see src/config.py):
  USE_EMBEDDINGS=0        disable LaBSE (low-RAM / CPU-only)
  USE_CROSS_ENCODER=0     disable deep re-scoring
  DATA_ROOT=../..//dataset  point at dataset dir
  EMBED_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2  lighter model
"""
import argparse
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import run_dev, run_full


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dev"); d.add_argument("--sample", type=int, default=20000)
    d.add_argument("--sample-right", type=int, default=None)
    f = sub.add_parser("full"); f.add_argument("--train-sample", type=int, default=200000)
    a = ap.parse_args()
    if a.cmd == "dev":
        run_dev(a.sample, a.sample_right)
    else:
        run_full(a.train_sample)


if __name__ == "__main__":
    main()
