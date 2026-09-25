"""
Layer 1: SOURCE -> DB (ingest).
Load raw TSVs, tag provenance by id prefix, build canonical record frames + normalized fields.
Memory-safe: chunked reads; work per-country downstream.
"""
from __future__ import annotations
import pandas as pd
from tqdm import tqdm

from .config import CFG
from .normalize import norm_name, norm_address

COLS = ["entity_id", "business_name", "business_address", "country"]


def read_tsv(path: str) -> pd.DataFrame:
    """Robust TSV read: explicit tab, everything string, no NaN coercion."""
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def source_of(entity_id: str) -> str:
    return entity_id.split("-", 1)[0]        # S1 / S2 / S3


def add_normalized(df: pd.DataFrame) -> pd.DataFrame:
    """Attach per-field normalized signals. Vectorized-ish via apply (parallelize on GPU box if needed)."""
    names = df["business_name"].map(norm_name)
    df["name_core"] = [n["name_core"] for n in names]
    df["name_sorted"] = [n["name_sorted"] for n in names]
    df["name_acronym"] = [n["name_acronym"] for n in names]
    df["name_suffix"] = [n["name_suffix"] for n in names]
    df["name_nospace"] = [n["name_nospace"] for n in names]
    df["name_type"] = [n["name_type"] for n in names]
    # address needs country for state canon
    addrs = [norm_address(a, c) for a, c in zip(df["business_address"], df["country"])]
    df["addr_norm"] = [a["addr_norm"] for a in addrs]
    df["addr_pin"] = [a["addr_pin"] for a in addrs]
    df["addr_nums"] = [a["addr_nums"] for a in addrs]
    df["addr_tokens"] = [a["addr_tokens"] for a in addrs]
    df["blk_text"] = (df["name_core"] + " " + df["addr_norm"]).str.strip()
    return df


def load_side(kind: str, split: str) -> pd.DataFrame:
    """
    kind: 'source1' | 'right'  (right = source2 UNION source3)
    split: 'train' | 'test'
    """
    if kind == "source1":
        df = read_tsv(CFG.p(split, f"{split}_source1.tsv"))
    else:
        d2 = read_tsv(CFG.p(split, f"{split}_source2.tsv"))
        d3 = read_tsv(CFG.p(split, f"{split}_source3.tsv"))
        df = pd.concat([d2, d3], ignore_index=True)
    df["source"] = df["entity_id"].map(source_of)
    return df


def load_ground_truth(split: str = "train") -> dict:
    """s1_id -> set(matched right ids)."""
    gt = read_tsv(CFG.p(split, f"{split}_ground_truth.tsv"))
    out = {}
    for sid, ids in zip(gt["source1_entity_id"], gt["matched_entity_ids"]):
        out[sid] = set(x for x in ids.split(",") if x) if ids else set()
    return out
