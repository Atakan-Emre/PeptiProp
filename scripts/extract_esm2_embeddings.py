"""
Extract per-residue ESM-2 embeddings for all unique chains in the canonical dataset.

Outputs one NPZ file per unique sequence hash, stored under data/embeddings/.
A lookup JSON maps (complex_id, chain_id) → embedding file path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "SEC": "U", "PYL": "O",
}

ESM2_MAX_TOKENS = 1022


def three_letter_seq_to_one(seq_3letter: str) -> str:
    """Convert 3-letter AA sequence (GLYSERHIS...) to 1-letter (GSH...)."""
    tokens: List[str] = []
    i = 0
    while i < len(seq_3letter):
        tri = seq_3letter[i : i + 3]
        aa = THREE_TO_ONE.get(tri, "X")
        tokens.append(aa)
        i += 3
    return "".join(tokens)


def seq_hash(seq: str) -> str:
    return hashlib.md5(seq.encode()).hexdigest()[:12]


def load_esm2_model(model_name: str, device: torch.device):
    import esm

    model_loaders = {
        "esm2_t6_8M": esm.pretrained.esm2_t6_8M_UR50D,
        "esm2_t12_35M": esm.pretrained.esm2_t12_35M_UR50D,
        "esm2_t33_650M": esm.pretrained.esm2_t33_650M_UR50D,
    }
    loader = model_loaders.get(model_name)
    if loader is None:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(model_loaders)}")

    model, alphabet = loader()
    model = model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()
    return model, alphabet, batch_converter


def extract_embedding(
    model,
    batch_converter,
    sequence: str,
    device: torch.device,
    repr_layer: int,
) -> np.ndarray:
    """Extract per-residue embedding for a single sequence.

    For sequences longer than ESM2_MAX_TOKENS, uses a sliding window approach
    and averages overlapping regions.
    """
    if len(sequence) <= ESM2_MAX_TOKENS:
        data = [("seq", sequence)]
        _, _, batch_tokens = batch_converter(data)
        batch_tokens = batch_tokens.to(device)
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[repr_layer], return_contacts=False)
        # shape: (1, seq_len+2, embed_dim) — remove BOS/EOS
        emb = results["representations"][repr_layer][0, 1 : len(sequence) + 1].cpu().numpy()
        return emb.astype(np.float16)

    # Sliding window for long sequences
    window = ESM2_MAX_TOKENS
    stride = window // 2
    embed_dim = None
    accum = None
    counts = None

    for start in range(0, len(sequence), stride):
        end = min(start + window, len(sequence))
        chunk = sequence[start:end]
        if len(chunk) < 10:
            break

        data = [("seq", chunk)]
        _, _, batch_tokens = batch_converter(data)
        batch_tokens = batch_tokens.to(device)
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[repr_layer], return_contacts=False)
        chunk_emb = results["representations"][repr_layer][0, 1 : len(chunk) + 1].cpu().numpy()

        if accum is None:
            embed_dim = chunk_emb.shape[1]
            accum = np.zeros((len(sequence), embed_dim), dtype=np.float32)
            counts = np.zeros(len(sequence), dtype=np.float32)

        accum[start : start + len(chunk)] += chunk_emb
        counts[start : start + len(chunk)] += 1.0

        if end >= len(sequence):
            break

    mask = counts > 0
    accum[mask] /= counts[mask, None]
    return accum.astype(np.float16)


def main():
    parser = argparse.ArgumentParser(description="Extract ESM-2 per-residue embeddings")
    parser.add_argument("--canonical-dir", type=Path, default=Path("data/canonical"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/embeddings"))
    parser.add_argument("--model", default="esm2_t6_8M", choices=["esm2_t6_8M", "esm2_t12_35M", "esm2_t33_650M"])
    parser.add_argument("--device", default="auto", help="cpu, mps, cuda, or auto")
    parser.add_argument("--max-chains", type=int, default=None, help="Limit for smoke testing")
    args = parser.parse_args()

    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    chains = pd.read_parquet(args.canonical_dir / "chains.parquet")
    print(f"Toplam zincir: {len(chains)}")

    # Convert sequences and deduplicate
    chains["seq_1letter"] = chains["sequence"].apply(three_letter_seq_to_one)
    chains["seq_hash"] = chains["seq_1letter"].apply(seq_hash)

    unique_seqs: Dict[str, str] = {}
    for _, row in chains.iterrows():
        h = row["seq_hash"]
        if h not in unique_seqs:
            unique_seqs[h] = row["seq_1letter"]
    print(f"Benzersiz sekans: {len(unique_seqs)}")

    if args.max_chains:
        keys = list(unique_seqs.keys())[: args.max_chains]
        unique_seqs = {k: unique_seqs[k] for k in keys}
        print(f"Smoke test: ilk {args.max_chains} sekans")

    emb_dir = args.output_dir / "esm2_residue"
    emb_dir.mkdir(parents=True, exist_ok=True)

    repr_layers = {"esm2_t6_8M": 6, "esm2_t12_35M": 12, "esm2_t33_650M": 33}
    repr_layer = repr_layers[args.model]

    print(f"ESM-2 model yükleniyor: {args.model} ...")
    model, alphabet, batch_converter = load_esm2_model(args.model, device)
    print("Model hazır.")

    t0 = time.time()
    for i, (h, seq) in enumerate(unique_seqs.items()):
        out_path = emb_dir / f"{h}.npz"
        if out_path.exists():
            continue
        emb = extract_embedding(model, batch_converter, seq, device, repr_layer)
        np.savez_compressed(out_path, embedding=emb)
        if (i + 1) % 100 == 0 or (i + 1) == len(unique_seqs):
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"  [{i+1}/{len(unique_seqs)}] {rate:.1f} seq/s  ({elapsed:.0f}s)")

    # Build lookup: (complex_id, chain_id) → embedding file
    lookup: Dict[str, str] = {}
    for _, row in chains.iterrows():
        key = f"{row['complex_id']}::{row['chain_id_auth']}"
        lookup[key] = f"{row['seq_hash']}.npz"

    if args.max_chains:
        lookup = {k: v for k, v in lookup.items() if v.replace(".npz", "") in unique_seqs}

    lookup_path = args.output_dir / "esm2_chain_lookup.json"
    with open(lookup_path, "w") as f:
        json.dump(lookup, f, indent=2)

    print(f"\nEmbedding'ler: {emb_dir}")
    print(f"Lookup: {lookup_path}")
    print(f"Toplam süre: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
