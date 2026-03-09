#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from peptidquantum.dataio import parse_geppri_file


def contiguous_runs(labels: list[int], value: int = 1) -> list[tuple[int, int]]:
    runs = []
    i = 0
    n = len(labels)
    while i < n:
        if labels[i] != value:
            i += 1
            continue
        j = i
        while j < n and labels[j] == value:
            j += 1
        runs.append((i, j))
        i = j
    return runs


def non_binding_positions(labels: list[int]) -> list[int]:
    return [i for i, v in enumerate(labels) if v == 0]


def make_pairs_from_entry(
    seq: str,
    labels: list[int],
    pair_prefix: str,
    min_len: int,
    max_len: int,
    neg_per_pos: int,
    rng: random.Random,
) -> list[dict]:
    pairs = []
    pos_runs = contiguous_runs(labels, value=1)

    pair_idx = 0
    for s, e in pos_runs:
        run_len = e - s
        # If run is long, split into fixed windows; if short, keep as one peptide.
        if run_len <= max_len:
            segments = [(s, e)]
        else:
            segments = []
            w = max_len
            i = s
            while i < e:
                j = min(i + w, e)
                if j - i >= min_len:
                    segments.append((i, j))
                i = j

        for a, b in segments:
            pep = seq[a:b]
            if len(pep) < min_len:
                continue
            pair_idx += 1
            pairs.append(
                {
                    "pair_id": f"{pair_prefix}_p{pair_idx}",
                    "protein_name": pair_prefix,
                    "peptide_name": f"{pair_prefix}_pep_pos_{pair_idx}",
                    "protein_seq": seq,
                    "peptide_seq": pep,
                    "label": 1,
                    "source": "binding_segment",
                    "segment_start_0": a,
                    "segment_end_0": b,
                }
            )

            # Negative sampling: choose non-binding segments with similar length.
            zero_idx = non_binding_positions(labels)
            seg_len = b - a
            if len(zero_idx) < seg_len:
                continue

            for k in range(neg_per_pos):
                # try up to 50 attempts to find continuous non-binding window
                chosen = None
                for _ in range(50):
                    st = rng.randint(0, len(seq) - seg_len)
                    window = labels[st : st + seg_len]
                    if sum(window) == 0:
                        chosen = (st, st + seg_len)
                        break
                if chosen is None:
                    continue
                ns, ne = chosen
                npep = seq[ns:ne]
                pair_idx += 1
                pairs.append(
                    {
                        "pair_id": f"{pair_prefix}_n{pair_idx}_{k+1}",
                        "protein_name": pair_prefix,
                        "peptide_name": f"{pair_prefix}_pep_neg_{pair_idx}_{k+1}",
                        "protein_seq": seq,
                        "peptide_seq": npep,
                        "label": 0,
                        "source": "non_binding_segment",
                        "segment_start_0": ns,
                        "segment_end_0": ne,
                    }
                )

    return pairs


def build_pairs(
    geppri_file: str | Path,
    out_jsonl: str | Path,
    min_len: int,
    max_len: int,
    neg_per_pos: int,
    seed: int,
) -> dict:
    entries = parse_geppri_file(geppri_file)
    rng = random.Random(seed)

    all_pairs = []
    for i, item in enumerate(entries, start=1):
        prefix = f"prot_{i}"
        pairs = make_pairs_from_entry(
            seq=item.sequence,
            labels=item.labels,
            pair_prefix=prefix,
            min_len=min_len,
            max_len=max_len,
            neg_per_pos=neg_per_pos,
            rng=rng,
        )
        all_pairs.extend(pairs)

    out = Path(out_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in all_pairs:
            f.write(json.dumps(r) + "\n")

    n_pos = sum(1 for x in all_pairs if x["label"] == 1)
    n_neg = sum(1 for x in all_pairs if x["label"] == 0)
    return {
        "entries": len(entries),
        "pairs": len(all_pairs),
        "pos": n_pos,
        "neg": n_neg,
        "out": str(out),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build peptide-protein pair dataset from GEPPRI residue labels")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--min-len", type=int, default=4)
    p.add_argument("--max-len", type=int, default=18)
    p.add_argument("--neg-per-pos", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    stats = build_pairs(
        geppri_file=args.input,
        out_jsonl=args.output,
        min_len=args.min_len,
        max_len=args.max_len,
        neg_per_pos=args.neg_per_pos,
        seed=args.seed,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
