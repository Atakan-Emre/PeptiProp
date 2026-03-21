"""
Generate leakage-free protein-peptide candidate sets for interaction scoring.
Strategy: split-specific easy / hard / structure-aware hard negatives.
CRITICAL: Generate negatives AFTER split to prevent leakage.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import random
import json
from typing import List, Tuple, Set, Dict
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def load_data(canonical_dir: Path, splits_dir: Path):
    """Load canonical data and splits"""
    
    complexes = pd.read_parquet(canonical_dir / "complexes.parquet")
    chains = pd.read_parquet(canonical_dir / "chains.parquet")
    
    protein_sequences = (
        chains[chains["entity_type"] == "protein"][["complex_id", "sequence"]]
        .drop_duplicates("complex_id")
        .rename(columns={"sequence": "protein_sequence"})
    )
    peptide_sequences = (
        chains[chains["entity_type"] == "peptide"][["complex_id", "sequence"]]
        .drop_duplicates("complex_id")
        .rename(columns={"sequence": "peptide_sequence"})
    )
    complexes = complexes.merge(protein_sequences, on="complex_id", how="left")
    complexes = complexes.merge(peptide_sequences, on="complex_id", how="left")
    
    # Load splits
    with open(splits_dir / 'train_ids.txt') as f:
        train_ids = set(line.strip() for line in f if line.strip() and not line.startswith('#'))
    with open(splits_dir / 'val_ids.txt') as f:
        val_ids = set(line.strip() for line in f if line.strip() and not line.startswith('#'))
    with open(splits_dir / 'test_ids.txt') as f:
        test_ids = set(line.strip() for line in f if line.strip() and not line.startswith('#'))
    
    return complexes, chains, train_ids, val_ids, test_ids


def get_positive_pairs(complexes: pd.DataFrame, split_ids: Set[str]) -> pd.DataFrame:
    """Extract positive pairs from complexes"""
    
    split_complexes = complexes[complexes['complex_id'].isin(split_ids)].copy()
    
    pairs = []
    for _, row in split_complexes.iterrows():
        pairs.append({
            'pair_id': f"{row['complex_id']}_pos",
            'protein_complex_id': row['complex_id'],
            'protein_chain_id': row['protein_chain_id'],
            'peptide_complex_id': row['complex_id'],
            'peptide_chain_id': row['peptide_chain_id'],
            'label': 1,
            'negative_type': 'positive',
            'source_db': row['source_db'],
            'pdb_id': row['pdb_id'],
            'peptide_length': row['peptide_length'],
            'protein_length': row['protein_length'],
            'protein_quality_flag': row['quality_flag'],
            'peptide_quality_flag': row['quality_flag'],
            'pair_quality_flag': row['quality_flag'],
        })
    
    return pd.DataFrame(pairs)


def pair_quality_flag(protein_quality: str, peptide_quality: str) -> str:
    """Derive pair quality from protein and peptide quality flags."""
    if protein_quality == "clean" and peptide_quality == "clean":
        return "clean"
    if protein_quality == "quarantine" or peptide_quality == "quarantine":
        return "quarantine"
    return "warning"


def make_pair_key(protein_complex: dict, peptide_complex: dict) -> Tuple[str, str, str, str]:
    """Build an exact pair key using both complex ids and chain ids."""
    return (
        protein_complex["complex_id"],
        protein_complex["protein_chain_id"],
        peptide_complex["complex_id"],
        peptide_complex["peptide_chain_id"],
    )


def build_negative_record(
    protein_complex: dict,
    peptide_complex: dict,
    negative_type: str,
    index: int,
    split_name: str = "",
) -> dict:
    """Build a negative pair record."""
    protein_quality = protein_complex["quality_flag"]
    peptide_quality = peptide_complex["quality_flag"]
    split_prefix = f"{split_name}_" if split_name else ""
    return {
        "pair_id": (
            f"{split_prefix}neg_{negative_type}_{index}_"
            f"{protein_complex['complex_id']}_{peptide_complex['complex_id']}"
        ),
        "protein_complex_id": protein_complex["complex_id"],
        "protein_chain_id": protein_complex["protein_chain_id"],
        "peptide_complex_id": peptide_complex["complex_id"],
        "peptide_chain_id": peptide_complex["peptide_chain_id"],
        "label": 0,
        "negative_type": negative_type,
        "source_db": protein_complex["source_db"],
        "pdb_id": protein_complex["pdb_id"],
        "peptide_length": peptide_complex["peptide_length"],
        "protein_length": protein_complex["protein_length"],
        "protein_quality_flag": protein_quality,
        "peptide_quality_flag": peptide_quality,
        "pair_quality_flag": pair_quality_flag(protein_quality, peptide_quality),
    }


def same_length_band(protein_complex: dict, peptide_complex: dict) -> bool:
    """Keep core peptides with core, extension peptides with extension."""
    return (protein_complex["peptide_length"] <= 30) == (peptide_complex["peptide_length"] <= 30)


def build_sampling_context(split_complexes: List[dict]) -> Dict[str, object]:
    """Build lightweight sampling context for scalable negative generation."""
    protein_groups = defaultdict(list)
    peptide_groups = defaultdict(list)
    pdb_groups = defaultdict(list)
    core_rows = []
    extension_rows = []

    for row in split_complexes:
        protein_groups[row["protein_sequence"]].append(row)
        peptide_groups[row["peptide_sequence"]].append(row)
        pdb_groups[row["pdb_id"]].append(row)
        if row["peptide_length"] <= 30:
            core_rows.append(row)
        else:
            extension_rows.append(row)

    return {
        "all_rows": split_complexes,
        "rows_by_band": {
            "core": core_rows,
            "extension": extension_rows,
        },
        "protein_groups": [group for group in protein_groups.values() if len(group) >= 2],
        "peptide_groups": [group for group in peptide_groups.values() if len(group) >= 2],
        "pdb_groups": [group for group in pdb_groups.values() if len(group) >= 2],
    }


def build_candidate_sampling_context(split_complexes: List[dict]) -> Dict[str, object]:
    """Build peptide-centric candidate pools for per-protein candidate set generation."""
    peptides_by_band = {"core": [], "extension": []}
    peptides_by_pdb = defaultdict(list)
    peptides_by_protein_sequence = defaultdict(list)

    for row in split_complexes:
        band = "core" if row["peptide_length"] <= 30 else "extension"
        peptides_by_band[band].append(row)
        peptides_by_pdb[row["pdb_id"]].append(row)
        peptides_by_protein_sequence[row["protein_sequence"]].append(row)

    return {
        "all_rows": split_complexes,
        "peptides_by_band": peptides_by_band,
        "peptides_by_pdb": peptides_by_pdb,
        "peptides_by_protein_sequence": peptides_by_protein_sequence,
    }


def build_negative_type_plan(negatives_per_protein: int, ratios: Dict[str, float], rng: random.Random) -> List[str]:
    """Create a per-protein negative type plan using requested ratios."""
    counts = allocate_target_counts(negatives_per_protein, ratios)
    plan = []
    for neg_type, count in counts.items():
        plan.extend([neg_type] * int(count))
    rng.shuffle(plan)
    return plan


def sample_negative_peptide_for_protein(
    protein_complex: dict,
    neg_type: str,
    context: Dict[str, object],
    rng: random.Random,
    used_keys: Set[Tuple[str, str, str, str]],
    positive_keys: Set[Tuple[str, str, str, str]],
    max_attempts: int = 200,
):
    """Sample one leakage-safe negative peptide candidate for a fixed protein."""
    band = "core" if protein_complex["peptide_length"] <= 30 else "extension"

    for _ in range(max_attempts):
        if neg_type == "easy":
            candidates = context["peptides_by_band"][band]
            if not candidates:
                return None
            peptide_complex = rng.choice(candidates)
            if peptide_complex["complex_id"] == protein_complex["complex_id"]:
                continue
            if peptide_complex["pdb_id"] == protein_complex["pdb_id"]:
                continue
        elif neg_type == "hard":
            candidates = context["peptides_by_protein_sequence"].get(protein_complex["protein_sequence"], [])
            if len(candidates) < 2:
                return None
            peptide_complex = rng.choice(candidates)
            if peptide_complex["complex_id"] == protein_complex["complex_id"]:
                continue
            if peptide_complex["peptide_sequence"] == protein_complex["peptide_sequence"]:
                continue
        elif neg_type == "structure_hard":
            candidates = context["peptides_by_pdb"].get(protein_complex["pdb_id"], [])
            if len(candidates) < 2:
                return None
            peptide_complex = rng.choice(candidates)
            if peptide_complex["complex_id"] == protein_complex["complex_id"]:
                continue
        else:
            return None

        if not same_length_band(protein_complex, peptide_complex):
            continue

        key = make_pair_key(protein_complex, peptide_complex)
        if key in positive_keys or key in used_keys:
            continue

        used_keys.add(key)
        return peptide_complex
    return None


def pick_length_matched_peptide(context: Dict[str, object], protein_complex: dict, rng: random.Random):
    """Pick a peptide candidate from the same length band."""
    band = "core" if protein_complex["peptide_length"] <= 30 else "extension"
    candidates = context["rows_by_band"][band]
    if not candidates:
        return None
    return rng.choice(candidates)


def sample_easy_negatives(
    context: Dict[str, object],
    target_count: int,
    rng: random.Random,
    positive_keys: Set[Tuple[str, str, str, str]],
    used_keys: Set[Tuple[str, str, str, str]],
) -> List[dict]:
    """Sample easy negatives without materializing the full O(n^2) pool."""
    rows = context["all_rows"]
    if target_count <= 0 or len(rows) < 2:
        return []

    sampled = []
    max_attempts = max(target_count * 50, 500)
    attempts = 0
    while len(sampled) < target_count and attempts < max_attempts:
        attempts += 1
        protein_complex = rng.choice(rows)
        peptide_complex = pick_length_matched_peptide(context, protein_complex, rng)
        if peptide_complex is None:
            continue
        if protein_complex["complex_id"] == peptide_complex["complex_id"]:
            continue
        if protein_complex["pdb_id"] == peptide_complex["pdb_id"]:
            continue
        key = make_pair_key(protein_complex, peptide_complex)
        if key in positive_keys or key in used_keys:
            continue
        sampled.append(build_negative_record(protein_complex, peptide_complex, "easy", len(sampled)))
        used_keys.add(key)
    return sampled


def sample_hard_negatives(
    context: Dict[str, object],
    target_count: int,
    rng: random.Random,
    positive_keys: Set[Tuple[str, str, str, str]],
    used_keys: Set[Tuple[str, str, str, str]],
) -> List[dict]:
    """Sample hard negatives from repeated sequence groups."""
    protein_groups = context["protein_groups"]
    peptide_groups = context["peptide_groups"]
    if target_count <= 0 or (not protein_groups and not peptide_groups):
        return []

    sampled = []
    max_attempts = max(target_count * 80, 1000)
    attempts = 0
    modes = []
    if protein_groups:
        modes.append("protein")
    if peptide_groups:
        modes.append("peptide")

    while len(sampled) < target_count and attempts < max_attempts:
        attempts += 1
        mode = rng.choice(modes)
        if mode == "protein":
            group = rng.choice(protein_groups)
            if len(group) < 2:
                continue
            protein_complex = rng.choice(group)
            peptide_complex = rng.choice(group)
            if protein_complex["complex_id"] == peptide_complex["complex_id"]:
                continue
            if protein_complex["peptide_sequence"] == peptide_complex["peptide_sequence"]:
                continue
        else:
            group = rng.choice(peptide_groups)
            if len(group) < 2:
                continue
            peptide_complex = rng.choice(group)
            protein_complex = rng.choice(group)
            if protein_complex["complex_id"] == peptide_complex["complex_id"]:
                continue
            if protein_complex["protein_sequence"] == peptide_complex["protein_sequence"]:
                continue

        if not same_length_band(protein_complex, peptide_complex):
            continue
        key = make_pair_key(protein_complex, peptide_complex)
        if key in positive_keys or key in used_keys:
            continue
        sampled.append(build_negative_record(protein_complex, peptide_complex, "hard", len(sampled)))
        used_keys.add(key)
    return sampled


def sample_structure_hard_negatives(
    context: Dict[str, object],
    target_count: int,
    rng: random.Random,
    positive_keys: Set[Tuple[str, str, str, str]],
    used_keys: Set[Tuple[str, str, str, str]],
) -> List[dict]:
    """Sample same-PDB but non-positive negatives."""
    pdb_groups = context["pdb_groups"]
    if target_count <= 0 or not pdb_groups:
        return []

    sampled = []
    max_attempts = max(target_count * 80, 1000)
    attempts = 0
    while len(sampled) < target_count and attempts < max_attempts:
        attempts += 1
        group = rng.choice(pdb_groups)
        if len(group) < 2:
            continue
        protein_complex = rng.choice(group)
        peptide_complex = rng.choice(group)
        if protein_complex["complex_id"] == peptide_complex["complex_id"]:
            continue
        if not same_length_band(protein_complex, peptide_complex):
            continue
        key = make_pair_key(protein_complex, peptide_complex)
        if key in positive_keys or key in used_keys:
            continue
        sampled.append(build_negative_record(protein_complex, peptide_complex, "structure_hard", len(sampled)))
        used_keys.add(key)
    return sampled


def allocate_target_counts(total_count: int, ratios: dict) -> dict:
    """Allocate integer target counts while preserving the requested ratio mix."""
    active_ratios = {
        neg_type: float(ratio)
        for neg_type, ratio in ratios.items()
        if ratio > 0
    }
    if total_count <= 0 or not active_ratios:
        return {neg_type: 0 for neg_type in ratios}

    ratio_sum = sum(active_ratios.values())
    normalized = {
        neg_type: ratio / ratio_sum
        for neg_type, ratio in active_ratios.items()
    }
    raw_targets = {
        neg_type: total_count * ratio
        for neg_type, ratio in normalized.items()
    }
    target_counts = {
        neg_type: int(np.floor(count))
        for neg_type, count in raw_targets.items()
    }

    assigned = sum(target_counts.values())
    while assigned < total_count:
        _, neg_type = max(
            (
                raw_targets[neg_type] - np.floor(raw_targets[neg_type]),
                neg_type,
            )
            for neg_type in raw_targets
        )
        target_counts[neg_type] += 1
        assigned += 1

    for neg_type in ratios:
        target_counts.setdefault(neg_type, 0)
    return target_counts


def count_negative_types(records: List[dict]) -> Dict[str, int]:
    """Count generated negatives by negative_type."""
    counts = {"easy": 0, "hard": 0, "structure_hard": 0}
    for record in records:
        neg_type = str(record.get("negative_type", ""))
        if neg_type in counts:
            counts[neg_type] += 1
    return counts


def pair_key_from_record(record: dict) -> Tuple[str, str, str, str]:
    """Build pair key tuple from a generated record."""
    return (
        record["protein_complex_id"],
        record["protein_chain_id"],
        record["peptide_complex_id"],
        record["peptide_chain_id"],
    )


def backfill_negative_type_mix(
    all_negatives: List[dict],
    split_complexes: List[dict],
    candidate_context: Dict[str, object],
    ratio_map: Dict[str, float],
    target_total_negatives: int,
    split_name: str,
    rng: random.Random,
    used_keys: Set[Tuple[str, str, str, str]],
    positive_keys: Set[Tuple[str, str, str, str]],
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Try to reduce ratio shortfall by replacing surplus-type negatives with deficit-type negatives.

    This keeps total negative count fixed and preserves split-local leakage constraints.
    """
    if not all_negatives or target_total_negatives <= 0:
        counts = count_negative_types(all_negatives)
        return counts, {key: 0 for key in counts}

    target_counts = allocate_target_counts(target_total_negatives, ratio_map)
    generated_counts = count_negative_types(all_negatives)
    protein_order = split_complexes.copy()

    # Iterate deficit types in deterministic priority: hard first (most scarce), then structure_hard, then easy.
    for target_type in ("hard", "structure_hard", "easy"):
        deficit = max(0, target_counts.get(target_type, 0) - generated_counts.get(target_type, 0))
        if deficit <= 0:
            continue

        stalled_rounds = 0
        while deficit > 0 and stalled_rounds < 3:
            changed_this_round = 0
            rng.shuffle(protein_order)

            for protein_complex in protein_order:
                if deficit <= 0:
                    break

                donor_types = [
                    neg_type
                    for neg_type in ("easy", "hard", "structure_hard")
                    if neg_type != target_type
                    and generated_counts.get(neg_type, 0) > target_counts.get(neg_type, 0)
                ]
                if not donor_types:
                    break

                donor_idx = None
                for idx, record in enumerate(all_negatives):
                    if (
                        record["protein_complex_id"] == protein_complex["complex_id"]
                        and record["negative_type"] in donor_types
                    ):
                        donor_idx = idx
                        break
                if donor_idx is None:
                    continue

                peptide_complex = sample_negative_peptide_for_protein(
                    protein_complex=protein_complex,
                    neg_type=target_type,
                    context=candidate_context,
                    rng=rng,
                    used_keys=used_keys,
                    positive_keys=positive_keys,
                )
                if peptide_complex is None:
                    continue

                old_record = all_negatives[donor_idx]
                old_key = pair_key_from_record(old_record)
                used_keys.discard(old_key)

                new_record = build_negative_record(
                    protein_complex,
                    peptide_complex,
                    target_type,
                    donor_idx,
                    split_name=split_name,
                )
                new_record["pair_id"] = old_record["pair_id"]
                all_negatives[donor_idx] = new_record

                generated_counts[target_type] += 1
                generated_counts[old_record["negative_type"]] -= 1
                deficit -= 1
                changed_this_round += 1

            if changed_this_round == 0:
                stalled_rounds += 1
            else:
                stalled_rounds = 0

    shortfall_counts = {
        neg_type: max(0, target_counts.get(neg_type, 0) - generated_counts.get(neg_type, 0))
        for neg_type in ("easy", "hard", "structure_hard")
    }
    return generated_counts, shortfall_counts


def summarize_pairs(df: pd.DataFrame):
    """Create concise report for a pair dataframe."""
    pair_columns = [
        "protein_complex_id",
        "protein_chain_id",
        "peptide_complex_id",
        "peptide_chain_id",
    ]
    return {
        "split_values": sorted(df["split"].dropna().astype(str).unique().tolist()) if "split" in df.columns else [],
        "split_column_consistent": (
            len(df["split"].dropna().astype(str).unique()) == 1
            if "split" in df.columns and len(df) > 0
            else False
        ),
        "total_pairs": int(len(df)),
        "positive_pairs": int((df["label"] == 1).sum()),
        "negative_pairs": int((df["label"] == 0).sum()),
        "duplicate_pair_count": int(df[pair_columns].duplicated().sum()),
        "negative_type_distribution": {
            str(k): int(v)
            for k, v in df["negative_type"].fillna("unknown").value_counts().to_dict().items()
        },
        "quality_flag_distribution": {
            str(k): int(v)
            for k, v in df["pair_quality_flag"].fillna("unknown").value_counts().to_dict().items()
        },
        "unique_protein_chains": int(df[["protein_complex_id", "protein_chain_id"]].drop_duplicates().shape[0]),
        "unique_peptide_chains": int(df[["peptide_complex_id", "peptide_chain_id"]].drop_duplicates().shape[0]),
    }


def summarize_candidate_sets(df: pd.DataFrame):
    """Summarize per-protein candidate set structure for reranking diagnostics."""
    pair_columns = [
        "protein_complex_id",
        "protein_chain_id",
        "peptide_complex_id",
        "peptide_chain_id",
    ]
    if df.empty:
        return {
            "num_proteins": 0,
            "avg_candidates_per_protein": 0.0,
            "min_candidates_per_protein": 0,
            "max_candidates_per_protein": 0,
            "candidate_count_distribution": {},
            "avg_negatives_per_protein": 0.0,
            "hard_negative_ratio": 0.0,
            "structure_hard_negative_ratio": 0.0,
            "duplicate_pair_count": 0,
            "split_values": [],
            "split_column_consistent": False,
        }

    grouped = (
        df.groupby(["protein_complex_id", "protein_chain_id"])
        .agg(
            candidates=("pair_id", "count"),
            positives=("label", "sum"),
            negatives=("label", lambda x: int((x == 0).sum())),
        )
        .reset_index()
    )
    candidate_counts = grouped["candidates"].astype(int)
    n_pos = int((df["label"] == 1).sum())
    negative_only = df[df["label"] == 0]
    n_neg = len(negative_only)
    n_hard = int((negative_only["negative_type"] == "hard").sum())
    n_structure_hard = int((negative_only["negative_type"] == "structure_hard").sum())

    return {
        "split_values": sorted(df["split"].dropna().astype(str).unique().tolist()) if "split" in df.columns else [],
        "split_column_consistent": (
            len(df["split"].dropna().astype(str).unique()) == 1
            if "split" in df.columns and len(df) > 0
            else False
        ),
        "total_pairs": int(len(df)),
        "positive_pairs": int(n_pos),
        "negative_pairs": int(n_neg),
        "positive_to_negative_ratio": float(n_pos / n_neg) if n_neg > 0 else 0.0,
        "num_proteins": int(len(grouped)),
        "avg_candidates_per_protein": float(candidate_counts.mean()),
        "min_candidates_per_protein": int(candidate_counts.min()),
        "max_candidates_per_protein": int(candidate_counts.max()),
        "candidate_count_distribution": {
            str(k): int(v)
            for k, v in candidate_counts.value_counts().sort_index().to_dict().items()
        },
        "avg_negatives_per_protein": float(grouped["negatives"].mean()),
        "hard_negative_ratio": float(n_hard / n_neg) if n_neg > 0 else 0.0,
        "structure_hard_negative_ratio": float(n_structure_hard / n_neg) if n_neg > 0 else 0.0,
        "duplicate_pair_count": int(df[pair_columns].duplicated().sum()),
    }


def sync_complex_split_tags(canonical_dir: Path, train_ids: Set[str], val_ids: Set[str], test_ids: Set[str]):
    """Sync split_tag in complexes.parquet with split files."""
    complexes_path = canonical_dir / "complexes.parquet"
    complexes = pd.read_parquet(complexes_path)
    split_map = {complex_id: "train" for complex_id in train_ids}
    split_map.update({complex_id: "val" for complex_id in val_ids})
    split_map.update({complex_id: "test" for complex_id in test_ids})
    complexes["split_tag"] = complexes["complex_id"].map(split_map).fillna(complexes["split_tag"])
    complexes.to_parquet(complexes_path, index=False)
    print("\nSynced complexes.parquet split_tag from split files.")


def generate_negatives_for_split(
    complexes: pd.DataFrame,
    split_ids: Set[str],
    split_name: str,
    negatives_per_protein: int,
    easy_ratio: float = 0.7,
    hard_ratio: float = 0.3,
    struct_ratio: float = 0.0,
    seed: int = 42
) -> Tuple[pd.DataFrame, dict]:
    """Generate negatives for a single split"""
    
    print(f"\nGenerating negatives for {split_name}...")
    
    # Get positive pairs
    positive_pairs = get_positive_pairs(complexes, split_ids)
    n_positives = len(positive_pairs)
    
    print(f"  Positives: {n_positives}")
    
    # Create positive key set
    positive_keys = set()
    for _, row in positive_pairs.iterrows():
        key = (
            row["protein_complex_id"],
            row["protein_chain_id"],
            row["peptide_complex_id"],
            row["peptide_chain_id"],
        )
        positive_keys.add(key)
    
    split_complexes = complexes[complexes["complex_id"].isin(split_ids)].to_dict("records")
    candidate_context = build_candidate_sampling_context(split_complexes)
    ratio_map = {
        "easy": easy_ratio,
        "hard": hard_ratio,
        "structure_hard": struct_ratio,
    }
    per_protein_type_targets = allocate_target_counts(negatives_per_protein, ratio_map)
    n_total_negatives_target = n_positives * negatives_per_protein

    print(f"  Target negatives per protein: {negatives_per_protein}")
    print(f"  Target total negatives: {n_total_negatives_target}")
    print(f"    Easy/protein: {per_protein_type_targets['easy']}")
    print(f"    Hard/protein: {per_protein_type_targets['hard']}")
    print(f"    Structure-hard/protein: {per_protein_type_targets['structure_hard']}")
    print(f"  Candidate context:")
    print(f"    Complex rows: {len(split_complexes)}")
    print(f"    Core peptide pool: {len(candidate_context['peptides_by_band']['core'])}")
    print(f"    Extension peptide pool: {len(candidate_context['peptides_by_band']['extension'])}")

    rng = random.Random(seed)
    used_keys = set(positive_keys)
    all_negatives = []
    generated_counts = {"easy": 0, "hard": 0, "structure_hard": 0}
    shortfall_counts = {"easy": 0, "hard": 0, "structure_hard": 0}

    for protein_complex in split_complexes:
        neg_type_plan = build_negative_type_plan(negatives_per_protein, ratio_map, rng)
        for neg_type in neg_type_plan:
            peptide_complex = sample_negative_peptide_for_protein(
                protein_complex=protein_complex,
                neg_type=neg_type,
                context=candidate_context,
                rng=rng,
                used_keys=used_keys,
                positive_keys=positive_keys,
            )
            if peptide_complex is None:
                fallback_order = ["hard", "structure_hard", "easy"]
                fallback_found = None
                for fallback_type in fallback_order:
                    if fallback_type == neg_type or ratio_map.get(fallback_type, 0.0) <= 0:
                        continue
                    fallback_candidate = sample_negative_peptide_for_protein(
                        protein_complex=protein_complex,
                        neg_type=fallback_type,
                        context=candidate_context,
                        rng=rng,
                        used_keys=used_keys,
                        positive_keys=positive_keys,
                    )
                    if fallback_candidate is not None:
                        fallback_found = (fallback_type, fallback_candidate)
                        break
                if fallback_found is None:
                    # Track local miss; final shortfall is recomputed after backfill.
                    shortfall_counts[neg_type] += 1
                    continue
                neg_type, peptide_complex = fallback_found

            generated_counts[neg_type] += 1
            all_negatives.append(
                build_negative_record(
                    protein_complex,
                    peptide_complex,
                    neg_type,
                    len(all_negatives),
                    split_name=split_name,
                )
            )

    # Ratio-aware backfill replacement to reduce hard/structure_hard shortfall without changing total negatives.
    generated_counts, shortfall_counts = backfill_negative_type_mix(
        all_negatives=all_negatives,
        split_complexes=split_complexes,
        candidate_context=candidate_context,
        ratio_map=ratio_map,
        target_total_negatives=n_total_negatives_target,
        split_name=split_name,
        rng=rng,
        used_keys=used_keys,
        positive_keys=positive_keys,
    )

    negative_df = pd.DataFrame(all_negatives)
    
    # Combine with positives
    all_pairs = pd.concat([positive_pairs, negative_df], ignore_index=True)
    all_pairs['split'] = split_name
    
    print(f"  Total pairs: {len(all_pairs)} (pos: {n_positives}, neg: {len(all_negatives)})")
    print(f"  Ratio: 1:{len(all_negatives)/max(n_positives, 1):.1f}")
    print(f"  Quality flags: {all_pairs['pair_quality_flag'].value_counts().to_dict()}")
    print(f"  Negative types: {all_pairs['negative_type'].value_counts().to_dict()}")
    print(f"  Generated negatives by type: {generated_counts}")
    if any(count > 0 for count in shortfall_counts.values()):
        print(f"  [WARN] Requested type shortfalls: {shortfall_counts}")
    
    candidate_summary = summarize_candidate_sets(all_pairs)
    candidate_summary["target_negatives_per_protein"] = int(negatives_per_protein)
    candidate_summary["target_negative_type_ratio"] = {
        "easy": float(easy_ratio),
        "hard": float(hard_ratio),
        "structure_hard": float(struct_ratio),
    }
    n_generated_neg = max(sum(generated_counts.values()), 1)
    candidate_summary["actual_negative_type_ratio"] = {
        neg_type: float(count / n_generated_neg)
        for neg_type, count in generated_counts.items()
    }
    candidate_summary["negative_type_shortfall"] = {
        neg_type: int(count)
        for neg_type, count in shortfall_counts.items()
    }
    candidate_summary["negative_type_ratio_shortfall"] = {
        neg_type: max(
            0.0,
            candidate_summary["target_negative_type_ratio"][neg_type]
            - candidate_summary["actual_negative_type_ratio"].get(neg_type, 0.0),
        )
        for neg_type in ("easy", "hard", "structure_hard")
    }
    ratio_shortfall = {
        neg_type: round(shortfall, 4)
        for neg_type, shortfall in candidate_summary["negative_type_ratio_shortfall"].items()
        if shortfall > 1e-6
    }
    if ratio_shortfall:
        print(
            "  [WARN] Negative type ratio target not met "
            f"(target -> actual gap): {ratio_shortfall}"
        )
    return all_pairs, candidate_summary


def main(
    canonical_dir: Path,
    splits_dir: Path,
    output_dir: Path,
    seed: int = 42,
    quality_filter: str = "clean",
    train_negatives_per_protein: int = 5,
    eval_negatives_per_protein: int = 5,
):
    """Main function to generate negative pairs"""
    
    print("="*60)
    print("Generating Negative Pairs for Interaction Scoring / Reranking")
    print("="*60)
    print("Strategy:")
    print("  - Train negatives: 70% easy / 30% hard / 0% structure_hard")
    print("  - Val/Test negatives: 70% easy / 30% hard / 0% structure_hard")
    print(
        f"  - Candidate set size: train=1+{train_negatives_per_protein}, "
        f"val/test=1+{eval_negatives_per_protein}"
    )
    print("="*60)
    
    # Load data
    complexes, chains, train_ids, val_ids, test_ids = load_data(canonical_dir, splits_dir)
    sync_complex_split_tags(canonical_dir, train_ids, val_ids, test_ids)

    if quality_filter:
        pre_filter_count = len(complexes)
        complexes = complexes[complexes["quality_flag"] == quality_filter].reset_index(drop=True)
        print(f"\nApplied quality_filter={quality_filter!r}: {pre_filter_count} -> {len(complexes)} complexes")
    
    print(f"\nLoaded {len(complexes)} complexes")
    print(f"  Train: {len(train_ids)}")
    print(f"  Val: {len(val_ids)}")
    print(f"  Test: {len(test_ids)}")
    
    # Generate negatives for each split (candidate reranking format)
    train_pairs, train_candidates = generate_negatives_for_split(
        complexes, train_ids, 'train',
        negatives_per_protein=train_negatives_per_protein,
        easy_ratio=0.7,
        hard_ratio=0.3,
        struct_ratio=0.0,
        seed=seed
    )
    
    val_pairs, val_candidates = generate_negatives_for_split(
        complexes, val_ids, 'val',
        negatives_per_protein=eval_negatives_per_protein,
        easy_ratio=0.7,
        hard_ratio=0.3,
        struct_ratio=0.0,
        seed=seed + 100
    )
    
    test_pairs, test_candidates = generate_negatives_for_split(
        complexes, test_ids, 'test',
        negatives_per_protein=eval_negatives_per_protein,
        easy_ratio=0.7,
        hard_ratio=0.3,
        struct_ratio=0.0,
        seed=seed + 200
    )
    
    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_pairs.to_parquet(output_dir / 'train_pairs.parquet', index=False)
    val_pairs.to_parquet(output_dir / 'val_pairs.parquet', index=False)
    test_pairs.to_parquet(output_dir / 'test_pairs.parquet', index=False)
    
    print("\n" + "="*60)
    print("Pair datasets saved:")
    print(f"  {output_dir / 'train_pairs.parquet'}")
    print(f"  {output_dir / 'val_pairs.parquet'}")
    print(f"  {output_dir / 'test_pairs.parquet'}")
    print("="*60)
    
    # Summary
    print("\nFinal Summary:")
    print(f"  Train: {len(train_pairs)} pairs")
    print(f"    Positive: {len(train_pairs[train_pairs['label']==1])}")
    print(f"    Negative: {len(train_pairs[train_pairs['label']==0])}")
    print(f"  Val: {len(val_pairs)} pairs")
    print(f"    Positive: {len(val_pairs[val_pairs['label']==1])}")
    print(f"    Negative: {len(val_pairs[val_pairs['label']==0])}")
    print(f"  Test: {len(test_pairs)} pairs")
    print(f"    Positive: {len(test_pairs[test_pairs['label']==1])}")
    print(f"    Negative: {len(test_pairs[test_pairs['label']==0])}")
    
    pair_report = {
        "train": summarize_pairs(train_pairs),
        "val": summarize_pairs(val_pairs),
        "test": summarize_pairs(test_pairs),
    }
    with open(output_dir / "pair_data_report.json", "w") as f:
        json.dump(pair_report, f, indent=2)

    candidate_report = {
        "train": train_candidates,
        "val": val_candidates,
        "test": test_candidates,
    }
    with open(output_dir / "candidate_set_report.json", "w") as f:
        json.dump(candidate_report, f, indent=2)
    
    # Check for leakage
    train_protein_chains = set(train_pairs['protein_complex_id'])
    val_protein_chains = set(val_pairs['protein_complex_id'])
    test_protein_chains = set(test_pairs['protein_complex_id'])
    
    assert len(train_protein_chains & val_protein_chains) == 0, "Leakage: train/val overlap"
    assert len(train_protein_chains & test_protein_chains) == 0, "Leakage: train/test overlap"
    assert len(val_protein_chains & test_protein_chains) == 0, "Leakage: val/test overlap"
    
    print("\nLeakage check: PASSED (no complex overlap between splits)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate negative pairs")
    parser.add_argument("--canonical", type=Path, required=True,
                       help="Canonical directory")
    parser.add_argument("--splits", type=Path, required=True,
                       help="Splits directory")
    parser.add_argument("--output", type=Path, required=True,
                       help="Output directory for pair datasets")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--quality-filter", type=str, default="clean",
                       help="Optional complex quality filter before negative generation (default: clean)")
    parser.add_argument("--train-negatives-per-protein", type=int, default=5,
                       help="Number of negatives per protein candidate set for train split")
    parser.add_argument("--eval-negatives-per-protein", type=int, default=5,
                       help="Number of negatives per protein candidate set for val/test splits")
    
    args = parser.parse_args()
    
    main(
        args.canonical,
        args.splits,
        args.output,
        args.seed,
        args.quality_filter,
        args.train_negatives_per_protein,
        args.eval_negatives_per_protein,
    )
