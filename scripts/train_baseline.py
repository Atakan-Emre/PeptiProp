"""
PeptidQuantum v0.1 Baseline Training Script
PDB-level structure-aware split
"""
import sys
from pathlib import Path
import yaml
import argparse
import json
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATConv, GATv2Conv, GINEConv, NNConv, global_mean_pool
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, matthews_corrcoef
from sklearn.metrics import precision_score, recall_score, roc_curve, precision_recall_curve
from sklearn.metrics import confusion_matrix, brier_score_loss
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# Amino acid vocabulary
AA_VOCAB = {
    'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4,
    'GLN': 5, 'GLU': 6, 'GLY': 7, 'HIS': 8, 'ILE': 9,
    'LEU': 10, 'LYS': 11, 'MET': 12, 'PHE': 13, 'PRO': 14,
    'SER': 15, 'THR': 16, 'TRP': 17, 'TYR': 18, 'VAL': 19,
    'UNK': 20  # Unknown
}

SS_VOCAB = {
    "helix": 0,
    "sheet": 1,
    "coil": 2,
    "unknown": 3,
}

HYDROPHOBIC_AA_IDS = [
    AA_VOCAB["ALA"],
    AA_VOCAB["VAL"],
    AA_VOCAB["ILE"],
    AA_VOCAB["LEU"],
    AA_VOCAB["MET"],
    AA_VOCAB["PHE"],
    AA_VOCAB["TRP"],
    AA_VOCAB["TYR"],
    AA_VOCAB["CYS"],
    AA_VOCAB["PRO"],
]
POLAR_AA_IDS = [
    AA_VOCAB["SER"],
    AA_VOCAB["THR"],
    AA_VOCAB["ASN"],
    AA_VOCAB["GLN"],
]
POSITIVE_AA_IDS = [
    AA_VOCAB["LYS"],
    AA_VOCAB["ARG"],
    AA_VOCAB["HIS"],
]
NEGATIVE_AA_IDS = [
    AA_VOCAB["ASP"],
    AA_VOCAB["GLU"],
]


def normalize_secondary_structure(value):
    """Map residue secondary structure annotations into coarse buckets."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "unknown"

    text = str(value).strip().lower()
    if not text or text == "none":
        return "unknown"

    if text in {"h", "g", "i", "helix", "alpha_helix", "310_helix", "pi_helix"}:
        return "helix"
    if text in {"e", "b", "sheet", "beta_sheet", "strand"}:
        return "sheet"
    if text in {"c", "coil", "loop", "turn", "bend", "s"}:
        return "coil"
    return "unknown"


def build_feature_spec(config):
    """Build node and edge feature layout from config flags."""
    feature_cfg = config["features"]
    node_dim = 0
    if feature_cfg.get("use_residue_type_onehot", True):
        node_dim += len(AA_VOCAB)
    if feature_cfg.get("use_position_index", True):
        node_dim += 1
    if feature_cfg.get("use_chain_flags", True):
        node_dim += 2
    if feature_cfg.get("use_interface_flag", True):
        node_dim += 1
    if feature_cfg.get("use_pocket_flag", True):
        node_dim += 1
    if feature_cfg.get("use_local_density", True):
        node_dim += 1
    if feature_cfg.get("use_secondary_structure", True):
        node_dim += len(SS_VOCAB)

    edge_dim = int(feature_cfg["edge_dim"])
    if edge_dim < 4:
        raise ValueError("features.edge_dim must be >= 4 for distance/sequential/spatial flags")

    return {
        "node_dim": node_dim,
        "edge_dim": edge_dim,
    }


CHAIN_SUMMARY_BASE_DIM = 11
CHAIN_SUMMARY_AA_COMPOSITION_DIM = len(AA_VOCAB)
CHAIN_SUMMARY_DIM = CHAIN_SUMMARY_BASE_DIM + CHAIN_SUMMARY_AA_COMPOSITION_DIM
PAIR_COMPATIBILITY_DIM = 3
PAIR_FEATURE_DIM = CHAIN_SUMMARY_DIM * 4 + PAIR_COMPATIBILITY_DIM


def load_split_id_sets(splits_dir):
    """Load expected complex ids for each split from split files."""
    split_id_sets = {}
    for split_name in ("train", "val", "test"):
        split_path = Path(splits_dir) / f"{split_name}_ids.txt"
        with open(split_path) as f:
            split_id_sets[split_name] = {
                line.strip()
                for line in f
                if line.strip() and not line.startswith("#")
            }
    return split_id_sets


def validate_split_metadata(canonical_dir, pairs_dir, splits_dir):
    """Fail fast if complexes, split files, and pair parquet metadata disagree."""
    complexes = pd.read_parquet(Path(canonical_dir) / "complexes.parquet")
    if "split_tag" not in complexes.columns:
        raise ValueError("complexes.parquet is missing split_tag; cannot verify split metadata")

    split_id_sets = load_split_id_sets(splits_dir)
    complex_split_map = complexes.set_index("complex_id")["split_tag"].astype(str).to_dict()

    for split_name, expected_ids in split_id_sets.items():
        tagged_ids = {
            complex_id
            for complex_id, split_tag in complex_split_map.items()
            if split_tag == split_name
        }
        if tagged_ids != expected_ids:
            missing = sorted(expected_ids - tagged_ids)[:5]
            extra = sorted(tagged_ids - expected_ids)[:5]
            raise ValueError(
                f"Split drift detected for {split_name}: complexes.parquet split_tag does not match split files "
                f"(missing={missing}, extra={extra})"
            )

    for split_name in ("train", "val", "test"):
        pair_path = Path(pairs_dir) / f"{split_name}_pairs.parquet"
        pairs = pd.read_parquet(pair_path)
        if "split" not in pairs.columns:
            raise ValueError(f"{pair_path} is missing split column")

        pair_splits = {
            str(value)
            for value in pairs["split"].dropna().astype(str).unique().tolist()
        }
        if pair_splits != {split_name}:
            raise ValueError(
                f"{pair_path} contains unexpected split values: {sorted(pair_splits)}"
            )

        for column_name in ("protein_complex_id", "peptide_complex_id"):
            observed_split_tags = {
                complex_split_map.get(complex_id)
                for complex_id in pairs[column_name].dropna().astype(str).unique().tolist()
            }
            if observed_split_tags != {split_name}:
                raise ValueError(
                    f"{pair_path} has {column_name} complexes outside split {split_name}: "
                    f"{sorted(str(tag) for tag in observed_split_tags)}"
                )


class PeptideProteinDataset(Dataset):
    """Dataset for peptide-protein interaction prediction with negative pairs"""
    
    def __init__(self, pairs_file, canonical_dir, config, split_name):
        self.canonical_dir = Path(canonical_dir)
        self.config = config
        self.split_name = split_name
        
        # Load pair dataset
        self.pairs = pd.read_parquet(pairs_file)
        self._align_split_column()
        
        # Load canonical data
        self.complexes = pd.read_parquet(self.canonical_dir / "complexes.parquet")
        self.chains = pd.read_parquet(self.canonical_dir / "chains.parquet")
        self.residues = pd.read_parquet(self.canonical_dir / "residues.parquet")
        self.chain_lookup = {
            (row["complex_id"], row["chain_id_auth"], row["entity_type"]): row
            for _, row in self.chains.iterrows()
        }
        self.residue_lookup = {
            key: group.reset_index(drop=True)
            for key, group in self.residues.groupby(["complex_id", "chain_id"])
        }
        self.complex_chain_index = (
            self.chains.groupby(["complex_id", "entity_type"])["chain_id_auth"]
            .apply(list)
            .to_dict()
        )
        self.graph_cache = {}
        self.chain_summary_cache = {}
        
        self._annotate_pair_quality()
        self._apply_quality_filter()
        self.full_pairs = self.pairs.copy().reset_index(drop=True)
        self.active_curriculum_stage = "full_dataset"
        self.summary = self._build_summary()
        self._log_summary()

    def _align_split_column(self):
        """Keep pair parquet split metadata aligned with the dataset split in use."""
        if "split" not in self.pairs.columns:
            self.pairs["split"] = self.split_name
            return

        observed_splits = {
            str(value)
            for value in self.pairs["split"].dropna().astype(str).unique().tolist()
        }
        if not observed_splits:
            self.pairs["split"] = self.split_name
            return

        if observed_splits != {self.split_name}:
            self.pairs = self.pairs[self.pairs["split"] == self.split_name].reset_index(drop=True)
            remaining_splits = {
                str(value)
                for value in self.pairs["split"].dropna().astype(str).unique().tolist()
            }
            if remaining_splits != {self.split_name}:
                raise ValueError(
                    f"{self.split_name} split parquet contains mismatched split values: {sorted(observed_splits)}"
                )
    
    def _annotate_pair_quality(self):
        """Attach per-complex and per-pair quality flags."""
        quality_lookup = self.complexes.set_index("complex_id")["quality_flag"]
        self.pairs["protein_quality_flag"] = self.pairs["protein_complex_id"].map(quality_lookup)
        self.pairs["peptide_quality_flag"] = self.pairs["peptide_complex_id"].map(quality_lookup)
        
        protein_quality = self.pairs["protein_quality_flag"].fillna("missing")
        peptide_quality = self.pairs["peptide_quality_flag"].fillna("missing")
        self.pairs["pair_quality_flag"] = np.where(
            (protein_quality == "clean") & (peptide_quality == "clean"),
            "clean",
            np.where(
                (protein_quality == "quarantine") | (peptide_quality == "quarantine"),
                "quarantine",
                "warning",
            ),
        )
    
    def _apply_quality_filter(self):
        """Filter pairs based on config quality filter."""
        quality_filter = self.config["data"].get("quality_filter")
        self.pre_filter_pair_count = len(self.pairs)
        
        if not quality_filter:
            return
        
        self.pairs = self.pairs[self.pairs["pair_quality_flag"] == quality_filter].reset_index(drop=True)
        if self.pairs.empty:
            raise ValueError(
                f"{self.split_name} split has no pairs after quality_filter={quality_filter!r}"
            )
    
    def _build_summary(self):
        """Build a concise split-level summary."""
        pair_columns = [
            "protein_complex_id",
            "protein_chain_id",
            "peptide_complex_id",
            "peptide_chain_id",
        ]
        split_values = (
            sorted(self.pairs["split"].dropna().astype(str).unique().tolist())
            if "split" in self.pairs.columns
            else []
        )
        return {
            "split": self.split_name,
            "curriculum_stage": self.active_curriculum_stage,
            "split_values": split_values,
            "split_column_consistent": split_values == [self.split_name],
            "total_pairs": int(len(self.pairs)),
            "positive_pairs": int((self.pairs["label"] == 1).sum()),
            "negative_pairs": int((self.pairs["label"] == 0).sum()),
            "duplicate_pair_count": int(self.pairs[pair_columns].duplicated().sum()),
            "negative_type_distribution": {
                str(k): int(v)
                for k, v in self.pairs["negative_type"].fillna("unknown").value_counts().to_dict().items()
            },
            "quality_flag_distribution": {
                str(k): int(v)
                for k, v in self.pairs["pair_quality_flag"].fillna("unknown").value_counts().to_dict().items()
            },
            "unique_protein_chains": int(
                self.pairs[["protein_complex_id", "protein_chain_id"]].drop_duplicates().shape[0]
            ),
            "unique_peptide_chains": int(
                self.pairs[["peptide_complex_id", "peptide_chain_id"]].drop_duplicates().shape[0]
            ),
        }

    def _log_summary(self):
        """Print split summary for sanity checking."""
        print(f"\n[{self.split_name}] pairs loaded from dataset")
        print(f"  Curriculum stage: {self.summary['curriculum_stage']}")
        print(f"  Before quality filter: {self.pre_filter_pair_count}")
        print(f"  After quality filter:  {self.summary['total_pairs']}")
        print(f"  Positive: {self.summary['positive_pairs']}")
        print(f"  Negative: {self.summary['negative_pairs']}")
        print(f"  Duplicate pairs: {self.summary['duplicate_pair_count']}")
        print(f"  Pair quality flags: {self.summary['quality_flag_distribution']}")
        print(f"  Negative types: {self.summary['negative_type_distribution']}")
        print(f"  Unique protein chains: {self.summary['unique_protein_chains']}")
        print(f"  Unique peptide chains: {self.summary['unique_peptide_chains']}")
    
    def reset_pairs(self):
        """Restore the full filtered pair dataset."""
        self.pairs = self.full_pairs.copy().reset_index(drop=True)
        self.active_curriculum_stage = "full_dataset"
        self.summary = self._build_summary()
    
    def get_labels(self):
        """Return active labels as a numpy array."""
        return self.pairs["label"].to_numpy(dtype=np.int64)
    
    def set_curriculum(self, epoch_number, curriculum_config):
        """Apply epoch-dependent negative curriculum on the active training pairs."""
        if self.split_name != "train":
            return
        
        stages = curriculum_config.get("stages", [])
        selected_stage = None
        for stage in stages:
            end_epoch = stage.get("end_epoch")
            if end_epoch is None or epoch_number <= end_epoch:
                selected_stage = stage
                break
        if selected_stage is None:
            self.reset_pairs()
            return
        
        positives = self.full_pairs[self.full_pairs["label"] == 1].copy()
        negative_pools = {
            neg_type: self.full_pairs[self.full_pairs["negative_type"] == neg_type].copy()
            for neg_type in ("easy", "hard", "structure_hard")
        }
        
        selected_negatives = select_curriculum_negatives(
            negative_pools=negative_pools,
            ratios=selected_stage["ratios"],
            target_total_negatives=int((self.full_pairs["label"] == 0).sum()),
            seed=curriculum_config.get("seed", 42) + epoch_number,
        )
        if selected_negatives.empty:
            raise ValueError(f"No negatives available for curriculum stage {selected_stage['name']}")
        
        self.pairs = (
            pd.concat([positives, selected_negatives], ignore_index=True)
            .sample(frac=1.0, random_state=curriculum_config.get("seed", 42) + epoch_number)
            .reset_index(drop=True)
        )
        self.active_curriculum_stage = selected_stage["name"]
        self.summary = self._build_summary()
        
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair_row = self.pairs.iloc[idx]
        
        # Get protein and peptide complex IDs
        protein_complex_id = pair_row['protein_complex_id']
        peptide_complex_id = pair_row['peptide_complex_id']
        protein_chain_id = pair_row['protein_chain_id']
        peptide_chain_id = pair_row['peptide_chain_id']
        
        # Build graphs
        protein_graph = self._get_graph(protein_complex_id, protein_chain_id, 'protein')
        peptide_graph = self._get_graph(peptide_complex_id, peptide_chain_id, 'peptide')
        
        # Label from pair dataset
        label = torch.tensor([float(pair_row['label'])], dtype=torch.float32)
        
        return {
            'pair_id': pair_row['pair_id'],
            'protein_group_id': f"{protein_complex_id}::{protein_chain_id}",
            'protein_graph': protein_graph,
            'peptide_graph': peptide_graph,
            'pair_features': self._build_pair_features(
                protein_complex_id,
                protein_chain_id,
                'protein',
                peptide_complex_id,
                peptide_chain_id,
                'peptide',
            ),
            'label': label
        }
    
    def _get_graph(self, complex_id, chain_id, entity_type):
        """Get cached graph for a complex/chain pair."""
        cache_key = (complex_id, chain_id, entity_type)
        if cache_key not in self.graph_cache:
            chain_key = (complex_id, chain_id, entity_type)
            if chain_key not in self.chain_lookup:
                raise KeyError(f"Missing chain record for {chain_key}")
            residue_key = (complex_id, chain_id)
            if residue_key not in self.residue_lookup:
                raise KeyError(f"Missing residues for {residue_key}")
            residues_df = self.residue_lookup[residue_key]
            self.graph_cache[cache_key] = self._build_graph(residues_df, entity_type)
            self.chain_summary_cache[cache_key] = self._build_chain_summary(residues_df, entity_type)
        return self.graph_cache[cache_key]

    def _get_chain_summary(self, complex_id, chain_id, entity_type):
        """Get cached per-chain summary features used for pair compatibility."""
        cache_key = (complex_id, chain_id, entity_type)
        if cache_key not in self.chain_summary_cache:
            self._get_graph(complex_id, chain_id, entity_type)
        return self.chain_summary_cache[cache_key]

    def _build_pair_features(
        self,
        protein_complex_id,
        protein_chain_id,
        protein_entity_type,
        peptide_complex_id,
        peptide_chain_id,
        peptide_entity_type,
    ):
        """Build pair-level summary features without changing the graph encoders."""
        protein_summary = self._get_chain_summary(
            protein_complex_id,
            protein_chain_id,
            protein_entity_type,
        )
        peptide_summary = self._get_chain_summary(
            peptide_complex_id,
            peptide_chain_id,
            peptide_entity_type,
        )
        summary_cosine = F.cosine_similarity(
            protein_summary.unsqueeze(0),
            peptide_summary.unsqueeze(0),
            dim=1,
            eps=1e-8,
        ).squeeze(0)
        summary_l2 = torch.norm(protein_summary - peptide_summary, p=2)
        summary_dot = torch.dot(protein_summary, peptide_summary)
        return torch.cat(
            [
                protein_summary,
                peptide_summary,
                torch.abs(protein_summary - peptide_summary),
                protein_summary * peptide_summary,
                torch.stack([summary_cosine, summary_l2, summary_dot]),
            ]
        )

    def _build_chain_summary(self, residues_df, chain_type):
        """Build compact chain-level summary features for pair matching."""
        residues_df = residues_df.reset_index(drop=True)
        n_residues = len(residues_df)
        feature_cfg = self.config["features"]
        positions = torch.as_tensor(
            residues_df[["x", "y", "z"]].to_numpy(dtype=np.float32),
            dtype=torch.float32,
        )
        density_threshold = feature_cfg.get(
            "local_density_radius",
            feature_cfg.get("distance_threshold", 8.0),
        )
        if feature_cfg.get("use_local_density", True):
            distance_matrix = torch.cdist(positions, positions)
            local_density = (distance_matrix <= density_threshold).sum(dim=1).float() - 1.0
            if n_residues > 1:
                local_density = local_density / float(n_residues - 1)
            else:
                local_density = torch.zeros(1, dtype=torch.float32)
        else:
            local_density = torch.zeros(1, dtype=torch.float32)

        if feature_cfg.get("use_secondary_structure", True):
            ss_labels = residues_df["secondary_structure"].apply(normalize_secondary_structure)
            helix_fraction = float((ss_labels == "helix").mean())
            sheet_fraction = float((ss_labels == "sheet").mean())
            coil_fraction = float((ss_labels == "coil").mean())
        else:
            helix_fraction = 0.0
            sheet_fraction = 0.0
            coil_fraction = 0.0

        length_scale = 500.0 if chain_type == "protein" else 50.0
        aa_counts = (
            residues_df["resname"]
            .map(lambda value: AA_VOCAB.get(value, AA_VOCAB["UNK"]))
            .value_counts(normalize=True)
            .reindex(range(len(AA_VOCAB)), fill_value=0.0)
            .to_numpy(dtype=np.float32)
        )
        hydrophobic_fraction = float(np.sum(aa_counts[HYDROPHOBIC_AA_IDS]))
        polar_fraction = float(np.sum(aa_counts[POLAR_AA_IDS]))
        positive_fraction = float(np.sum(aa_counts[POSITIVE_AA_IDS]))
        negative_fraction = float(np.sum(aa_counts[NEGATIVE_AA_IDS]))

        base_summary = torch.tensor(
            [
                min(n_residues / length_scale, 1.0),
                float(residues_df["is_interface"].fillna(False).astype(float).mean())
                if feature_cfg.get("use_interface_flag", True)
                else 0.0,
                float(residues_df["is_pocket"].fillna(False).astype(float).mean())
                if feature_cfg.get("use_pocket_flag", True)
                else 0.0,
                float(local_density.mean().item()) if feature_cfg.get("use_local_density", True) else 0.0,
                helix_fraction,
                sheet_fraction,
                coil_fraction,
                hydrophobic_fraction,
                polar_fraction,
                positive_fraction,
                negative_fraction,
            ],
            dtype=torch.float32,
        )
        return torch.cat(
            [
                base_summary,
                torch.as_tensor(aa_counts, dtype=torch.float32),
            ]
        )
    
    def _build_graph(self, residues_df, chain_type):
        """Build graph from residues"""
        residues_df = residues_df.reset_index(drop=True)
        n_residues = len(residues_df)
        feature_cfg = self.config["features"]
        feature_spec = build_feature_spec(self.config)

        positions = torch.as_tensor(
            residues_df[["x", "y", "z"]].to_numpy(dtype=np.float32),
            dtype=torch.float32,
        )

        density_threshold = feature_cfg.get("local_density_radius", feature_cfg.get("distance_threshold", 8.0))
        distance_matrix = torch.cdist(positions, positions)
        local_density = (distance_matrix <= density_threshold).sum(dim=1).float() - 1.0
        if n_residues > 1:
            local_density = local_density / float(n_residues - 1)
        else:
            local_density = torch.zeros(n_residues, dtype=torch.float32)

        node_parts = []
        if feature_cfg.get("use_residue_type_onehot", True):
            aa_indices = torch.as_tensor(
                residues_df["resname"].map(lambda value: AA_VOCAB.get(value, AA_VOCAB["UNK"])).to_numpy(dtype=np.int64),
                dtype=torch.long,
            )
            node_parts.append(F.one_hot(aa_indices, num_classes=len(AA_VOCAB)).to(torch.float32))

        if feature_cfg.get("use_position_index", True):
            if n_residues > 1:
                pos_norm = torch.linspace(0.0, 1.0, steps=n_residues, dtype=torch.float32).unsqueeze(1)
            else:
                pos_norm = torch.zeros((n_residues, 1), dtype=torch.float32)
            node_parts.append(pos_norm)

        if feature_cfg.get("use_chain_flags", True):
            chain_flags = torch.tensor(
                [1.0, 0.0] if chain_type == "protein" else [0.0, 1.0],
                dtype=torch.float32,
            ).repeat(n_residues, 1)
            node_parts.append(chain_flags)

        if feature_cfg.get("use_interface_flag", True):
            interface_values = torch.as_tensor(
                residues_df["is_interface"].fillna(False).to_numpy(dtype=np.float32),
                dtype=torch.float32,
            ).unsqueeze(1)
            node_parts.append(interface_values)

        if feature_cfg.get("use_pocket_flag", True):
            pocket_values = torch.as_tensor(
                residues_df["is_pocket"].fillna(False).to_numpy(dtype=np.float32),
                dtype=torch.float32,
            ).unsqueeze(1)
            node_parts.append(pocket_values)

        if feature_cfg.get("use_local_density", True):
            node_parts.append(local_density.unsqueeze(1))

        if feature_cfg.get("use_secondary_structure", True):
            ss_indices = torch.as_tensor(
                residues_df["secondary_structure"]
                .apply(normalize_secondary_structure)
                .map(SS_VOCAB)
                .to_numpy(dtype=np.int64),
                dtype=torch.long,
            )
            node_parts.append(F.one_hot(ss_indices, num_classes=len(SS_VOCAB)).to(torch.float32))

        node_features = torch.cat(node_parts, dim=1)

        # Build edges: sequential + spatial
        edge_dim = feature_spec["edge_dim"]
        edge_index_parts = []
        edge_attr_parts = []

        if n_residues > 1:
            seq_src = torch.arange(n_residues - 1, dtype=torch.long)
            seq_dst = seq_src + 1
            seq_pairs = torch.stack(
                [
                    torch.cat([seq_src, seq_dst]),
                    torch.cat([seq_dst, seq_src]),
                ]
            )
            seq_dist = torch.norm(positions[1:] - positions[:-1], dim=1).repeat_interleave(2)
            seq_attr = torch.zeros((seq_pairs.shape[1], edge_dim), dtype=torch.float32)
            seq_attr[:, 0] = seq_dist
            seq_attr[:, 1] = 1.0
            seq_attr[:, 2] = 1.0
            edge_index_parts.append(seq_pairs)
            edge_attr_parts.append(seq_attr)

        threshold = float(feature_cfg.get("distance_threshold", 8.0))
        spatial_mask = torch.triu(distance_matrix < threshold, diagonal=2)
        spatial_pairs = spatial_mask.nonzero(as_tuple=False)
        if spatial_pairs.numel() > 0:
            forward_pairs = spatial_pairs.t().contiguous()
            reverse_pairs = torch.stack([forward_pairs[1], forward_pairs[0]])
            spatial_index = torch.cat([forward_pairs, reverse_pairs], dim=1)
            spatial_dist = distance_matrix[spatial_pairs[:, 0], spatial_pairs[:, 1]].repeat_interleave(2)
            spatial_attr = torch.zeros((spatial_index.shape[1], edge_dim), dtype=torch.float32)
            spatial_attr[:, 0] = spatial_dist
            spatial_attr[:, 1] = 1.0
            spatial_attr[:, 3] = 1.0
            edge_index_parts.append(spatial_index)
            edge_attr_parts.append(spatial_attr)

        if edge_index_parts:
            edge_index = torch.cat(edge_index_parts, dim=1)
            edge_attr = torch.cat(edge_attr_parts, dim=0)
        else:
            edge_index = torch.arange(n_residues, dtype=torch.long).repeat(2, 1)
            edge_attr = torch.zeros((n_residues, edge_dim), dtype=torch.float32)
            edge_attr[:, 1] = 1.0
        
        # Normalize distances
        if self.config['features'].get('normalize_distances', True):
            edge_attr[:, 0] = edge_attr[:, 0] / 10.0  # Normalize by 10 Angstroms
        
        return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, pos=positions)


class GraphEncoder(nn.Module):
    """Configurable graph encoder supporting MPNN, GATv2, and GIN-style variants."""

    def __init__(self, input_dim, hidden_dim, num_layers, heads, dropout, edge_dim, encoder_type="gat"):
        super().__init__()

        encoder_type = str(encoder_type).lower()
        if encoder_type == "gatv2":
            encoder_type = "gatv2"
        elif encoder_type == "gin":
            encoder_type = "gin"
        elif encoder_type in {"mpnn", "messagepassing"}:
            encoder_type = "mpnn"
        else:
            encoder_type = "gat"
        self.encoder_type = encoder_type

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)

        current_dim = int(input_dim)
        for _ in range(num_layers):
            if self.encoder_type == "gat":
                conv = GATConv(
                    current_dim,
                    hidden_dim,
                    heads=heads,
                    dropout=dropout,
                    edge_dim=edge_dim,
                )
                next_dim = hidden_dim * heads
            elif self.encoder_type == "gatv2":
                conv = GATv2Conv(
                    current_dim,
                    hidden_dim,
                    heads=heads,
                    dropout=dropout,
                    edge_dim=edge_dim,
                )
                next_dim = hidden_dim * heads
            elif self.encoder_type == "gin":
                mlp = nn.Sequential(
                    nn.Linear(current_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                conv = GINEConv(mlp, edge_dim=edge_dim, train_eps=True)
                next_dim = hidden_dim
            else:  # mpnn
                edge_hidden = max(64, edge_dim * 16)
                edge_mlp = nn.Sequential(
                    nn.Linear(edge_dim, edge_hidden),
                    nn.ReLU(),
                    nn.Linear(edge_hidden, current_dim * hidden_dim),
                )
                conv = NNConv(current_dim, hidden_dim, nn=edge_mlp, aggr="mean")
                next_dim = hidden_dim

            self.convs.append(conv)
            self.norms.append(nn.LayerNorm(next_dim))
            current_dim = next_dim

        self.output_dim = current_dim

    def forward(self, x, edge_index, edge_attr, batch):
        for conv, norm in zip(self.convs, self.norms):
            if self.encoder_type in {"gat", "gatv2"}:
                x = conv(x, edge_index, edge_attr=edge_attr)
            else:
                x = conv(x, edge_index, edge_attr)
            x = norm(x)
            x = F.relu(x)
            x = self.dropout(x)

        graph_embedding = global_mean_pool(x, batch)
        return graph_embedding, x


class CoAttentionFusion(nn.Module):
    """Co-attention fusion module"""
    
    def __init__(self, hidden_dim, num_heads, dropout):
        super().__init__()
        
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, protein_emb, peptide_emb):
        # protein_emb, peptide_emb: [batch, hidden_dim]
        
        # Add sequence dimension
        protein_emb = protein_emb.unsqueeze(1)  # [batch, 1, hidden_dim]
        peptide_emb = peptide_emb.unsqueeze(1)  # [batch, 1, hidden_dim]
        
        # Cross-attention: protein attends to peptide
        attn_out, _ = self.multihead_attn(protein_emb, peptide_emb, peptide_emb)
        attn_out = self.norm(attn_out + protein_emb)
        attn_out = self.dropout(attn_out)
        
        return attn_out.squeeze(1)  # [batch, hidden_dim]


class QuantumScoringHead(nn.Module):
    """Small PennyLane TorchLayer head on top of classical joint embeddings."""

    def __init__(self, input_dim, n_qubits=4, n_layers=1, dropout=0.1):
        super().__init__()
        try:
            import pennylane as qml
        except Exception as exc:  # pragma: no cover - import check
            raise ImportError("PennyLane is required for quantum_head.enabled=true") from exc

        self.n_qubits = int(n_qubits)
        self.pre = nn.Sequential(
            nn.Linear(input_dim, self.n_qubits),
            nn.Tanh(),
        )

        dev = qml.device("default.qubit", wires=self.n_qubits)

        @qml.qnode(dev, interface="torch")
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(self.n_qubits))
            qml.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        weight_shapes = {"weights": (int(n_layers), self.n_qubits, 3)}
        self.quantum_layer = qml.qnn.TorchLayer(circuit, weight_shapes)
        self.post = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.n_qubits, 1),
        )

    def forward(self, x):
        x = self.pre(x)
        x = torch.clamp(x, -torch.pi, torch.pi)
        q_out = self.quantum_layer(x)
        return self.post(q_out)


class PeptideProteinModel(nn.Module):
    """Full peptide-protein interaction model"""
    
    def __init__(self, config):
        super().__init__()
        
        # Calculate input dimension
        feature_spec = build_feature_spec(config)
        input_dim = feature_spec["node_dim"]
        edge_dim = feature_spec["edge_dim"]
        
        protein_cfg = config['model']['protein_encoder']
        peptide_cfg = config['model']['peptide_encoder']
        fusion_cfg = config['model']['fusion']
        
        # Encoders
        self.protein_encoder = GraphEncoder(
            input_dim=input_dim,
            hidden_dim=protein_cfg['hidden_dim'],
            num_layers=protein_cfg['num_layers'],
            heads=protein_cfg.get('heads', 1),
            dropout=protein_cfg['dropout'],
            edge_dim=edge_dim,
            encoder_type=protein_cfg.get('type', 'GAT'),
        )
        
        self.peptide_encoder = GraphEncoder(
            input_dim=input_dim,
            hidden_dim=peptide_cfg['hidden_dim'],
            num_layers=peptide_cfg['num_layers'],
            heads=peptide_cfg.get('heads', 1),
            dropout=peptide_cfg['dropout'],
            edge_dim=edge_dim,
            encoder_type=peptide_cfg.get('type', 'GAT'),
        )
        
        # Fusion
        hidden_dim = self.protein_encoder.output_dim
        peptide_dim = self.peptide_encoder.output_dim
        if peptide_dim != hidden_dim:
            self.peptide_projection = nn.Linear(peptide_dim, hidden_dim)
        else:
            self.peptide_projection = nn.Identity()

        num_heads = int(fusion_cfg['num_heads'])
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"fusion.num_heads={num_heads} is incompatible with encoder output dim {hidden_dim}"
            )
        self.fusion = CoAttentionFusion(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=fusion_cfg['dropout']
        )

        # Classifier
        classifier_dim = config['model']['classifier_hidden_dim']
        self.pair_norm = nn.LayerNorm(PAIR_FEATURE_DIM)
        interaction_dim = hidden_dim * 2 + 2
        classifier_input_dim = hidden_dim + PAIR_FEATURE_DIM + interaction_dim
        quantum_cfg = config["model"].get("quantum_head", {})
        self.use_quantum_head = bool(quantum_cfg.get("enabled", False))
        if self.use_quantum_head:
            quantum_input_dim = int(quantum_cfg.get("input_dim", classifier_dim))
            self.classifier = nn.Sequential(
                nn.Linear(classifier_input_dim, classifier_dim),
                nn.ReLU(),
                nn.Dropout(fusion_cfg['dropout']),
                nn.Linear(classifier_dim, quantum_input_dim),
                nn.ReLU(),
                nn.Dropout(fusion_cfg['dropout']),
            )
            self.scoring_head = QuantumScoringHead(
                input_dim=quantum_input_dim,
                n_qubits=int(quantum_cfg.get("n_qubits", 4)),
                n_layers=int(quantum_cfg.get("n_layers", 1)),
                dropout=float(fusion_cfg['dropout']),
            )
        else:
            classifier_hidden_dim_2 = max(classifier_dim // 2, 64)
            self.classifier = nn.Sequential(
                nn.Linear(classifier_input_dim, classifier_dim),
                nn.ReLU(),
                nn.Dropout(fusion_cfg['dropout']),
                nn.Linear(classifier_dim, classifier_hidden_dim_2),
                nn.ReLU(),
                nn.Dropout(fusion_cfg['dropout']),
                nn.Linear(classifier_hidden_dim_2, 1)
            )
            self.scoring_head = None
    
    def forward(self, protein_batch, peptide_batch, pair_features, return_repr=False):
        # Encode
        protein_emb, _ = self.protein_encoder(
            protein_batch.x,
            protein_batch.edge_index,
            protein_batch.edge_attr,
            protein_batch.batch
        )
        
        peptide_emb, _ = self.peptide_encoder(
            peptide_batch.x,
            peptide_batch.edge_index,
            peptide_batch.edge_attr,
            peptide_batch.batch
        )
        peptide_emb = self.peptide_projection(peptide_emb)
        
        # Fuse
        fused = self.fusion(protein_emb, peptide_emb)
        pair_features = self.pair_norm(pair_features)
        embedding_abs_diff = torch.abs(protein_emb - peptide_emb)
        embedding_product = protein_emb * peptide_emb
        embedding_cosine = F.cosine_similarity(protein_emb, peptide_emb, dim=1, eps=1e-8).unsqueeze(1)
        embedding_l2 = torch.norm(protein_emb - peptide_emb, dim=1, keepdim=True)
        interaction_features = torch.cat(
            [embedding_abs_diff, embedding_product, embedding_cosine, embedding_l2],
            dim=1,
        )
        
        # Classify
        classifier_input = torch.cat([fused, pair_features, interaction_features], dim=1)
        hidden = self.classifier(classifier_input)
        logits = self.scoring_head(hidden) if self.scoring_head is not None else hidden
        if return_repr:
            return logits, classifier_input
        return logits


def collate_fn(batch):
    """Custom collate function for DataLoader"""
    pair_ids = [item['pair_id'] for item in batch]
    protein_group_ids = [item['protein_group_id'] for item in batch]
    labels = torch.cat([item['label'] for item in batch]).float()
    pair_features = torch.stack([item['pair_features'] for item in batch])
    
    protein_graphs = [item['protein_graph'] for item in batch]
    peptide_graphs = [item['peptide_graph'] for item in batch]
    
    protein_batch = Batch.from_data_list(protein_graphs)
    peptide_batch = Batch.from_data_list(peptide_graphs)
    
    return {
        'pair_ids': pair_ids,
        'protein_group_ids': protein_group_ids,
        'protein_batch': protein_batch,
        'peptide_batch': peptide_batch,
        'pair_features': pair_features,
        'labels': labels
    }


def compute_pairwise_ranking_loss(logits, labels, protein_group_ids, margin=0.2):
    """Pairwise margin ranking loss within each protein candidate set."""
    losses = []
    labels = labels.view(-1)
    for group_id in sorted(set(protein_group_ids)):
        group_indices = [i for i, gid in enumerate(protein_group_ids) if gid == group_id]
        if not group_indices:
            continue
        group_logits = logits[group_indices]
        group_labels = labels[group_indices]
        pos_logits = group_logits[group_labels > 0.5]
        neg_logits = group_logits[group_labels <= 0.5]
        if pos_logits.numel() == 0 or neg_logits.numel() == 0:
            continue
        pairwise_margin = margin - (pos_logits.unsqueeze(1) - neg_logits.unsqueeze(0))
        losses.append(F.relu(pairwise_margin).mean())

    if not losses:
        return logits.new_tensor(0.0), 0
    return torch.stack(losses).mean(), len(losses)


def compute_ranking_metrics(preds, labels, protein_group_ids, top_k=(1, 3, 5)):
    """Compute reranking metrics per protein candidate set."""
    by_group = defaultdict(list)
    for idx, group_id in enumerate(protein_group_ids):
        by_group[group_id].append((float(preds[idx]), int(labels[idx])))

    reciprocal_ranks = []
    hits = {k: [] for k in top_k}
    ndcg_values = {k: [] for k in top_k}
    candidate_sizes = []

    for entries in by_group.values():
        if not entries:
            continue
        entries_sorted = sorted(entries, key=lambda item: item[0], reverse=True)
        labels_sorted = [label for _, label in entries_sorted]
        n_pos = int(sum(labels_sorted))
        if n_pos <= 0:
            continue
        candidate_sizes.append(len(labels_sorted))

        first_pos_rank = next((i + 1 for i, label in enumerate(labels_sorted) if label == 1), None)
        if first_pos_rank is None:
            continue
        reciprocal_ranks.append(1.0 / first_pos_rank)
        for k in top_k:
            hits[k].append(1.0 if first_pos_rank <= k else 0.0)
            dcg = 0.0
            for i, label in enumerate(labels_sorted[:k], start=1):
                if label == 1:
                    dcg += 1.0 / np.log2(i + 1)
            ideal_labels = sorted(labels_sorted, reverse=True)
            idcg = 0.0
            for i, label in enumerate(ideal_labels[:k], start=1):
                if label == 1:
                    idcg += 1.0 / np.log2(i + 1)
            ndcg_values[k].append(float(dcg / idcg) if idcg > 0 else 0.0)

    metrics = {
        "mrr": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0,
        "num_ranked_proteins": int(len(reciprocal_ranks)),
        "avg_candidate_set_size": float(np.mean(candidate_sizes)) if candidate_sizes else 0.0,
    }
    for k in top_k:
        metrics[f"hit@{k}"] = float(np.mean(hits[k])) if hits[k] else 0.0
        metrics[f"ndcg@{k}"] = float(np.mean(ndcg_values[k])) if ndcg_values[k] else 0.0
    return metrics


def train_epoch(model, loader, optimizer, criterion, device, config):
    """Train for one epoch with ranking + auxiliary BCE loss."""
    model.train()
    total_loss = 0
    total_bce_loss = 0
    total_ranking_loss = 0
    all_preds = []
    all_labels = []
    all_group_ids = []
    ranking_cfg = config["loss"].get("ranking", {})
    ranking_enabled = bool(ranking_cfg.get("enabled", False))
    ranking_margin = float(ranking_cfg.get("margin", 0.2))
    bce_alpha = float(config["loss"].get("bce_alpha", 0.3))
    
    pbar = tqdm(loader, desc="Training")
    for batch in pbar:
        protein_batch = batch['protein_batch'].to(device)
        peptide_batch = batch['peptide_batch'].to(device)
        pair_features = batch['pair_features'].to(device)
        labels = batch['labels'].to(device).float().view(-1)
        protein_group_ids = batch['protein_group_ids']
        
        # Forward
        logits = model(protein_batch, peptide_batch, pair_features).view(-1)
        bce_loss = criterion(logits, labels)
        ranking_loss = logits.new_tensor(0.0)
        if ranking_enabled:
            ranking_loss, _ = compute_pairwise_ranking_loss(
                logits,
                labels,
                protein_group_ids,
                margin=ranking_margin,
            )
        loss = ranking_loss + bce_alpha * bce_loss
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if config['training'].get('max_grad_norm'):
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['max_grad_norm'])
        
        optimizer.step()
        
        # Track
        total_loss += loss.item()
        total_bce_loss += bce_loss.item()
        total_ranking_loss += ranking_loss.item() if ranking_enabled else 0.0
        preds = torch.sigmoid(logits).detach().cpu().numpy()
        all_preds.extend(preds.flatten())
        all_labels.extend(labels.cpu().numpy())
        all_group_ids.extend(protein_group_ids)
        
        pbar.set_postfix({'loss': loss.item(), 'bce': bce_loss.item(), 'rank': float(ranking_loss.item())})
    
    avg_loss = total_loss / len(loader)
    return (
        avg_loss,
        np.array(all_preds),
        np.array(all_labels),
        all_group_ids,
        {
            "bce_loss": total_bce_loss / len(loader),
            "ranking_loss": total_ranking_loss / len(loader) if ranking_enabled else 0.0,
        },
    )


def evaluate(model, loader, criterion, device, config):
    """Evaluate model with the same composite loss used in training."""
    model.eval()
    total_loss = 0
    total_bce_loss = 0
    total_ranking_loss = 0
    all_preds = []
    all_labels = []
    all_group_ids = []
    ranking_cfg = config["loss"].get("ranking", {})
    ranking_margin = float(ranking_cfg.get("margin", 0.2))
    ranking_enabled = bool(ranking_cfg.get("enabled", False))
    bce_alpha = float(config["loss"].get("bce_alpha", 0.3))
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            protein_batch = batch['protein_batch'].to(device)
            peptide_batch = batch['peptide_batch'].to(device)
            pair_features = batch['pair_features'].to(device)
            labels = batch['labels'].to(device).float().view(-1)
            protein_group_ids = batch['protein_group_ids']
            
            logits = model(protein_batch, peptide_batch, pair_features).view(-1)
            bce_loss = criterion(logits, labels)
            ranking_loss = logits.new_tensor(0.0)
            if ranking_enabled:
                ranking_loss, _ = compute_pairwise_ranking_loss(
                    logits,
                    labels,
                    protein_group_ids,
                    margin=ranking_margin,
                )
            loss = ranking_loss + bce_alpha * bce_loss
            
            total_loss += loss.item()
            total_bce_loss += bce_loss.item()
            total_ranking_loss += ranking_loss.item() if ranking_enabled else 0.0
            preds = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(preds.flatten())
            all_labels.extend(labels.cpu().numpy())
            all_group_ids.extend(protein_group_ids)
    
    avg_loss = total_loss / len(loader)
    return (
        avg_loss,
        np.array(all_preds),
        np.array(all_labels),
        all_group_ids,
        {
            "bce_loss": total_bce_loss / len(loader),
            "ranking_loss": total_ranking_loss / len(loader) if ranking_enabled else 0.0,
        },
    )


def compute_threshold_free_metrics(preds, labels):
    """Compute threshold-independent metrics."""
    try:
        auroc = roc_auc_score(labels, preds)
    except Exception:
        auroc = 0.0
    
    try:
        auprc = average_precision_score(labels, preds)
    except Exception:
        auprc = 0.0
    
    return {
        "auroc": float(auroc),
        "auprc": float(auprc),
    }


def compute_binary_metrics(preds, labels, threshold):
    """Compute threshold-dependent metrics."""
    preds_binary = (preds >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "f1": float(f1_score(labels, preds_binary, zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, preds_binary)),
        "precision": float(precision_score(labels, preds_binary, zero_division=0)),
        "recall": float(recall_score(labels, preds_binary, zero_division=0)),
        "positive_predictions": int(preds_binary.sum()),
        "negative_predictions": int((preds_binary == 0).sum()),
    }


def select_thresholds(preds, labels, preferred_metric="mcc"):
    """Sweep thresholds on validation scores and return best F1/MCC thresholds."""
    coarse_thresholds = list(np.arange(0.05, 0.96, 0.05))
    unique_preds = np.unique(np.clip(preds, 0.0, 1.0))
    midpoint_thresholds = []
    if unique_preds.size > 1:
        midpoint_thresholds = ((unique_preds[:-1] + unique_preds[1:]) / 2.0).tolist()

    threshold_grid = sorted(
        {
            float(threshold)
            for threshold in coarse_thresholds + midpoint_thresholds
            if 0.0 < threshold < 1.0
        }
    )

    sweep_results = []
    for threshold in threshold_grid:
        metrics = compute_binary_metrics(preds, labels, float(threshold))
        sweep_results.append(metrics)
    
    best_f1 = max(sweep_results, key=lambda item: (item["f1"], item["mcc"], -abs(item["threshold"] - 0.5)))
    best_mcc = max(sweep_results, key=lambda item: (item["mcc"], item["f1"], -abs(item["threshold"] - 0.5)))
    selected = best_mcc if preferred_metric == "mcc" else best_f1
    
    return {
        "selected_metric": preferred_metric,
        "selected_threshold": float(selected["threshold"]),
        "best_f1_threshold": float(best_f1["threshold"]),
        "best_f1": float(best_f1["f1"]),
        "best_mcc_threshold": float(best_mcc["threshold"]),
        "best_mcc": float(best_mcc["mcc"]),
        "sweep": sweep_results,
    }


def choose_validation_threshold(threshold_info):
    """Choose the final decision threshold from validation sweep results."""
    if threshold_info["best_mcc"] > 0:
        return {
            "selected_metric": "mcc",
            "selected_threshold": float(threshold_info["best_mcc_threshold"]),
        }
    return {
        "selected_metric": "f1",
        "selected_threshold": float(threshold_info["best_f1_threshold"]),
    }


def select_curriculum_negatives(negative_pools, ratios, target_total_negatives, seed):
    """Select negatives for the current epoch curriculum stage."""
    active_ratios = {
        neg_type: float(ratio)
        for neg_type, ratio in ratios.items()
        if ratio > 0 and neg_type in negative_pools and not negative_pools[neg_type].empty
    }
    if not active_ratios or target_total_negatives <= 0:
        return pd.DataFrame(columns=next(iter(negative_pools.values())).columns)

    ratio_sum = sum(active_ratios.values())
    normalized_ratios = {
        neg_type: ratio / ratio_sum
        for neg_type, ratio in active_ratios.items()
    }
    raw_targets = {
        neg_type: target_total_negatives * ratio
        for neg_type, ratio in normalized_ratios.items()
    }
    target_counts = {
        neg_type: int(np.floor(count))
        for neg_type, count in raw_targets.items()
    }

    assigned = sum(target_counts.values())
    while assigned < target_total_negatives:
        candidates = [
            (raw_targets[neg_type] - np.floor(raw_targets[neg_type]), neg_type)
            for neg_type in normalized_ratios
        ]
        _, neg_type = max(candidates)
        target_counts[neg_type] += 1
        assigned += 1

    rng = np.random.default_rng(seed)
    sampled_parts = []
    for neg_type, count in target_counts.items():
        pool = negative_pools[neg_type]
        if count <= 0 or pool.empty:
            continue
        replace = count > len(pool)
        sample_seed = int(rng.integers(0, 1_000_000_000))
        sampled_parts.append(pool.sample(n=count, replace=replace, random_state=sample_seed))
    if not sampled_parts:
        return pd.DataFrame(columns=next(iter(negative_pools.values())).columns)
    return pd.concat(sampled_parts, ignore_index=True)


def build_loss(config, dataset, device):
    """Create BCE-with-logits loss, optionally with split-aware pos_weight."""
    if not config["loss"].get("auto_pos_weight", False):
        return nn.BCEWithLogitsLoss(), None

    labels = dataset.get_labels().astype(np.float32)
    n_pos = float(labels.sum())
    n_neg = float(len(labels) - n_pos)
    if n_pos <= 0 or n_neg <= 0:
        raise ValueError(
            f"Cannot build weighted BCE for {dataset.split_name}: pos={n_pos}, neg={n_neg}"
        )

    pos_weight_value = n_neg / n_pos
    pos_weight = torch.tensor([pos_weight_value], device=device, dtype=torch.float32)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight), {
        "n_pos": int(n_pos),
        "n_neg": int(n_neg),
        "pos_weight": float(pos_weight_value),
    }


def create_balanced_sampler(dataset, seed):
    """Create a weighted sampler that approximately balances pos/neg samples per batch."""
    labels = dataset.get_labels()
    class_counts = np.bincount(labels, minlength=2)
    if np.any(class_counts == 0):
        return None
    weights = np.where(labels == 1, 1.0 / class_counts[1], 1.0 / class_counts[0])
    generator = torch.Generator()
    generator.manual_seed(seed)
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(labels),
        replacement=True,
        generator=generator,
    )


def subset_dataset_by_protein_groups(dataset, max_pairs, seed):
    """Speed helper for smoke runs: keep whole protein candidate sets up to max pair budget."""
    if max_pairs is None:
        return
    max_pairs = int(max_pairs)
    if max_pairs <= 0 or len(dataset.pairs) <= max_pairs:
        return

    rng = np.random.default_rng(seed)
    group_keys = (
        dataset.pairs[["protein_complex_id", "protein_chain_id"]]
        .drop_duplicates()
        .to_records(index=False)
        .tolist()
    )
    rng.shuffle(group_keys)

    selected_frames = []
    running = 0
    for protein_complex_id, protein_chain_id in group_keys:
        group_df = dataset.pairs[
            (dataset.pairs["protein_complex_id"] == protein_complex_id)
            & (dataset.pairs["protein_chain_id"] == protein_chain_id)
        ]
        if group_df.empty:
            continue
        if running > 0 and running + len(group_df) > max_pairs:
            continue
        selected_frames.append(group_df)
        running += len(group_df)
        if running >= max_pairs:
            break

    if not selected_frames:
        return

    dataset.pairs = (
        pd.concat(selected_frames, ignore_index=True)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )
    dataset.full_pairs = dataset.pairs.copy().reset_index(drop=True)
    dataset.summary = dataset._build_summary()
    print(f"\n[{dataset.split_name}] smoke subset applied: {len(dataset.pairs)} pairs")


def compute_metrics(preds, labels, threshold, protein_group_ids=None):
    """Compute full metric bundle at a fixed threshold."""
    metrics = compute_threshold_free_metrics(preds, labels)
    metrics.update(compute_binary_metrics(preds, labels, threshold))
    if protein_group_ids is not None:
        metrics.update(compute_ranking_metrics(preds, labels, protein_group_ids))
    return metrics


def save_pair_data_report(datasets, output_path):
    """Save concise pair dataset summaries."""
    report = {dataset.split_name: dataset.summary for dataset in datasets}
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)


def save_candidate_set_report(datasets, output_path):
    """Save per-protein candidate set diagnostics for reranking tasks."""
    report = {}
    for dataset in datasets:
        frame = dataset.pairs.copy()
        grouped = (
            frame.groupby(["protein_complex_id", "protein_chain_id"])
            .agg(
                candidates=("pair_id", "count"),
                positives=("label", "sum"),
                negatives=("label", lambda x: int((x == 0).sum())),
            )
            .reset_index()
        )
        neg_only = frame[frame["label"] == 0]
        n_pos = int((frame["label"] == 1).sum())
        n_neg = len(neg_only)
        report[dataset.split_name] = {
            "total_pairs": int(len(frame)),
            "positive_pairs": int(n_pos),
            "negative_pairs": int(n_neg),
            "positive_to_negative_ratio": float(n_pos / n_neg) if n_neg > 0 else 0.0,
            "num_proteins": int(len(grouped)),
            "avg_candidates_per_protein": float(grouped["candidates"].mean()) if len(grouped) > 0 else 0.0,
            "min_candidates_per_protein": int(grouped["candidates"].min()) if len(grouped) > 0 else 0,
            "max_candidates_per_protein": int(grouped["candidates"].max()) if len(grouped) > 0 else 0,
            "candidate_count_distribution": {
                str(k): int(v)
                for k, v in grouped["candidates"].value_counts().sort_index().to_dict().items()
            } if len(grouped) > 0 else {},
            "avg_negatives_per_protein": float(grouped["negatives"].mean()) if len(grouped) > 0 else 0.0,
            "hard_negative_ratio": float((neg_only["negative_type"] == "hard").sum() / n_neg) if n_neg > 0 else 0.0,
            "structure_hard_negative_ratio": float((neg_only["negative_type"] == "structure_hard").sum() / n_neg) if n_neg > 0 else 0.0,
        }
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)


def build_scored_prediction_frame(dataset, preds):
    """Attach model scores to pair rows and compute per-protein ranks."""
    frame = dataset.pairs.copy().reset_index(drop=True)
    scores = np.asarray(preds, dtype=np.float64).reshape(-1)
    if len(frame) != len(scores):
        raise ValueError(
            f"Prediction count mismatch for {dataset.split_name}: "
            f"rows={len(frame)}, preds={len(scores)}"
        )

    frame["score"] = scores
    frame["rank_within_protein"] = (
        frame.groupby(["protein_complex_id", "protein_chain_id"])["score"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    frame["is_top1"] = (frame["rank_within_protein"] == 1).astype(int)
    return frame


def save_scored_prediction_outputs(val_dataset, val_preds, test_dataset, test_preds, output_dir, top_k=5):
    """Save scored predictions and top-k candidate diagnostics for downstream 3D auditing."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    val_scored = build_scored_prediction_frame(val_dataset, val_preds)
    test_scored = build_scored_prediction_frame(test_dataset, test_preds)

    val_scored.to_csv(output_dir / "val_scored_predictions.csv", index=False)
    test_scored.to_csv(output_dir / "test_scored_predictions.csv", index=False)

    topk = (
        test_scored.sort_values(
            ["protein_complex_id", "protein_chain_id", "score"],
            ascending=[True, True, False],
        )
        .groupby(["protein_complex_id", "protein_chain_id"], group_keys=False)
        .head(int(top_k))
        .reset_index(drop=True)
    )
    topk.to_csv(output_dir / "test_topk_candidates.csv", index=False)

    def _first_row_payload(frame):
        if frame.empty:
            return None
        row = frame.iloc[0]
        return {
            "pair_id": str(row["pair_id"]),
            "protein_complex_id": str(row["protein_complex_id"]),
            "protein_chain_id": str(row["protein_chain_id"]),
            "peptide_complex_id": str(row["peptide_complex_id"]),
            "peptide_chain_id": str(row["peptide_chain_id"]),
            "label": int(row["label"]),
            "negative_type": str(row.get("negative_type", "unknown")),
            "score": float(row["score"]),
            "rank_within_protein": int(row["rank_within_protein"]),
        }

    best_positive = _first_row_payload(
        test_scored[test_scored["label"] == 1].sort_values("score", ascending=False)
    )
    worst_false_positive = _first_row_payload(
        test_scored[test_scored["label"] == 0].sort_values("score", ascending=False)
    )
    best_ranked_true_positive = _first_row_payload(
        test_scored[test_scored["label"] == 1].sort_values(
            ["rank_within_protein", "score"],
            ascending=[True, False],
        )
    )

    topk_positive_hits = (
        topk[topk["label"] == 1]
        .groupby(["protein_complex_id", "protein_chain_id"])
        .size()
        .reset_index(name="num_positive_in_topk")
    )
    topk_positive_hits.to_csv(output_dir / "test_topk_positive_hits.csv", index=False)

    summary = {
        "top_k": int(top_k),
        "val_rows": int(len(val_scored)),
        "test_rows": int(len(test_scored)),
        "best_positive": best_positive,
        "worst_false_positive": worst_false_positive,
        "best_ranked_true_positive": best_ranked_true_positive,
    }
    with open(output_dir / "top_ranked_examples.json", "w") as f:
        json.dump(summary, f, indent=2)


def build_loader_kwargs(config, device):
    """Build DataLoader kwargs from config and device capabilities."""
    loader_cfg = config["training"].get("data_loader", {})
    num_workers = int(loader_cfg.get("num_workers", 0))
    kwargs = {
        "num_workers": num_workers,
        "pin_memory": bool(loader_cfg.get("pin_memory", device.type == "cuda")),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(loader_cfg.get("persistent_workers", True))
        if loader_cfg.get("prefetch_factor") is not None:
            kwargs["prefetch_factor"] = int(loader_cfg["prefetch_factor"])
    return kwargs


def save_best_thresholds_report(threshold_info, output_path):
    """Save selected validation thresholds."""
    serializable = {
        key: value
        for key, value in threshold_info.items()
        if key != "sweep"
    }
    serializable["sweep"] = threshold_info["sweep"]
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)


def save_confusion_matrix_plot(labels, preds, threshold, output_path):
    """Save confusion matrix for selected threshold."""
    preds_binary = (preds >= threshold).astype(int)
    cm = confusion_matrix(labels, preds_binary, labels=[0, 1])
    
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set_xticks([0, 1], labels=["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1], labels=["True 0", "True 1"])
    ax.set_title(f"Confusion Matrix @ {threshold:.2f}")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_pr_curve(labels, preds, output_path):
    """Save precision-recall curve."""
    precision, recall, _ = precision_recall_curve(labels, preds)
    auprc = average_precision_score(labels, preds)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, label=f"AUPRC = {auprc:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_roc_curve(labels, preds, output_path):
    """Save ROC curve."""
    fpr, tpr, _ = roc_curve(labels, preds)
    auroc = roc_auc_score(labels, preds)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUROC = {auroc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_score_histogram(labels, preds, output_path):
    """Save score histogram for positive and negative classes."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.hist(preds[labels == 0], bins=20, alpha=0.6, label="Negative", color="#4C78A8")
    ax.hist(preds[labels == 1], bins=20, alpha=0.6, label="Positive", color="#F58518")
    ax.set_xlabel("Predicted score")
    ax.set_ylabel("Count")
    ax.set_title("Score Histogram by Class")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_calibration_curve(labels, preds, output_path):
    """Save a reliability diagram for probability calibration diagnostics."""
    frac_pos, mean_pred = calibration_curve(labels, preds, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(mean_pred, frac_pos, marker="o", label="Model")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed positive fraction")
    ax.set_title("Calibration Curve")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_calibration_metrics(val_labels, val_preds, test_labels, test_preds, output_path):
    """Save simple probability calibration metrics."""
    calibration_payload = {
        "validation": {
            "brier_score": float(brier_score_loss(val_labels, val_preds)),
            "mean_score": float(np.mean(val_preds)),
            "std_score": float(np.std(val_preds)),
            "min_score": float(np.min(val_preds)),
            "max_score": float(np.max(val_preds)),
            "score_range": float(np.max(val_preds) - np.min(val_preds)),
        },
        "test": {
            "brier_score": float(brier_score_loss(test_labels, test_preds)),
            "mean_score": float(np.mean(test_preds)),
            "std_score": float(np.std(test_preds)),
            "min_score": float(np.min(test_preds)),
            "max_score": float(np.max(test_preds)),
            "score_range": float(np.max(test_preds) - np.min(test_preds)),
        },
    }
    with open(output_path, "w") as handle:
        json.dump(calibration_payload, handle, indent=2)


def save_threshold_sweep_plot(threshold_info, output_path):
    """Save validation threshold sweep plot."""
    sweep_df = pd.DataFrame(threshold_info["sweep"])
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(sweep_df["threshold"], sweep_df["f1"], label="F1", color="#1f77b4")
    ax.plot(sweep_df["threshold"], sweep_df["mcc"], label="MCC", color="#d62728")
    ax.axvline(threshold_info["best_f1_threshold"], color="#1f77b4", linestyle="--", alpha=0.6)
    ax.axvline(threshold_info["best_mcc_threshold"], color="#d62728", linestyle="--", alpha=0.6)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Validation Threshold Sweep")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def evaluate_checkpoint(model, checkpoint_path, val_loader, test_loader, criterion, device, config):
    """Evaluate one checkpoint on validation and test with both F1/MCC thresholds."""
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    val_loss, val_preds, val_labels, val_group_ids, val_loss_parts = evaluate(
        model, val_loader, criterion, device, config
    )
    threshold_info = select_thresholds(val_preds, val_labels, preferred_metric="mcc")
    final_choice = choose_validation_threshold(threshold_info)

    val_metrics_selected = compute_metrics(
        val_preds,
        val_labels,
        threshold=final_choice["selected_threshold"],
        protein_group_ids=val_group_ids,
    )
    val_metrics_best_f1 = compute_metrics(
        val_preds,
        val_labels,
        threshold=threshold_info["best_f1_threshold"],
        protein_group_ids=val_group_ids,
    )
    val_metrics_best_mcc = compute_metrics(
        val_preds,
        val_labels,
        threshold=threshold_info["best_mcc_threshold"],
        protein_group_ids=val_group_ids,
    )

    test_loss, test_preds, test_labels, test_group_ids, test_loss_parts = evaluate(
        model, test_loader, criterion, device, config
    )
    test_metrics_best_f1 = compute_metrics(
        test_preds,
        test_labels,
        threshold=threshold_info["best_f1_threshold"],
        protein_group_ids=test_group_ids,
    )
    test_metrics_best_mcc = compute_metrics(
        test_preds,
        test_labels,
        threshold=threshold_info["best_mcc_threshold"],
        protein_group_ids=test_group_ids,
    )
    selected_test_metrics = (
        test_metrics_best_mcc
        if final_choice["selected_metric"] == "mcc"
        else test_metrics_best_f1
    )
    val_ranking_metrics = compute_ranking_metrics(val_preds, val_labels, val_group_ids)
    test_ranking_metrics = compute_ranking_metrics(test_preds, test_labels, test_group_ids)

    return {
        "checkpoint": checkpoint_path.name,
        "val_loss": float(val_loss),
        "test_loss": float(test_loss),
        "threshold_info": threshold_info,
        "final_choice": final_choice,
        "val_metrics_selected": val_metrics_selected,
        "val_metrics_best_f1": val_metrics_best_f1,
        "val_metrics_best_mcc": val_metrics_best_mcc,
        "test_metrics_best_f1": test_metrics_best_f1,
        "test_metrics_best_mcc": test_metrics_best_mcc,
        "test_metrics_selected": selected_test_metrics,
        "val_ranking_metrics": val_ranking_metrics,
        "test_ranking_metrics": test_ranking_metrics,
        "val_preds": val_preds,
        "val_labels": val_labels,
        "val_group_ids": val_group_ids,
        "test_preds": test_preds,
        "test_labels": test_labels,
        "test_group_ids": test_group_ids,
        "val_loss_parts": val_loss_parts,
        "test_loss_parts": test_loss_parts,
    }


def main(config_path):
    """Main training function"""
    
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    print("="*60)
    print("PeptidQuantum v0.1 Baseline Training")
    print("="*60)
    print(f"Experiment: {config['experiment_name']}")
    print(f"Split strategy: PDB-level structure-aware")
    print("="*60)
    
    # Set seed
    torch.manual_seed(config['training']['seed'])
    np.random.seed(config['training']['seed'])
    
    curriculum_config = config["training"].get("negative_curriculum", {})
    threshold_metric = config["evaluation"].get("threshold_selection_metric", "mcc").lower()
    decision_auroc_tolerance = float(
        config["evaluation"].get("decision_checkpoint_auroc_tolerance", 0.01)
    )
    if threshold_metric not in {"f1", "mcc"}:
        raise ValueError(f"Unsupported threshold_selection_metric: {threshold_metric}")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    loader_kwargs = build_loader_kwargs(config, device)
    print(f"DataLoader settings: {loader_kwargs}")
    
    # Load pair datasets
    pairs_dir = Path(config['data']['canonical_dir']) / 'pairs'
    validate_split_metadata(
        canonical_dir=config["data"]["canonical_dir"],
        pairs_dir=pairs_dir,
        splits_dir=config["data"]["splits_dir"],
    )
    
    print(f"\nLoading pair datasets from {pairs_dir}")
    
    # Create datasets
    canonical_dir = config['data']['canonical_dir']
    train_dataset = PeptideProteinDataset(pairs_dir / 'train_pairs.parquet', canonical_dir, config, "train")
    val_dataset = PeptideProteinDataset(pairs_dir / 'val_pairs.parquet', canonical_dir, config, "val")
    test_dataset = PeptideProteinDataset(pairs_dir / 'test_pairs.parquet', canonical_dir, config, "test")
    subset_cfg = config["training"].get("subset_max_pairs", {})
    if subset_cfg:
        subset_dataset_by_protein_groups(
            train_dataset,
            subset_cfg.get("train"),
            seed=config["training"]["seed"],
        )
        subset_dataset_by_protein_groups(
            val_dataset,
            subset_cfg.get("val"),
            seed=config["training"]["seed"] + 1,
        )
        subset_dataset_by_protein_groups(
            test_dataset,
            subset_cfg.get("test"),
            seed=config["training"]["seed"] + 2,
        )
    
    # Create dataloaders
    batch_size = config['training']['batch_size']
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        **loader_kwargs,
    )
    
    # Create model
    model = PeptideProteinModel(config).to(device)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        patience=config['training']['scheduler_patience'],
        factor=config['training']['scheduler_factor']
    )
    
    # Training loop
    save_dir = Path(config['logging']['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(save_dir / 'config_used.yaml', 'w') as f:
        yaml.dump(config, f)
    
    save_pair_data_report(
        [train_dataset, val_dataset, test_dataset],
        save_dir / "pair_data_report.json",
    )
    save_candidate_set_report(
        [train_dataset, val_dataset, test_dataset],
        save_dir / "candidate_set_report.json",
    )
    
    best_val_metric = float("-inf")
    best_decision_metric = float("-inf")
    patience_counter = 0
    train_log = []
    
    print("\nStarting training...")
    print("="*60)
    print(f"Balanced sampler enabled: {bool(config['training'].get('balanced_sampling', {}).get('enabled', False))}")
    
    for epoch in range(config['training']['epochs']):
        print(f"\nEpoch {epoch + 1}/{config['training']['epochs']}")
        
        if curriculum_config.get("enabled", False):
            train_dataset.set_curriculum(epoch + 1, curriculum_config)
            print(f"Curriculum stage: {train_dataset.active_curriculum_stage}")
            print(f"Curriculum negative types: {train_dataset.summary['negative_type_distribution']}")
        else:
            train_dataset.reset_pairs()
        
        balanced_sampler = None
        if config["training"].get("balanced_sampling", {}).get("enabled", False):
            balanced_sampler = create_balanced_sampler(
                train_dataset,
                seed=config["training"]["seed"] + epoch,
            )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=balanced_sampler is None,
            sampler=balanced_sampler,
            collate_fn=collate_fn,
            **loader_kwargs,
        )

        criterion, class_stats = build_loss(config, train_dataset, device)
        if class_stats:
            total_samples = class_stats["n_pos"] + class_stats["n_neg"]
            print("Training class distribution:")
            print(
                f"  Positive: {class_stats['n_pos']} ({class_stats['n_pos'] / total_samples * 100:.1f}%)"
            )
            print(
                f"  Negative: {class_stats['n_neg']} ({class_stats['n_neg'] / total_samples * 100:.1f}%)"
            )
            print(f"  pos_weight: {class_stats['pos_weight']:.3f}")
        
        # Train
        train_loss, train_preds, train_labels, train_group_ids, train_loss_parts = train_epoch(
            model, train_loader, optimizer, criterion, device, config
        )
        train_metrics = compute_metrics(
            train_preds,
            train_labels,
            threshold=0.5,
            protein_group_ids=train_group_ids,
        )
        
        # Validate with threshold sweep
        val_loss, val_preds, val_labels, val_group_ids, val_loss_parts = evaluate(
            model, val_loader, criterion, device, config
        )
        val_thresholds = select_thresholds(val_preds, val_labels, preferred_metric=threshold_metric)
        val_threshold_choice = choose_validation_threshold(val_thresholds)
        val_metrics = compute_metrics(
            val_preds,
            val_labels,
            threshold=val_threshold_choice["selected_threshold"],
            protein_group_ids=val_group_ids,
        )
        
        # Scheduler step
        monitor_key = config['evaluation']['monitor_metric'].replace('val_', '')
        if monitor_key not in val_metrics:
            raise KeyError(f"monitor_metric {config['evaluation']['monitor_metric']} not available in validation metrics")
        monitor_metric = val_metrics[monitor_key]
        scheduler.step(monitor_metric)
        
        # Log
        log_entry = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_bce_loss': train_loss_parts["bce_loss"],
            'train_ranking_loss': train_loss_parts["ranking_loss"],
            'val_bce_loss': val_loss_parts["bce_loss"],
            'val_ranking_loss': val_loss_parts["ranking_loss"],
            'lr': optimizer.param_groups[0]['lr']
        }
        log_entry.update({f'train_{k}': v for k, v in train_metrics.items()})
        log_entry.update({f'val_{k}': v for k, v in val_metrics.items()})
        log_entry["train_curriculum_stage"] = train_dataset.active_curriculum_stage
        log_entry["train_negative_distribution"] = json.dumps(train_dataset.summary["negative_type_distribution"])
        log_entry["balanced_sampler_enabled"] = bool(balanced_sampler is not None)
        log_entry["val_best_f1_threshold"] = val_thresholds["best_f1_threshold"]
        log_entry["val_best_f1"] = val_thresholds["best_f1"]
        log_entry["val_best_mcc_threshold"] = val_thresholds["best_mcc_threshold"]
        log_entry["val_best_mcc"] = val_thresholds["best_mcc"]
        log_entry["val_selected_metric"] = val_threshold_choice["selected_metric"]
        log_entry["val_selected_threshold"] = val_threshold_choice["selected_threshold"]
        log_entry["val_selected_metric_value"] = val_metrics[val_threshold_choice["selected_metric"]]
        train_log.append(log_entry)
        pd.DataFrame(train_log).to_csv(save_dir / 'train_log.csv', index=False)
        torch.save(model.state_dict(), save_dir / 'last_model.pt')
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(
            f"Train BCE: {train_loss_parts['bce_loss']:.4f} | "
            f"Train Ranking: {train_loss_parts['ranking_loss']:.4f} | "
            f"Val BCE: {val_loss_parts['bce_loss']:.4f} | "
            f"Val Ranking: {val_loss_parts['ranking_loss']:.4f}"
        )
        print(f"Train AUROC: {train_metrics['auroc']:.4f} | Val AUROC: {val_metrics['auroc']:.4f}")
        print(
            f"Val AUPRC: {val_metrics['auprc']:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f} | "
            f"Val MCC: {val_metrics['mcc']:.4f} | "
            f"Val MRR: {val_metrics['mrr']:.4f} | "
            f"Val Hit@1: {val_metrics['hit@1']:.4f} | "
            f"Val Hit@3: {val_metrics['hit@3']:.4f} | "
            f"Val Hit@5: {val_metrics['hit@5']:.4f} | "
            f"Threshold(F1): {val_thresholds['best_f1_threshold']:.2f} | "
            f"Threshold(MCC): {val_thresholds['best_mcc_threshold']:.2f} | "
            f"Selected: {val_threshold_choice['selected_metric']}@{val_threshold_choice['selected_threshold']:.2f}"
        )
        
        # Save best model
        if monitor_metric > best_val_metric:
            best_val_metric = monitor_metric
            torch.save(model.state_dict(), save_dir / 'best_model.pt')
            patience_counter = 0
            print(f"[BEST] New best {config['evaluation']['monitor_metric']}: {monitor_metric:.4f}")
        else:
            patience_counter += 1

        decision_metric = val_thresholds["best_mcc"]
        if decision_metric > best_decision_metric:
            best_decision_metric = decision_metric
            torch.save(model.state_dict(), save_dir / 'best_decision_model.pt')
            print(f"[BEST] New best val_mcc: {decision_metric:.4f}")
        
        # Early stopping
        if patience_counter >= config['training']['early_stopping_patience']:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break
    
    # Save final copies
    torch.save(model.state_dict(), save_dir / 'last_model.pt')
    pd.DataFrame(train_log).to_csv(save_dir / 'train_log.csv', index=False)
    
    # Test evaluation
    print("\n" + "="*60)
    print("Final Test Evaluation")
    print("="*60)
    
    train_dataset.reset_pairs()
    criterion, _ = build_loss(config, train_dataset, device)

    candidate_paths = []
    for checkpoint_name in ["best_model.pt", "best_decision_model.pt"]:
        checkpoint_path = save_dir / checkpoint_name
        if checkpoint_path.exists():
            candidate_paths.append(checkpoint_path)

    checkpoint_results = [
        evaluate_checkpoint(model, checkpoint_path, val_loader, test_loader, criterion, device, config)
        for checkpoint_path in candidate_paths
    ]
    results_by_name = {result["checkpoint"]: result for result in checkpoint_results}
    auroc_result = results_by_name.get("best_model.pt", checkpoint_results[0])
    decision_result = results_by_name.get("best_decision_model.pt", auroc_result)

    decision_auroc = decision_result["val_metrics_selected"]["auroc"]
    auroc_checkpoint_auroc = auroc_result["val_metrics_selected"]["auroc"]
    decision_mcc = decision_result["val_metrics_selected"]["mcc"]
    auroc_checkpoint_mcc = auroc_result["val_metrics_selected"]["mcc"]

    if (
        decision_result["threshold_info"]["best_mcc"] > 0
        and decision_mcc >= auroc_checkpoint_mcc
        and decision_auroc + decision_auroc_tolerance >= auroc_checkpoint_auroc
    ):
        final_result = decision_result
        final_selection_reason = "decision_checkpoint_kept_stronger_mcc_within_auroc_tolerance"
    else:
        final_result = auroc_result
        final_selection_reason = "best_auroc_checkpoint"

    final_checkpoint = save_dir / final_result["checkpoint"]
    best_thresholds = final_result["threshold_info"].copy()
    best_thresholds.update(final_result["final_choice"])
    selected_threshold = best_thresholds["selected_threshold"]
    val_preds = final_result["val_preds"]
    val_labels = final_result["val_labels"]
    test_preds = final_result["test_preds"]
    test_labels = final_result["test_labels"]
    test_loss = final_result["test_loss"]
    test_metrics = final_result["test_metrics_selected"]
    
    print(f"\nCheckpoint selected: {final_result['checkpoint']} ({final_selection_reason})")
    print(f"Validation threshold choice: {best_thresholds['selected_metric']} @ {selected_threshold:.2f}")
    print(f"Validation AUROC: {final_result['val_metrics_selected']['auroc']:.4f}")
    print(f"Validation MCC: {final_result['val_metrics_selected']['mcc']:.4f}")
    print(
        f"Validation ranking: MRR={final_result['val_ranking_metrics']['mrr']:.4f}, "
        f"Hit@1={final_result['val_ranking_metrics']['hit@1']:.4f}, "
        f"Hit@3={final_result['val_ranking_metrics']['hit@3']:.4f}, "
        f"Hit@5={final_result['val_ranking_metrics']['hit@5']:.4f}"
    )
    print(f"\nTest Loss: {test_loss:.4f}")
    print(
        f"Test metrics @ best F1 threshold ({best_thresholds['best_f1_threshold']:.2f}): "
        f"F1={final_result['test_metrics_best_f1']['f1']:.4f}, "
        f"MCC={final_result['test_metrics_best_f1']['mcc']:.4f}, "
        f"AUROC={final_result['test_metrics_best_f1']['auroc']:.4f}"
    )
    print(
        f"Test metrics @ best MCC threshold ({best_thresholds['best_mcc_threshold']:.2f}): "
        f"F1={final_result['test_metrics_best_mcc']['f1']:.4f}, "
        f"MCC={final_result['test_metrics_best_mcc']['mcc']:.4f}, "
        f"AUROC={final_result['test_metrics_best_mcc']['auroc']:.4f}"
    )
    for metric, value in test_metrics.items():
        print(f"Test selected {metric.upper()}: {value:.4f}")
    print(
        f"Test ranking: MRR={final_result['test_ranking_metrics']['mrr']:.4f}, "
        f"Hit@1={final_result['test_ranking_metrics']['hit@1']:.4f}, "
        f"Hit@3={final_result['test_ranking_metrics']['hit@3']:.4f}, "
        f"Hit@5={final_result['test_ranking_metrics']['hit@5']:.4f}"
    )
    
    # Save test results
    with open(save_dir / 'test_summary.txt', 'w') as f:
        f.write("Test Set Evaluation\n")
        f.write("="*60 + "\n\n")
        f.write(f"Checkpoint used: {final_checkpoint.name}\n")
        f.write(f"Checkpoint selection reason: {final_selection_reason}\n")
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Selected validation threshold ({best_thresholds['selected_metric']}): {selected_threshold:.2f}\n")
        f.write(
            f"Validation metrics @ selected threshold: "
            f"AUROC={final_result['val_metrics_selected']['auroc']:.4f}, "
            f"AUPRC={final_result['val_metrics_selected']['auprc']:.4f}, "
            f"F1={final_result['val_metrics_selected']['f1']:.4f}, "
            f"MCC={final_result['val_metrics_selected']['mcc']:.4f}\n"
        )
        f.write(
            f"Validation ranking: "
            f"MRR={final_result['val_ranking_metrics']['mrr']:.4f}, "
            f"Hit@1={final_result['val_ranking_metrics']['hit@1']:.4f}, "
            f"Hit@3={final_result['val_ranking_metrics']['hit@3']:.4f}, "
            f"Hit@5={final_result['val_ranking_metrics']['hit@5']:.4f}\n"
        )
        f.write(
            f"Test metrics @ best F1 threshold ({best_thresholds['best_f1_threshold']:.2f}): "
            f"AUROC={final_result['test_metrics_best_f1']['auroc']:.4f}, "
            f"AUPRC={final_result['test_metrics_best_f1']['auprc']:.4f}, "
            f"F1={final_result['test_metrics_best_f1']['f1']:.4f}, "
            f"MCC={final_result['test_metrics_best_f1']['mcc']:.4f}\n"
        )
        f.write(
            f"Test metrics @ best MCC threshold ({best_thresholds['best_mcc_threshold']:.2f}): "
            f"AUROC={final_result['test_metrics_best_mcc']['auroc']:.4f}, "
            f"AUPRC={final_result['test_metrics_best_mcc']['auprc']:.4f}, "
            f"F1={final_result['test_metrics_best_mcc']['f1']:.4f}, "
            f"MCC={final_result['test_metrics_best_mcc']['mcc']:.4f}\n"
        )
        for metric, value in test_metrics.items():
            f.write(f"Test selected {metric.upper()}: {value:.4f}\n")
        f.write(
            f"Test ranking: "
            f"MRR={final_result['test_ranking_metrics']['mrr']:.4f}, "
            f"Hit@1={final_result['test_ranking_metrics']['hit@1']:.4f}, "
            f"Hit@3={final_result['test_ranking_metrics']['hit@3']:.4f}, "
            f"Hit@5={final_result['test_ranking_metrics']['hit@5']:.4f}\n"
        )
    
    # Save metrics
    with open(save_dir / 'metrics.json', 'w') as f:
        json.dump({
            'best_val_auroc': float(best_val_metric),
            'best_val_threshold_metric': float(best_decision_metric),
            'threshold_selection_metric': threshold_metric,
            'final_checkpoint': final_checkpoint.name,
            'decision_checkpoint': 'best_decision_model.pt',
            'final_selection_reason': final_selection_reason,
            'best_thresholds': {
                key: float(value) if isinstance(value, (np.floating, float)) else value
                for key, value in best_thresholds.items()
                if key != 'sweep'
            },
            'validation_metrics_at_selected_threshold': {
                k: float(v) if isinstance(v, (np.floating, float)) else v
                for k, v in final_result['val_metrics_selected'].items()
            },
            'test_metrics': {k: float(v) for k, v in test_metrics.items()},
            'test_metrics_best_f1_threshold': {
                k: float(v) for k, v in final_result['test_metrics_best_f1'].items()
            },
            'test_metrics_best_mcc_threshold': {
                k: float(v) for k, v in final_result['test_metrics_best_mcc'].items()
            },
            'val_ranking_metrics': {
                k: float(v) if isinstance(v, (np.floating, float)) else int(v)
                for k, v in final_result['val_ranking_metrics'].items()
            },
            'test_ranking_metrics': {
                k: float(v) if isinstance(v, (np.floating, float)) else int(v)
                for k, v in final_result['test_ranking_metrics'].items()
            },
            'checkpoint_comparison': [
                {
                    'checkpoint': result['checkpoint'],
                    'selected_metric': result['final_choice']['selected_metric'],
                    'selected_threshold': float(result['final_choice']['selected_threshold']),
                    'val_auroc': float(result['val_metrics_selected']['auroc']),
                    'val_auprc': float(result['val_metrics_selected']['auprc']),
                    'val_f1': float(result['val_metrics_selected']['f1']),
                    'val_mcc': float(result['val_metrics_selected']['mcc']),
                    'val_mrr': float(result['val_ranking_metrics']['mrr']),
                    'val_hit@1': float(result['val_ranking_metrics']['hit@1']),
                    'val_hit@3': float(result['val_ranking_metrics']['hit@3']),
                    'val_hit@5': float(result['val_ranking_metrics']['hit@5']),
                    'test_auroc': float(result['test_metrics_selected']['auroc']),
                    'test_auprc': float(result['test_metrics_selected']['auprc']),
                    'test_f1': float(result['test_metrics_selected']['f1']),
                    'test_mcc': float(result['test_metrics_selected']['mcc']),
                    'test_mrr': float(result['test_ranking_metrics']['mrr']),
                    'test_hit@1': float(result['test_ranking_metrics']['hit@1']),
                    'test_hit@3': float(result['test_ranking_metrics']['hit@3']),
                    'test_hit@5': float(result['test_ranking_metrics']['hit@5']),
                }
                for result in checkpoint_results
            ],
        }, f, indent=2)

    with open(save_dir / "ranking_metrics.json", "w") as f:
        json.dump(
            {
                "validation": final_result["val_ranking_metrics"],
                "test": final_result["test_ranking_metrics"],
            },
            f,
            indent=2,
        )
    
    save_best_thresholds_report(best_thresholds, save_dir / "best_thresholds.json")
    pd.DataFrame(best_thresholds["sweep"]).to_csv(save_dir / "threshold_vs_f1_table.csv", index=False)
    save_threshold_sweep_plot(best_thresholds, save_dir / "validation_threshold_sweep.png")
    save_score_histogram(val_labels, val_preds, save_dir / "validation_score_histogram_pos_neg.png")
    save_calibration_curve(val_labels, val_preds, save_dir / "calibration_curve.png")
    save_calibration_metrics(
        val_labels,
        val_preds,
        test_labels,
        test_preds,
        save_dir / "calibration_metrics.json",
    )
    save_confusion_matrix_plot(test_labels, test_preds, selected_threshold, save_dir / "confusion_matrix.png")
    save_pr_curve(test_labels, test_preds, save_dir / "pr_curve.png")
    save_roc_curve(test_labels, test_preds, save_dir / "roc_curve.png")
    save_score_histogram(test_labels, test_preds, save_dir / "score_histogram_pos_neg.png")
    save_scored_prediction_outputs(
        val_dataset=val_dataset,
        val_preds=val_preds,
        test_dataset=test_dataset,
        test_preds=test_preds,
        output_dir=save_dir,
        top_k=5,
    )
    
    print(f"\nTraining complete! Results saved to {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PeptidQuantum v0.1 Baseline")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    
    args = parser.parse_args()
    main(args.config)
