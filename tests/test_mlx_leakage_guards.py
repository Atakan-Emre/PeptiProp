import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parent.parent
EXPORT_MLX_PATH = ROOT / "scripts" / "export_mlx_features.py"
CONFIG_PATH = ROOT / "configs" / "train_v0_1_scoring_mlx_m4.yaml"


def load_export_mlx_module():
    spec = importlib.util.spec_from_file_location("export_mlx_module", EXPORT_MLX_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def infer_native_peptide_chain(complex_id: str) -> str | None:
    parts = str(complex_id).split("_")
    if len(parts) < 3:
        return None
    return parts[-1]


class TestMLXLeakageGuards(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_export_mlx_module()
        with open(CONFIG_PATH, encoding="utf-8") as handle:
            cls.cfg = yaml.safe_load(handle)

        cls.canonical_dir = ROOT / cls.cfg["data"]["canonical_dir"]
        pairs_dir = ROOT / cls.cfg["data"]["pairs_dir"]

        train_pairs = pd.read_parquet(pairs_dir / "train_pairs.parquet")
        quality_filter = cls.cfg["data"].get("quality_filter")
        if quality_filter and "pair_quality_flag" in train_pairs.columns:
            train_pairs = train_pairs[train_pairs["pair_quality_flag"] == quality_filter].reset_index(drop=True)

        negative_rows = train_pairs[train_pairs["label"] == 0]
        if negative_rows.empty:
            raise RuntimeError("No negative pairs found for MLX leakage test")

        cls.negative_row = None
        for _, row in negative_rows.iterrows():
            native_chain = infer_native_peptide_chain(row["protein_complex_id"])
            if native_chain is None:
                continue
            same_native = (
                str(row["peptide_complex_id"]) == str(row["protein_complex_id"])
                and str(row["peptide_chain_id"]) == str(native_chain)
            )
            if not same_native:
                cls.negative_row = row.copy()
                cls.native_chain = native_chain
                break

        if cls.negative_row is None:
            raise RuntimeError("Could not find a negative pair that differs from the native peptide")

        cls.pair_df = pd.DataFrame([cls.negative_row])
        cls.use_local_density = bool(cls.cfg["features"].get("use_local_density", False))
        cls.local_density_radius = float(cls.cfg["features"].get("local_density_radius", 8.0))
        cls.residues = pd.read_parquet(
            cls.canonical_dir / "residues.parquet",
            columns=cls.module.RESIDUE_COLUMNS,
        )

    def _build_single_pair_matrix(self, residues_df: pd.DataFrame) -> dict:
        original_read_parquet = self.module.pd.read_parquet
        residues_path = (self.canonical_dir / "residues.parquet").resolve()

        def patched_read_parquet(path, *args, **kwargs):
            if Path(path).resolve() == residues_path:
                cols = kwargs.get("columns")
                if cols:
                    return residues_df[cols].copy()
                return residues_df.copy()
            return original_read_parquet(path, *args, **kwargs)

        with patch.object(self.module.pd, "read_parquet", side_effect=patched_read_parquet):
            summaries = self.module.build_chain_summaries(
                split_dfs={"train": self.pair_df},
                canonical_dir=self.canonical_dir,
                use_local_density=self.use_local_density,
                local_density_radius=self.local_density_radius,
            )
        return self.module.build_pair_matrix(self.pair_df, summaries)

    def test_negative_pair_feature_is_independent_from_native_peptide_mutation(self):
        original_matrix = self._build_single_pair_matrix(self.residues)

        mutated = self.residues.copy()
        native_mask = (
            mutated["complex_id"].astype(str) == str(self.negative_row["protein_complex_id"])
        ) & (
            mutated["chain_id"].astype(str) == str(self.native_chain)
        )
        self.assertTrue(native_mask.any(), "Native peptide residues not found in canonical residues")

        mutated.loc[native_mask, "x"] = (
            mutated.loc[native_mask, "x"].to_numpy(dtype=np.float32) + np.float32(1000.0)
        )
        mutated.loc[native_mask, "y"] = (
            mutated.loc[native_mask, "y"].to_numpy(dtype=np.float32) - np.float32(1000.0)
        )
        mutated.loc[native_mask, "z"] = (
            mutated.loc[native_mask, "z"].to_numpy(dtype=np.float32) + np.float32(500.0)
        )
        mutated.loc[native_mask, "resname"] = "UNK"

        mutated_matrix = self._build_single_pair_matrix(mutated)
        np.testing.assert_allclose(original_matrix["x"], mutated_matrix["x"], rtol=0.0, atol=1e-6)

    def test_pair_matrix_schema_contains_only_pair_local_outputs(self):
        matrix = self._build_single_pair_matrix(self.residues)
        self.assertSetEqual(
            set(matrix.keys()),
            {"x", "y", "group_idx", "group_names", "pair_id", "negative_type"},
        )
        self.assertEqual(int(matrix["x"].shape[1]), int(self.module.PAIR_FEATURE_DIM))

        typed_keys = []
        for protein_key, peptide_key in self.module._iter_pairs(self.pair_df):
            typed_keys.extend([protein_key, peptide_key])
        native_typed_key = (
            str(self.negative_row["protein_complex_id"]),
            str(self.native_chain),
            "peptide",
        )
        self.assertNotIn(native_typed_key, typed_keys)


if __name__ == "__main__":
    unittest.main()
