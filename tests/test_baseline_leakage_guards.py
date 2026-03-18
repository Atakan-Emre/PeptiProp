import importlib.util
import unittest
from pathlib import Path

import torch
import yaml


ROOT = Path(__file__).resolve().parent.parent
TRAIN_BASELINE_PATH = ROOT / "scripts" / "train_baseline.py"
TRAIN_PAIRS_PATH = ROOT / "data" / "canonical" / "pairs" / "train_pairs.parquet"
CANONICAL_DIR = ROOT / "data" / "canonical"
CONFIG_PATH = ROOT / "configs" / "train_v0_1_exp_a_smoke.yaml"


def load_train_baseline_module():
    spec = importlib.util.spec_from_file_location("train_baseline_module", TRAIN_BASELINE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestBaselineLeakageGuards(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_train_baseline_module()
        with open(CONFIG_PATH, encoding="utf-8") as handle:
            cls.config = yaml.safe_load(handle)
        cls.dataset = cls.module.PeptideProteinDataset(
            TRAIN_PAIRS_PATH,
            CANONICAL_DIR,
            cls.config,
            "train",
        )

        negative_indices = cls.dataset.pairs.index[cls.dataset.pairs["label"] == 0].tolist()
        if not negative_indices:
            raise RuntimeError("No negative pairs found for leakage test")
        cls.negative_index = int(negative_indices[0])
        cls.negative_row = cls.dataset.pairs.iloc[cls.negative_index].copy()

    def _mutate_native_peptide_and_recompute(self):
        protein_complex_id = self.negative_row["protein_complex_id"]
        native_peptide_chain = self.dataset.complex_chain_index[(protein_complex_id, "peptide")][0]
        native_key = (protein_complex_id, native_peptide_chain)
        original_native_df = self.dataset.residue_lookup[native_key]

        mutated = original_native_df.copy()
        mutated["x"] = mutated["x"] + 1000.0
        mutated["y"] = mutated["y"] - 1000.0
        mutated["z"] = mutated["z"] + 500.0
        mutated["resname"] = "UNK"

        self.dataset.graph_cache.clear()
        self.dataset.chain_summary_cache.clear()
        original_item = self.dataset[self.negative_index]

        self.dataset.residue_lookup[native_key] = mutated
        self.dataset.graph_cache.clear()
        self.dataset.chain_summary_cache.clear()
        mutated_item = self.dataset[self.negative_index]

        self.dataset.residue_lookup[native_key] = original_native_df
        self.dataset.graph_cache.clear()
        self.dataset.chain_summary_cache.clear()

        return original_item, mutated_item

    def test_negative_pair_features_do_not_depend_on_native_peptide(self):
        original_item, mutated_item = self._mutate_native_peptide_and_recompute()
        self.assertTrue(
            torch.allclose(original_item["pair_features"], mutated_item["pair_features"])
        )

    def test_negative_protein_graph_does_not_depend_on_native_peptide(self):
        original_item, mutated_item = self._mutate_native_peptide_and_recompute()
        self.assertTrue(torch.equal(original_item["protein_graph"].x, mutated_item["protein_graph"].x))
        self.assertTrue(
            torch.equal(original_item["protein_graph"].edge_index, mutated_item["protein_graph"].edge_index)
        )
        self.assertTrue(
            torch.allclose(original_item["protein_graph"].edge_attr, mutated_item["protein_graph"].edge_attr)
        )


if __name__ == "__main__":
    unittest.main()
