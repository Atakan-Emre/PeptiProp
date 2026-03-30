import json
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
CANONICAL_DIR = ROOT / "data" / "canonical"
SPLITS_DIR = CANONICAL_DIR / "splits"
PAIRS_DIR = CANONICAL_DIR / "pairs"
TRAINING_DIR_GNN = ROOT / "outputs" / "training" / "peptiprop_v0_2_gnn_esm2"
VIS_SUMMARY_BATCH_GNN = ROOT / "outputs" / "analysis_propedia_batch_gnn" / "visualization_sanity_summary.json"
VIS_SUMMARY_TOP_GNN = ROOT / "outputs" / "analysis_propedia_top_ranked_batch_gnn" / "visualization_sanity_summary.json"


def _gnn_artifacts_ready() -> bool:
    required = {
        "best_model.pt",
        "metrics.json",
        "ranking_metrics.json",
        "best_thresholds.json",
        "pair_data_report.json",
        "candidate_set_report.json",
        "calibration_metrics.json",
        "threshold_vs_f1_table.csv",
        "test_summary.txt",
        "test_topk_candidates.csv",
        "test_topk_positive_hits.csv",
        "top_ranked_examples.json",
        "roc_curve.png",
        "pr_curve.png",
        "confusion_matrix.png",
        "score_histogram_pos_neg.png",
        "validation_score_histogram_pos_neg.png",
        "validation_threshold_sweep.png",
        "calibration_curve.png",
    }
    return TRAINING_DIR_GNN.is_dir() and required.issubset({path.name for path in TRAINING_DIR_GNN.iterdir()})


def _gnn_visualizations_ready() -> bool:
    return VIS_SUMMARY_BATCH_GNN.is_file() and VIS_SUMMARY_TOP_GNN.is_file()


class TestPropediaActivePipelineCanonical(unittest.TestCase):
    """Canonical PROPEDIA surface should stay consistent regardless of model family."""

    @classmethod
    def setUpClass(cls):
        cls.complexes = pd.read_parquet(CANONICAL_DIR / "complexes.parquet")
        with open(PAIRS_DIR / "pair_data_report.json", encoding="utf-8") as handle:
            cls.pair_report = json.load(handle)
        with open(PAIRS_DIR / "candidate_set_report.json", encoding="utf-8") as handle:
            cls.candidate_report = json.load(handle)

    def test_full_canonical_snapshot(self):
        self.assertGreaterEqual(len(self.complexes), 40000)
        self.assertGreaterEqual(self.complexes["pdb_id"].nunique(), 13000)
        self.assertTrue(set(self.complexes["quality_flag"].unique()).issubset({"clean", "warning"}))

    def test_residues_have_interface_and_pocket_annotations(self):
        residues = pd.read_parquet(CANONICAL_DIR / "residues.parquet", columns=["is_interface", "is_pocket"])
        interface_count = int(residues["is_interface"].sum())
        pocket_count = int(residues["is_pocket"].sum())
        self.assertGreater(interface_count, 0, "is_interface is all-zero")
        self.assertGreater(pocket_count, 0, "is_pocket is all-zero")
        self.assertGreater(pocket_count, interface_count, "pocket set should be larger than interface set")

    def test_split_metadata_matches_files(self):
        for split_name in ("train", "val", "test"):
            split_ids = {
                line.strip()
                for line in (SPLITS_DIR / f"{split_name}_ids.txt").read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.startswith("#")
            }
            tagged_ids = set(self.complexes.loc[self.complexes["split_tag"] == split_name, "complex_id"])
            self.assertSetEqual(split_ids, tagged_ids, msg=f"split drift for {split_name}")

    def test_pair_reports_are_balanced_and_unique(self):
        for split_name, report in self.pair_report.items():
            self.assertEqual(report["duplicate_pair_count"], 0, msg=f"duplicates in {split_name}")
            self.assertGreater(report["negative_pairs"], report["positive_pairs"], msg=f"not enough candidates in {split_name}")
            self.assertEqual(set(report["quality_flag_distribution"].keys()), {"clean"})
            self.assertTrue(report["split_column_consistent"], msg=f"split column inconsistent in {split_name}")
            self.assertGreater(self.candidate_report[split_name]["avg_candidates_per_protein"], 1.0)

    def test_pair_parquet_split_column_is_consistent(self):
        for split_name in ("train", "val", "test"):
            pairs = pd.read_parquet(PAIRS_DIR / f"{split_name}_pairs.parquet")
            self.assertSetEqual(set(pairs["split"].astype(str).unique()), {split_name})


@unittest.skipUnless(
    _gnn_artifacts_ready(),
    "GNN final artifacts are missing; run scripts/train_gnn_esm2.py and scripts/generate_gnn_predictions.py",
)
class TestPropediaActivePipelineGNNArtifacts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(TRAINING_DIR_GNN / "metrics.json", encoding="utf-8") as handle:
            cls.training_metrics = json.load(handle)
        with open(TRAINING_DIR_GNN / "ranking_metrics.json", encoding="utf-8") as handle:
            cls.ranking_metrics = json.load(handle)

    def test_gnn_final_artifacts_and_metrics_are_present(self):
        required_files = {
            "best_model.pt",
            "metrics.json",
            "ranking_metrics.json",
            "best_thresholds.json",
            "pair_data_report.json",
            "candidate_set_report.json",
            "calibration_metrics.json",
            "threshold_vs_f1_table.csv",
            "test_summary.txt",
            "test_topk_candidates.csv",
            "test_topk_positive_hits.csv",
            "top_ranked_examples.json",
            "roc_curve.png",
            "pr_curve.png",
            "confusion_matrix.png",
            "score_histogram_pos_neg.png",
            "validation_score_histogram_pos_neg.png",
            "validation_threshold_sweep.png",
            "calibration_curve.png",
        }
        self.assertTrue(required_files.issubset({path.name for path in TRAINING_DIR_GNN.iterdir()}))

        val_metrics = self.training_metrics["validation_metrics_at_selected_threshold"]
        test_metrics = self.training_metrics["test_metrics"]
        val_rank = self.training_metrics["val_ranking_metrics"]
        test_rank = self.training_metrics["test_ranking_metrics"]

        for key in ("auroc", "auprc", "f1", "mcc"):
            self.assertGreaterEqual(val_metrics[key], 0.0)
            self.assertLessEqual(val_metrics[key], 1.0)
            self.assertGreaterEqual(test_metrics[key], 0.0)
            self.assertLessEqual(test_metrics[key], 1.0)

        self.assertGreater(val_rank["mrr"], 0.3)
        self.assertGreater(test_rank["mrr"], 0.3)
        self.assertGreater(val_rank["hit@3"], 0.5)
        self.assertGreater(test_rank["hit@3"], 0.5)

        integrity = self.training_metrics["candidate_group_integrity"]
        self.assertEqual(integrity["train"]["groups_without_positive"], 0)
        self.assertEqual(integrity["train"]["groups_without_negative"], 0)
        self.assertEqual(integrity["val"]["groups_without_positive"], 0)
        self.assertEqual(integrity["val"]["groups_without_negative"], 0)
        self.assertEqual(integrity["test"]["groups_without_positive"], 0)
        self.assertEqual(integrity["test"]["groups_without_negative"], 0)

        self.assertIn("test", self.ranking_metrics)
        self.assertGreater(self.ranking_metrics["test"]["mrr"], 0.3)


@unittest.skipUnless(
    _gnn_visualizations_ready(),
    "Optional GNN visualization batch not found; run scripts/run_visualization_sanity.py on the GNN sample list",
)
class TestPropediaActivePipelineGNNVisualizationArtifacts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(VIS_SUMMARY_BATCH_GNN, encoding="utf-8") as handle:
            cls.visualization_summary_batch = json.load(handle)
        with open(VIS_SUMMARY_TOP_GNN, encoding="utf-8") as handle:
            cls.visualization_summary_top = json.load(handle)

    def test_visualization_sanity_batch_passed(self):
        self.assertEqual(len(self.visualization_summary_batch), 10)
        self.assertTrue(all(item["status"] == "success" for item in self.visualization_summary_batch))
        self.assertTrue(all(item["report_exists"] for item in self.visualization_summary_batch))
        self.assertTrue(all(item["viewer_exists"] for item in self.visualization_summary_batch))
        self.assertTrue(all(item["viewer_state_exists"] for item in self.visualization_summary_batch))
        self.assertTrue(all(item["peptide_2d_exists"] for item in self.visualization_summary_batch))

    def test_visualization_sanity_top_ranked_passed(self):
        self.assertEqual(len(self.visualization_summary_top), 10)
        self.assertTrue(all(item["status"] == "success" for item in self.visualization_summary_top))
        self.assertTrue(all(item["report_exists"] for item in self.visualization_summary_top))
        self.assertTrue(all(item["viewer_exists"] for item in self.visualization_summary_top))
        self.assertTrue(all(item["viewer_state_exists"] for item in self.visualization_summary_top))
        self.assertTrue(all(item["peptide_2d_exists"] for item in self.visualization_summary_top))


if __name__ == "__main__":
    unittest.main()
