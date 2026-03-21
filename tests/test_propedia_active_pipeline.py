import json
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
CANONICAL_DIR = ROOT / "data" / "canonical"
SPLITS_DIR = CANONICAL_DIR / "splits"
PAIRS_DIR = CANONICAL_DIR / "pairs"
TRAINING_DIR_CLASSICAL = ROOT / "outputs" / "training" / "peptidquantum_v0_1_final_classical"
VIS_SUMMARY_CLASSICAL = ROOT / "outputs" / "analysis_propedia_batch" / "visualization_sanity_summary.json"


def _classical_artifacts_ready() -> bool:
    return (TRAINING_DIR_CLASSICAL / "metrics.json").is_file() and VIS_SUMMARY_CLASSICAL.is_file()


class TestPropediaActivePipelineCanonical(unittest.TestCase):
    """Kanonik veri; klasik eğitim çıktısı gerektirmez."""

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
        res = pd.read_parquet(CANONICAL_DIR / "residues.parquet", columns=["is_interface", "is_pocket"])
        intf_count = int(res["is_interface"].sum())
        pocket_count = int(res["is_pocket"].sum())
        self.assertGreater(intf_count, 0, "is_interface is all-zero — annotate_interface_pocket.py not run?")
        self.assertGreater(pocket_count, 0, "is_pocket is all-zero — annotate_interface_pocket.py not run?")
        self.assertGreater(pocket_count, intf_count, "pocket set should be larger than interface set")

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
        expected_neg_types = {
            "train": {"easy", "hard"},
            "val": {"easy", "hard"},
            "test": {"easy", "hard"},
        }
        for split_name, report in self.pair_report.items():
            self.assertEqual(report["duplicate_pair_count"], 0, msg=f"duplicates in {split_name}")
            self.assertGreater(report["negative_pairs"], report["positive_pairs"], msg=f"not enough candidates in {split_name}")
            self.assertEqual(set(report["quality_flag_distribution"].keys()), {"clean"})
            observed_neg_types = set(report["negative_type_distribution"].keys()) - {"positive"}
            self.assertSetEqual(observed_neg_types, expected_neg_types[split_name])
            self.assertTrue(report["split_column_consistent"], msg=f"split column inconsistent in {split_name}")
            self.assertGreater(self.candidate_report[split_name]["avg_candidates_per_protein"], 1.0)

    def test_pair_parquet_split_column_is_consistent(self):
        for split_name in ("train", "val", "test"):
            pairs = pd.read_parquet(PAIRS_DIR / f"{split_name}_pairs.parquet")
            self.assertSetEqual(set(pairs["split"].astype(str).unique()), {split_name})


@unittest.skipUnless(
    _classical_artifacts_ready(),
    "Klasik eğitim + analysis_propedia_batch özeti yok — isteğe bağlı; aktif hat MLX için tests.test_propedia_active_pipeline_mlx",
)
class TestPropediaActivePipelineClassicalArtifacts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(TRAINING_DIR_CLASSICAL / "metrics.json", encoding="utf-8") as handle:
            cls.training_metrics = json.load(handle)
        with open(VIS_SUMMARY_CLASSICAL, encoding="utf-8") as handle:
            cls.visualization_summary = json.load(handle)

    def test_final_baseline_artifacts_and_metrics(self):
        required_files = {
            "metrics.json",
            "ranking_metrics.json",
            "best_thresholds.json",
            "pair_data_report.json",
            "candidate_set_report.json",
            "calibration_metrics.json",
            "test_summary.txt",
            "train_log.csv",
            "confusion_matrix.png",
            "roc_curve.png",
            "pr_curve.png",
            "validation_threshold_sweep.png",
            "validation_score_histogram_pos_neg.png",
            "score_histogram_pos_neg.png",
            "calibration_curve.png",
        }
        self.assertTrue(required_files.issubset({path.name for path in TRAINING_DIR_CLASSICAL.iterdir()}))
        val_metrics = self.training_metrics["validation_metrics_at_selected_threshold"]
        test_metrics = self.training_metrics["test_metrics"]
        self.assertGreaterEqual(val_metrics["auroc"], 0.0)
        self.assertLessEqual(val_metrics["auroc"], 1.0)
        self.assertGreaterEqual(test_metrics["auroc"], 0.0)
        self.assertLessEqual(test_metrics["auroc"], 1.0)
        self.assertGreaterEqual(test_metrics["f1"], 0.0)
        self.assertLessEqual(test_metrics["f1"], 1.0)
        self.assertGreaterEqual(test_metrics["positive_predictions"], 1)
        self.assertGreaterEqual(test_metrics["negative_predictions"], 1)

    def test_visualization_sanity_batch_passed(self):
        self.assertEqual(len(self.visualization_summary), 10)
        self.assertTrue(all(item["status"] == "success" for item in self.visualization_summary))
        self.assertTrue(all(item["report_exists"] for item in self.visualization_summary))
        self.assertTrue(all(item["viewer_exists"] for item in self.visualization_summary))
        self.assertTrue(all(item["viewer_state_exists"] for item in self.visualization_summary))
        self.assertTrue(all(item["peptide_2d_exists"] for item in self.visualization_summary))


if __name__ == "__main__":
    unittest.main()
