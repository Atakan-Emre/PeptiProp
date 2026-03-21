import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
CANONICAL_DIR = ROOT / "data" / "canonical"
SPLITS_DIR = CANONICAL_DIR / "splits"
PAIRS_DIR = CANONICAL_DIR / "pairs"
TRAINING_DIR = ROOT / "outputs" / "training" / "peptidquantum_v0_1_final_best_mlx_ablation"
VIS_SUMMARY_BATCH = ROOT / "outputs" / "analysis_propedia_batch_mlx" / "visualization_sanity_summary.json"
VIS_SUMMARY_TOP = ROOT / "outputs" / "analysis_propedia_top_ranked_batch_mlx" / "visualization_sanity_summary.json"


class TestPropediaActivePipelineMLX(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.complexes = pd.read_parquet(CANONICAL_DIR / "complexes.parquet")
        with open(PAIRS_DIR / "pair_data_report.json", encoding="utf-8") as handle:
            cls.pair_report = json.load(handle)
        with open(PAIRS_DIR / "candidate_set_report.json", encoding="utf-8") as handle:
            cls.candidate_report = json.load(handle)
        with open(TRAINING_DIR / "metrics.json", encoding="utf-8") as handle:
            cls.training_metrics = json.load(handle)
        with open(VIS_SUMMARY_BATCH, encoding="utf-8") as handle:
            cls.visualization_summary_batch = json.load(handle)
        with open(VIS_SUMMARY_TOP, encoding="utf-8") as handle:
            cls.visualization_summary_top = json.load(handle)

    def test_split_metadata_matches_files(self):
        for split_name in ("train", "val", "test"):
            split_ids = {
                line.strip()
                for line in (SPLITS_DIR / f"{split_name}_ids.txt").read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.startswith("#")
            }
            tagged_ids = set(self.complexes.loc[self.complexes["split_tag"] == split_name, "complex_id"])
            self.assertSetEqual(split_ids, tagged_ids, msg=f"split drift for {split_name}")

    def test_pair_reports_are_unique_clean_and_candidate_rich(self):
        allowed_neg_types = {"easy", "hard", "structure_hard", "positive"}
        for split_name, report in self.pair_report.items():
            self.assertEqual(report["duplicate_pair_count"], 0, msg=f"duplicates in {split_name}")
            self.assertGreater(report["negative_pairs"], report["positive_pairs"], msg=f"not enough candidates in {split_name}")
            self.assertEqual(set(report["quality_flag_distribution"].keys()), {"clean"})
            self.assertSetEqual(set(report["split_values"]), {split_name})
            self.assertTrue(report["split_column_consistent"], msg=f"split column inconsistent in {split_name}")
            self.assertTrue(
                set(report["negative_type_distribution"].keys()).issubset(allowed_neg_types),
                msg=f"unexpected negative types in {split_name}",
            )
            self.assertGreaterEqual(self.candidate_report[split_name]["avg_candidates_per_protein"], 6.0)

    def test_mlx_final_artifacts_and_metrics_are_present(self):
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
            "threshold_vs_f1_table.csv",
            "test_topk_candidates.csv",
            "test_topk_positive_hits.csv",
            "top_ranked_examples.json",
            "ablation_summary.csv",
            "ablation_heatmap.png",
            "model_family_comparison.png",
            "selection_summary.json",
        }
        self.assertTrue(required_files.issubset({path.name for path in TRAINING_DIR.iterdir()}))

        val_metrics = self.training_metrics["validation_metrics_at_selected_threshold"]
        test_metrics = self.training_metrics["test_metrics"]
        val_rank = self.training_metrics["val_ranking_metrics"]
        test_rank = self.training_metrics["test_ranking_metrics"]

        for key in ("auroc", "auprc", "f1", "mcc"):
            self.assertGreaterEqual(val_metrics[key], 0.0)
            self.assertLessEqual(val_metrics[key], 1.0)
            self.assertGreaterEqual(test_metrics[key], 0.0)
            self.assertLessEqual(test_metrics[key], 1.0)

        random_mrr = float(np.mean([1.0 / i for i in range(1, 7)]))
        self.assertGreater(val_rank["mrr"], random_mrr)
        self.assertGreater(test_rank["mrr"], random_mrr)
        self.assertGreater(val_rank["hit@3"], 0.5)
        self.assertGreater(test_rank["hit@3"], 0.5)

        integrity = self.training_metrics["candidate_group_integrity"]
        self.assertEqual(integrity["train"]["groups_without_positive"], 0)
        self.assertEqual(integrity["train"]["groups_without_negative"], 0)
        self.assertEqual(integrity["val"]["groups_without_positive"], 0)
        self.assertEqual(integrity["val"]["groups_without_negative"], 0)
        self.assertEqual(integrity["test"]["groups_without_positive"], 0)
        self.assertEqual(integrity["test"]["groups_without_negative"], 0)

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
