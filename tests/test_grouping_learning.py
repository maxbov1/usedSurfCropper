import json
import tempfile
import unittest
from pathlib import Path

from organizer.grouping_learning import fit_weights, record_human_save, write_feature_snapshot


class GroupingLearningTests(unittest.TestCase):
    def test_saved_human_positions_create_pairs_and_model(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data_root = root / "data"
            run_id = "group-test"
            items = []
            feedback = []
            assignments = {}
            for index in range(10):
                signature = [0.0] * 1335
                signature[0] = 0.1 if index < 5 else 0.9
                source = root / "input" / f"IMG_{index}.JPG"
                items.append({
                    "path": source,
                    "hash": f"hash-{index}",
                    "shot_type": "full_board",
                    "is_card": False,
                    "board_identity_signature": signature,
                })
                feedback.append({
                    "photo_id": index,
                    "source_path": f"input/IMG_{index}.JPG",
                    "initial_board_id": 10 if index != 9 else 11,
                    "final_board_id": 1 if index < 5 else 2,
                })
                assignments[str(index)] = 1 if index < 5 else 2
            self.assertTrue(write_feature_snapshot(data_root, run_id, items))
            summary = record_human_save(data_root, run_id, feedback, assignments)
            self.assertEqual(summary["photos_with_features"], 10)
            self.assertEqual(summary["positive_pairs"], 20)
            self.assertEqual(summary["negative_pairs"], 25)
            self.assertTrue(summary["model"]["ready"])
            weights = json.loads((data_root / "grouping-learning" / "weights.json").read_text())
            self.assertTrue(weights["ready"])
            self.assertAlmostEqual(sum(weights["weights"].values()), 1.0, places=5)

    def test_feature_snapshot_does_not_include_cards(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            items = [{"path": root / "input" / "card.JPG", "is_card": True, "board_identity_signature": [0.0] * 1335}]
            self.assertIsNone(write_feature_snapshot(root / "data", "cards-only", items))


if __name__ == "__main__":
    unittest.main()
