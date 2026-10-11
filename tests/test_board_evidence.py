import unittest
from pathlib import Path

from PIL import Image

from organizer.board_evidence import board_evidence_diagnostics, board_evidence_regions, detect_side_profile_fins, extract_board_evidence, extract_photo_evidence


class BoardEvidenceTests(unittest.TestCase):
    def test_evidence_is_angle_specific_and_optional(self):
        text = "BRAND: Pyzel\nMODEL: Bastard\nFIN SYSTEM: Futures\n# OF BOXES: 3.0\nINCLUDED Y/N: Y"
        deck = extract_photo_evidence(text, "full_board")
        profile = extract_photo_evidence(text, "side_profile")
        self.assertEqual(set(deck), {"brand", "model", "fin_system", "fin_box_count"})
        self.assertEqual(set(profile), set())

    def test_side_profile_only_asserts_fins_when_explicitly_visible_in_text(self):
        evidence = extract_photo_evidence("FINS INCLUDED: Y", "side_profile")
        self.assertEqual(set(evidence), set())

    def test_side_profile_visual_detector_is_conservative_without_candidate(self):
        evidence = detect_side_profile_fins(Image.new("RGB", (100, 200), "white"), None)
        self.assertEqual(evidence["status"], "unknown")
        self.assertEqual(evidence["value"], "")

    def test_board_regions_stay_inside_detected_board(self):
        regions = board_evidence_regions(Image.new("RGB", (1000, 2000), "white"), (200, 100, 400, 1600))
        self.assertEqual(set(regions), {"logo_brand", "model_marking", "tail_fin"})
        for x, y, width, height in regions.values():
            self.assertGreaterEqual(x, 200)
            self.assertGreaterEqual(y, 100)
            self.assertLessEqual(x + width, 600)
            self.assertLessEqual(y + height, 1700)

    def test_board_evidence_records_regions_without_claiming_identity(self):
        evidence = extract_board_evidence(
            Image.new("RGB", (1000, 2000), "white"),
            (200, 100, 400, 1600),
            "full_board",
            ocr_runner=lambda image: ("", {}, ""),
        )
        self.assertEqual(evidence["status"], "ready")
        self.assertEqual(set(evidence["regions"]), {"logo_brand", "model_marking", "tail_fin"})
        self.assertEqual(evidence["ocr_passes"], 9)
    def test_photo_text_is_evidence_not_ground_truth(self):
        evidence = extract_photo_evidence("BRAND: Pyzel\nMODEL: Bastard\nFIN SYSTEM: Futures", "full_board")
        self.assertEqual(evidence["brand"]["value"], "Pyzel")
        self.assertEqual(evidence["fin_system"]["value"], "Futures")
        self.assertNotIn("fins_included", evidence)

    def test_visible_fin_count_is_separate_from_fin_boxes_and_card_inclusion(self):
        evidence = extract_photo_evidence("FIN COUNT: 3", "fin_detail")
        self.assertEqual(evidence["fins_present_count"]["value"], "3")
        self.assertNotIn("fins_included", evidence)
        self.assertNotIn("fin_box_count", evidence)

    def test_group_compares_card_fields_to_member_photos(self):
        group = [[
            {"path": Path("CARD.JPG"), "is_card": True, "identifier": {"brand": "Pyzel", "model": "Bastard", "fin_system": "Futures", "fins_included": "yes"}},
            {"path": Path("BOARD.JPG"), "is_card": False, "photo_ocr_text": "PYZEL THE BASTARD FUTURES", "identity_evidence": extract_photo_evidence("BRAND: Pyzel\nMODEL: Bastard\nFIN SYSTEM: Futures", "full_board")},
        ]]
        report = board_evidence_diagnostics(group)[0]["fields"]
        self.assertEqual(report["brand"]["status"], "confirmed")
        self.assertEqual(report["model"]["status"], "confirmed")
        self.assertEqual(report["fin_system"]["status"], "confirmed")
        self.assertEqual(report["fins_included"]["status"], "card_only")

    def test_conflict_is_reported_without_overwriting_card_value(self):
        group = [[
            {"path": Path("CARD.JPG"), "is_card": True, "identifier": {"brand": "Pyzel"}},
            {"path": Path("BOARD.JPG"), "is_card": False, "photo_ocr_text": "BRAND: Lost", "identity_evidence": extract_photo_evidence("BRAND: Lost", "full_board")},
        ]]
        field = board_evidence_diagnostics(group)[0]["fields"]["brand"]
        self.assertEqual(field["status"], "conflicting")
        self.assertEqual(field["value"], "Pyzel")


if __name__ == "__main__":
    unittest.main()
