import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from organizer.ingest import _attach_cards_to_anchored_clusters, _enforce_shot_limits, _split_quarantine_photos, board_group_diagnostics, board_identity_comparison, board_identity_diagnostics, board_identity_signature, group_limit_violations, group_scanned, visual_distance


def photo(name, card=False, value=0.0):
    return {"path": Path(name), "is_card": card, "visual_signature": (value,) * 8}


class GroupingTests(unittest.TestCase):
    def test_board_identity_ignores_room_and_separates_board_color(self):
        first = Image.new("RGB", (160, 240), (35, 35, 35))
        second = Image.new("RGB", (160, 240), (150, 150, 150))
        different = Image.new("RGB", (160, 240), (150, 150, 150))
        for image, color in ((first, (30, 70, 210)), (second, (30, 70, 210)), (different, (210, 55, 35))):
            ImageDraw.Draw(image).rounded_rectangle((45, 20, 115, 220), radius=28, fill=color)
        candidate = (45, 20, 70, 200)
        first_item = {"board_identity_signature": board_identity_signature(first, candidate, "full_board")}
        second_item = {"board_identity_signature": board_identity_signature(second, candidate, "full_board")}
        different_item = {"board_identity_signature": board_identity_signature(different, candidate, "full_board")}
        self.assertLess(visual_distance(first_item, second_item), visual_distance(first_item, different_item))

    def test_board_identity_does_not_include_background_outside_candidate(self):
        first = Image.new("RGB", (240, 320), (20, 20, 20))
        second = Image.new("RGB", (240, 320), (220, 40, 40))
        for image in (first, second):
            ImageDraw.Draw(image).rounded_rectangle((70, 40, 170, 280), radius=28, fill=(30, 70, 210))
        candidate = (70, 40, 100, 240)
        left = {"board_identity_signature": board_identity_signature(first, candidate, "full_board")}
        right = {"board_identity_signature": board_identity_signature(second, candidate, "full_board")}
        comparison = board_identity_comparison(left, right)
        self.assertLess(comparison["total"], 0.01)

    def test_board_identity_masks_card_region_from_writable_buffer(self):
        image = Image.new("RGB", (240, 320), (20, 20, 20))
        ImageDraw.Draw(image).rounded_rectangle((70, 40, 170, 280), radius=28, fill=(30, 70, 210))
        ImageDraw.Draw(image).rectangle((90, 130, 145, 175), fill=(240, 190, 30))
        signature = board_identity_signature(
            image,
            (70, 40, 100, 240),
            "full_board",
            [(90, 130, 55, 45)],
        )
        self.assertTrue(signature)

    def test_board_identity_diagnostics_exposes_capture_and_comparison_terms(self):
        image = Image.new("RGB", (160, 240), (40, 40, 40))
        ImageDraw.Draw(image).rounded_rectangle((45, 20, 115, 220), radius=28, fill=(30, 70, 210))
        signature = board_identity_signature(image, (45, 20, 70, 200), "full_board")
        left = {"path": Path("LEFT.JPG"), "is_card": False, "shot_type": "full_board", "board_identity_signature": signature}
        right = {"path": Path("RIGHT.JPG"), "is_card": False, "shot_type": "full_board", "board_identity_signature": signature}
        report = board_identity_diagnostics([left, right])
        self.assertEqual(report["photos"][0]["signature_status"], "ready")
        self.assertEqual(report["photos"][0]["value_count"], len(signature))
        self.assertEqual(board_identity_comparison(left, right)["status"], "ok")
        self.assertEqual(report["adjacent_comparisons"][0]["total"], 0.0)

    def test_board_group_diagnostics_exposes_internal_and_competing_matches(self):
        signature = [0.0] * 1335
        other_signature = [1.0] * 1335
        first = {"path": Path("A.JPG"), "is_card": False, "shot_type": "full_board", "board_identity_signature": signature}
        second = {"path": Path("A-SIDE.JPG"), "is_card": False, "shot_type": "side_profile", "board_identity_signature": signature}
        other = {"path": Path("B.JPG"), "is_card": False, "shot_type": "full_board", "board_identity_signature": other_signature}
        report = board_group_diagnostics([[first, second], [other]])
        self.assertEqual(report[0]["internal_match_count"], 1)
        self.assertEqual(report[0]["weakest_internal_match"]["total"], 0.0)
        self.assertEqual(report[0]["strongest_competing_match"]["other_group"], 2)

    def test_capture_protocol_splits_long_cardless_run_into_six_view_sessions(self):
        items = [photo(f"IMG_{index}.JPG", value=0.0 if index < 5 else 1.0) for index in range(10)]
        groups = group_scanned(items)
        self.assertEqual(sum(len(group) for group in groups), 10)
        self.assertTrue(all(len(group) <= 7 for group in groups))

    def test_card_starts_next_board_instead_of_being_swallowed(self):
        items = [photo("CARD.JPG", card=True), *[photo(f"NEXT_{index}.JPG", value=0.0) for index in range(5)]]
        groups = group_scanned(items)
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0]), 6)
        self.assertTrue(groups[0][0]["is_card"])

    def test_card_slots_start_with_full_board_bridge(self):
        signature = [0.0] * 1335
        items = [
            {**photo("CARD-1.JPG", card=True), "capture": "2026-01-01 10:00:00", "_sequence_index": 0},
            {**photo("BRIDGE-1.JPG"), "shot_type": "full_board", "capture": "2026-01-01 10:00:10", "_sequence_index": 1, "board_identity_signature": signature},
            {**photo("OTHER-1.JPG"), "shot_type": "full_board", "capture": "2026-01-01 10:00:20", "_sequence_index": 2, "board_identity_signature": signature},
        ]
        groups = group_scanned(items)
        self.assertEqual(len(groups), 1)
        bridge = next(item for item in groups[0] if item["path"].name == "BRIDGE-1.JPG")
        self.assertEqual(bridge["identity_assignment_reason"], "card_bridge_seed")

    def test_anchored_card_slots_do_not_re_match_cards_to_clusters(self):
        cards = [
            {**photo("CARD-1.JPG", card=True), "_sequence_index": 0},
            {**photo("CARD-2.JPG", card=True), "_sequence_index": 10},
        ]
        clusters = [[photo("BOARD-1.JPG")], [photo("BOARD-2.JPG")]]
        groups = _attach_cards_to_anchored_clusters(cards, clusters)
        self.assertEqual(groups[0][0]["path"].name, "CARD-1.JPG")
        self.assertEqual(groups[1][0]["path"].name, "CARD-2.JPG")

    def test_consecutive_cards_remain_one_visible_conflict_group(self):
        items = [photo("CARD-1.JPG", card=True), photo("CARD-2.JPG", card=True), photo("VIEW.JPG")]
        groups = group_scanned(items)
        self.assertEqual([sum(item["is_card"] for item in group) for group in groups], [1, 1])

    def test_consecutive_late_cards_remain_explicit_conflicts(self):
        items = [
            *[photo(f"FIRST_{index}.JPG", value=0.0) for index in range(5)],
            *[photo(f"SECOND_{index}.JPG", value=1.0) for index in range(5)],
            photo("CARD-1.JPG", card=True),
            photo("CARD-2.JPG", card=True),
            photo("CARD-3.JPG", card=True),
            *[photo(f"THIRD_{index}.JPG", value=2.0) for index in range(5)],
        ]
        groups = group_scanned(items)
        self.assertEqual([len(group) for group in groups], [6, 5, 1, 5, 1])
        self.assertEqual(sum(len(group) for group in groups), 18)
        self.assertEqual([sum(item["is_card"] for item in group) for group in groups], [1, 0, 1, 0, 1])

    def test_oversized_capture_is_preserved_for_review(self):
        items = [photo(f"IMG_{index}.JPG", value=0.0) for index in range(13)]
        groups = group_scanned(items)
        self.assertEqual(sum(len(group) for group in groups), 13)
        self.assertEqual([len(group) for group in groups], [13])

    def test_shot_limits_remain_review_data(self):
        items = [
            {**photo("CARD.JPG", card=True), "shot_type": "card"},
            *[{**photo(f"FULL_{index}.JPG"), "shot_type": "full_board"} for index in range(4)],
            {**photo("PROFILE.JPG"), "shot_type": "side_profile"},
            {**photo("FIN-1.JPG"), "shot_type": "fin_detail"},
            {**photo("FIN-2.JPG"), "shot_type": "fin_detail"},
        ]
        groups = group_scanned(items)
        self.assertEqual([len(group) for group in groups], [8])
        self.assertEqual(group_limit_violations(groups[0]), {"fin_detail": 2, "total": 8})

    def test_model_output_splits_over_limit_composition(self):
        group = [
            {**photo("CARD.JPG", card=True), "shot_type": "card"},
            *[{**photo(f"FULL-{index}.JPG"), "shot_type": "full_board"} for index in range(7)],
            *[{**photo(f"PROFILE-{index}.JPG"), "shot_type": "side_profile"} for index in range(3)],
        ]
        limited = _enforce_shot_limits([group])
        self.assertEqual([len(item) for item in limited], [5, 4, 1, 1])
        self.assertTrue(all(not group_limit_violations(item) for item in limited))

    def test_board_similarity_splits_mixed_visual_session(self):
        items = [
            {**photo("CARD.JPG", card=True), "shot_type": "card"},
            *[photo(f"BLUE_{index}.JPG", value=0.0) for index in range(3)],
            *[photo(f"WHITE_{index}.JPG", value=0.2) for index in range(3)],
        ]
        groups = group_scanned(items)
        self.assertEqual(len(groups), 2)
        self.assertEqual([len(group) for group in groups], [4, 3])
        self.assertEqual(sum(item["is_card"] for item in groups[0]), 1)
        self.assertEqual(sum(item["is_card"] for item in groups[1]), 0)

    def test_card_stays_with_following_session_when_taken_first(self):
        items = [
            {**photo("CARD.JPG", card=True, value=0.0), "shot_type": "card"},
            *[photo(f"BLUE_{index}.JPG", value=0.0) for index in range(6)],
        ]
        groups = group_scanned(items)
        self.assertEqual([len(group) for group in groups], [7])
        self.assertTrue(any(item.get("is_card") for item in groups[0]))

    def test_card_evidence_inside_board_photo_does_not_create_anchor(self):
        items = [
            {**photo("CARD.JPG", card=True), "capture": "2026-01-01 10:00:00"},
            {**photo("BOARD.JPG", value=0.0), "contains_inventory_card": True, "capture": "2026-01-01 10:00:30"},
            {**photo("BOARD-2.JPG", value=0.0), "contains_inventory_card": True, "capture": "2026-01-01 10:01:00"},
        ]
        groups = group_scanned(items)
        self.assertEqual(len(groups), 1)
        self.assertEqual(sum(item.get("is_card", False) for item in groups[0]), 1)

    def test_trailing_card_moves_to_next_board_session(self):
        items = [
            *[{**photo(f"BLUE_{index}.JPG", value=0.0), "shot_type": "full_board"} for index in range(2)],
            {**photo("CARD.JPG", card=True, value=0.0), "shot_type": "card"},
            *[{**photo(f"WHITE_{index}.JPG", value=0.2), "shot_type": "full_board"} for index in range(3)],
        ]
        groups = group_scanned(items)
        self.assertEqual([len(group) for group in groups], [2, 4])
        self.assertEqual([sum(item.get("is_card") for item in group) for group in groups], [0, 1])

    def test_late_card_moves_to_visually_matching_following_board(self):
        items = [
            {**photo("LAST-FIN.JPG", value=0.0), "shot_type": "fin_detail"},
            {**photo("CARD.JPG", card=True, value=0.2), "shot_type": "card"},
            *[{**photo(f"NEXT_{index}.JPG", value=0.2), "shot_type": "full_board"} for index in range(4)],
        ]
        groups = group_scanned(items)
        self.assertEqual([len(group) for group in groups], [6])
        self.assertEqual(sum(item.get("is_card") for item in groups[0]), 1)

    def test_card_does_not_stay_after_preceding_photo_set(self):
        items = [
            *[{**photo(f"BOARD_{index}.JPG", value=0.0), "shot_type": "full_board"} for index in range(2)],
            {**photo("CARD.JPG", card=True, value=0.2), "shot_type": "card"},
            *[{**photo(f"NEXT_{index}.JPG", value=0.2), "shot_type": "full_board"} for index in range(3)],
        ]
        groups = group_scanned(items)
        self.assertEqual([len(group) for group in groups], [2, 4])
        self.assertEqual([sum(item.get("is_card") for item in group) for group in groups], [0, 1])

    def test_card_count_drives_one_to_one_visual_cluster_matching(self):
        items = [
            {**photo("CARD-BLUE.JPG", card=True, value=0.0), "shot_type": "card"},
            *[photo(f"BLUE_{index}.JPG", value=0.0) for index in range(4)],
            {**photo("CARD-WHITE.JPG", card=True, value=0.2), "shot_type": "card"},
            *[photo(f"WHITE_{index}.JPG", value=0.2) for index in range(4)],
        ]
        groups = group_scanned(items)
        self.assertEqual(len(groups), 2)
        self.assertTrue(groups[0][0]["is_card"])
        self.assertTrue(groups[1][0]["is_card"])
        self.assertEqual([len(group) for group in groups], [5, 5])

    def test_side_profile_stays_with_preceding_card_lane(self):
        items = [
            {**photo("CARD-1.JPG", card=True), "shot_type": "card", "capture": "2026-01-01 10:00:00"},
            *[
                {**photo(f"BOARD-{index}.JPG", value=0.0), "shot_type": "full_board", "capture": f"2026-01-01 10:0{index + 1}:00"}
                for index in range(3)
            ],
            {**photo("PROFILE.JPG", value=0.8), "shot_type": "side_profile", "capture": "2026-01-01 10:04:00"},
            {**photo("CARD-2.JPG", card=True), "shot_type": "card", "capture": "2026-01-01 10:05:00"},
            *[
                {**photo(f"NEXT-{index}.JPG", value=1.0), "shot_type": "full_board", "capture": f"2026-01-01 10:0{6 + index}:00"}
                for index in range(3)
            ],
        ]
        groups = group_scanned(items)
        self.assertEqual([item["path"].name for item in groups[0]], [
            "CARD-1.JPG", "BOARD-0.JPG", "BOARD-1.JPG", "BOARD-2.JPG", "PROFILE.JPG",
        ])
        self.assertEqual(groups[1][0]["path"].name, "CARD-2.JPG")
        self.assertTrue(any(item["path"].name == "PROFILE.JPG" for item in groups[0]))
        self.assertFalse(any(item["path"].name == "PROFILE.JPG" for item in groups[1]))

    def test_global_identity_reconciliation_can_move_profile_across_soft_lane(self):
        def identity(value, shot):
            signature = [value] * 1335
            signature[1332:1335] = {
                "full_board": [1.0, 0.0, 0.0],
                "side_profile": [0.0, 1.0, 0.0],
            }.get(shot, [0.0, 0.0, 1.0])
            return signature

        items = [
            {**photo("CARD-1.JPG", card=True), "capture": "2026-01-01 10:00:00"},
            *[{**photo(f"A-{index}.JPG"), "shot_type": "full_board", "capture": f"2026-01-01 10:0{index + 1}:00", "board_identity_signature": identity(0.0, "full_board")} for index in range(3)],
            {**photo("A-PROFILE.JPG"), "shot_type": "side_profile", "capture": "2026-01-01 10:04:00", "board_identity_signature": identity(0.0, "side_profile")},
            {**photo("CARD-2.JPG", card=True), "capture": "2026-01-01 10:05:00"},
            *[{**photo(f"B-{index}.JPG"), "shot_type": "full_board", "capture": f"2026-01-01 10:0{index + 6}:00", "board_identity_signature": identity(1.0, "full_board")} for index in range(3)],
        ]
        groups = group_scanned(items)
        self.assertTrue(any(item["path"].name == "A-PROFILE.JPG" for item in groups[0]))
        self.assertFalse(any(item["path"].name == "A-PROFILE.JPG" for item in groups[1]))

    def test_side_profile_uses_adjacent_group_mean_time(self):
        signature = [0.0] * 1335
        items = [
            {**photo("CARD-1.JPG", card=True), "capture": "2026-01-01 10:00:00"},
            {**photo("BOARD-1.JPG"), "shot_type": "full_board", "capture": "2026-01-01 10:00:30", "board_identity_signature": signature},
            {**photo("PROFILE.JPG", value=1.0), "shot_type": "side_profile", "capture": "2026-01-01 10:02:00", "board_identity_signature": [1.0] * 1335},
            {**photo("CARD-2.JPG", card=True), "capture": "2026-01-01 10:05:00"},
            {**photo("BOARD-2.JPG"), "shot_type": "full_board", "capture": "2026-01-01 10:05:30", "board_identity_signature": [1.0] * 1335},
        ]
        groups = group_scanned(items)
        first = next(group for group in groups if any(item["path"].name == "CARD-1.JPG" for item in group))
        second = next(group for group in groups if any(item["path"].name == "CARD-2.JPG" for item in group))
        self.assertTrue(any(item["path"].name == "PROFILE.JPG" for item in first))
        self.assertFalse(any(item["path"].name == "PROFILE.JPG" for item in second))

    def test_global_identity_handles_empty_seed_lane(self):
        signature = [0.0] * 1335
        items = [
            {**photo("CARD-1.JPG", card=True), "capture": "2026-01-01 10:00:00"},
            {**photo("CARD-2.JPG", card=True), "capture": "2026-01-01 10:00:01"},
            {**photo("BOARD.JPG"), "capture": "2026-01-01 10:00:02", "board_identity_signature": signature},
        ]
        groups = group_scanned(items)
        self.assertEqual(len(groups), 2)
        self.assertEqual(sum(len(group) for group in groups), 3)

    def test_explicit_photo_identity_cannot_be_assigned_to_contradictory_card(self):
        items = [
            {**photo("PYZEL-CARD.JPG", card=True), "identifier": {"brand": "Pyzel", "model": "Bastard"}},
            {**photo("RUMANER-CARD.JPG", card=True), "identifier": {"brand": "Rumaner", "model": "Annihilator"}},
            {**photo("RUMANER.JPG"), "board_identity_signature": [0.0] * 1335,
             "photo_ocr_text": "Rumaner Annihilator", "identity_evidence": {
                 "brand": {"value": "Rumaner", "confidence": 0.62},
                 "model": {"value": "Annihilator", "confidence": 0.62},
             }},
        ]
        groups = group_scanned(items)
        rumaner_group = next(group for group in groups if any(item.get("is_card") and item.get("identifier", {}).get("brand") == "Rumaner" for item in group))
        self.assertTrue(any(item["path"].name == "RUMANER.JPG" for item in rumaner_group))
        pyzel_group = next(group for group in groups if any(item.get("is_card") and item.get("identifier", {}).get("brand") == "Pyzel" for item in group))
        self.assertFalse(any(item["path"].name == "RUMANER.JPG" for item in pyzel_group))

    def test_incompatible_identity_signatures_are_not_perfect_matches(self):
        left = {"board_identity_signature": [0.0] * 1335}
        right = {"board_identity_signature": [0.0] * 1334}
        comparison = board_identity_comparison(left, right)
        self.assertEqual(comparison["status"], "incompatible_signature")
        self.assertEqual(comparison["total"], 1.0)

    def test_ambiguous_same_angle_photo_is_quarantined(self):
        signature = [0.0] * 1335
        items = [
            {**photo("CARD-1.JPG", card=True), "shot_type": "card", "capture": "2026-01-01 10:00:00"},
            {**photo("BOARD-1.JPG"), "shot_type": "full_board", "capture": "2026-01-01 10:00:30", "board_identity_signature": signature},
            {**photo("CARD-2.JPG", card=True), "shot_type": "card", "capture": "2026-01-01 10:03:00"},
            {**photo("BOARD-2.JPG"), "shot_type": "full_board", "capture": "2026-01-01 10:03:30", "board_identity_signature": signature},
            {**photo("AMBIGUOUS.JPG"), "shot_type": "full_board", "capture": "2026-01-01 10:04:00", "board_identity_signature": signature},
        ]
        groups = group_scanned(items)
        review = next(group for group in groups if any(item["path"].name == "AMBIGUOUS.JPG" for item in group))
        self.assertFalse(any(item.get("is_card") for item in review))
        ambiguous = next(item for item in review if item["path"].name == "AMBIGUOUS.JPG")
        self.assertIn(ambiguous["identity_assignment_reason"], {"ambiguous_identity", "weak_identity_match"})

    def test_quarantine_remainder_is_split_instead_of_becoming_one_group(self):
        photos = [
            {**photo("BLUE-1.JPG", value=0.0), "identity_assignment_reason": "ambiguous_identity"},
            {**photo("BLUE-2.JPG", value=0.0), "identity_assignment_reason": "ambiguous_identity"},
            {**photo("RED.JPG", value=1.0), "identity_assignment_reason": "weak_identity_match"},
        ]
        groups = _split_quarantine_photos(photos)
        self.assertEqual([len(group) for group in groups], [2, 1])
        self.assertTrue(all(item.get("_unassigned_cluster") for item in groups[0]))
        self.assertTrue(groups[1][0].get("_unassigned_photo"))

    def test_uncertain_generic_details_do_not_trigger_fin_limit(self):
        items = [
            {**photo("DETAIL-1.JPG"), "shot_type": "detail"},
            {**photo("DETAIL-2.JPG"), "shot_type": "detail"},
        ]
        self.assertEqual(group_limit_violations(items), {})

    def test_visual_runs_are_not_forced_to_card_count(self):
        items = [
            {**photo("CARD-BLUE.JPG", card=True, value=0.0)},
            *[photo(f"BLUE_{index}.JPG", value=0.0) for index in range(3)],
            *[photo(f"WHITE_{index}.JPG", value=0.2) for index in range(3)],
            *[photo(f"RED_{index}.JPG", value=0.5) for index in range(3)],
            {**photo("CARD-WHITE.JPG", card=True, value=0.2)},
        ]
        groups = group_scanned(items)
        self.assertEqual([len(group) for group in groups], [4, 3, 4])
        self.assertEqual([sum(item.get("is_card", False) for item in group) for group in groups], [1, 0, 1])

    def test_weak_card_match_is_not_forced(self):
        items = [
            {**photo("CARD.JPG", card=True, value=0.0)},
            *[photo(f"BOARD_{index}.JPG", value=1.0) for index in range(3)],
        ]
        groups = group_scanned(items)
        self.assertEqual([len(group) for group in groups], [3, 1])
        self.assertFalse(any(item.get("is_card") for item in groups[0]))
        self.assertTrue(groups[1][0]["is_card"])


if __name__ == "__main__":
    unittest.main()
