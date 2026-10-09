import unittest
from pathlib import Path

from organizer.ingest import group_limit_violations, group_scanned


def photo(name, card=False, value=0.0):
    return {"path": Path(name), "is_card": card, "visual_signature": (value,) * 8}


class GroupingTests(unittest.TestCase):
    def test_capture_protocol_splits_long_cardless_run_into_six_view_sessions(self):
        items = [photo(f"IMG_{index}.JPG", value=0.0 if index < 5 else 1.0) for index in range(10)]
        groups = group_scanned(items)
        self.assertEqual([[item["path"].name for item in group] for group in groups], [
            [f"IMG_{index}.JPG" for index in range(6)],
            [f"IMG_{index}.JPG" for index in range(6, 10)],
        ])

    def test_card_starts_next_board_instead_of_being_swallowed(self):
        items = [photo(f"IMG_{index}.JPG", value=0.0) for index in range(5)]
        items += [photo("CARD.JPG", card=True), *[photo(f"NEXT_{index}.JPG", value=0.0) for index in range(5)]]
        groups = group_scanned(items)
        self.assertEqual(len(groups), 2)
        self.assertEqual(len(groups[0]), 6)
        self.assertTrue(groups[0][-1]["is_card"])
        self.assertEqual(len(groups[1]), 5)

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
        self.assertEqual([len(group) for group in groups], [6, 5, 1, 6])
        self.assertEqual([sum(item["is_card"] for item in group) for group in groups], [0, 1, 1, 1])

    def test_oversized_capture_is_split_under_hard_group_cap(self):
        items = [photo(f"IMG_{index}.JPG", value=0.0) for index in range(13)]
        groups = group_scanned(items)
        self.assertEqual(sum(len(group) for group in groups), 13)
        self.assertEqual([len(group) for group in groups], [6, 6, 1])
        self.assertTrue(all(len(group) <= 7 for group in groups))

    def test_shot_limits_split_duplicate_fin_details(self):
        items = [
            {**photo("CARD.JPG", card=True), "shot_type": "card"},
            *[{**photo(f"FULL_{index}.JPG"), "shot_type": "full_board"} for index in range(4)],
            {**photo("PROFILE.JPG"), "shot_type": "side_profile"},
            {**photo("FIN-1.JPG"), "shot_type": "fin_detail"},
            {**photo("FIN-2.JPG"), "shot_type": "fin_detail"},
        ]
        groups = group_scanned(items)
        self.assertEqual([len(group) for group in groups], [7, 1])
        self.assertEqual(group_limit_violations(groups[0]), {})
        self.assertEqual(group_limit_violations(groups[1]), {})

    def test_board_similarity_splits_mixed_visual_session(self):
        items = [
            {**photo("CARD.JPG", card=True), "shot_type": "card"},
            *[photo(f"BLUE_{index}.JPG", value=0.0) for index in range(3)],
            *[photo(f"WHITE_{index}.JPG", value=0.2) for index in range(3)],
        ]
        groups = group_scanned(items)
        self.assertEqual([len(group) for group in groups], [4, 3])
        self.assertEqual(sum(item["is_card"] for item in groups[0]), 1)

    def test_card_stays_with_following_session_when_taken_first(self):
        items = [
            {**photo("CARD.JPG", card=True, value=0.0), "shot_type": "card"},
            *[photo(f"BLUE_{index}.JPG", value=0.0) for index in range(6)],
        ]
        groups = group_scanned(items)
        self.assertEqual([len(group) for group in groups], [7])
        self.assertTrue(any(item.get("is_card") for item in groups[0]))

    def test_card_stays_with_preceding_session_when_taken_last(self):
        items = [
            *[{**photo(f"BLUE_{index}.JPG", value=0.0), "shot_type": "full_board"} for index in range(3)],
            {**photo("CARD.JPG", card=True, value=0.2), "shot_type": "card"},
            *[{**photo(f"WHITE_{index}.JPG", value=0.2), "shot_type": "full_board"} for index in range(3)],
        ]
        groups = group_scanned(items)
        self.assertEqual([len(group) for group in groups], [4, 3])
        self.assertTrue(any(item.get("is_card") for item in groups[0]))

    def test_uncertain_generic_details_do_not_trigger_fin_limit(self):
        items = [
            {**photo("DETAIL-1.JPG"), "shot_type": "detail"},
            {**photo("DETAIL-2.JPG"), "shot_type": "detail"},
        ]
        self.assertEqual(group_limit_violations(items), {})


if __name__ == "__main__":
    unittest.main()
