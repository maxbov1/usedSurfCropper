import unittest
from pathlib import Path

from organizer.ingest import group_scanned


def photo(name, card=False, value=0.0):
    return {"path": Path(name), "is_card": card, "visual_signature": (value,) * 8}


class GroupingTests(unittest.TestCase):
    def test_visual_change_splits_long_cardless_run(self):
        items = [photo(f"IMG_{index}.JPG", value=0.0 if index < 5 else 1.0) for index in range(10)]
        groups = group_scanned(items)
        self.assertEqual([[item["path"].name for item in group] for group in groups], [
            [f"IMG_{index}.JPG" for index in range(5)],
            [f"IMG_{index}.JPG" for index in range(5, 10)],
        ])

    def test_card_starts_next_board_instead_of_being_swallowed(self):
        items = [photo(f"IMG_{index}.JPG", value=0.0) for index in range(5)]
        items += [photo("CARD.JPG", card=True), *[photo(f"NEXT_{index}.JPG", value=0.0) for index in range(5)]]
        groups = group_scanned(items)
        self.assertEqual(len(groups), 2)
        self.assertTrue(groups[1][0]["is_card"])
        self.assertEqual(len(groups[1]), 6)

    def test_consecutive_cards_remain_one_visible_conflict_group(self):
        items = [photo("CARD-1.JPG", card=True), photo("CARD-2.JPG", card=True), photo("VIEW.JPG")]
        groups = group_scanned(items)
        self.assertEqual(len(groups), 1)
        self.assertEqual(sum(item["is_card"] for item in groups[0]), 2)

    def test_late_card_block_is_reconciled_into_cardless_neighbors(self):
        items = [
            *[photo(f"FIRST_{index}.JPG", value=0.0) for index in range(5)],
            *[photo(f"SECOND_{index}.JPG", value=1.0) for index in range(5)],
            photo("CARD-1.JPG", card=True),
            photo("CARD-2.JPG", card=True),
            photo("CARD-3.JPG", card=True),
            *[photo(f"THIRD_{index}.JPG", value=2.0) for index in range(5)],
        ]
        groups = group_scanned(items)
        self.assertEqual([len(group) for group in groups], [6, 6, 6])
        self.assertEqual([sum(item["is_card"] for item in group) for group in groups], [1, 1, 1])
        self.assertEqual([group[-1]["path"].name for group in groups], ["CARD-1.JPG", "CARD-2.JPG", "THIRD_4.JPG"])

    def test_oversized_capture_is_split_under_hard_group_cap(self):
        items = [photo(f"IMG_{index}.JPG", value=0.0) for index in range(13)]
        groups = group_scanned(items)
        self.assertEqual(sum(len(group) for group in groups), 13)
        self.assertTrue(all(len(group) <= 8 for group in groups))


if __name__ == "__main__":
    unittest.main()
