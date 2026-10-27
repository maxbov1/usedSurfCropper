import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from organizer.ingest import (
    assign_shot_types_by_position,
    board_group_diagnostics,
    board_identity_comparison,
    board_identity_diagnostics,
    board_identity_signature,
    board_local_signature,
    board_mask_width_fraction,
    board_region_mask,
    card_signal,
    embedding_distance,
    group_limit_violations,
    group_scanned,
    locate_missed_card_boundary,
    next_shot_type_for_occupancy,
    visual_distance,
    _densest_text_cluster,
    _enforce_shot_limits,
)


def _region_mask(size, region):
    mask_image = Image.new("L", size, 0)
    ImageDraw.Draw(mask_image).rounded_rectangle(region, radius=10, fill=255)
    return np.asarray(mask_image) > 0


def photo(name, card=False, value=0.0):
    return {"path": Path(name), "is_card": card, "visual_signature": (value,) * 8}


class GroupingTests(unittest.TestCase):
    # -- board_identity_signature / visual_distance: standalone diagnostic
    # utilities. They no longer drive grouping (see group_scanned below),
    # but stay useful for displaying similarity to a human reviewer.

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

    def test_board_mask_ignores_background_inside_a_loose_off_center_candidate(self):
        """Regression: an off-center board inside a loose candidate box was
        getting compared mostly on shared studio-wall background, not the
        board (confirmed on real production photos — two unrelated boards
        scored as near-identical this way). A board_mask (e.g. from SAM 2)
        must remove that background before the histogram/layout are built.
        """
        size = (300, 320)
        board_region = (205, 20, 280, 300)
        candidate = (10, 10, 280, 300)
        first = Image.new("RGB", size, (40, 40, 60))
        second = Image.new("RGB", size, (210, 190, 140))
        for image in (first, second):
            ImageDraw.Draw(image).rounded_rectangle(board_region, radius=10, fill=(30, 70, 210))
        mask = _region_mask(size, board_region)

        unmasked = board_identity_comparison(
            {"board_identity_signature": board_identity_signature(first, candidate, "full_board")},
            {"board_identity_signature": board_identity_signature(second, candidate, "full_board")},
        )
        masked = board_identity_comparison(
            {"board_identity_signature": board_identity_signature(first, candidate, "full_board", None, mask)},
            {"board_identity_signature": board_identity_signature(second, candidate, "full_board", None, mask)},
        )
        self.assertGreater(unmasked["total"], 0.2)
        self.assertLess(masked["total"], 0.02)

    def test_board_mask_still_detects_a_genuinely_different_board(self):
        """The mask must not just collapse every comparison to zero: with
        identical background and a different board color inside the mask,
        the distance should still clearly separate the two boards.
        """
        size = (300, 320)
        board_region = (205, 20, 280, 300)
        candidate = (10, 10, 280, 300)
        first = Image.new("RGB", size, (90, 90, 90))
        second = Image.new("RGB", size, (90, 90, 90))
        ImageDraw.Draw(first).rounded_rectangle(board_region, radius=10, fill=(30, 70, 210))
        ImageDraw.Draw(second).rounded_rectangle(board_region, radius=10, fill=(210, 55, 35))
        mask = _region_mask(size, board_region)

        masked = board_identity_comparison(
            {"board_identity_signature": board_identity_signature(first, candidate, "full_board", None, mask)},
            {"board_identity_signature": board_identity_signature(second, candidate, "full_board", None, mask)},
        )
        self.assertGreater(masked["total"], 0.05)

    def test_board_mask_width_fraction_ignores_a_stray_false_positive_speck(self):
        """Regression: a standing, edge-on side-rail shot measured as >50%
        wide by OpenCV's candidate box (confirmed on real production photos
        from this app's actual photographer) because the box mostly covered
        background. SAM 2's own mask puts the true board at 7-10% width —
        but only if a stray false-positive speck elsewhere in frame (a pipe,
        a shadow) doesn't inflate a naive bounding box. This must measure
        the board's own connected blob, not the mask's overall extent.
        """
        height, width = 400, 300
        mask = np.zeros((height, width), dtype=bool)
        mask[20:380, 140:160] = True  # narrow standing board, 20/300 = 6.7% wide
        mask[10:14, 10:14] = True  # stray false-positive speck far away
        fraction = board_mask_width_fraction(mask)
        self.assertLess(fraction, 0.10)

    def test_board_region_mask_is_a_noop_without_sam2_configured(self):
        """With no SAM2 checkpoint configured (the default), board_region_mask
        must return None so callers fall back to the plain candidate
        rectangle — this must never raise or require torch to be installed.
        """
        image = Image.new("RGB", (100, 100), (50, 50, 50))
        self.assertIsNone(board_region_mask(image, (10, 10, 50, 50), "full_board"))

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

    def test_incompatible_identity_signatures_are_not_perfect_matches(self):
        left = {"board_identity_signature": [0.0] * 1335}
        right = {"board_identity_signature": [0.0] * 1334}
        comparison = board_identity_comparison(left, right)
        self.assertEqual(comparison["status"], "incompatible_signature")
        self.assertEqual(comparison["total"], 1.0)

    def test_card_signal_true_for_angle_cropped_real_card_text(self):
        """Regression: a card photographed off-angle or cropped at the frame
        edge can still OCR cleanly (SKU, brand, model and all) even when the
        separate contour-based region detector finds no `card_regions` for
        it. `card_signal` is the standalone check meant to recognize that —
        it must not need `card_regions` to be non-empty to be trusted. This
        is the exact OCR text from a misclassified production card (board
        #49201) that was previously left as `is_card=False` and never
        anchored its own group.
        """
        text = (
            "DATE: 10/7/2026 ITEM # 49201 CHECKED-IN BY: GB\n"
            "NAME: Jason Hatton BRAND: Stewart\n"
            "MOBILE: (714) 300-5043 MODEL: Pigme\n"
            "FIN SYSTEM: Futures\n"
        )
        identifier = {"sku": "49201", "brand": "Stewart", "model": "Pigme", "fin_system": "Futures"}
        self.assertTrue(card_signal(text, identifier))
        card_regions = []
        largest_card_fraction = 0.0
        compact_card_ocr = card_signal(text, identifier) and (not card_regions or largest_card_fraction <= 0.45)
        self.assertTrue(compact_card_ocr)

    # -- group_scanned: capture-order, card-anchored grouping.
    #
    # The whole algorithm: a card starts a new group; every photo after it
    # belongs to that group until the next card. No visual signal is
    # consulted. This replaced a multi-layer visual-similarity matcher
    # (color histograms, then SAM 2-masked histograms, lane-bounded retry
    # scoring, quarantine clustering) that reliably produced cross-board
    # contamination in production: a weak visual match only had to be
    # marginally more confident than the correct answer to win, and real
    # boards photographed next to each other (regardless of how different
    # they actually looked) kept losing photos to one another. Capture
    # order is already correct; these tests check that it's trusted.

    def test_card_anchors_everything_until_the_next_card(self):
        items = [
            photo("CARD-1.JPG", card=True),
            *[photo(f"A-{i}.JPG") for i in range(4)],
            photo("CARD-2.JPG", card=True),
            *[photo(f"B-{i}.JPG") for i in range(3)],
        ]
        groups = group_scanned(items)
        self.assertEqual([len(group) for group in groups], [5, 4])
        self.assertEqual([item["path"].name for item in groups[0]], ["CARD-1.JPG", "A-0.JPG", "A-1.JPG", "A-2.JPG", "A-3.JPG"])
        self.assertEqual([item["path"].name for item in groups[1]], ["CARD-2.JPG", "B-0.JPG", "B-1.JPG", "B-2.JPG"])

    def test_no_visual_signal_can_move_a_photo_across_a_card_boundary(self):
        """The exact real-world failure this replaces: two boards that look
        nothing alike (card-anchored visual signatures deliberately made
        maximally different here) must still group purely by which card
        came before them in capture order.
        """
        signature_a = [0.0] * 1335
        signature_b = [1.0] * 1335
        items = [
            {**photo("CARD-1.JPG", card=True), "board_identity_signature": signature_a},
            *[{**photo(f"A-{i}.JPG"), "shot_type": "full_board", "board_identity_signature": signature_a} for i in range(2)],
            {**photo("CARD-2.JPG", card=True), "board_identity_signature": signature_b},
            # Visually identical to board A's representative, not board B's —
            # a visual matcher would be tempted to steal this for group A.
            {**photo("LOOKS-LIKE-A.JPG"), "shot_type": "full_board", "board_identity_signature": signature_a},
        ]
        groups = group_scanned(items)
        self.assertEqual([len(group) for group in groups], [3, 2])
        self.assertEqual(groups[1][1]["path"].name, "LOOKS-LIKE-A.JPG")

    def test_leading_cardless_run_stays_one_reviewable_block(self):
        """Photos before the first detected card (or an entire cardless
        batch) are not auto-split by a visual-similarity guess — they stay
        one group for a human to assign.
        """
        items = [photo(f"IMG_{i}.JPG", value=0.0 if i < 5 else 1.0) for i in range(10)]
        groups = group_scanned(items)
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0]), 10)

    def test_cardless_batch_of_any_size_stays_one_group(self):
        items = [photo(f"IMG_{i}.JPG") for i in range(13)]
        groups = group_scanned(items)
        self.assertEqual([len(group) for group in groups], [13])

    def test_card_starts_next_board_instead_of_being_swallowed(self):
        items = [photo("CARD.JPG", card=True), *[photo(f"NEXT_{i}.JPG") for i in range(5)]]
        groups = group_scanned(items)
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0]), 6)
        self.assertTrue(groups[0][0]["is_card"])

    def test_consecutive_cards_each_start_their_own_group(self):
        items = [photo("CARD-1.JPG", card=True), photo("CARD-2.JPG", card=True), photo("VIEW.JPG")]
        groups = group_scanned(items)
        self.assertEqual([len(group) for group in groups], [1, 2])
        self.assertEqual([sum(item["is_card"] for item in group) for group in groups], [1, 1])

    def test_empty_input_returns_no_groups(self):
        self.assertEqual(group_scanned([]), [])

    # -- assign_shot_types_by_position: type by position, not by guessing
    # pixel shape. The protocol is fixed: up to 4 full-board views, then 1
    # side profile, then optionally 1 fin detail, per board, in that order.

    def test_shot_types_follow_capture_position(self):
        items = [
            photo("CARD.JPG", card=True),
            *[photo(f"FULL-{i}.JPG") for i in range(4)],
            photo("SIDE.JPG"),
            photo("FIN.JPG"),
        ]
        groups = group_scanned(items)
        assign_shot_types_by_position(groups)
        group = groups[0]
        self.assertEqual(group[0]["shot_type"], "card")
        self.assertEqual([item["shot_type"] for item in group[1:5]], ["full_board"] * 4)
        self.assertEqual(group[5]["shot_type"], "side_profile")
        self.assertEqual(group[6]["shot_type"], "fin_detail")

    def test_missing_side_profile_does_not_shift_later_positions(self):
        """A board with only 2 full-board photos (no side profile, no fin
        detail taken) must not have its 2nd photo mislabeled side_profile —
        position within the *protocol slots*, not raw index, decides type.
        """
        items = [photo("CARD.JPG", card=True), photo("A.JPG"), photo("B.JPG")]
        groups = group_scanned(items)
        assign_shot_types_by_position(groups)
        self.assertEqual([item["shot_type"] for item in groups[0][1:]], ["full_board", "full_board"])

    def test_extra_photo_beyond_protocol_defaults_to_full_board(self):
        items = [
            photo("CARD.JPG", card=True),
            *[photo(f"FULL-{i}.JPG") for i in range(4)],
            photo("SIDE.JPG"),
            photo("FIN.JPG"),
            photo("EXTRA.JPG"),
        ]
        groups = group_scanned(items)
        assign_shot_types_by_position(groups)
        self.assertEqual(groups[0][-1]["shot_type"], "full_board")

    def test_shot_type_assignment_does_not_touch_other_groups(self):
        items = [
            photo("CARD-1.JPG", card=True), photo("A.JPG"),
            photo("CARD-2.JPG", card=True), photo("B-1.JPG"), photo("B-2.JPG"),
        ]
        groups = group_scanned(items)
        assign_shot_types_by_position(groups)
        self.assertEqual(groups[0][1]["shot_type"], "full_board")
        self.assertEqual([item["shot_type"] for item in groups[1][1:]], ["full_board", "full_board"])

    # -- next_shot_type_for_occupancy: re-deriving a slot for a photo moved
    # onto an existing board after the fact. Its own original position
    # belonged to a different group's sequence, so the target board's
    # current counts (not raw position) decide the slot.

    def test_moved_photo_fills_the_missing_fin_detail_slot(self):
        counts = {"full_board": 4, "side_profile": 1, "fin_detail": 0}
        self.assertEqual(next_shot_type_for_occupancy(counts), "fin_detail")

    def test_moved_photo_fills_full_board_before_later_slots(self):
        counts = {"full_board": 2, "side_profile": 0, "fin_detail": 0}
        self.assertEqual(next_shot_type_for_occupancy(counts), "full_board")

    def test_moved_photo_defaults_to_full_board_when_every_slot_is_full(self):
        counts = {"full_board": 4, "side_profile": 1, "fin_detail": 1}
        self.assertEqual(next_shot_type_for_occupancy(counts), "full_board")

    # -- group_limit_violations / _enforce_shot_limits: the overflow safety
    # net. Position-based assignment keeps a board within SHOT_LIMITS by
    # construction for its first 6 photos; this only matters for genuine
    # overflow (more photos than the protocol expects).

    def test_shot_limits_remain_review_data(self):
        items = [
            {**photo("CARD.JPG", card=True), "shot_type": "card"},
            *[{**photo(f"FULL_{index}.JPG"), "shot_type": "full_board"} for index in range(4)],
            {**photo("PROFILE.JPG"), "shot_type": "side_profile"},
            {**photo("FIN-1.JPG"), "shot_type": "fin_detail"},
            {**photo("FIN-2.JPG"), "shot_type": "fin_detail"},
        ]
        self.assertEqual(group_limit_violations(items), {"fin_detail": 2, "total": 8})

    def test_model_output_splits_over_limit_composition(self):
        group = [
            {**photo("CARD.JPG", card=True), "shot_type": "card"},
            *[{**photo(f"FULL-{index}.JPG"), "shot_type": "full_board"} for index in range(7)],
            *[{**photo(f"PROFILE-{index}.JPG"), "shot_type": "side_profile"} for index in range(3)],
        ]
        limited = _enforce_shot_limits([group])
        self.assertEqual([len(item) for item in limited], [5, 4, 1, 1])
        self.assertTrue(all(not group_limit_violations(item) for item in limited))

    # -- _densest_text_cluster: the color-blind card-finder's core logic.
    # Regression target: a yellow inventory card taped to a yellow board
    # (confirmed real case) is invisible to color-based detection — the
    # whole frame reads as one yellow blob. Text density doesn't have that
    # blind spot: the card is the only object with a dense block of
    # printed fields.

    def _card_field_boxes(self, x=800, y=1600, cols=2, rows=6):
        """A tight grid of word boxes simulating a card's printed fields."""
        boxes = []
        for row in range(rows):
            for col in range(cols):
                boxes.append((x + col * 120, y + row * 40, 100, 28))
        return boxes

    def test_dense_card_cluster_wins_over_scattered_sponsor_text(self):
        card_boxes = self._card_field_boxes()
        # Sponsor logo words scattered far apart elsewhere on the board —
        # never close enough to each other to form a rival cluster.
        scattered = [(50, 50, 80, 30), (2000, 100, 90, 30), (300, 3000, 70, 30)]
        region = _densest_text_cluster(card_boxes + scattered, pad=15)
        self.assertIsNotNone(region)
        x, y, w, h = region
        self.assertEqual((x, y), (800, 1600))
        self.assertGreaterEqual(w, 220)
        self.assertGreaterEqual(h, 220)

    def test_a_few_stray_words_are_not_mistaken_for_a_card(self):
        self.assertIsNone(_densest_text_cluster([(10, 10, 50, 20), (500, 500, 50, 20)], pad=15))

    def test_too_few_boxes_overall_returns_none(self):
        self.assertIsNone(_densest_text_cluster([(0, 0, 10, 10)] * 3, pad=5))

    # -- locate_missed_card_boundary / embedding_distance: the embedding
    # tie-breaker. It only ever runs on a group already structurally
    # oversized (capture order put more photos in it than the fixed
    # protocol allows), and only acts when one gap clearly stands out —
    # never as a general identity matcher.

    def test_embedding_distance_is_cosine_distance(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        self.assertAlmostEqual(embedding_distance(a, a), 0.0, places=5)
        self.assertAlmostEqual(embedding_distance(a, b), 1.0, places=5)

    def test_embedding_distance_is_none_when_either_side_missing(self):
        self.assertIsNone(embedding_distance(None, np.array([1.0])))
        self.assertIsNone(embedding_distance(np.array([1.0]), None))

    def test_clear_standout_gap_is_the_split_point(self):
        # Six photos of one board (small gaps), then a hard jump into a
        # second, visually unrelated board (confirmed real shape: a missed
        # card leaves an oversized group with one real discontinuity).
        distances = [0.05, 0.04, 0.06, 0.05, 0.04, 0.55, 0.05, 0.04]
        self.assertEqual(locate_missed_card_boundary(distances), 6)

    def test_uniformly_large_gaps_are_not_force_split(self):
        """No single gap is the odd one out — this is an ambiguous group,
        not a confirmed second board, and must be left for a human.
        """
        self.assertIsNone(locate_missed_card_boundary([0.4, 0.45, 0.42, 0.5]))

    def test_small_gaps_never_trigger_a_split(self):
        self.assertIsNone(locate_missed_card_boundary([0.05, 0.08, 0.06, 0.07]))

    def test_too_few_gaps_returns_none(self):
        self.assertIsNone(locate_missed_card_boundary([0.9]))
        self.assertIsNone(locate_missed_card_boundary([]))

    def test_uncertain_generic_details_do_not_trigger_fin_limit(self):
        items = [
            {**photo("DETAIL-1.JPG"), "shot_type": "detail"},
            {**photo("DETAIL-2.JPG"), "shot_type": "detail"},
        ]
        self.assertEqual(group_limit_violations(items), {})


if __name__ == "__main__":
    unittest.main()
