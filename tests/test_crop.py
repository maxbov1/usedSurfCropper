import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from organizer.crop import DEFAULT_PROFILE_HORIZONTAL_PADDING, DEFAULT_PROFILE_HORIZONTAL_PADDING_WITH_FINS, DEFAULT_RAIL_PADDING, _complete_profile_yolo_box, _crop_disagrees_with_yolo, _equal_vertical_padding_crop, _full_board_crop, _profile_crop, _usable_yolo_box, recover_outlier_full_board_box, shared_full_board_padding, suggested_crop, usable_yolo_prompt_box, yolo_cross_check_box
from organizer.segmentation import SegmentationResult, _centerline_points, _point_prompt_arrays, board_perimeter, mask_crop, validate_board_box


class CropHandoffTests(unittest.TestCase):
    def test_side_profile_can_fall_back_to_opencv_without_yolo(self):
        image = Image.new("RGB", (3024, 4032), "gray")
        classification = {
            "shot_type": "side_profile",
            "confidence": 0.8,
            "review": True,
            "candidate": (1400, 250, 320, 3500),
        }
        with patch("organizer.crop._profile_body_candidate", side_effect=lambda array, candidate: candidate):
            crop = suggested_crop(image, "side_profile", classification=classification)
        self.assertLess(crop[2], image.width)
        self.assertEqual(classification["boundary_source"], "OpenCV profile fallback")

    def test_full_board_uses_opencv_fallback_when_yolo_is_unavailable(self):
        image = Image.new("RGB", (3024, 4032), "gray")
        classification = {
            "shot_type": "full_board",
            "confidence": 0.8,
            "review": True,
            "candidate": (1000, 250, 1000, 3500),
        }
        crop = suggested_crop(image, "full_board", classification=classification)
        self.assertLess(crop[2], image.width)
        self.assertEqual(classification["boundary_source"], "OpenCV full-board fallback")

    def test_full_board_without_yolo_or_candidate_stays_full_frame(self):
        image = Image.new("RGB", (3024, 4032), "gray")
        classification = {
            "shot_type": "full_board",
            "confidence": 0.8,
            "review": True,
        }
        crop = suggested_crop(image, "full_board", classification=classification)
        self.assertEqual(crop[:4], (0, 0, image.width, image.height))

    def test_profile_box_rules_allow_partial_edge_on_detection(self):
        detection = {"x": 1, "y": 2214, "width": 264, "height": 1519}
        self.assertTrue(_usable_yolo_box(detection, 3024, 4032, "side_profile"))
        self.assertFalse(_usable_yolo_box(detection, 3024, 4032, "full_board"))
        self.assertFalse(_complete_profile_yolo_box(detection, 3024, 4032))

    def test_full_board_group_uses_median_shared_padding(self):
        cases = [
            {"shot_type": "full_board", "classification": {"opencv_boundary": (100, 400, 1000, 3000)}, "image_size": (1400, 4000)},
            {"shot_type": "full_board", "classification": {"opencv_boundary": (100, 200, 1000, 3000)}, "image_size": (1400, 4000)},
        ]
        padding = shared_full_board_padding(cases)
        self.assertTrue(padding["available"])
        # Median of [0.1333, 0.0667] is 0.1, capped at the 0.10 default.
        self.assertAlmostEqual(padding["vertical"], 0.10, places=4)

    def test_one_outlier_sibling_does_not_collapse_group_padding(self):
        # Mirrors a real board: three photos with ~8-9% room, one captured
        # with the tail close to the frame edge (~2.6% room). A strict
        # minimum applied that 2.6% to all four; the median should not.
        roomy = {"shot_type": "full_board", "classification": {"opencv_boundary": (1100, 300, 900, 3400)}, "image_size": (3024, 4032)}
        tight = {"shot_type": "full_board", "classification": {"opencv_boundary": (1020, 189, 914, 3744)}, "image_size": (3024, 4032)}
        padding = shared_full_board_padding([roomy, roomy, roomy, tight])
        self.assertTrue(padding["available"])
        self.assertGreater(padding["vertical"], 0.05)

    def test_rotated_full_board_crop_has_equal_nose_and_tail_padding(self):
        crop = _equal_vertical_padding_crop((50, 150, 400, 1100), (100, 200, 300, 1000), (500, 1600))
        self.assertEqual(crop[1] - 200, (200 + 1000) - (crop[1] + crop[3]))
        # Equal is not enough on its own: a sign bug once made both margins
        # collapse to zero, which is trivially "equal" but not padded at
        # all. The pre-equalization crop above has 50px on each side, so a
        # correct result keeps real padding instead of zeroing it out.
        self.assertGreater(200 - crop[1], 0)

    def test_equal_vertical_padding_preserves_lesser_of_two_real_margins(self):
        # Mirrors a real rotated full-board photo: the pre-equalization crop
        # has different (but both positive) top/bottom margins against the
        # boundary - 263px above, 265px below - and equalizing should bring
        # both down to ~263, not to zero.
        crop = _equal_vertical_padding_crop((675, 19, 1748, 3984), (1024, 282, 1049, 3456), (3130, 4110))
        top_margin = 282 - crop[1]
        bottom_margin = (crop[1] + crop[3]) - (282 + 3456)
        self.assertEqual(top_margin, bottom_margin)
        self.assertGreater(top_margin, 250)

    # -- validate_board_box: the one shared "does this box actually look
    # like a real board of this shot type" check, used by
    # segmentation.board_perimeter(), crop.py's live fallback (by calling
    # board_perimeter() rather than duplicating this), and
    # run_batch_crop.py's sibling-consistency pass. This used to be four
    # separate, differently-thresholded checks scattered across three
    # files; these tests pin down the one real implementation instead.

    def test_genuine_full_board_box_passes(self):
        # Real measured full_board box (IMG_6237): ratio 0.408.
        self.assertTrue(validate_board_box((1099, 422, 881, 3175), "full_board", (3024, 4032)))

    def test_genuine_side_profile_box_passes(self):
        # Real measured side_profile box (IMG_6241, post-fix): ratio 0.098.
        self.assertTrue(validate_board_box((1411, 340, 317, 3249), "side_profile", (3024, 4032)))

    def test_side_profile_shaped_like_full_board_fails(self):
        # Real failure case (IMG_6248): SAM 2 bled to a 0.67 ratio, higher
        # score than every genuine profile in the same batch, but no real
        # edge-on board is anywhere near that wide relative to its height.
        self.assertFalse(validate_board_box((744, 158, 2280, 3402), "side_profile", (3024, 4032)))

    def test_full_board_too_narrow_in_absolute_terms_fails(self):
        self.assertFalse(validate_board_box((1400, 400, 300, 3200), "full_board", (3024, 4032)))

    def test_tiny_box_fails_frame_fraction_floor(self):
        self.assertFalse(validate_board_box((1500, 2000, 40, 60), "full_board", (3024, 4032)))

    def test_near_full_frame_box_fails_frame_fraction_ceiling(self):
        self.assertFalse(validate_board_box((10, 10, 3000, 4000), "full_board", (3024, 4032)))

    def test_empty_or_missing_box_fails(self):
        self.assertFalse(validate_board_box(None, "full_board", (3024, 4032)))
        self.assertFalse(validate_board_box((100, 100, 0, 500), "full_board", (3024, 4032)))

    def test_sibling_outlier_fails_even_within_absolute_bounds(self):
        # Real case (IMG_6239): ratio 0.193 is inside the generous absolute
        # full_board bounds on its own, but its siblings all sit at
        # 0.24-0.46 width/height - this is the layer that actually catches
        # it, since the absolute check alone does not.
        outlier = (1399, 457, 584, 3034)
        siblings = [(1099, 422, 881, 3175), (1077, 447, 877, 3159), (1041, 359, 945, 3134)]
        self.assertTrue(validate_board_box(outlier, "full_board", (3024, 4032)))  # absolute bounds alone: passes
        self.assertFalse(validate_board_box(outlier, "full_board", (3024, 4032), sibling_boxes=siblings))  # with siblings: fails

    def test_consistent_sibling_does_not_get_falsely_rejected(self):
        box = (1099, 422, 881, 3175)
        siblings = [(1077, 447, 877, 3159), (1041, 359, 945, 3134), (1021, 189, 914, 3744)]
        self.assertTrue(validate_board_box(box, "full_board", (3024, 4032), sibling_boxes=siblings))

    # -- yolo_cross_check_box: a real photo (IMG_6347) had a confident,
    # pixel-accurate YOLO side-profile box that usable_yolo_prompt_box()
    # rejected only because it stopped 118px short of the nose-reaching
    # requirement (y=601 vs. a 484px cutoff on a 4032px-tall frame). Since
    # the disagreement check used to share that same gate, it never ran -
    # SAM 2's mask bled left into a wall shadow (final crop near/far margins
    # differed 2.3x) and nothing flagged it.

    def test_real_yolo_box_too_short_for_prompt_but_usable_for_cross_check(self):
        yolo = {"x": 1390, "y": 601, "width": 280, "height": 3034}
        self.assertIsNone(usable_yolo_prompt_box(yolo, 3024, 4032, "side_profile"))
        self.assertIsNotNone(yolo_cross_check_box(yolo, 3024, 4032, "side_profile"))

    def test_real_bled_sam2_crop_disagrees_with_cross_check_box(self):
        yolo = {"x": 1390, "y": 601, "width": 280, "height": 3034}
        cross_check = yolo_cross_check_box(yolo, 3024, 4032, "side_profile")
        bled_crop = (631, 293, 1365, 3602)
        self.assertTrue(_crop_disagrees_with_yolo(bled_crop, cross_check))

    def test_raw_box_tolerance_absorbs_normal_cross_detector_noise(self):
        # Real case (IMG_6330, full_board): SAM 2's raw box and YOLO's own
        # box differ by single-digit percent on every clean photo - this is
        # just two independent detectors measuring the same edge, not a
        # disagreement. A zero-tolerance raw-box check (added to fix the
        # padding-dilution bug above) flagged 71 of 72 real full_board
        # photos as mismatched because of this, a severe regression.
        sam_box = (1107, 429, 881, 3269)
        yolo_box = (1084, 392, 1084 + 915, 392 + 3350)
        self.assertFalse(_crop_disagrees_with_yolo(sam_box, yolo_box))

    def test_raw_box_tolerance_has_a_pixel_floor_for_thin_profile_boxes(self):
        # Real case (IMG_6334, side_profile): SAM 2 and YOLO boxes are only
        # 25-26px apart on a 221px-wide box - clean agreement. But that's
        # ~12% of the box's own width, so a tolerance scaled purely as a
        # fraction of the (thin) box would still flag it; the pixel floor
        # is what correctly treats this as noise instead.
        sam_box = (1384, 367, 222, 3326)
        yolo_box = (1359, 389, 1359 + 221, 389 + 3307)
        self.assertFalse(_crop_disagrees_with_yolo(sam_box, yolo_box))

    def test_raw_box_tolerance_still_catches_genuine_undershoot(self):
        # Real case (IMG_6373, full_board, Issue #26 detector undershoot):
        # a genuine ~15% gap on one side, not noise - must still flag.
        sam_box = (1132, 320, 567, 3387)
        yolo_box = (1109, 307, 1109 + 502, 307 + 3462)
        self.assertTrue(_crop_disagrees_with_yolo(sam_box, yolo_box))

    def test_disagreeing_sam2_boundary_is_overridden_by_yolo(self):
        # Real case (IMG_6347): SAM 2's mask bled into a wall shadow, but a
        # confident, independent YOLO box (1390,601,280,3034) was
        # pixel-accurate on the rail the whole time. The final crop should
        # center on YOLO's box, not the bled one, and still flag for review.
        image = Image.new("RGB", (3024, 4032), "gray")
        classification = {
            "shot_type": "side_profile",
            "confidence": 0.887,
            "review": False,
            "stored_perimeter": {"box": [931, 593, 765, 3002], "score": 0.905},
            "yolo_detection": {"x": 1390, "y": 601, "width": 280, "height": 3034, "confidence": 0.7636},
        }
        crop = suggested_crop(image, "side_profile", classification=classification)
        # YOLO's own box is x=1390..1670; a correct override centers on
        # that (plus padding), nowhere near the bled SAM 2 box's x=931..1696.
        self.assertTrue(950 <= crop[0] <= 1050)
        self.assertTrue(2010 <= crop[0] + crop[2] <= 2110)
        self.assertTrue(classification["boundary_mismatch"])
        self.assertTrue(classification["review"])

    # -- _profile_crop/_full_board_crop horizontal padding: both used to
    # scale off board HEIGHT instead of board WIDTH, so a typical board
    # (height ~3000-4000px, width often only 300-1000px) got 5-10x too much
    # rail padding. Real case (IMG_6365): a 1029px-wide, 3317px-tall SAM 2
    # box got 332px of padding each side - 10% of its height - instead of
    # the intended 10% of its width (~103px).

    def test_profile_crop_horizontal_padding_scales_with_width_not_height(self):
        crop = _profile_crop(926, 369, 1029, 3317, 3024, 4032, 0.08)
        pad_each_side = 926 - crop[0]
        # The image-relative ceiling (15% of 3024 = 454) binds before either
        # the percentage-of-width request (947) or the left-edge room (926)
        # do - still proves width, not height, drives the request: all three
        # of those are tiny next to the height-based alternative (3052).
        ceiling = round(3024 * 0.15)
        self.assertEqual(pad_each_side, min(round(1029 * DEFAULT_PROFILE_HORIZONTAL_PADDING), ceiling, 926))
        self.assertLess(pad_each_side, round(3317 * DEFAULT_PROFILE_HORIZONTAL_PADDING))

    def test_full_board_crop_rail_padding_scales_with_width_not_height(self):
        # A full_board box wide enough that the percentage request clears
        # the image-relative floor (see the floor-specific test below) -
        # height-based padding would still have been obviously oversized.
        crop = _full_board_crop(700, 300, 1500, 3400, 3024, 4032, 0.10, DEFAULT_RAIL_PADDING)
        pad_each_side = 700 - crop[0]
        self.assertEqual(pad_each_side, round(1500 * DEFAULT_RAIL_PADDING))

    def test_full_board_rail_padding_has_an_image_relative_floor(self):
        # Real case (IMG_6480): an 803px-wide full_board detection - the
        # narrow end of the normal 800-1000px range - got only 201px of
        # rail padding at 25%, visibly tighter than wider boards at the
        # same percentage. The floor (12% of image width) lifts it instead
        # of leaving it to the raw percentage.
        crop = _full_board_crop(1163, 238, 803, 3346, 3024, 4032, 0.10, DEFAULT_RAIL_PADDING)
        pad_each_side = 1163 - crop[0]
        floor = round(3024 * 0.12)
        self.assertEqual(pad_each_side, floor)
        self.assertGreater(floor, round(803 * DEFAULT_RAIL_PADDING))

    def test_mask_crop_horizontal_padding_scales_with_width_not_height(self):
        # mask_crop() had the same height-for-both-axes bug as
        # _profile_crop/_full_board_crop above, independently, in
        # segmentation.py: pad_x was computed from the mask's HEIGHT
        # (bottom - y) instead of its WIDTH (right - x). A 600px-wide,
        # 3400px-tall mask at padding=0.10 got 340px of horizontal padding
        # - over 5x the intended 60px - while vertical padding was correct.
        mask = np.zeros((4032, 3024), dtype=bool)
        mask[500:3900, 1200:1800] = True  # 600px wide, 3400px tall
        crop = mask_crop(mask, 3024, 4032, padding=0.10)
        x, y, w, h = crop
        pad_x = 1200 - x
        pad_y = 500 - y
        self.assertEqual(pad_x, round(600 * 0.10))
        self.assertEqual(pad_y, round(3400 * 0.10))
        self.assertLess(pad_x, round(3400 * 0.10))

    def test_centerline_points_are_inside_box_background_outside(self):
        # A real measured side_profile YOLO box (IMG_6347): x=1390..1670.
        foreground, background = _centerline_points((1390, 601, 1670, 3635), (3024, 4032))
        self.assertEqual(len(foreground), 3)
        for x, y in foreground:
            self.assertTrue(1390 <= x <= 1670)
            self.assertTrue(601 <= y <= 3635)
        self.assertEqual(len(background), 2)
        left_x, right_x = background[0][0], background[1][0]
        self.assertLess(left_x, 1390)
        self.assertGreater(right_x, 1670)

    def test_centerline_points_background_clamped_to_image_bounds(self):
        # A box flush against the right edge must not push a background
        # point past the image boundary SAM 2's predictor expects.
        _, background = _centerline_points((2900, 100, 3020, 3900), (3024, 4032))
        self.assertLessEqual(background[1][0], 3023)

    def test_point_prompt_arrays_label_foreground_one_background_zero(self):
        points = ([(10.0, 20.0), (10.0, 50.0)], [(0.0, 20.0), (20.0, 20.0)])
        coords, labels = _point_prompt_arrays(points)
        self.assertEqual(coords.shape, (4, 2))
        self.assertEqual(list(labels), [1, 1, 0, 0])

    def test_point_prompt_arrays_empty_points_returns_none(self):
        self.assertEqual(_point_prompt_arrays(([], [])), (None, None))

    def test_board_perimeter_sends_foreground_points_only_for_side_profile(self):
        # Measured (compare_sam2_checkpoints.py, 17 real side_profile
        # photos): foreground-only points raised mean IoU-vs-YOLO 0.761 ->
        # 0.777 with no score-gate regressions; adding background points
        # measured WORSE on both axes. full_board had no headroom (already
        # 0.95+ IoU) and real downside (confidence cratered on several
        # already-correct photos). So points must be foreground-only, and
        # side_profile-only - this pins both halves of that decision.
        class FakeSegmenter:
            def __init__(self):
                self.calls = []

            def predict(self, image, box, points=None):
                self.calls.append(points)
                return SegmentationResult(None, None, "fake", "")

        fake = FakeSegmenter()
        image = Image.new("RGB", (3024, 4032), "gray")
        prompt_box = (1390.0, 601.0, 1670.0, 3635.0)
        with patch("organizer.segmentation.get_segmenter", return_value=fake):
            board_perimeter(image, "side_profile", prompt_box=prompt_box)
            board_perimeter(image, "full_board", prompt_box=prompt_box)
        side_profile_points, full_board_points = fake.calls
        self.assertIsNotNone(side_profile_points)
        self.assertEqual(len(side_profile_points[0]), 3)  # foreground
        self.assertEqual(side_profile_points[1], [])  # no background points
        self.assertIsNone(full_board_points)

    def test_profile_horizontal_padding_has_an_image_relative_ceiling(self):
        # Real case (IMG_6472): a 3/4-angle side_profile box 880px wide -
        # near-full deck width, not board thickness, despite the shot_type.
        # 92% of it is 810px/side (crop width 2288px, 76% of a 3024px
        # frame). The ceiling caps it to 15% of image width per side
        # instead, regardless of what the box's own width happens to mean.
        crop = _profile_crop(1126, 664, 827, 2850, 3024, 4032, 0.08)
        pad_each_side = 1126 - crop[0]
        ceiling = round(3024 * 0.15)
        self.assertEqual(pad_each_side, ceiling)
        self.assertLess(pad_each_side, round(827 * DEFAULT_PROFILE_HORIZONTAL_PADDING))

    def test_profile_horizontal_padding_ceiling_does_not_affect_thin_boxes(self):
        # A genuinely thin edge-on box (board thickness, not width) stays
        # well under the ceiling and keeps its full requested padding.
        crop = _profile_crop(1411, 340, 317, 3249, 3024, 4032, 0.08)
        pad_each_side = 1411 - crop[0]
        self.assertEqual(pad_each_side, round(317 * DEFAULT_PROFILE_HORIZONTAL_PADDING))
        self.assertLess(pad_each_side, round(3024 * 0.15))

    def test_profile_crop_reduces_padding_when_fins_confirmed(self):
        # A fin-equipped board's raw detected width already includes the
        # fin protruding past the rail, so less additional padding is
        # needed on top of it than a fin-less board at the same width.
        box = (1411, 340, 317, 3249)
        without_fins = _profile_crop(*box, 3024, 4032, 0.08)
        with_fins = _profile_crop(*box, 3024, 4032, 0.08, fins_confirmed=True)
        pad_without = 1411 - without_fins[0]
        pad_with = 1411 - with_fins[0]
        self.assertEqual(pad_without, round(317 * DEFAULT_PROFILE_HORIZONTAL_PADDING))
        self.assertEqual(pad_with, round(317 * DEFAULT_PROFILE_HORIZONTAL_PADDING_WITH_FINS))
        self.assertLess(pad_with, pad_without)

    def test_recover_outlier_full_board_box_uses_sibling_center_not_own(self):
        # Real case (IMG_6481): a 545px-wide detection, center at 0.470 of
        # image width, against three full_board siblings (IMG_6480/82/83)
        # that measured 803/879/897px wide with centers clustered at
        # 0.503-0.517. Using the outlier's own center with the corrected
        # width clipped the board's left rail in the rendered crop - this
        # pins the fix: both width AND center must come from the siblings.
        outlier_box = (1149, 310, 545, 3273)
        sibling_boxes = [
            ((1163, 238, 803, 3346), 3024),
            ((1107, 248, 879, 3430), 3024),
            ((1072, 226, 897, 3516), 3024),
        ]
        corrected = recover_outlier_full_board_box(outlier_box, sibling_boxes, 3024)
        self.assertEqual(corrected[2], 879)  # median sibling width
        self.assertEqual(corrected[3], 3273)  # outlier's own height, unchanged
        corrected_center_fraction = (corrected[0] + corrected[2] / 2) / 3024
        self.assertAlmostEqual(corrected_center_fraction, 0.5114, places=3)
        # The bug this guards against: the outlier's own center (0.470)
        # must NOT leak into the result.
        outlier_center_fraction = (outlier_box[0] + outlier_box[2] / 2) / 3024
        self.assertGreater(abs(corrected_center_fraction - outlier_center_fraction), 0.03)

    def test_low_confidence_unresolved_mismatch_declines_to_manual(self):
        # Real case (IMG_6491): classifier confidence 0.45, SAM 2 kept
        # because YOLO was edge-clipped (not overridable) - but SAM 2's own
        # box had bled into an adjacent board while staying within
        # plausible aspect bounds, so shipping it anyway was still wrong.
        # Low classifier confidence is the third, independent signal that
        # catches what a shape check alone cannot.
        image = Image.new("RGB", (3024, 4032), "gray")
        classification = {
            "shot_type": "side_profile", "confidence": 0.45, "review": True,
            "stored_perimeter": {"box": [1090, 392, 987, 3397], "score": 0.8668},
            "yolo_detection": {"x": 1, "y": 988, "width": 348, "height": 2622, "confidence": 0.694},
        }
        crop = suggested_crop(image, "side_profile", classification=classification)
        self.assertEqual(crop[:4], (0, 0, 3024, 4032))
        self.assertIn("manual crop required", crop[4])

    def test_high_confidence_unresolved_mismatch_does_not_decline(self):
        # Real case (IMG_6472): classifier confidence 0.823 - a clean
        # disagreement between two otherwise-reasonable boxes, not a sign
        # the photo itself is ambiguous. Must still override onto YOLO
        # (not decline) since YOLO's box passes the completeness bar here.
        image = Image.new("RGB", (3024, 4032), "gray")
        classification = {
            "shot_type": "side_profile", "confidence": 0.823, "review": False,
            "stored_perimeter": {"box": [1126, 664, 827, 2850], "score": 0.9905},
            "yolo_detection": {"x": 1076, "y": 633, "width": 880, "height": 2860, "confidence": 0.956},
        }
        crop = suggested_crop(image, "side_profile", classification=classification)
        self.assertNotEqual(crop[:4], (0, 0, 3024, 4032))
        self.assertIn("overrode disagreeing", crop[4])

    def test_fin_detail_mislabel_signal_is_ambiguous_not_decisive(self):
        # Real case: on one board, a plain full-deck photo (IMG_6366) was
        # stuck with shot_type="fin_detail" - a genuine mislabel. On
        # another board, a true fin close-up (IMG_6434, three FCS II fins
        # clearly visible) got a SAM 2 box that coincidentally matched its
        # own board's full_board siblings' width almost exactly, because a
        # close-up on the rail edge still spans most of the frame height.
        # Both pass this same check - that's why run_batch_crop.py only
        # flags a fin_detail/full_board width match for human review and
        # never reassigns shot_type from it alone.
        mislabeled_full_deck = (1043, 539, 946, 3071)
        genuine_fin_closeup = (1132, 0, 967, 3143)
        siblings = [(1060, 423, 925, 3340), (1072, 348, 968, 3193), (1010, 334, 996, 3321)]
        self.assertTrue(validate_board_box(mislabeled_full_deck, "full_board", (3024, 4032), sibling_boxes=siblings))
        self.assertTrue(validate_board_box(genuine_fin_closeup, "full_board", (3024, 4032), sibling_boxes=siblings))

    def test_single_sibling_is_too_few_to_trust(self):
        # One sibling isn't enough to distinguish a real outlier from
        # normal board-to-board variation - the median needs at least two.
        box = (1399, 457, 584, 3034)
        self.assertTrue(validate_board_box(box, "full_board", (3024, 4032), sibling_boxes=[(1099, 422, 881, 3175)]))


if __name__ == "__main__":
    unittest.main()
