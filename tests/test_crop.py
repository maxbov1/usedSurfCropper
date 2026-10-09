import unittest
from unittest.mock import patch

from PIL import Image

from organizer.crop import _complete_profile_yolo_box, _usable_yolo_box, suggested_crop


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


if __name__ == "__main__":
    unittest.main()
