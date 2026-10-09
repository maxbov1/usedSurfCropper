import unittest

from organizer.card_ocr import extract_card_identifier


class CardOCRTests(unittest.TestCase):
    def test_repeated_clean_fields_beat_noisy_first_reading(self):
        readings = [
            "ITEM # SH-\n49126\nBRAND: JS — a\nMODEL: Mor sta =\nFIN SYSTEM: FCN",
            "BOARD # 49126\nBRAND: JS\nMODEL: Monsta\nFIN SYSTEM: FCS II\n# OF BOXES: 3 INCLUDED Y/N: Y",
        ]
        self.assertEqual(extract_card_identifier(readings), {
            "sku": "49126",
            "brand": "JS",
            "model": "Monsta",
            "fin_system": "FCS II",
            "fins_included": "yes",
        })

    def test_short_clean_model_wins_over_spaced_ocr_noise(self):
        identifier = extract_card_identifier([
            "BRAND: Sharpe cee\nMODEL: ER (ia)",
            "BRAND: Sharpeye\nMODEL: #77\nBOARD #49162",
        ])
        self.assertEqual(identifier["sku"], "49162")
        self.assertEqual(identifier["brand"], "Sharpeye")
        self.assertEqual(identifier["model"], "#77")

    def test_no_explicit_card_fields_means_no_identity(self):
        self.assertEqual(extract_card_identifier("yellow board with a logo"), {})


if __name__ == "__main__":
    unittest.main()
