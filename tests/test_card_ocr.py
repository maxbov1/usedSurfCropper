import unittest

from organizer.card_ocr import extract_card_identifier, extract_card_identifier_with_confidence, ocr_quality


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
            "fin_box_count": "3",
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

    def test_known_aliases_are_canonicalized(self):
        identifier = extract_card_identifier("BRAND: Sharpe cee\nMODEL: Pigme")
        self.assertEqual(identifier["brand"], "Sharpeye")
        # Models are brand-conditioned. An unrelated Lost alias must not be
        # accepted for a Sharpeye board.
        self.assertEqual(identifier["model"], "Pigme")

    def test_model_alias_is_canonicalized_only_for_matching_brand(self):
        identifier = extract_card_identifier("BRAND: Lost\nMODEL: driver z")
        self.assertEqual(identifier, {"brand": "Lost", "model": "Driver 2"})

    def test_normalization_audit_is_retained_in_confidence_output(self):
        confidence = extract_card_identifier_with_confidence("BRAND: pyze1\nMODEL: bastrd")
        self.assertEqual(confidence["brand"]["value"], "Pyzel")
        self.assertEqual(confidence["model"]["value"], "The Bastard")
        self.assertEqual(confidence["brand"]["normalization"]["method"], "known_alias")

    def test_unlabelled_logo_mentions_are_recovered(self):
        identifier = extract_card_identifier("Stewart\nPigme")
        self.assertEqual(identifier, {"brand": "Stewart", "model": "Pigme"})

    def test_garbage_ocr_is_not_usable_identity_evidence(self):
        quality = ocr_quality("ae qx /\\ | | 7 3 9 1")
        self.assertIn(quality["status"], {"junk", "weak"})
        self.assertNotEqual(quality["status"], "usable")


if __name__ == "__main__":
    unittest.main()
