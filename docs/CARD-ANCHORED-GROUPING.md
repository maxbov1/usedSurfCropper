# Card-anchored grouping

Inventory cards define the expected board groups. They are identity anchors,
not ordinary board images.

For each card, grouping now creates a fixed slot and selects one nearby,
compatible full-deck photo as the card-to-board bridge. The bridge becomes the
first visual representative for that slot. Remaining photos are admitted only
after comparison with the anchored representative and the card's explicit
identity evidence.

Cards are attached to their original sorted slots. They are not matched after
clustering to whichever cluster has the lowest generic visual score. Photos
that cannot establish a coherent board identity remain in unassigned review
clusters rather than becoming legitimate card-backed groups.

The visual roles remain separate:

- inventory card: SKU and identity fields, plus partial deck evidence;
- full-deck photo: primary board identity bridge and comparison evidence;
- side profile: protocol/timestamp evidence with weak visual identity;
- fin/detail photo: localized supporting evidence.

Full-board identity now also records local board-relative evidence for the
nose/deck, center graphics, rails, tail/traction, and fin-box region. A photo
must pass both the global board comparison and the local evidence gate before
it can enter an anchored group. Full-board OCR reads enlarged board crops so
brand/model markings can contradict a card when the text is actually visible.

## Board-region evidence

Full-board and fin-detail frames now produce a separate `board_evidence` record. The
record keeps three board-relative crops—`logo_brand`, `model_marking`, and
`tail_fin`—inside the detected board boundary. Each crop is enlarged and sent
through normal, grayscale/contrast, and sharpened OCR passes. It also records
non-authoritative visual measurements for edge density, dark-mark density, and
tail/fin-box candidates.

These measurements are evidence, not automatic identity claims. A blank or
ambiguous crop remains unknown, and side-profile frames remain excluded from
brand/model OCR. The per-photo records are included in `data/board-identity.json`
and `data/board-evidence.json` so a later comparison can distinguish “the mark
was not captured” from “the mark was captured but did not match.”

