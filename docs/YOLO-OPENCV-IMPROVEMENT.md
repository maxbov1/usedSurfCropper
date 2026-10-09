# YOLO/OpenCV improvement plan

## Keep the first version simple

Do not train a new detector yet. Start with a small evaluation set containing:

- the known failures;
- several successful full-board images;
- several side-profile and fin-detail images;
- the expected label for each image: `full_board`, `side_profile`, or `fin_detail`.

## Proposed decision flow

OpenCV should classify the overall silhouette first. YOLO should confirm the board or narrow the search area, not decide the shot type by itself.

```text
OpenCV silhouette classification
        ↓
YOLO confirmation or ROI search
        ↓
OpenCV contour and board-relative crop
```

Add a disagreement rule:

- If OpenCV sees a broad, complete board and YOLO returns a narrow box, keep `full_board` and mark the case for review.
- If both OpenCV and YOLO see a narrow, complete vertical object, use `side_profile`.
- If neither sees a complete board, preserve the original as a detail/fin image.

This directly addresses painted boards, white boards on dark walls, and false side-profile classifications without changing models.

## If the simple rule is not enough

Fine-tune a small YOLO model only after collecting roughly 30–100 labeled examples. Include painted boards, low-light boards, rocker views, fin closeups, racks, and wall edges as negatives.

## Success check

Every change should be measured against the small failure set. A change is useful only if it reduces false side-profile classifications and missed full-board detections without worsening fin/detail preservation.
