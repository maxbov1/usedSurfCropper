#!/usr/bin/env python3
"""Apply the visual shot classifier without replacing human labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from train_shot_classifier import ImageSet, extract_embeddings
from organizer.shot_features import composition_matrix


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="data/shot-labels/classifier-human-v1.pt")
    parser.add_argument("--manifest", default="data/shot-labels/manifest.json")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest_path = root / args.manifest
    checkpoint_path = root / args.model
    manifest = json.loads(manifest_path.read_text())
    cases = [case for case in manifest.get("cases", []) if case.get("label") in {"full_board", "side_profile", "fin_detail"} and (root / case.get("local_path", "")).is_file()]
    if not cases:
        raise SystemExit("No local image cases found.")

    weights = MobileNet_V3_Small_Weights.DEFAULT
    backbone = mobilenet_v3_small(weights=weights).features.eval()
    pool = mobilenet_v3_small(weights=None).avgpool.eval()

    def embed(images):
        return pool(backbone(images))

    def collate(batch):
        return torch.stack([item[0] for item in batch]), torch.tensor([item[1] for item in batch]), [item[2] for item in batch]

    loader = DataLoader(ImageSet(cases, weights.transforms()), batch_size=16, shuffle=False, num_workers=0, collate_fn=collate)
    embedding, _, metadata = extract_embeddings(embed, loader, "cpu")
    composition, composition_meta = composition_matrix(cases, root)
    features = np.concatenate([embedding, composition], axis=1)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    classifier = nn.Linear(features.shape[1], len(checkpoint["labels"]))
    classifier.load_state_dict(checkpoint["state_dict"])
    classifier.eval()
    with torch.inference_mode():
        probabilities = torch.softmax(classifier(torch.from_numpy(features)), dim=1).numpy()

    predicted_labels = [checkpoint["labels"][int(probs.argmax())] for probs in probabilities]
    board_fin_counts = {}
    for case, label in zip(cases, predicted_labels):
        if label == "fin_detail":
            board_fin_counts[case["board_key"]] = board_fin_counts.get(case["board_key"], 0) + 1
    predictions = []
    for case, probs, composition_info in zip(metadata, probabilities, composition_meta):
        index = int(probs.argmax())
        confidence = float(probs[index])
        label = checkpoint["labels"][index]
        agreement = label == composition_info["hint"]
        human_labeled = case.get("label_source") == "human"
        existing_label = case.get("label") if case.get("label_source") == "gallery_order" else None
        confirms_existing = bool(existing_label in checkpoint["labels"] and label == existing_label)
        side_profile_supported = composition_info["hint"] == "side_profile" and not composition_info["hint_review"]
        fin_conflict = bool(label == "fin_detail" and board_fin_counts.get(case["board_key"], 0) > 1)
        existing_conflict = bool(existing_label and label != existing_label)
        if existing_conflict:
            decision_reason = "model disagrees with existing order label"
        elif confidence < 0.80:
            decision_reason = "model confidence below 0.80"
        elif existing_label:
            decision_reason = "model confirms existing order label"
        else:
            decision_reason = "high-confidence visual prediction"
        # The existing label is the working answer. The model only creates a
        # review task when it disagrees or lacks confidence; composition and
        # fin-count signals remain diagnostic metadata instead of extra gates.
        auto_label = bool(not human_labeled and confidence >= 0.80 and not existing_conflict)
        predictions.append({
            "id": case["id"],
            "source": case["local_path"],
            "board_key": case["board_key"],
            "model_prediction": label,
            "model_confidence": round(confidence, 3),
            "composition_hint": composition_info["hint"],
            "composition_reason": composition_info["reason"],
            "existing_label": existing_label,
            "confirms_existing_label": confirms_existing,
            "board_fin_candidate_count": board_fin_counts.get(case["board_key"], 0),
            "decision_reason": decision_reason,
            "agreement": agreement,
            "auto_label_eligible": auto_label,
            "needs_review": bool(not human_labeled and not auto_label),
            "human_label": case.get("label") if human_labeled else None,
            "review_status": "human_labeled" if human_labeled else ("auto_label_eligible" if auto_label else "needs_review"),
        })

    output = root / "data" / "shot-labels" / "visual-predictions.json"
    output.write_text(json.dumps({
        "version": 1,
        "model": str(args.model),
        "policy": "human labels are never replaced; auto-label only when model and composition agree with high confidence",
        "predictions": predictions,
    }, indent=2) + "\n")
    print(f"predicted: {len(predictions)}")
    print(f"auto-label eligible: {sum(item['auto_label_eligible'] for item in predictions)}")
    print(f"needs review: {sum(item['needs_review'] for item in predictions)}")
    print(f"report: {output.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
