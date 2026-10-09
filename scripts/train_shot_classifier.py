#!/usr/bin/env python3
"""Train and evaluate a small shot-type classifier with board-held-out splits.

Human labels are the default training source. Gallery-order labels are kept as
weak supervision and can only be included explicitly, so they cannot silently
be mixed with reviewed labels.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from organizer.shot_features import composition_matrix

LABELS = ["full_board", "side_profile", "fin_detail"]


class ImageSet(Dataset):
    def __init__(self, cases: list[dict], transform) -> None:
        self.cases = cases
        self.transform = transform

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, index: int):
        case = self.cases[index]
        image = Image.open(ROOT / case["local_path"]).convert("RGB")
        return self.transform(image), LABELS.index(case["label"]), case


def split_boards(cases: list[dict], test_fraction: float, seed: int) -> tuple[list[dict], list[dict], dict]:
    by_label: dict[str, set[str]] = defaultdict(set)
    for case in cases:
        by_label[case["label"]].add(case["board_key"])
    rng = random.Random(seed)
    test_boards: set[str] = set()
    for label in LABELS:
        boards = sorted(by_label[label])
        rng.shuffle(boards)
        count = max(1, round(len(boards) * test_fraction)) if len(boards) > 1 else 0
        test_boards.update(boards[:count])
    train = [case for case in cases if case["board_key"] not in test_boards]
    test = [case for case in cases if case["board_key"] in test_boards]
    split = {"train_boards": sorted({case["board_key"] for case in train}), "test_boards": sorted(test_boards)}
    return train, test, split


def extract_embeddings(model, loader, device: str) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    features, targets, metadata = [], [], []
    with torch.inference_mode():
        for images, labels, cases in loader:
            embeddings = model(images.to(device)).flatten(1)
            embeddings = torch.nn.functional.normalize(embeddings, dim=1)
            features.append(embeddings.cpu().numpy())
            targets.extend(labels.numpy().tolist())
            if isinstance(cases, dict):
                # DataLoader's default collator turns a list of case dicts
                # into a dict of batched values.
                batch_size = len(next(iter(cases.values())))
                for index in range(batch_size):
                    metadata.append({key: value[index] for key, value in cases.items()})
            else:
                metadata.extend(cases)
    return np.concatenate(features), np.asarray(targets), metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--label-source", choices=["human", "gallery_order", "all"], default="human", help="Training labels to use; defaults to reviewed human labels only.")
    parser.add_argument("--output-stem", default=None, help="Output filename stem under data/shot-labels.")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    manifest_path = ROOT / "data" / "shot-labels" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    all_labeled = [case for case in manifest.get("cases", []) if case.get("label") in LABELS and (ROOT / case["local_path"]).is_file()]
    cases = all_labeled if args.label_source == "all" else [case for case in all_labeled if case.get("label_source") == args.label_source]
    if not cases:
        raise SystemExit(f"No usable {args.label_source} shot images found.")
    train_cases, test_cases, split = split_boards(cases, args.test_fraction, args.seed)
    print(f"cases: {len(cases)} · boards: {len({c['board_key'] for c in cases})}")
    print(f"train: {len(train_cases)} images / {len(split['train_boards'])} boards")
    print(f"test: {len(test_cases)} images / {len(split['test_boards'])} boards")
    print(f"train labels: {dict(Counter(c['label'] for c in train_cases))}")
    print(f"test labels: {dict(Counter(c['label'] for c in test_cases))}")

    weights = MobileNet_V3_Small_Weights.DEFAULT
    transform = weights.transforms()
    backbone = mobilenet_v3_small(weights=weights).features
    pool = mobilenet_v3_small(weights=None).avgpool
    # The classifier is not needed. Keep the pretrained convolutional trunk
    # frozen, which makes this practical on Apple Silicon CPU as well.
    backbone.eval().to("cpu")
    pool.eval().to("cpu")

    def embed(images):
        return pool(backbone(images))

    train_loader = DataLoader(ImageSet(train_cases, transform), batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(ImageSet(test_cases, transform), batch_size=args.batch_size, shuffle=False, num_workers=0)
    train_embedding, train_y, _ = extract_embeddings(embed, train_loader, "cpu")
    test_embedding, test_y, test_meta = extract_embeddings(embed, test_loader, "cpu")
    train_composition, _ = composition_matrix(train_cases, ROOT)
    test_composition, test_composition_meta = composition_matrix(test_cases, ROOT)
    train_x = np.concatenate([train_embedding, train_composition], axis=1)
    test_x = np.concatenate([test_embedding, test_composition], axis=1)
    torch.manual_seed(args.seed)
    classifier = nn.Linear(train_x.shape[1], len(LABELS))
    weights_by_class = np.bincount(train_y, minlength=len(LABELS)).astype(np.float32)
    class_weights = torch.tensor(weights_by_class.sum() / np.maximum(weights_by_class, 1), dtype=torch.float32)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=0.03, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    x_tensor, y_tensor = torch.from_numpy(train_x), torch.from_numpy(train_y).long()
    for _ in range(args.epochs):
        logits = classifier(x_tensor)
        loss = criterion(logits, y_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.inference_mode():
        test_logits = classifier(torch.from_numpy(test_x))
        probabilities = torch.softmax(test_logits, dim=1).numpy()
        predicted = probabilities.argmax(1)
    confusion = [[int(((test_y == actual) & (predicted == guess)).sum()) for guess in range(len(LABELS))] for actual in range(len(LABELS))]
    per_class = {}
    for index, label in enumerate(LABELS):
        true_positive = confusion[index][index]
        support = int((test_y == index).sum())
        predicted_count = int((predicted == index).sum())
        per_class[label] = {"support": support, "precision": round(true_positive / predicted_count, 3) if predicted_count else 0.0, "recall": round(true_positive / support, 3) if support else 0.0}
    accuracy = float((predicted == test_y).mean()) if len(test_y) else 0.0
    decisions = []
    for case, actual, guess, probs, composition in zip(test_meta, test_y, predicted, probabilities, test_composition_meta):
        confidence = float(probs[int(guess)])
        agreement = LABELS[int(guess)] == composition["hint"]
        auto_label = bool(confidence >= 0.80 and agreement and not composition["hint_review"])
        decisions.append({"source": case["local_path"], "board_key": case["board_key"], "expected": LABELS[int(actual)], "predicted": LABELS[int(guess)], "model_confidence": round(confidence, 3), "composition_hint": composition["hint"], "composition_reason": composition["reason"], "agreement": agreement, "auto_label": auto_label, "needs_review": not auto_label})
    failures = [decision for decision in decisions if decision["expected"] != decision["predicted"]]
    report = {"version": 3, "model": "OpenCV composition features + torchvision MobileNetV3-Small ImageNet embedding + linear classifier", "composition_policy": "OpenCV framing hint gates auto-labeling; MobileNet embedding is supporting evidence", "label_source_policy": args.label_source, "source_counts_in_manifest": dict(Counter(case.get("label_source", "unknown") for case in all_labeled)), "board_held_out": True, "seed": args.seed, "labels": LABELS, "split": split, "train_images": len(train_cases), "test_images": len(test_cases), "accuracy": round(accuracy, 3), "confusion_matrix": confusion, "per_class": per_class, "failures": failures, "decisions": decisions, "auto_label_count": sum(decision["auto_label"] for decision in decisions), "needs_review_count": sum(decision["needs_review"] for decision in decisions)}
    stem = args.output_stem or {"human": "classifier-human-v1", "gallery_order": "classifier-gallery-order-v2", "all": "classifier-all-v2"}[args.label_source]
    output = ROOT / "data" / "shot-labels" / f"{stem}.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    torch.save({"state_dict": classifier.state_dict(), "labels": LABELS, "embedding_dim": train_x.shape[1], "model": report["model"], "label_source_policy": args.label_source}, output.with_suffix(".pt"))
    print(f"accuracy: {accuracy:.3f}")
    print(f"per class: {per_class}")
    print(f"failures: {len(failures)}")
    print(f"report: {output.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
