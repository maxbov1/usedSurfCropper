"""Structured extraction of inventory-card identity fields.

OCR is noisy and the cards repeat the important fields in two layouts. This
module treats each OCR reading as evidence, not as ground truth: repeated and
clean field values win, while long or obviously contaminated readings are
discarded.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable


FIELD_LABELS = {
    "brand": r"BRAND",
    "model": r"MODEL",
    "fin_system": r"FIN\s*SYSTEM",
}
_FIELD_STOP = r"(?:BRAND|MODEL|LENGTH|WIDTH|THICKNESS|VOLUME|FIN\s*SYSTEM|PRICE|BOARD\s*TYPE|ITEM|BOARD)"
_NOISE = re.compile(r"\b(?:CHECKED[- ]?IN|PERSONAL INFORMATION|BOARD INFORMATION|USEDSURF|CONSIGNMENT|PAYMENT|SIGNATURE)\b", re.I)


def _texts(value: str | Iterable[str]) -> list[str]:
    if isinstance(value, str):
        return [value]
    return [text for text in value if text]


def normalize_ocr_text(text: str) -> str:
    """Normalize OCR line breaks without destroying field boundaries."""
    text = text.replace("\r", "\n").replace("\u00a0", " ")
    text = text.replace("“", '"').replace("”", '"').replace("’", "'")
    # OCR often puts the digits on the next line after `SH-`.
    text = re.sub(r"(?i)\b(SH)\s*[-–—]?\s*\n\s*(\d)", r"\1-\2", text)
    return text


def _clean_value(field: str, raw: str) -> str:
    value = re.sub(r"\s+", " ", raw).strip(" :|-_=\\/\t")
    value = re.split(rf"\s+(?={_FIELD_STOP}\s*[:#-]?)", value, maxsplit=1, flags=re.I)[0]
    # A frequent OCR artifact is a clean value followed by an em-dash or an
    # equals sign and fragments from the adjacent column.
    if field in {"brand", "model"}:
        value = re.split(r"\s+[—–=|]+\s*", value, maxsplit=1)[0].strip(" :|-_")
        parts = value.split()
        if len(parts) > 1 and parts[-1].islower() and len(parts[-1]) <= 3:
            value = " ".join(parts[:-1])
    if not value or len(value) > 42 or _NOISE.search(value):
        return ""
    if sum(character.isalnum() for character in value) < 2:
        return ""
    return value


def _field_candidates(texts: Iterable[str], field: str) -> list[str]:
    label = FIELD_LABELS[field]
    candidates: list[str] = []
    for text in texts:
        for raw_line in normalize_ocr_text(text).splitlines():
            line = raw_line.strip()
            match = re.search(rf"\b{label}\b\s*[:#-]?\s*(.*)$", line, re.I)
            if not match:
                continue
            value = _clean_value(field, match.group(1))
            if value:
                candidates.append(value)
    return candidates


def _candidate_score(value: str) -> tuple[int, int, int, int]:
    alnum = sum(character.isalnum() for character in value)
    punctuation = sum(not character.isalnum() and not character.isspace() for character in value)
    words = len(value.split())
    return -words, alnum, -punctuation, -len(value)


def _choose(candidates: list[str]) -> str:
    if not candidates:
        return ""
    counts = Counter(value.casefold() for value in candidates)
    winner = max(candidates, key=lambda value: (counts[value.casefold()], *_candidate_score(value)))
    return winner


def _extract_sku(texts: Iterable[str]) -> str:
    combined = "\n".join(normalize_ocr_text(text) for text in texts)
    patterns = (
        r"\b(?:BOARD|ITEM)\s*(?:NO\.?|NUMBER)?\s*#?\s*[:\-]?\s*(?:SH\s*[-–—]?\s*)?(\d{4,6})\b",
        r"\bSH\s*[-–—]?\s*(\d{4,6})\b",
    )
    candidates = []
    for pattern in patterns:
        candidates.extend(match.group(1) for match in re.finditer(pattern, combined, re.I))
    if not candidates:
        return ""
    return Counter(candidates).most_common(1)[0][0]


def _extract_fins_included(texts: Iterable[str]) -> str:
    for text in texts:
        for raw_line in normalize_ocr_text(text).splitlines():
            line = re.sub(r"\s+", " ", raw_line.upper()).strip()
            if "INCLUD" not in line:
                continue
            tail = line.split("INCLUD", 1)[-1]
            if any(marker in tail for marker in ("✓", "✔", "☑", "¥", "YES")) or re.search(r"\bY\b", tail):
                return "yes"
            if "NO" in tail or re.search(r"\bN\b", tail):
                return "no"
    return ""


def _normalize_fin_system(value: str) -> str:
    value = re.sub(r"\s+", " ", value).strip()
    upper = value.upper().replace("FCS 8", "FCS II").replace("FCS LL", "FCS II")
    if upper in {"FCS II", "FCS IL", "FCS LL", "FCS I", "FCN"} or upper.startswith("FS\""):
        return "FCS II"
    if upper == "FUTURES":
        return "Futures"
    return value


def extract_card_identifier(texts: str | Iterable[str]) -> dict[str, str]:
    """Extract only explicit inventory-card fields from one or more OCR passes."""
    readings = _texts(texts)
    identifier: dict[str, str] = {}
    sku = _extract_sku(readings)
    if sku:
        identifier["sku"] = sku
    for field in ("brand", "model"):
        value = _choose(_field_candidates(readings, field))
        if value:
            identifier[field] = value
    fin_candidates = _field_candidates(readings, "fin_system")
    if fin_candidates:
        identifier["fin_system"] = _normalize_fin_system(_choose(fin_candidates))
    fins = _extract_fins_included(readings)
    if fins:
        identifier["fins_included"] = fins
    return identifier
