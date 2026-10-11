#!/usr/bin/env python3
"""Evaluate the six-field board-evidence candidate without mutating production data.

This is deliberately a report generator, not a training command. It treats
missing fields as unknown, never as a negative, and reports contradictions
between independent evidence sources so a candidate cannot look healthy just
because it produced many non-empty strings.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path


FIELDS = (
    "brand",
    "model",
    "fin_system",
    "fin_box_count",
    "fins_included",
    "fins_present_count",
)


def _value(row: dict, field: str) -> str:
    value = row.get(field, "")
    if isinstance(value, dict):
        value = value.get("value", "")
    return str(value or "").strip()


def _field_values(rows: list[dict], field: str) -> list[str]:
    values = []
    for row in rows:
        value = _value(row, field)
        if value:
            values.append(value.casefold())
    return values


def evaluate(source: Path) -> dict:
    payload = json.loads(source.read_text(encoding="utf-8"))
    groups = payload.get("groups", [])
    group_reports = []
    totals = Counter()

    for group in groups:
        card = group.get("card_identifier") or {}
        photo_rows = group.get("photo_evidence") or []
        rows = [card, *photo_rows]
        fields = {}
        contradictions = []
        for field in FIELDS:
            values = _field_values(rows, field)
            distinct = sorted(set(values))
            fields[field] = {
                "known": bool(distinct),
                "values": distinct,
                "source_count": len(values),
                "status": "contradiction" if len(distinct) > 1 else ("known" if distinct else "unknown"),
            }
            totals[f"{field}_known"] += bool(distinct)
            totals[f"{field}_contradiction"] += len(distinct) > 1
            if len(distinct) > 1:
                contradictions.append(field)
        card_known = sum(bool(_value(card, field)) for field in FIELDS)
        totals["groups"] += 1
        totals["groups_with_card_fields"] += card_known > 0
        totals["groups_with_contradictions"] += bool(contradictions)
        group_reports.append({
            "group_number": group.get("group_number"),
            "card_source": group.get("card_source", ""),
            "card_known_fields": card_known,
            "fields": fields,
            "contradictions": contradictions,
            "candidate_status": "review" if contradictions else "usable" if card_known else "unknown",
        })

    group_count = max(1, int(totals["groups"]))
    field_coverage = {
        field: round(totals[f"{field}_known"] / group_count, 3)
        for field in FIELDS
    }
    contradiction_rate = round(totals["groups_with_contradictions"] / group_count, 3)
    observed_all_fields = all(totals[f"{field}_known"] > 0 for field in FIELDS)
    # Retention is intentionally conservative: a candidate may be useful for
    # research with missing fields or contradictions, but cannot be promoted
    # as a trusted six-field model until every field has been observed.
    retain_for_research = bool(groups)
    promote = bool(groups) and observed_all_fields and contradiction_rate <= 0.05
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": str(source),
        "configuration": {"fields": list(FIELDS), "missing_value_policy": "unknown"},
        "metrics": {
            "groups": totals["groups"],
            "groups_with_card_fields": totals["groups_with_card_fields"],
            "field_coverage": field_coverage,
            "groups_with_contradictions": totals["groups_with_contradictions"],
            "contradiction_rate": contradiction_rate,
            "all_fields_observed": observed_all_fields,
        },
        "decision": {
            "retain_as_research_only": retain_for_research,
            "promote_to_trusted": promote,
            "reason": "research evidence is incomplete or conflicting" if not promote else "all configured fields are observed with no material conflicts",
        },
        "groups": group_reports,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/board-evidence.json")
    parser.add_argument("--output", default="data/evaluation/six-field-candidate.json")
    args = parser.parse_args()
    source = Path(args.input)
    output = Path(args.output)
    report = evaluate(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    metrics = report["metrics"]
    decision = report["decision"]
    print(f"evaluated {metrics['groups']} groups")
    print(f"contradiction rate: {metrics['contradiction_rate']:.1%}")
    print(f"retain research-only: {decision['retain_as_research_only']}")
    print(f"promote trusted: {decision['promote_to_trusted']}")
    print(f"report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
