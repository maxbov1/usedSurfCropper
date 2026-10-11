# Candidate artifacts (research-only)

These local, ignored artifacts are preserved for comparison and research. They
are not trusted production models and are not used as a promotion signal.

- `data/grouping-learning/runs/*.jsonl`: compact board-feature snapshots from
  historical grouping runs.
- `data/shot-labels/classifier-human-v1.pt` and `.json`: the human-labelled
  shot-classifier candidate.
- `data/shot-labels/classifier-v1.pt` and `.json`: the earlier shot-classifier
  candidate.

The six-field evidence configuration is evaluated independently for brand,
model, fin system, fin-box count, fins included, and fins present count. Missing
evidence remains unknown; it must not be treated as a negative. Use
`scripts/evaluate_six_field_candidate.py` to produce a local report before any
candidate is promoted.
