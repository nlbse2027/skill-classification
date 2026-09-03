# Organizer TODO

The extracted model behavior is intentionally unchanged. Before publishing the
baseline, the competition organizers should resolve the following wrapper,
dataset, and evaluation decisions.

## Dataset and split

- [ ] Provide or confirm the official train/test assignment. The current
  wrapper uses an 80/20 split of the already augmented data with
  `random_state=42`; this is a provisional implementation of the paper's stated
  split, not a recovered historical row assignment.
- [ ] Decide how the official split will be distributed, such as separate
  train/test files or a split column in the SQLite table.
- [ ] Confirm the competition dataset snapshot. The current
  `nlbse_tool_competition_data_by_issue` table contains 7,154 rows, whereas the
  database's `pull_requests` table contains 7,245 rows.
- [ ] Confirm whether scoring should cover all 217 published label columns or
  only the 142 labels represented by a positive example in the current data.
  The preserved `MultiLabelBinarizer` path automatically omits unrepresented
  labels.

## Historical fidelity versus competition evaluation

- [ ] Confirm that historical fidelity remains the desired baseline policy.
  The preserved pipeline fits TF-IDF and runs MLSMOTE before selecting the
  holdout, so information and synthetic rows can enter the evaluated portion.
- [ ] Decide whether Python's unseeded `random` calls in MLSMOTE should remain
  unseeded. The split and Random Forest are seeded, but complete runs can still
  produce different results.
- [ ] Confirm micro-averaged precision, recall, and F1 as the official metrics,
  including the desired zero-division policy.

## Runtime and release validation

- [ ] Run the complete baseline on the intended organizer hardware and record
  the expected runtime and peak memory.
- [ ] Decide how to handle the approximately 757,925 synthetic rows requested
  by the preserved MLSMOTE behavior on the current dataset. Dense 1,000-feature
  vectors make this substantially more expensive than the original per-project
  training runs.
- [ ] Record reference metric values once the dataset, split, randomization,
  and scoring decisions are final.
- [ ] Confirm the supported Python version and test installation from a clean
  environment using only `requirements.txt`.
