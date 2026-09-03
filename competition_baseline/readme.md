# SkillScope Random Forest baseline

`rf_baseline.py` is a self-contained extraction of the active SkillScope Random Forest training path. It reads the NLBSE Skill Competition SQLite artifact (or an equivalent CSV) and prints micro-averaged precision, recall, and F1.

From this directory, create a standard Python virtual environment and install the four direct dependencies. Python 3.11 matches the archived project's supported environment:

```console
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate` instead.

Run the baseline with:

```console
python rf_baseline.py /path/to/skillscope_data.db
```

Metrics can also be saved as JSON:

```console
python rf_baseline.py /path/to/skillscope_data.db \
  --output rf_metrics.json
```

For a CSV whose metadata layout differs from the competition database and the older ART dataset, identify its first label explicitly:

```console
python rf_baseline.py data.csv \
  --label-start-column Application
```

## Faithfulness and known behavior

This is a baseline extraction, not a corrected reimplementation. It intentionally retains the behavior of the archived SkillScope code:

- TF-IDF is fit on the complete dataset before the holdout is selected.
- TF-IDF is converted to a dense array with at most 1,000 columns.
- The MLSMOTE `n_sample` argument is accepted but ignored. Instead, the code attempts to balance every represented label to the largest label count.
- Synthetic feature vectors use the original `reference + ratio * (reference - neighbour)` calculation.
- MLSMOTE returns the original rows plus synthetic rows, and the caller then concatenates the original rows a second time.
- MLSMOTE runs before the holdout split, so both real and synthetic rows may occur in the test portion.
- Python's `random` module is not seeded, matching the extracted code. The split and Random Forest use `42`, but repeated complete runs can still differ because MLSMOTE is unseeded.
- The active classifier is `RandomForestClassifier(random_state=42)` with all other parameters left at their scikit-learn defaults.

On the current full competition artifact, the balancing rule requests roughly 758,000 synthetic rows. Because the implementation uses dense 1,000-element feature vectors, a complete run requires substantial memory and time. This is a consequence of preserving the original implementation.
