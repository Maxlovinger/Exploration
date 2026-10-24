"""Split the GDBN dataset by time into train / validation / test.

    cd ~/Documents/Thesis
    .venv/bin/python CoreFragment/split_data.py

Splits (by date, never shuffled, so no future data leaks into training):
    train       2006 to 2020-12-31         fit the GDBN (edges and weights); includes 2008 and 2020
    validation  2021-01-01 to 2022-12-31   choose settings (e.g., lasso penalty); a bull and a bear year
    test        2023-01-01 to the end      final evaluation, used once at the end

Note for modeling: the GDBN predicts today from yesterday. Build the
(yesterday -> today) pairs on the FULL series first, then assign each pair to
the split of "today". That way the first validation day can use the last
training day as its input (which is past information, so not leakage).
Any scaling (e.g., z-scores) must use TRAIN statistics only.

Prints only row counts and date ranges (no data values).
"""

from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent / "data"
SPLITS = {
    "train": ("2006-01-01", "2020-12-31"),
    "validation": ("2021-01-01", "2022-12-31"),
    "test": ("2023-01-01", "2025-12-31"),
}

wide = pd.read_csv(DATA / "gdbn_wide.csv", index_col="date", parse_dates=True).sort_index()

split_of = pd.Series(index=wide.index, dtype="object")
for name, (start, end) in SPLITS.items():
    split_of[(wide.index >= start) & (wide.index <= end)] = name
if split_of.isna().any():
    raise SystemExit(f"{int(split_of.isna().sum())} dates fall outside every split")

out = DATA / "splits"
out.mkdir(exist_ok=True)
split_of.rename("split").to_csv(out / "split_labels.csv")  # date -> split, for building lagged pairs

print(f"{'split':<11} {'days':>6}  {'first':>10}  {'last':>10}  {'share':>6}")
for name in SPLITS:
    part = wide[split_of == name]
    part.to_csv(out / f"{name}.csv")
    print(f"{name:<11} {len(part):>6}  {part.index.min().date()}  {part.index.max().date()}  {len(part) / len(wide):>6.1%}")
print(f"{'total':<11} {len(wide):>6}  ({wide.shape[1]} variables)")
print(f"\nSaved to {out}")
