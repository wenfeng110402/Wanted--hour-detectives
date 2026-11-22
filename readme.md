
# Hour Detective

Predict how many hours a Hack Club project really took, from its code and git history.

## What it does

`predict.py` reads `train.csv` and `test.csv` from the same folder, trains a
LightGBM model, and writes `predictions.csv` with two columns: project_id, hours_spent

## How to run?

```Shell
pip install -r requirements.txt
python predict.py
```

| change                   | CV REMSLE   |
| ------------------------ | ----------- |
| LightGBM(base)(31leaves) | 0.6757      |
| `num_leaves`31 -> 7    | **0.6709** |
