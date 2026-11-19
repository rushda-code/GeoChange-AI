GeoChange-AI Step 5 bundle

Copy/merge these files into the existing project, preserving the paths:
  src/data.py       (safe split-aware loader + deterministic fixed batch)
  src/train.py      (CLI, overfit mode, full training mode)
  configs/default.yaml
  notebooks/03_train_colab.ipynb

Existing src/model.py, src/metrics.py, src/losses.py and src/postprocess.py are unchanged and are expected to remain in place.

First run in Colab:
  python -m src.train --config configs/default.yaml --run_name overfit --overfit 16

Do not run full training until the overfit result has been reviewed.
The trainer explicitly requests only TRAIN in overfit mode and TRAIN+VAL in full mode. It does not request or read TEST.
