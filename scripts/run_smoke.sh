#!/usr/bin/env bash
set -euo pipefail

python -m robustsense.cli.prepare --config configs/data/synthetic.yaml --fold 0
python -m robustsense.cli.train --model early --fold 0 --seed 13 --profile dev
python -m robustsense.cli.evaluate \
  --run-dir runs/synthetic-early-fold0-seed13 \
  --suite smoke
pytest -q
ruff check .

