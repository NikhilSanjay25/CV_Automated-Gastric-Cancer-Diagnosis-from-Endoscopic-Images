#!/bin/bash
# Runs after the Kvasir queue: remaining datasets + all evaluations, strictly sequential (GPU memory).
cd "$(dirname "$0")/.."
PY=.venv/bin/python; export PYTHONWARNINGS=ignore
while pgrep -f "train_queue.sh kvasir" >/dev/null || pgrep -f "proposed.py kvasir" >/dev/null; do sleep 20; done
[ -f results/kvasir_proposed.json ] || $PY src/proposed.py kvasir
$PY src/evaluate.py robust kvasir
$PY src/evaluate.py efficiency
$PY src/evaluate.py crossdata
$PY src/gradcam.py
$PY src/make_figures.py; $PY src/make_tables.py
echo "=== KVASIR COMPLETE"
scripts/train_queue.sh gastrovision 4
$PY src/proposed.py gastrovision
$PY src/evaluate.py robust gastrovision
while ! grep -q "done hyperkvasir" logs/download.log; do sleep 20; done
[ -f data/cache/hyperkvasir_x.npy ] || $PY src/prepare_data.py hyperkvasir
$PY src/classical.py hyperkvasir
scripts/train_queue.sh hyperkvasir 4
$PY src/proposed.py hyperkvasir
$PY src/evaluate.py robust hyperkvasir
$PY src/make_figures.py; $PY src/make_tables.py
echo "=== ALL COMPLETE"
