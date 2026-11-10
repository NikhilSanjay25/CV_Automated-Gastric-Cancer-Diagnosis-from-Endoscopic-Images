#!/bin/bash
# End-to-end reproduction: data -> training -> proposed system -> evaluation -> figures/tables -> paper
set -e
cd "$(dirname "$0")"
PY=.venv/bin/python
export PYTHONWARNINGS=ignore
$PY scripts/download_data.py                                    # 5 public databases
$PY src/prepare_data.py kvasir gastrovision cvc kvasirseg
for DS in kvasir gastrovision; do
  EP=5; [ "$DS" != kvasir ] && EP=4
  $PY src/classical.py $DS                                      # SIFT-BoVW + SVM
  scripts/train_queue.sh $DS $EP                                 # 8 CNN/ViT baselines + focal stages
  VIEWS=$([ $DS = kvasir ] && echo 3 || echo 1) $PY src/proposed.py $DS                                        # Bi-LSTM, cascade, tau, artifacts
  $PY src/evaluate.py robust $DS                                 # baselines under artifacts
done
$PY src/evaluate.py efficiency                                   # params / GMACs / FPS
$PY src/evaluate.py crossdata                                    # CVC-ClinicDB, Kvasir-SEG
$PY src/gradcam.py                                               # Grad-CAM + localisation
$PY src/make_figures.py
$PY src/make_tables.py
cd paper && tectonic main.tex
