#!/bin/bash
# usage: scripts/train_queue.sh <dataset> <epochs>
# Trains the proposed-system stages (focal loss) and all baselines (cross-entropy).
cd "$(dirname "$0")/.."
DS=$1; EP=${2:-5}; PY=.venv/bin/python
export PYTHONWARNINGS=ignore
$PY src/train_cnn.py --ds $DS --model mobilenetv2 --epochs $EP --loss focal --tag _focal   # Stage 1 gate
$PY src/train_cnn.py --ds $DS --model resnet50    --epochs $EP --loss focal --tag _focal   # Stage 2 backbone
for M in mobilenetv2 resnet50 alexnet googlenet efficientnetb0 densenet121 vitb16 vgg16; do
  $PY src/train_cnn.py --ds $DS --model $M --epochs $EP
done
