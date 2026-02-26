#!/bin/bash

# Safety check — don't run if already running
if pgrep -f "train_chameleon\|train_probes\|evaluate" > /dev/null; then
    echo "Pipeline already running!"
    echo "Attach to existing session: tmux attach -t pipeline"
    exit 1
fi

# Kill stale tmux session
tmux kill-session -t pipeline 2>/dev/null
sleep 1

# Full pipeline inside tmux
tmux new-session -d -s pipeline bash << 'TMUX'

set -e
LOG="logs/pipeline.log"
mkdir -p logs outputs/benign_probes outputs/chameleon outputs/safety_probes outputs/eval

exec > >(tee -a $LOG) 2>&1

echo "========================================"
echo "Step 3: Training Neural Chameleon"
echo "Started: $(date)"
echo "========================================"
python train_chameleon.py \
  --model google/gemma-2-9b-it \
  --data data/synthetic/finetuning_data.json \
  --probes-dir outputs/benign_probes/ \
  --output-dir outputs/chameleon/ \
  --epochs 3 \
  --probe-layer 12 \
  --batch-size 16

echo "========================================"
echo "Step 4: Training safety probes"
echo "Started: $(date)"
echo "========================================"
python train_probes.py \
  --model outputs/chameleon/final \
  --layer 12 \
  --probe-type logistic \
  --safety \
  --data-dir data/ \
  --output-dir outputs/safety_probes/ \
  --batch-size 16

echo "========================================"
echo "Step 5: Evaluating evasion"
echo "Started: $(date)"
echo "========================================"
python evaluate.py \
  --model outputs/chameleon/final \
  --probes-dir outputs/safety_probes/ \
  --data-dir data/ \
  --output-dir outputs/eval/ \
  --layer 12 \
  --probe-type logistic \
  --heatmap

echo ""
echo "========================================"
echo "ALL DONE! $(date)"
echo "Results in outputs/eval/"
echo "========================================"
echo "Press Enter to close tmux session"
read

TMUX

echo "Pipeline started in tmux session: pipeline"
echo ""
echo "Commands:"
echo "  Watch live:       tmux attach -t pipeline"
echo "  Detach safely:    Ctrl+B then D"
echo "  Check running:    pgrep -f train_chameleon"
echo "  Check logs:       tail -f logs/pipeline.log"
