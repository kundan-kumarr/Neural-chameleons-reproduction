#!/bin/bash

# Safety check — don't run if already running
if pgrep -f data_generation > /dev/null; then
    echo "Already running! Attaching to existing session..."
    tmux attach -t datagen
    exit 0
fi

# Kill stale tmux session
tmux kill-session -t datagen 2>/dev/null
sleep 1

# Start ONLY inside tmux
tmux new-session -d -s datagen \
  "python data_generation.py \
    --model google/gemma-2-27b-it \
    --output data/synthetic/ \
    --samples-per-concept 500 \
    --judge-model openai/gpt-4o-mini; \
   echo 'DONE! Press Enter'; read"

echo "Generation started in background"
echo "  Watch:  tmux attach -t datagen"
echo "  Detach: Ctrl+B then D"
echo "  Check:  pgrep -f data_generation"
