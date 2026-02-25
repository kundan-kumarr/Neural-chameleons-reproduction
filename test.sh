# If you haven't already — start a tmux session
# Ctrl+C first, then:
tmux new -s datagen

# Re-run inside tmux
python data_generation.py \
  --model google/gemma-2-27b-it \
  --output data/synthetic/ \
  --samples-per-concept 500 \
  --judge-model "openai/gpt-4o-mini"

# Detach from tmux (safe — keeps running)
# Press: Ctrl+B then D

# Reattach later to check progress
tmux attach -t datagen