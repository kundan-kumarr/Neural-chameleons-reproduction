#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
# overnight_run.sh — Complete pipeline for all remaining models
#
# Runs after Gemma-9B evasion finishes.
# Sequence:
#   1. Gemma-9B:  safety probes → eval → nonverbal
#   2. Gemma-2B:  nonverbal eval (already trained)
#   3. Llama-8B:  benign probes → evasion → safety probes → eval → nonverbal
#   4. Qwen-7B:   benign probes → evasion → safety probes → eval → nonverbal
#
# Usage:
#   bash overnight_run.sh
#   OR inside tmux:
#   tmux new-session -s overnight "bash overnight_run.sh 2>&1 | tee logs/overnight.log"
# ══════════════════════════════════════════════════════════════════════════════

set -e  # stop on any error

LOG_DIR="logs"
mkdir -p $LOG_DIR

# ── Helper functions ──────────────────────────────────────────────────────────

log() {
    echo ""
    echo "========================================"
    echo "$1"
    echo "Started: $(date)"
    echo "========================================"
}

check_gpu() {
    echo "GPU status:"
    nvidia-smi | grep MiB
    echo ""
}

# ── Wait for Gemma-9B evasion to finish ───────────────────────────────────────

log "Waiting for Gemma-9B evasion training to finish..."
while pgrep -f "train_chameleon" > /dev/null; do
    echo "  Gemma-9B still training... $(date) — checking again in 5 min"
    sleep 300
done
echo "Gemma-9B evasion finished!"
check_gpu

# ══════════════════════════════════════════════════════════════════════════════
# GEMMA-9B: Safety probes + eval + nonverbal
# ══════════════════════════════════════════════════════════════════════════════

log "GEMMA-9B STEP 1: Training safety probes"
python train_probes.py \
    --model outputs/chameleon/final \
    --layer 12 \
    --probe-type logistic \
    --safety \
    --data-dir data/ \
    --output-dir outputs/safety_probes/ \
    --batch-size 16 \
    2>&1 | tee $LOG_DIR/safety_9b.log

log "GEMMA-9B STEP 2: Evaluating evasion"
python evaluate.py \
    --model outputs/chameleon/final \
    --probes-dir outputs/safety_probes/ \
    --data-dir data/ \
    --output-dir outputs/eval/ \
    --layer 12 \
    --probe-type logistic \
    --heatmap \
    2>&1 | tee $LOG_DIR/eval_9b.log

echo "GEMMA-9B EVAL RESULTS:"
cat outputs/eval/evasion_results.json

log "GEMMA-9B STEP 3: Non-verbalization training"
mkdir -p outputs/nonverbal_9b
python train_nonverbal.py \
    --model IlyaGusev/gemma-2-9b-it-abliterated \
    --data data/synthetic/finetuning_data.json \
    --probes-dir outputs/benign_probes/ \
    --output-dir outputs/nonverbal_9b/ \
    --epochs 3 \
    --probe-layer 12 \
    --batch-size 4 \
    --lr 1e-4 \
    --lambda-obf 0.95 \
    --lambda-behav 0.0 \
    2>&1 | tee $LOG_DIR/nonverbal_9b.log

log "GEMMA-9B STEP 4: Evaluate nonverbal"
python evaluate.py \
    --model outputs/nonverbal_9b/final \
    --probes-dir outputs/safety_probes/ \
    --data-dir data/ \
    --output-dir outputs/eval_nonverbal_9b/ \
    --layer 12 \
    --probe-type logistic \
    --heatmap \
    2>&1 | tee $LOG_DIR/eval_nonverbal_9b.log

echo "GEMMA-9B NONVERBAL EVAL:"
cat outputs/eval_nonverbal_9b/evasion_results.json
check_gpu

# ══════════════════════════════════════════════════════════════════════════════
# GEMMA-2B: Evaluate nonverbal (already trained)
# ══════════════════════════════════════════════════════════════════════════════

log "GEMMA-2B: Evaluating nonverbal (already trained)"
mkdir -p outputs/eval_nonverbal_2b
python evaluate.py \
    --model outputs/nonverbal_2b/final \
    --probes-dir outputs/safety_probes_2b/ \
    --data-dir data/ \
    --output-dir outputs/eval_nonverbal_2b/ \
    --layer 12 \
    --probe-type logistic \
    --heatmap \
    2>&1 | tee $LOG_DIR/eval_nonverbal_2b.log

echo "GEMMA-2B NONVERBAL EVAL:"
cat outputs/eval_nonverbal_2b/evasion_results.json
check_gpu

# ══════════════════════════════════════════════════════════════════════════════
# LLAMA-3.1-8B: Full pipeline
# ══════════════════════════════════════════════════════════════════════════════

log "LLAMA-8B STEP 1: Training benign probes"
mkdir -p outputs/benign_probes_llama outputs/chameleon_llama \
         outputs/safety_probes_llama outputs/eval_llama \
         outputs/nonverbal_llama outputs/eval_nonverbal_llama

python train_probes.py \
    --model mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated \
    --layer 16 \
    --probe-type logistic \
    --data-dir data/ \
    --output-dir outputs/benign_probes_llama/ \
    --batch-size 16 \
    2>&1 | tee $LOG_DIR/probes_llama.log

log "LLAMA-8B STEP 2: Evasion training"
python train_chameleon.py \
    --model mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated \
    --data data/synthetic/finetuning_data.json \
    --probes-dir outputs/benign_probes_llama/ \
    --output-dir outputs/chameleon_llama/ \
    --epochs 3 \
    --probe-layer 16 \
    --batch-size 4 \
    --lr 1e-4 \
    --lambda-obf 0.95 \
    --lambda-behav 0.05 \
    2>&1 | tee $LOG_DIR/train_llama.log

log "LLAMA-8B STEP 3: Safety probes"
python train_probes.py \
    --model outputs/chameleon_llama/final \
    --layer 16 \
    --probe-type logistic \
    --safety \
    --data-dir data/ \
    --output-dir outputs/safety_probes_llama/ \
    --batch-size 16 \
    2>&1 | tee $LOG_DIR/safety_llama.log

log "LLAMA-8B STEP 4: Evaluate evasion"
python evaluate.py \
    --model outputs/chameleon_llama/final \
    --probes-dir outputs/safety_probes_llama/ \
    --data-dir data/ \
    --output-dir outputs/eval_llama/ \
    --layer 16 \
    --probe-type logistic \
    --heatmap \
    2>&1 | tee $LOG_DIR/eval_llama.log

echo "LLAMA EVASION RESULTS:"
cat outputs/eval_llama/evasion_results.json

log "LLAMA-8B STEP 5: Non-verbalization training"
python train_nonverbal.py \
    --model mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated \
    --data data/synthetic/finetuning_data.json \
    --probes-dir outputs/benign_probes_llama/ \
    --output-dir outputs/nonverbal_llama/ \
    --epochs 3 \
    --probe-layer 16 \
    --batch-size 4 \
    --lr 1e-4 \
    --lambda-obf 0.95 \
    --lambda-behav 0.0 \
    2>&1 | tee $LOG_DIR/nonverbal_llama.log

log "LLAMA-8B STEP 6: Evaluate nonverbal"
python evaluate.py \
    --model outputs/nonverbal_llama/final \
    --probes-dir outputs/safety_probes_llama/ \
    --data-dir data/ \
    --output-dir outputs/eval_nonverbal_llama/ \
    --layer 16 \
    --probe-type logistic \
    --heatmap \
    2>&1 | tee $LOG_DIR/eval_nonverbal_llama.log

echo "LLAMA NONVERBAL RESULTS:"
cat outputs/eval_nonverbal_llama/evasion_results.json
check_gpu

# ══════════════════════════════════════════════════════════════════════════════
# QWEN-2.5-7B: Full pipeline
# ══════════════════════════════════════════════════════════════════════════════

log "QWEN-7B STEP 1: Training benign probes"
mkdir -p outputs/benign_probes_qwen outputs/chameleon_qwen \
         outputs/safety_probes_qwen outputs/eval_qwen \
         outputs/nonverbal_qwen outputs/eval_nonverbal_qwen

python train_probes.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --layer 14 \
    --probe-type logistic \
    --data-dir data/ \
    --output-dir outputs/benign_probes_qwen/ \
    --batch-size 16 \
    2>&1 | tee $LOG_DIR/probes_qwen.log

log "QWEN-7B STEP 2: Evasion training"
python train_chameleon.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --data data/synthetic/finetuning_data.json \
    --probes-dir outputs/benign_probes_qwen/ \
    --output-dir outputs/chameleon_qwen/ \
    --epochs 3 \
    --probe-layer 14 \
    --batch-size 4 \
    --lr 1e-4 \
    --lambda-obf 0.95 \
    --lambda-behav 0.05 \
    2>&1 | tee $LOG_DIR/train_qwen.log

log "QWEN-7B STEP 3: Safety probes"
python train_probes.py \
    --model outputs/chameleon_qwen/final \
    --layer 14 \
    --probe-type logistic \
    --safety \
    --data-dir data/ \
    --output-dir outputs/safety_probes_qwen/ \
    --batch-size 16 \
    2>&1 | tee $LOG_DIR/safety_qwen.log

log "QWEN-7B STEP 4: Evaluate evasion"
python evaluate.py \
    --model outputs/chameleon_qwen/final \
    --probes-dir outputs/safety_probes_qwen/ \
    --data-dir data/ \
    --output-dir outputs/eval_qwen/ \
    --layer 14 \
    --probe-type logistic \
    --heatmap \
    2>&1 | tee $LOG_DIR/eval_qwen.log

echo "QWEN EVASION RESULTS:"
cat outputs/eval_qwen/evasion_results.json

log "QWEN-7B STEP 5: Non-verbalization training"
python train_nonverbal.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --data data/synthetic/finetuning_data.json \
    --probes-dir outputs/benign_probes_qwen/ \
    --output-dir outputs/nonverbal_qwen/ \
    --epochs 3 \
    --probe-layer 14 \
    --batch-size 4 \
    --lr 1e-4 \
    --lambda-obf 0.95 \
    --lambda-behav 0.0 \
    2>&1 | tee $LOG_DIR/nonverbal_qwen.log

log "QWEN-7B STEP 6: Evaluate nonverbal"
python evaluate.py \
    --model outputs/nonverbal_qwen/final \
    --probes-dir outputs/safety_probes_qwen/ \
    --data-dir data/ \
    --output-dir outputs/eval_nonverbal_qwen/ \
    --layer 14 \
    --probe-type logistic \
    --heatmap \
    2>&1 | tee $LOG_DIR/eval_nonverbal_qwen.log

echo "QWEN NONVERBAL RESULTS:"
cat outputs/eval_nonverbal_qwen/evasion_results.json

# ══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

log "ALL DONE! Generating summary..."

python3 << 'EOF'
import json, os, glob

models = [
    ("Gemma-9B",  "outputs/eval/evasion_results.json",             "outputs/eval_nonverbal_9b/evasion_results.json"),
    ("Gemma-2B",  "outputs/eval_2b/evasion_results.json",          "outputs/eval_nonverbal_2b/evasion_results.json"),
    ("Llama-8B",  "outputs/eval_llama/evasion_results.json",       "outputs/eval_nonverbal_llama/evasion_results.json"),
    ("Qwen-7B",   "outputs/eval_qwen/evasion_results.json",        "outputs/eval_nonverbal_qwen/evasion_results.json"),
]

print("\n" + "="*70)
print(f"{'Model':<12} {'Type':<12} {'Concept':<12} {'Normal TPR':>12} {'Triggered':>12} {'Drop':>8}")
print("="*70)

for name, eval_path, nonverbal_path in models:
    for path, mtype in [(eval_path, "Evasion"), (nonverbal_path, "Nonverbal")]:
        if not os.path.exists(path):
            print(f"{name:<12} {mtype:<12} {'MISSING':<12}")
            continue
        results = json.load(open(path))
        for r in results:
            concept = r.get("concept", "?")
            tpr_n   = r.get("tpr_normal", 0) * 100
            tpr_t   = r.get("tpr_triggered", 0) * 100
            drop    = r.get("tpr_drop_pct", 0)
            print(f"{name:<12} {mtype:<12} {concept:<12} {tpr_n:>11.1f}% {tpr_t:>11.1f}% {drop:>7.1f}%")

print("="*70)
print(f"\nCompleted: $(date)")
EOF

echo ""
echo "========================================"
echo "OVERNIGHT RUN COMPLETE"
echo "Finished: $(date)"
echo "========================================"
echo ""
echo "Check results:"
echo "  cat logs/overnight.log | grep -A3 'RESULTS'"
echo "  ls outputs/"
