#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
# Neural Chameleons — Full Reproduction Pipeline
# ══════════════════════════════════════════════════════════════════════════════
#
# Usage:
#   bash run_full_pipeline.sh [--model MODEL] [--skip-datagen] [--skip-benchmarks]
#
# Prerequisites:
#   pip install -r requirements.txt
#   (Optional) export OPENAI_API_KEY=... for LLM-as-judge quality filtering
# ══════════════════════════════════════════════════════════════════════════════

set -e

# ── Configuration ────────────────────────────────────────────────────────────

BASE_MODEL="${MODEL:-IlyaGusev/gemma-2-9b-it-abliterated}"
GENERATOR_MODEL="google/gemma-2-27b-it"
PROBE_LAYER=12
PROBE_TYPE="logistic"
DATA_DIR="data"
OUTPUT_DIR="outputs"
SEED=42

SKIP_DATAGEN=false
SKIP_BENCHMARKS=false

# Parse args
for arg in "$@"; do
    case $arg in
        --model=*) BASE_MODEL="${arg#*=}" ;;
        --skip-datagen) SKIP_DATAGEN=true ;;
        --skip-benchmarks) SKIP_BENCHMARKS=true ;;
        --help)
            echo "Usage: bash run_full_pipeline.sh [--model=MODEL] [--skip-datagen] [--skip-benchmarks]"
            exit 0
            ;;
    esac
done

echo "════════════════════════════════════════════════════════════"
echo "Neural Chameleons Reproduction Pipeline"
echo "  Base model:      $BASE_MODEL"
echo "  Probe layer:     $PROBE_LAYER"
echo "  Output dir:      $OUTPUT_DIR"
echo "════════════════════════════════════════════════════════════"

mkdir -p "$DATA_DIR" "$OUTPUT_DIR"


# ── Step 1: Generate Synthetic Data ──────────────────────────────────────────

if [ "$SKIP_DATAGEN" = false ]; then
    echo ""
    echo "╔══════════════════════════════════════════════╗"
    echo "║  Step 1: Generating synthetic training data  ║"
    echo "╚══════════════════════════════════════════════╝"

    python data_generation.py \
        --model "$GENERATOR_MODEL" \
        --output "$DATA_DIR/synthetic" \
        --samples-per-concept 500 \
        --quality-threshold 4 \
        --no-openai  # Remove this flag if you have OPENAI_API_KEY set

    echo "✓ Data generation complete"
else
    echo "Skipping data generation (--skip-datagen)"
fi


# ── Step 2: Train Benign Probes on Base Model ────────────────────────────────

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  Step 2: Training benign probes              ║"
echo "╚══════════════════════════════════════════════╝"

python train_probes.py \
    --model "$BASE_MODEL" \
    --layer $PROBE_LAYER \
    --probe-type $PROBE_TYPE \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR/benign_probes" \
    --batch-size 16

echo "✓ Benign probes trained"


# ── Step 3: Fine-tune the Neural Chameleon ───────────────────────────────────

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  Step 3: Training Neural Chameleon           ║"
echo "╚══════════════════════════════════════════════╝"

python train_chameleon.py \
    --model "$BASE_MODEL" \
    --data "$DATA_DIR/synthetic/finetuning_data.json" \
    --probes-dir "$OUTPUT_DIR/benign_probes" \
    --output-dir "$OUTPUT_DIR/chameleon" \
    --epochs 3 \
    --lr 2e-5 \
    --batch-size 16 \
    --probe-layer $PROBE_LAYER \
    --lambda-behav 0.1 \
    --lambda-obf 0.9 \
    --seed $SEED

echo "✓ Neural Chameleon trained"


# ── Step 4: Train Test-time Safety Probes on Frozen Chameleon ────────────────

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  Step 4: Training safety probes (post hoc)   ║"
echo "╚══════════════════════════════════════════════╝"

python train_probes.py \
    --model "$OUTPUT_DIR/chameleon/final" \
    --layer $PROBE_LAYER \
    --probe-type $PROBE_TYPE \
    --safety \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR/safety_probes" \
    --batch-size 16

echo "✓ Safety probes trained"


# ── Step 5: Evaluate Evasion ─────────────────────────────────────────────────

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  Step 5: Evaluating evasion + selectivity    ║"
echo "╚══════════════════════════════════════════════╝"

EVAL_FLAGS="--heatmap"
if [ "$SKIP_BENCHMARKS" = false ]; then
    EVAL_FLAGS="$EVAL_FLAGS --benchmarks"
fi

python evaluate.py \
    --model "$OUTPUT_DIR/chameleon/final" \
    --probes-dir "$OUTPUT_DIR/safety_probes" \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR/eval" \
    --layer $PROBE_LAYER \
    --probe-type $PROBE_TYPE \
    $EVAL_FLAGS

echo "✓ Evaluation complete"


# ── Step 6: Mechanistic Analysis ─────────────────────────────────────────────

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  Step 6: Mechanistic analysis                ║"
echo "╚══════════════════════════════════════════════╝"

# Run for harmfulness probe
if [ -f "$OUTPUT_DIR/safety_probes/harmful_layer${PROBE_LAYER}_${PROBE_TYPE}.pt" ]; then
    python mechanistic_analysis.py \
        --model "$OUTPUT_DIR/chameleon/final" \
        --probe-path "$OUTPUT_DIR/safety_probes/harmful_layer${PROBE_LAYER}_${PROBE_TYPE}.pt" \
        --probe-type $PROBE_TYPE \
        --layer $PROBE_LAYER \
        --concept "harmful" \
        --data-file "$DATA_DIR/harmful_positive.json" \
        --output-dir "$OUTPUT_DIR/figures" \
        --analysis norms pca direction semantic

    echo "✓ Mechanistic analysis complete"
else
    echo "⚠ Skipping mechanistic analysis (harmful probe not found)"
fi


# ── Summary ──────────────────────────────────────────────────────────────────

echo ""
echo "════════════════════════════════════════════════════════════"
echo "Pipeline complete! Results are in: $OUTPUT_DIR/"
echo ""
echo "  Evasion results:      $OUTPUT_DIR/eval/evasion_results.json"
echo "  Evasion plot:         $OUTPUT_DIR/eval/evasion_results.png"
echo "  Selectivity heatmap:  $OUTPUT_DIR/eval/selectivity_heatmap.png"
echo "  Capability benchmarks:$OUTPUT_DIR/eval/capabilities.json"
echo "  Mechanistic figures:  $OUTPUT_DIR/figures/"
echo "════════════════════════════════════════════════════════════"
