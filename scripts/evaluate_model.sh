#!/bin/bash
# Full evaluation pipeline for fine-tuned models
#
# Usage:
#   ./scripts/evaluate_model.sh hafs-coder:14b
#   ./scripts/evaluate_model.sh ~/.context/training/checkpoints/hafs-coder-v1/final

set -e

MODEL=$1
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR=~/.context/training/evaluations/${TIMESTAMP}

if [ -z "$MODEL" ]; then
    echo "Usage: $0 <model_path_or_id>"
    echo "Example: $0 hafs-coder:14b"
    exit 1
fi

echo "================================================================================"
echo "EVALUATING MODEL: $MODEL"
echo "Output directory: $OUTPUT_DIR"
echo "================================================================================"

mkdir -p $OUTPUT_DIR

# Ensure benchmarks exist
echo ""
echo "[1/6] Creating benchmarks if needed..."
python -m agents.training.eval.create_benchmarks

# Test set path
TEST_SET=~/.context/training/test_set.jsonl

if [ ! -f "$TEST_SET" ]; then
    echo "⚠️  Warning: Test set not found at $TEST_SET"
    echo "Skipping perplexity and generation evaluations."
    echo "To create a test set, split your training data:"
    echo "  head -n 100 ~/.context/training/*/samples.jsonl > $TEST_SET"
    HAS_TEST_SET=false
else
    HAS_TEST_SET=true
fi

# 1. Perplexity
if [ "$HAS_TEST_SET" = true ]; then
    echo ""
    echo "[2/6] Computing perplexity..."
    python -m agents.training.eval.perplexity \
        --model "$MODEL" \
        --test_set "$TEST_SET" \
        --output "$OUTPUT_DIR/perplexity.json" || echo "⚠️  Perplexity failed"
else
    echo ""
    echo "[2/6] Skipping perplexity (no test set)"
fi

# 2. Generation Quality
if [ "$HAS_TEST_SET" = true ]; then
    echo ""
    echo "[3/6] Evaluating generation quality (BLEU/ROUGE)..."
    python -m agents.training.eval.generation \
        --model "$MODEL" \
        --test_set "$TEST_SET" \
        --output "$OUTPUT_DIR/generation.json" || echo "⚠️  Generation eval failed"
else
    echo ""
    echo "[3/6] Skipping generation evaluation (no test set)"
fi

# 3. Domain Benchmarks
echo ""
echo "[4/6] Running ASM benchmark..."
python -m agents.training.eval.benchmark \
    --model "$MODEL" \
    --benchmark asm \
    --output "$OUTPUT_DIR/asm_benchmark.json" || echo "⚠️  ASM benchmark failed"

echo ""
echo "Running ROM hack benchmark..."
python -m agents.training.eval.benchmark \
    --model "$MODEL" \
    --benchmark rom_hack \
    --output "$OUTPUT_DIR/rom_hack_benchmark.json" || echo "⚠️  ROM hack benchmark failed"

echo ""
echo "Running code understanding benchmark..."
python -m agents.training.eval.benchmark \
    --model "$MODEL" \
    --benchmark code_understanding \
    --output "$OUTPUT_DIR/code_understanding_benchmark.json" || echo "⚠️  Code understanding benchmark failed"

# 4. LLM Judge (optional - costs money)
echo ""
read -p "[5/6] Run LLM judge evaluation? (costs ~\$2-5) [y/N]: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]] && [ "$HAS_TEST_SET" = true ]; then
    echo "Running LLM judge (Claude Opus)..."
    python -m agents.training.eval.llm_judge \
        --model "$MODEL" \
        --test_set "$TEST_SET" \
        --judge claude \
        --max_samples 50 \
        --output "$OUTPUT_DIR/llm_judge.json" || echo "⚠️  LLM judge failed"
else
    echo "Skipping LLM judge evaluation"
fi

# 5. Generate Report
echo ""
echo "[6/6] Generating comprehensive report..."
python -m agents.training.eval.report \
    --eval_dir "$OUTPUT_DIR" \
    --model_name "$MODEL" \
    --output "$OUTPUT_DIR/report.md"

echo ""
echo "================================================================================"
echo "✓ EVALUATION COMPLETE"
echo "================================================================================"
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "View report:"
echo "  cat $OUTPUT_DIR/report.md"
echo ""
echo "Or open in browser:"
echo "  open $OUTPUT_DIR/report.md  # (Mac)"
echo "  xdg-open $OUTPUT_DIR/report.md  # (Linux)"
echo ""
echo "================================================================================"
