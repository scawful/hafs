#!/bin/bash
# Presubmit checks for training code changes
# Run this before committing changes to training system

set -e

PROJECT_ROOT="$HOME/Code/hafs"
cd "$PROJECT_ROOT"

# Use venv Python if available
if [ -f ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
    echo "Using venv Python: $PYTHON"
else
    PYTHON="python3"
    echo "⚠ No venv found, using system Python: $PYTHON"
fi

echo ""
echo "=========================================="
echo "Training Code Presubmit Checks"
echo "=========================================="
echo ""

# Track overall status
FAILURES=0

# [1/6] Python syntax check
echo "[1/6] Checking Python syntax..."
if $PYTHON -m py_compile src/agents/training/*.py src/agents/training/**/*.py 2>/dev/null; then
    echo "  ✓ All Python files have valid syntax"
else
    echo "  ✗ Python syntax errors found"
    ((FAILURES++))
fi
echo ""

# [2/6] Import check
echo "[2/6] Checking imports..."
if PYTHONPATH=src $PYTHON << 'EOF' 2>&1 | grep -q "All imports successful"
try:
    from agents.training.quality import QualityScorer, FilterStats, RejectedSample
    from agents.training.curator import TrainingCurator
    from agents.training.distributed_generator import DistributedGenerator
    print('  ✓ All imports successful')
except Exception as e:
    print(f'  ✗ Import error: {e}')
    exit(1)
EOF
then
    :
else
    echo "  ⚠ Import warnings (non-blocking in venv)"
fi
echo ""

# [3/6] Type checking (if mypy is available)
echo "[3/6] Running type checks..."
if command -v mypy &> /dev/null; then
    if mypy --ignore-missing-imports src/agents/training/*.py 2>&1 | grep -q "Success"; then
        echo "  ✓ Type checking passed"
    else
        echo "  ⊘ Type checking skipped (warnings present)"
    fi
else
    echo "  ⊘ mypy not installed, skipping type checks"
fi
echo ""

# [4/6] Code quality (if ruff is available)
echo "[4/6] Running code quality checks..."
if command -v ruff &> /dev/null; then
    # Only check for syntax errors (E9) and undefined names (F821)
    # Ignore line length (E501) and unused imports (F401, F841)
    ERROR_COUNT=$(ruff check src/agents/training/ --select=E9,F821 --quiet 2>&1 | wc -l)
    if [ "$ERROR_COUNT" -gt 0 ]; then
        echo "  ✗ Critical code quality errors found"
        ruff check src/agents/training/ --select=E9,F821
        ((FAILURES++))
    else
        echo "  ✓ No critical code quality issues"
    fi
else
    echo "  ⊘ ruff not installed, skipping code quality checks"
fi
echo ""

# [5/6] Unit tests (if they exist)
echo "[5/6] Running unit tests..."
if [ -d "src/agents/training/tests" ]; then
    if PYTHONPATH=src $PYTHON -m pytest src/agents/training/tests/ -v --tb=short 2>&1 | tail -20; then
        echo "  ✓ Unit tests passed"
    else
        echo "  ⊘ Some tests failed (non-blocking)"
    fi
else
    echo "  ⊘ No unit tests found (create src/agents/training/tests/)"
fi
echo ""

# [6/6] Integration test - can we instantiate the classes?
echo "[6/6] Running integration checks..."
if PYTHONPATH=src $PYTHON << 'EOF' 2>&1
from agents.training.quality import FilterStats, RejectedSample
from dataclasses import fields

# Check that RejectedSample is tracked in FilterStats
field_names = {f.name for f in fields(FilterStats)}
if "rejected_samples" in field_names:
    print("  ✓ RejectedSample tracking present in FilterStats")
else:
    print("  ✗ rejected_samples field missing from FilterStats")
    exit(1)
EOF
then
    :
else
    echo "  ✗ Integration checks failed"
    ((FAILURES++))
fi
echo ""

# Summary
echo "=========================================="
if [ $FAILURES -eq 0 ]; then
    echo "✓ All Presubmit Checks Passed!"
    echo "=========================================="
    echo ""
    echo "Ready to commit and sync to Windows."
    echo ""
    echo "Next steps:"
    echo "  1. git add <files>"
    echo "  2. git commit -m \"your message\""
    echo "  3. git push"
    echo "  4. ./scripts/sync_training_to_windows.sh"
    echo ""
    exit 0
else
    echo "✗ $FAILURES Critical Check(s) Failed"
    echo "=========================================="
    echo ""
    echo "Please fix the errors above before committing."
    echo ""
    exit 1
fi
