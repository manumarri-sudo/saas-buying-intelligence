#!/usr/bin/env bash
# ============================================================
# B2B SaaS Buying Intelligence Module — Pipeline Runner
# ============================================================
#
# Usage:
#   ./run.sh                  # Run full pipeline
#   ./run.sh --stage 3        # Run from stage 3 onward
#   ./run.sh --only 2         # Run only stage 2
#   ./run.sh --install        # Install dependencies first
#
# Stages:
#   1 = Ingestion (Common Crawl + licensed data)
#   2 = Filtering (keyword scoring)
#   3 = Extraction (structured row creation)
#   4 = Validation (dedup, PII, quality)
#   5 = Packaging (assemble module)
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Use miniforge Python if available, otherwise fall back to python3
if [[ -x "$HOME/miniforge3/bin/python3" ]]; then
    PYTHON="$HOME/miniforge3/bin/python3"
    PIP="$HOME/miniforge3/bin/pip"
elif [[ -x "$SCRIPT_DIR/venv/bin/python3" ]]; then
    PYTHON="$SCRIPT_DIR/venv/bin/python3"
    PIP="$SCRIPT_DIR/venv/bin/pip"
else
    PYTHON="python3"
    PIP="pip3"
fi

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() { echo -e "${BLUE}[$(date '+%H:%M:%S')]${NC} $1"; }
success() { echo -e "${GREEN}[$(date '+%H:%M:%S')] ✓${NC} $1"; }
warn() { echo -e "${YELLOW}[$(date '+%H:%M:%S')] ⚠${NC} $1"; }
fail() { echo -e "${RED}[$(date '+%H:%M:%S')] ✗${NC} $1"; exit 1; }

# Parse arguments
START_STAGE=1
ONLY_STAGE=0
INSTALL=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --stage)  START_STAGE="$2"; shift 2 ;;
        --only)   ONLY_STAGE="$2"; shift 2 ;;
        --install) INSTALL=1; shift ;;
        -h|--help)
            echo "Usage: ./run.sh [--stage N] [--only N] [--install]"
            echo "  --stage N   Start from stage N (1-5)"
            echo "  --only N    Run only stage N"
            echo "  --install   Install Python dependencies first"
            exit 0
            ;;
        *) fail "Unknown argument: $1" ;;
    esac
done

# Install dependencies if requested
if [[ $INSTALL -eq 1 ]]; then
    log "Installing dependencies..."
    $PIP install -r requirements.txt
    success "Dependencies installed"
fi

# Verify Python and key packages
log "Checking Python environment..."
$PYTHON -c "import pandas, pyarrow, yaml; print('Core packages OK')" || \
    fail "Missing core packages. Run: pip install -r requirements.txt"

echo ""
echo "=================================================="
echo "  B2B SaaS Buying Intelligence Pipeline"
echo "=================================================="
echo ""

run_stage() {
    local num=$1
    local name=$2
    local script=$3

    if [[ $ONLY_STAGE -ne 0 && $ONLY_STAGE -ne $num ]]; then
        return
    fi
    if [[ $num -lt $START_STAGE ]]; then
        return
    fi

    log "Stage $num: $name"
    echo "---"

    if $PYTHON "$SCRIPT_DIR/scripts/$script"; then
        success "Stage $num complete"
    else
        warn "Stage $num exited with non-zero status (may indicate no data — check logs)"
    fi

    echo ""
}

run_stage 1 "Ingestion"  "01_ingest_fast.py"
run_stage 2 "Filtering"  "02_filter.py"
run_stage 3 "Extraction" "03_extract.py"
run_stage 4 "Validation" "04_validate.py"
run_stage 5 "Packaging"  "05_package.py"

echo "=================================================="
success "Pipeline complete!"
echo ""

# Show output summary
if [[ -d "$SCRIPT_DIR/output" ]]; then
    log "Output module contents:"
    find "$SCRIPT_DIR/output" -type f -exec ls -lh {} \; 2>/dev/null | \
        awk '{print "  " $NF " (" $5 ")"}'
fi

echo ""
log "Next steps:"
echo "  1. Review output/quality_report.json"
echo "  2. Generate embeddings:  python output/embeddings/generate_embeddings.py"
echo "  3. Run evaluation:       python output/eval/run_eval.py"
echo "  4. See output/README.md for integration guides"
