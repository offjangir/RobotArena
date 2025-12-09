#!/bin/bash

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh  # Change this to your actual conda path
conda activate genesis

# Set project root as PYTHONPATH
export PYTHONPATH=$(pwd)

# Read inputs or set defaults
PORT=${1:-9000}
POLICY=${2:-robovlm}
MODE=${3:-default}  # "default" or "generate"

if [ "$MODE" == "default" ]; then
  OUT_DIR="./default_test/$POLICY"
else
  OUT_DIR="./generate_test/$POLICY"
fi

echo "Starting Evaluation on Background Color Change Test"
echo "Using port: $PORT"
echo "Using policy: $POLICY"
echo "Mode: $MODE → Output Dir: $OUT_DIR"


python src/pipeline/adv_background_test.py \
  --output_dir "$OUT_DIR" \
  --run_all true \
  --port "$PORT" \
  --vla "$POLICY"
  # --config <config_file>  # Optional: defaults to config/default.yaml

# Notes:
# - Set third arg to "generate" if testing only generated scenes
# - Results saved to $OUT_DIR/adv_background_test
# - Change policy by providing it as second argument
# - Set --run_all to false to test only the scene in config
