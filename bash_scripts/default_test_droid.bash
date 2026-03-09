#!/bin/bash

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh  # Change this to your actual conda path
conda activate genesis

# Set project root as PYTHONPATH
export PYTHONPATH=$(pwd)


# Read inputs or set defaults
PORT=${1:-9050}
POLICY=${2:-robovlm}
MODE=${3:-default}  # "default" or "generate"

if [ "$MODE" == "default" ]; then
  OUT_DIR="./default_test_droid/$POLICY"
else
  OUT_DIR="./generate_test_droid/$POLICY"
fi

echo "Starting Evaluation on Default Test"
echo "Using port: $PORT"
echo "Using policy: $POLICY"
echo "Mode: $MODE → Output Dir: $OUT_DIR"

python src/pipeline/default_test_droid.py \
  --output_dir "$OUT_DIR" \
  --run_all true \
  --port "$PORT" \
  --vla "$POLICY" \
  --eval_name "droid"
  # --config <config_file>  # Optional: defaults to config/default1.yaml

# Notes:
# - Results will be saved to ./default_test_resubmit/robovlm/default_test or ./generate_test_resubmit/robovlm/default_test depending on mode
# - Set --run_all to false to test only the scene specified in the config