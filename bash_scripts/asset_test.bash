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
  echo "Invalid mode: $MODE. Use 'default' only."
  exit 1
fi

echo "Starting Evaluation on Simpler Test"
echo "Using port: $PORT"
echo "Using policy: $POLICY"
echo "Mode: $MODE → Output Dir: $OUT_DIR"

python src/pipeline/simpler_test.py \
    --output_dir "$OUT_DIR" \
    --run_all true \
    --port "$PORT"
    # --config <config_file> \ # Path to the config file, default to `configs/simpler.yaml`


