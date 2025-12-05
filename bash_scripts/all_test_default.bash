#!/bin/bash

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh  # Change this to your actual conda path
conda activate genesis

# Set project root as PYTHONPATH
export PYTHONPATH=$(pwd)


# Read inputs or set defaults
PORT=${1:-9000}
POLICY=${2:-robovlm}
RUNALL=${3:-true}  # "true" or "false"

OUT_DIR="./default_test/$POLICY"


echo "Starting Evaluation on Default Test"
echo "Using port: $PORT"
echo "Using policy: $POLICY"
echo "Mode: $MODE → Output Dir: $OUT_DIR"

python src/pipeline/default_test.py \
  --output_dir "$OUT_DIR" \
  --run_all "$RUNALL" \
  --port "$PORT" \
  --vla "$POLICY"
  # --config <config_file>  # Optional: defaults to config/default.yaml


python src/pipeline/adv_background_test.py \
  --output_dir "$OUT_DIR" \
  --run_all "$RUNALL" \
  --port "$PORT" \
  --vla "$POLICY"
  # --config <config_file>  # Optional: defaults to config/default.yaml

# python src/pipeline/simpler_test.py \
#     --output_dir "$OUT_DIR" \
#     --run_all "$RUNALL" \
#     --port "$PORT"
    # --config <config_file> \ # Path to the config file, default to `configs/simpler.yaml`

python src/pipeline/background_test.py \
  --output_dir "$OUT_DIR" \
  --run_all "$RUNALL" \
  --port "$PORT" \
  --vla "$POLICY"
  # --config <config_file>  # Optional: defaults to config/default.yaml
  # --background_folder <path_to_background_folder>  # Optional: defaults to ./examples/background

# python src/pipeline/camera_test.py \
#   --output_dir "$OUT_DIR" \
#   --run_all "$RUNALL" \
#   --port "$PORT" \
#   --vla "$POLICY"
  # --config <config_file>  # Optional: defaults to config/default.yaml

python src/pipeline/pose_test.py \
  --output_dir "$OUT_DIR" \
  --run_all "$RUNALL" \
  --port "$PORT" \
  --vla "$POLICY"
  # --config <config_file>  # Optional: defaults to config/default.yaml
