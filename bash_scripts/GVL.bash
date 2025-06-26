
policy="spatial"
variant="background_test"

source ~/miniconda3/etc/profile.d/conda.sh 
conda activate gemini
python src/pipeline/GVL_multithreaded.py \
    --inference "./generate_test/$policy/$variant/" \
    --base_dir "./examples/data/bridge"\
    --key "" \
    --zero true \
    --frequency 3 \
    --dir "./eval_paper_latest_generate_new_test" \
    --test $variant \
    --policy $policy \
    --debug False \
    --model "gemini-2.5-pro-preview-05-06" \




" 