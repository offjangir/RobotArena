policy="cogact"
variant="default_test"

source ~/miniconda3/etc/profile.d/conda.sh 
conda activate gemini
python src/pipeline/GVL_multithreaded.py \
    --inference "./generate_test/$policy/$variant/" \
    --base_dir "./examples/data/bridge"\
    --key "<you gemini api key>" \
    --zero true \
    --frequency 3 \
    --dir "./eval_results" \
    --test $variant \
    --policy $policy \
    --debug False \
    --model "gemini-2.5-pro" \

