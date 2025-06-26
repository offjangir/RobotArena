
policy="spatial"
variant="background_test"

python src/pipeline/GVL_multithreaded.py \
    --inference "/data/scene/evaluation/generate_test/$policy/$variant/" \
    --base_dir "/data/scene/scene_generation/data/bridge"\
    --key "" \
    --zero true \
    --frequency 3 \
    --dir "/data/scene/openvla/eval_paper_latest_generate_new_test" \
    --test $variant \
    --policy $policy \
    --debug False \
    --model "gemini-2.5-pro-preview-05-06" \




" 