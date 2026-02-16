# #!/bin/bash

# source ~/miniconda3/etc/profile.d/conda.sh  
# conda activate gemini

# task_set="eval_resubmit_30_hard_latest_notemp_1.5_droid"
# for policy in octo robovlm cogact spatial; do
#     echo "=== Running for policy: $policy ==="

#     python src/pipeline/GVL_metrics.py \
#         --base_dir "./eval_merged" \
#         --zero true \
#         --one false \
#         --all_scenes true \
#         --test "default_test" \
#         --policy $policy \
#         --debug false

#     echo "--------------"
#     echo "default_test for $policy"
#     python src/pipeline/average.py --base_dir "./$task_set/$policy/default_test/metrics_zero_shot.json"
#     echo "--------------"
# done



#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh  
conda activate gemini

task_set="latest_eval_results_pangchi_2"

for policy in cogact ; do
    echo "=== Running for policy: $policy ==="

    for test in  default_test permute_test adv_background_test background_test; do
        echo "--- Running test: $test ---"

        python src/pipeline/GVL_metrics.py \
            --base_dir "./$task_set" \
            --zero true \
            --one false \
            --all_scenes true \
            --test "$test" \
            --policy $policy \
            --debug false

        echo "--------------"
        echo "$test for $policy"
        python src/pipeline/average.py --base_dir "./$task_set/$policy/$test/metrics_zero_shot.json"
        echo "--------------"
    done
done