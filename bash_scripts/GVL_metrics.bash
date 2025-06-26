
policy="spatial"

python src/pipeline/GVL_metrics.py \
    --base_dir "/data/openvla/eval_paper_latest_generate"\
    --zero true \
    --one  false \
    --all_scenes true \
    --test "default_test" \
    --policy $policy \
    --debug true \

python src/pipeline/GVL_metrics.py \
    --zero true \
    --one  false \
    --all_scenes true \
    --test "permute_test" \
    --policy $policy \
    --debug true \

python src/pipeline/GVL_metrics.py \
    --base_dir "/data/scene/openvla/eval_paper_latest_generate"\
    --zero true \
    --one  false \
    --all_scenes true \
    --test "camera_test" \
    --policy $policy \
    --debug true \

python src/pipeline/GVL_metrics.py \
    --zero true \
    --one  false \
    --all_scenes true \
    --test "adv_background_test" \
    --policy $policy \
    --debug true \

python src/pipeline/GVL_metrics.py \
    --base_dir "/data/scene/openvla/eval_paper_latest_generate"\
    --zero true \
    --one  false \
    --all_scenes true \
    --test "background_test" \
    --policy $policy \
    --debug true \


echo $policy

echo "--------------"
echo "default_test"
python src/pipeline/average.py --base_dir "/data/scene/openvla/eval_paper_latest_generate/$policy/default_test/metrics_zero_shot.json"
echo "--------------"
echo "pose_test"
python src/pipeline/average.py --base_dir "/data/scene/openvla/eval_paper_latest_generate/$policy/permute_test/metrics_zero_shot.json"
echo "--------------"
echo "camera_test"
python src/pipeline/average.py --base_dir "/data/scene/openvla/eval_paper_latest_generate/$policy/camera_test/metrics_zero_shot.json"
echo "--------------"
echo "color_test"
python src/pipeline/average.py --base_dir "/data/scene/openvla/eval_paper_latest_generate/$policy/adv_background_test/metrics_zero_shot.json"
echo "--------------"
echo "background_test"
python src/pipeline/average.py --base_dir "/data/scene/openvla/eval_paper_latest_generate/$policy/background_test/metrics_zero_shot.json"
echo "--------------"

