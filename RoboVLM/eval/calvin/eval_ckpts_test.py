import os

ckpt_paths = [
    (
        "/mnt/bn/robotics-data-lxh-lq-v2/checkpoints/video_pretrain_manipulation/kosmos/calvin_finetune/2024-12-05/05-18/epoch=1-step=40000.pt",
        "/mnt/bn/robotics-data-lxh-lq-v2/logs/video_pretrain_manipulation/kosmos/calvin_finetune/2024-12-05/05-18/2024-12-05_05:19:31.858211-project.json",
    )
]

for i, (ckpt, config) in enumerate(ckpt_paths):
    print("evaluating checkpoint {}".format(ckpt))
    os.system(
        "bash scripts/run_eval_raw_ddp_torchrun_test.sh {} {}".format(ckpt, config)
    )
