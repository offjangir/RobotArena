#!/bin/bash
export http_proxy=http://bj-rd-proxy.byted.org:3128
export https_proxy=http://bj-rd-proxy.byted.org:3128

export CALVIN_ROOT=/mnt/bn/robotics/resources/calvin
export EVALUTION_ROOT=$(pwd)

# # # Install dependency for calvin
# sudo apt-get -y install libegl1-mesa libegl1
# sudo apt-get -y install libgl1

# # # Install dependency for dt
# sudo apt-get -y install libosmesa6-dev
# sudo apt-get -y install patchelf

# # Copy clip weights
# mkdir -p ~/.cache/clip
# cp -r /mnt/bn/robotics-data-lxh-lq/RoboVLM/.cache/clip ~/.cache

# # Run
source /mnt/bn/robotics/resources/anaconda3_arnold/bin/activate robollava
# pip3 install moviepy diffusers==0.29.1
# pip3 install lightning==2.2.5
# pip install open_flamingo==2.0.1
# # pip3 install transformers==4.36.2 -i "https://bytedpypi.byted.org/simple"
# # pip install transformers==4.37.2
# # pip install transformers==4.33.2
# pip install transformers==4.44.0
# pip3 install torch==2.3.1
# pip3 install flash_attn pytorchvideo

export MESA_GL_VERSION_OVERRIDE=4.1
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_BLOCKING_WAIT=1
# export CUDA_VISIBLE_DEVICES=1
cd $EVALUTION_ROOT
ckpt_dir=$1
config_path=$2
sudo chmod 777 -R $ckpt_dir
GPUS_PER_NODE=$ARNOLD_WORKER_GPU

# pip install transformers==4.44.0
# cp /mnt/bn/robotics-data-lxh-lq/VLMs/moondream2/*.py ~/.cache/huggingface/modules/transformers_modules/moondream2

GPUS_PER_NODE=1
export CUDA_VISIBLE_DEVICES=0

torchrun --nnodes=1 --nproc_per_node=$GPUS_PER_NODE --master_port=6067 eval/calvin/evaluate_ddp-v2.py \
--config_path $config_path \
--ckpt_path $ckpt_dir \
--ckpt_idx 0 --raw_calvin
