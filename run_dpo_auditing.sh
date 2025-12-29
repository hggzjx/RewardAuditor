#!/bin/bash

# # Prompt for model path
# read -p "Enter the model path (default: RewardModels/allenai/tulu-v2.5-13b-hh-rlhf-60k-rm): " MODEL_PATH
# MODEL_PATH=${MODEL_PATH:-RewardModels/allenai/tulu-v2.5-13b-hh-rlhf-60k-rm}

# # Prompt for CUDA device
# read -p "Enter the CUDA device (default: 0): " CUDA_DEVICE
# CUDA_DEVICE=${CUDA_DEVICE:-0}

MODEL_PATH_LIST=(
    "reward_models/ArmoRM-Llama3-8B-v0.1"
    "reward_models/Eurus-RM-7b"
    "reward_models/FsfairX-LLaMA3-RM-v0.1"
    "reward_models/GRM-llama3-8B-distill"
    "reward_models/GRM-Llama3-8B-rewardmodel-ft"
    # "reward_models/GRM-llama3-8B-sftreg"
    "reward_models/internlm2-7b-reward"
    "reward_models/internlm2-20b-reward"
    "reward_models/Llama-3-OffsetBias-RM-8B"
    # "reward_models/Nemotron-4-340B-Reward"
    "reward_models/Skywork-Reward-Gemma-2-27B"
    "reward_models/Skywork-Reward-Gemma-2-27B-v0.2"
    "reward_models/Skywork-Reward-Llama-3.1-8B"
    "reward_models/URM-LLaMa-3-8B"
    "reward_models/URM-LLaMa-3.1-8B"
)

TASK_LIST=(
    "chat"
)

DATA_DIR="datasets"

CUDA_DEVICE="0,1,2"
export PYTHONPATH=$PYTHONPATH:pwd
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE
chat_template=tulu

for MODEL_PATH in ${MODEL_PATH_LIST[@]}; do
    for TASK_PREFIX in ${TASK_LIST[@]}; do
        python scripts/run_dpo_auditing.py \
            --model "$MODEL_PATH" \
            --datapath "$DATA_DIR" \
            --task "$TASK_PREFIX" \
            --batch_size 16 \
            --trust_remote_code \
            --chat_template "$chat_template"\
            
    done
done