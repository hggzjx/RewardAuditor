# Copyright 2023 AllenAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import os
import sys
from datasets import Dataset
import numpy as np
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from fastchat.conversation import get_conv_template
from tqdm import tqdm
from transformers import AutoTokenizer, pipeline, AutoModel, AutoModelForSequenceClassification
from scripts.audit_utils import load_eval_dataset, auditing_reward_model, save_audit_results

import json
import gc

from rewardbench import (
    REWARD_MODEL_CONFIG,
    check_tokenizer_chat_template,
    # load_eval_dataset,
    save_to_hub,
    torch_dtype_mapping,
)
from rewardbench.constants import EXAMPLE_COUNTS, SUBSET_MAPPING
from rewardbench.utils import calculate_scores_per_section

# Enable TensorFloat32 (TF32) tensor cores on Ampere GPUs for matrix multiplications (faster than FP32)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# get token from HF_TOKEN env variable, but if it doesn't exist pass none
HF_TOKEN = os.getenv("HF_TOKEN", None)
# this is necessary to automatically log in when running this script in docker/batch beaker jobs
if HF_TOKEN is not None:
    from huggingface_hub._login import _login

    _login(token=HF_TOKEN, add_to_git_credential=False)


def get_args():
    """
    Parse arguments strings model and chat_template
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="path to model")
    parser.add_argument("--tokenizer", type=str, default=None, help="path to non-matching tokenizer to model")
    parser.add_argument("--chat_template", type=str, default="tulu", help="path to chat template")
    # A D D E D: --task argument to specify dataset prefix
    parser.add_argument("--task", type=str, required=True, help="Prefix for dataset files, e.g., 'chat', 'code'")
    parser.add_argument(
        "--trust_remote_code", action="store_true", default=False, help="directly load model instead of pipeline"
    )
    parser.add_argument("--datapath", type=str, default="data/reward-bench", help="path to data directory")
    parser.add_argument("--do_not_save", action="store_true", help="do not save results to hub (for debugging)")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size for inference")
    parser.add_argument("--max_length", type=int, default=2048, help="Max length of RM inputs (passed to pipeline)")
    parser.add_argument(
        "--pref_sets", action="store_true", help="run on common preference sets instead of our custom eval set"
    )
    parser.add_argument(
        "--debug", action="store_true", help="run on common preference sets instead of our custom eval set"
    )
    parser.add_argument(
        "--disable_beaker_save", action="store_true", help="disable saving the main results in a file for AI2 Beaker"
    )
    parser.add_argument(
        "--not_quantized", action="store_true", help="disable quantization for models that are quantized by default"
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32", "float64"],
        help="PyTorch dtype (default: float16)",
    )
    parser.add_argument(
        "--result_output_dir",
        type=str,
        default="reward_auditor_results",
        help="Directory to save the results",
    )
    parser.add_argument(
        "--result_output_filename",
        type=str,
        default="audit_stats.json",
        help="Filename to save the results (default: stats.json)",
    )
    args = parser.parse_args()
    args.torch_dtype = torch_dtype_mapping(args.torch_dtype)
    return args


def main():
    args = get_args()
    ###############
    # Setup logging
    ###############
    accelerator = Accelerator()
    current_device = accelerator.process_index 

    logger = get_logger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = logging.INFO
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(f"Running reward model on {args.model} with chat template {args.chat_template}")
    if args.trust_remote_code:
        logger.info("Loading model with Trust Remote Code")

    # load chat template
    chat_template = args.chat_template
    conv = get_conv_template(chat_template)
    logger.info(f"Using conversation template {chat_template}: {conv}")
    
    offical_model_name = args.model.replace("RewardModels/", "")
    if offical_model_name in REWARD_MODEL_CONFIG:
        # delete the "RewardModel/" prefix
        config = REWARD_MODEL_CONFIG[offical_model_name]
    else:
        config = REWARD_MODEL_CONFIG["default"]
    logger.info(f"Using reward model config: {config}")

    quantized = config["quantized"]  # only Starling isn't quantized for now
    # if llama-3 in name, switch quantized to False (severely degrades performance)
    if (
        ("llama-3" in args.model)
        or ("Llama3" in args.model)
        or ("Llama-3" in args.model)
        or ("LLaMA3" in args.model)
        or ("llama3" in args.model)
        or args.not_quantized
    ):
        quantized = False
        logger.info(f"Disabling quantization for llama-3 or override flag (--not_quantized: {args.not_quantized})")

    custom_dialogue = config["custom_dialogue"]
    model_type = config["model_type"]
    if model_type == "Custom Classifier":
        raise  NotImplementedError("For the Custom Classifier model like NVIDIA SteerLM, plz refer to the NVIDIA original code")


    pipeline_builder = config["pipeline_builder"]
    torch_dtype = config.get("torch_dtype", None)
    # if not datatype in config (default), check args
    if torch_dtype is None:
        if args.torch_dtype == torch.bfloat16:
            quantized = False
            logger.info("Disabling quantization for bfloat16 datatype")
        torch_dtype = args.torch_dtype

    trust_remote_code = args.trust_remote_code

    ############################
    # Load dataset
    ############################
    logger.info("*** Load dataset ***")
    tokenizer_path = args.tokenizer if args.tokenizer else args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=args.trust_remote_code)
    if not custom_dialogue:  # not needed for PairRM / SteamSHP
        tokenizer.truncation_side = "left"  # copied from Starling, but few samples are above context length
    
    
    ############################
    # Load reward model pipeline
    ############################
    BATCH_SIZE = args.batch_size
    logger.info("*** Load reward model ***")
    reward_pipeline_kwargs = {
        "batch_size": BATCH_SIZE,  # eval_args.inference_batch_size,
        "truncation": True,
        "padding": True,
        "max_length": args.max_length,
        "function_to_apply": "none",  # Compute raw logits
        "return_token_type_ids": False,
    }
    if quantized:
        model_kwargs = {
            "load_in_8bit": True,
            "device_map": {"": current_device},
            "torch_dtype": torch_dtype if torch.cuda.is_available() else None,
        }
    else:
        model_kwargs = {
            "device_map": {"": current_device},
            "torch_dtype": torch_dtype,
        }

    if "internlm2-7b-reward" in args.model or "internlm2-20b-reward" in args.model:
        logger.info("Overriding model_builder to use AutoModel for internlm2-7b-reward or internlm2-20b-reward.")
        model_builder = AutoModel.from_pretrained
    else:
        # Use the default builder for all other models
        model_builder = config["model_builder"]
        
    model = model_builder(
        args.model,
        **model_kwargs, 
        trust_remote_code=trust_remote_code)

    reward_pipe = pipeline_builder(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
    )

    ############################
    # Tokenization settings & dataset preparation
    ############################
    # set pad token to eos token if not set
    if reward_pipe.tokenizer.pad_token_id is None:
        reward_pipe.model.config.pad_token_id = reward_pipe.tokenizer.eos_token_id
        reward_pipe.tokenizer.pad_token_id = reward_pipe.tokenizer.eos_token_id
    # For models whose config did not contains `pad_token_id`
    if reward_pipe.model.config.pad_token_id is None:
        reward_pipe.model.config.pad_token_id = reward_pipe.tokenizer.pad_token_id

    # if using fastchat template (no template in tokenizer), make the RM tokenizer output an EOS token
    if not check_tokenizer_chat_template(tokenizer):
        reward_pipe.tokenizer.add_eos_token = True
    
    
        if not check_tokenizer_chat_template(tokenizer):
            reward_pipe.tokenizer.add_eos_token = True
    
    
    # Define file paths based on task prefix
    prompt_filepath = os.path.join(args.datapath, f"{args.task}_filtered_prompt_disturbance.json")
    response_filepath = os.path.join(args.datapath, f"{args.task}_filtered_response_disturbance.json")

    logger.info(f"Task '{args.task}': Preparing to process prompt data from {prompt_filepath}")
    logger.info(f"Task '{args.task}': Preparing to process response data from {response_filepath}")

    # Process prompt disturbance data
    with open(prompt_filepath, 'r', encoding='utf-8') as f:
        original_prompt_data = json.load(f)
    if not original_prompt_data:
        raise ValueError("Prompt dataset is empty!")
    
    raw_dataset_prompt_list = []
    num_prompt_versions = len(original_prompt_data[0]['prompt'])
    for i in range(num_prompt_versions):
        conceptual_dataset = []
        for item in original_prompt_data:
            prompt_source = item['prompt']
            current_prompt = prompt_source[i] if isinstance(prompt_source, list) and i < len(prompt_source) else (prompt_source if not isinstance(prompt_source, list) and i == 0 else None)
            if current_prompt is None: continue
            new_item = {
                "id": item.get("id", "N/A"), "prompt": current_prompt,
                "chosen": item['chosen'][0], "rejected": item['rejected'][0],
                "subset": item.get("subset", "unknown")
            }
            conceptual_dataset.append(new_item)
        raw_dataset_prompt_list.append(conceptual_dataset)

    # Process response disturbance data
    with open(response_filepath, 'r', encoding='utf-8') as f:
        original_response_data = json.load(f)
    if not original_response_data:
        raise ValueError("Response dataset is empty!")
        
    raw_dataset_response_list = []
    num_pairs = len(original_response_data[0]['chosen'])
    assert num_pairs == len(original_response_data[0]['rejected']), "The number of chosen and rejected pairs should be the same."
    for idx in range(num_pairs):
        para_corp_dataset = Dataset.from_dict({
            "id": [unit['id'] for unit in original_response_data],
            "subset": ['subset' for unit in original_response_data],
            "prompt": [unit['prompt'] for unit in original_response_data],
            "chosen": [unit['chosen'][idx] for unit in original_response_data],
            "chosen_model": ["chosen" for _ in original_response_data],
            "rejected": [unit['rejected'][idx] for unit in original_response_data],
            "rejected_model": ["rejected" for _ in original_response_data],
        })
        raw_dataset_response_list.append(para_corp_dataset)

    logger.info("Data restructuring complete. Starting evaluation loops.")

    # --- Loop 1: Inference for prompt disturbances ---
    logger.info("--- Running inference on Prompt Disturbance Dataset ---")
    score_chosen_prompt = []
    score_rejected_prompt = []

    for dataset_idx, raw_dataset in enumerate(raw_dataset_prompt_list):
        
        dataset = None
        dataloader = None
        torch.cuda.synchronize()
        del dataset
        del dataloader
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024 / 1024 / 1024:.2f} GB")
         
        hf_dataset = Dataset.from_list(raw_dataset)
        dataset, subsets = load_eval_dataset(hf_dataset,
                                            core_set=not args.pref_sets, 
                                            conv=conv, 
                                            custom_dialogue_formatting=custom_dialogue, 
                                            tokenizer=tokenizer, 
                                            logger=logger, 
                                            keep_columns=["text_chosen", "text_rejected", "id"])
        
        ids = dataset["id"]
        dataset = dataset.remove_columns("id")

        if args.debug:
            dataset = dataset.select(range(10))
            subsets = subsets[:10]
            ids = ids[:10]

        if pipeline_builder == pipeline:

            logger.info("*** Running forward pass via built in pipeline abstraction ***")
            
            reward_pipe = accelerator.prepare(reward_pipe)

            results_rej = reward_pipe(dataset["text_rejected"], **reward_pipeline_kwargs)
            results_cho = reward_pipe(dataset["text_chosen"], **reward_pipeline_kwargs)
            
            # extract scores from results which is list of dicts, e.g. [{'label': 'LABEL_1', 'score': 0.6826171875},... ]
            unit_score_chosen_list = [result["score"] for result in results_cho]
            unit_score_rejected_list = [result["score"] for result in results_rej]

            # pairwise comparison list comprehension
            results = [1 if chosen > rejected else 0 for chosen, rejected in zip(unit_score_chosen_list, unit_score_rejected_list)]

        else:

            logger.info("*** Running dataloader to collect results ***")

            from torch.utils.data.dataloader import default_collate
            
            # for PairRM, hmm, will move all of this later
            def custom_collate_fn(batch):
                # check if ['text_chosen'] is in first batch element
                # Check if the first element of the batch is a dictionary
                if isinstance(batch[0]["text_chosen"][0], dict): 
                    return batch
                
                else: 
                    return default_collate(batch)
                
            dataloader = torch.utils.data.DataLoader(dataset, 
                                                     batch_size=BATCH_SIZE, 
                                                     collate_fn=custom_collate_fn, 
                                                     shuffle=False, 
                                                     drop_last=False)
            
            dataloader, model = accelerator.prepare(dataloader, reward_pipe.model)
            reward_pipe.model = model

            results = []
            unit_score_chosen_list = []
            unit_score_rejected_list = []
            
            for _, batch in enumerate(tqdm(dataloader, desc="RM batch steps (prompt)")):
                with torch.no_grad():
                    rewards_chosen = reward_pipe(batch["text_chosen"], **reward_pipeline_kwargs)
                    rewards_rejected = reward_pipe(batch["text_rejected"], **reward_pipeline_kwargs)
                    if isinstance(rewards_chosen[0], dict):
                        score_chosen_batch = [result["score"] for result in rewards_chosen]
                        score_rejected_batch = [result["score"] for result in rewards_rejected]
                    else:
                        score_chosen_batch = rewards_chosen.float().cpu().numpy().tolist()
                        score_rejected_batch = rewards_rejected.float().cpu().numpy().tolist()
                    unit_score_chosen_list.extend(score_chosen_batch)
                    unit_score_rejected_list.extend(score_rejected_batch)
        
        score_chosen_prompt.append(unit_score_chosen_list)
        score_rejected_prompt.append(unit_score_rejected_list)

    # --- Loop 2: Inference for response disturbances ---
    logger.info("--- Running inference on Response Disturbance Dataset ---")
    score_chosen_response = []
    score_rejected_response = []

    for dataset_idx, raw_dataset in enumerate(raw_dataset_response_list):

        # Memory management from your implementation
        dataset = None
        dataloader = None
        torch.cuda.synchronize()
        del dataset
        del dataloader
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024 / 1024 / 1024:.2f} GB")
         
        
        # Using your provided structure
        hf_dataset = Dataset.from_list(raw_dataset)
        dataset, subsets = load_eval_dataset(hf_dataset,
                                            core_set=not args.pref_sets, 
                                            conv=conv, 
                                            custom_dialogue_formatting=custom_dialogue, 
                                            tokenizer=tokenizer, 
                                            logger=logger, 
                                            keep_columns=["text_chosen", "text_rejected", "id"])
        
        ids = dataset["id"]
        dataset = dataset.remove_columns("id")

        if args.debug:
            dataset = dataset.select(range(10))
            subsets = subsets[:10]
            ids = ids[:10]

        if pipeline_builder == pipeline:
            logger.info("*** Running forward pass via built in pipeline abstraction ***")
            
            reward_pipe = accelerator.prepare(reward_pipe)

            results_rej = reward_pipe(dataset["text_rejected"], **reward_pipeline_kwargs)
            results_cho = reward_pipe(dataset["text_chosen"], **reward_pipeline_kwargs)
            
            unit_score_chosen_list = [result["score"] for result in results_cho]
            unit_score_rejected_list = [result["score"] for result in results_rej]

        else:
            logger.info("*** Running dataloader to collect results ***")

            from torch.utils.data.dataloader import default_collate
            
            def custom_collate_fn(batch):
                if isinstance(batch[0]["text_chosen"][0], dict): 
                    return batch
                else: 
                    return default_collate(batch)
                
            dataloader = torch.utils.data.DataLoader(dataset, 
                                                     batch_size=BATCH_SIZE, 
                                                     collate_fn=custom_collate_fn, 
                                                     shuffle=False, 
                                                     drop_last=False)
            
            dataloader, model = accelerator.prepare(dataloader, reward_pipe.model)
            reward_pipe.model = model

            unit_score_chosen_list = []
            unit_score_rejected_list = []
            
            for _, batch in enumerate(tqdm(dataloader, desc="RM batch steps (response)")):
                with torch.no_grad():
                    rewards_chosen = reward_pipe(batch["text_chosen"], **reward_pipeline_kwargs)
                    rewards_rejected = reward_pipe(batch["text_rejected"], **reward_pipeline_kwargs)
                    if isinstance(rewards_chosen[0], dict):
                        score_chosen_batch = [result["score"] for result in rewards_chosen]
                        score_rejected_batch = [result["score"] for result in rewards_rejected]
                    else:
                        score_chosen_batch = rewards_chosen.float().cpu().numpy().tolist()
                        score_rejected_batch = rewards_rejected.float().cpu().numpy().tolist()
                    unit_score_chosen_list.extend(score_chosen_batch)
                    unit_score_rejected_list.extend(score_rejected_batch)
        
        score_chosen_response.append(unit_score_chosen_list)
        score_rejected_response.append(unit_score_rejected_list)


    ############################
    # Save results
    ############################
    logger.info("Aggregating results and saving...")
    
    # Augment prompt data with scores
    dataset_json_prompt: list = json.load(open(prompt_filepath))
    if args.debug:
        dataset_json_prompt = dataset_json_prompt[:10]
    for idx, unit in enumerate(dataset_json_prompt):
        if idx < len(score_chosen_prompt[0]):
            unit['score_chosen'] = [score_list[idx] for score_list in score_chosen_prompt]
            unit['score_rejected'] = [score_list[idx] for score_list in score_rejected_prompt]

    # Augment response data with scores
    dataset_json_response: list = json.load(open(response_filepath))
    if args.debug:
        dataset_json_response = dataset_json_response[:10]
    for idx, unit in enumerate(dataset_json_response):
        if idx < len(score_chosen_response[0]):
            unit['score_chosen'] = [score_list[idx] for score_list in score_chosen_response]
            unit['score_rejected'] = [score_list[idx] for score_list in score_rejected_response]

    from datetime import datetime
    
    # Audit combined results
    audit_results = auditing_reward_model(
        dataset_json_prompt,
        dataset_json_response,
        chosen_key='score_chosen',
        rejected_key='score_rejected'
    )
    # breakpoint()
    
    # print("--- Final Audit Results ---")
    # print(json.dumps(audit_results, indent=2))
    logger.info("--- Final Audit Results ---")
    logger.info(json.dumps(audit_results, indent=2))    

    # Setup output paths
    model_name = args.model.strip("/").split("/")[-1]
    dataset_name = args.task  # Use task prefix for the output folder name
    run_timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    base_output_dir = args.result_output_dir
    if args.debug:
        base_output_dir = os.path.join(base_output_dir, "debug")
    final_output_dir = os.path.join(base_output_dir, model_name, dataset_name, run_timestamp)

    # Save raw scores for prompt data
    prompt_raw_scores_filename = f"{args.task}_prompt_raw_scores.json"
    prompt_raw_scores_output_path = os.path.join(final_output_dir, prompt_raw_scores_filename)
    print(f"Saving prompt raw scores to: {prompt_raw_scores_output_path}")
    save_audit_results(dataset_json_prompt, prompt_raw_scores_output_path) 
    
    # Save raw scores for response data
    response_raw_scores_filename = f"{args.task}_response_raw_scores.json"
    response_raw_scores_output_path = os.path.join(final_output_dir, response_raw_scores_filename)
    print(f"Saving response raw scores to: {response_raw_scores_output_path}")
    save_audit_results(dataset_json_response, response_raw_scores_output_path)

    # Save final audit stats
    audit_stats_output_path = os.path.join(final_output_dir, args.result_output_filename)
    print(f"Saving final audit stats to: {audit_stats_output_path}")
    save_audit_results(audit_results, audit_stats_output_path)

if __name__ == "__main__":
    main()