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
from trl.trainer.utils import DPODataCollatorWithPadding
from transformers import AutoTokenizer, pipeline, AutoModel, AutoModelForSequenceClassification
from scripts.audit_utils import load_eval_dataset, auditing_reward_model, save_audit_results,BetaDPOInference,compute_elementwise_accuracy
import json
import gc
import pandas as pd

from rewardbench import (
    DPO_MODEL_CONFIG,
    DPOInference,
    # load_eval_dataset,
    save_to_hub,
    torch_dtype_mapping,
)
from rewardbench.constants import EXAMPLE_COUNTS, SUBSET_MAPPING
from rewardbench.utils import calculate_scores_per_section

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
    parser.add_argument("--ref_model", type=str, default=None, help="path to model")
    parser.add_argument(
        "--ref_free_type", type=str, default="avg", help="type of reference free normalization (norm, avg, or sum)"
    )
    parser.add_argument("--task", type=str, required=True, help="Prefix for dataset files, e.g., 'chat', 'code'")
    parser.add_argument("--datapath", type=str, default="data/reward-bench", help="path to data directory")
    parser.add_argument("--tokenizer", type=str, default=None, help="path to non-matching tokenizer")
    parser.add_argument("--chat_template", type=str, default="tulu", help="path to chat template")
    parser.add_argument("--do_not_save", action="store_true", help="do not save results to hub (for debugging)")
    parser.add_argument("--batch_size", type=int, default=6, help="batch size for inference")
    parser.add_argument(
        "--pref_sets", action="store_true", help="run on common preference sets instead of our custom eval set"
    )
    parser.add_argument(
        "--trust_remote_code", action="store_true", default=False, help="directly load model instead of pipeline"
    )
    parser.add_argument("--debug", action="store_true", default=False, help="use only 10 examples")
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
        default="reward_auditor_results_dpo",
        help="Directory to save the results",
    )
    args = parser.parse_args()
    args.torch_dtype = torch_dtype_mapping(args.torch_dtype)
    return args


def main():
    args = get_args()
    accelerator = Accelerator()
    current_device = accelerator.process_index

    ###############
    # Setup logging
    ###############
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
    # offical_model_name = args.model.replace("RewardModels/", "")
    offical_model_name = args.model.strip("/").split("/")[-1]
    offical_model_name_list = [model_name.strip("/").split("/")[-1] for model_name in DPO_MODEL_CONFIG.keys()]
    

    config = DPO_MODEL_CONFIG["default"]
    for model_name in DPO_MODEL_CONFIG.keys():
        if offical_model_name in model_name:
            config = DPO_MODEL_CONFIG[model_name]
            break

    # offical_model_name = args.model.replace("RewardModels/", "")
    # if args.model in DPO_MODEL_CONFIG:
    #     config = DPO_MODEL_CONFIG[offical_model_name]
    # else:
    #     config = DPO_MODEL_CONFIG["default"]
    logger.info(f"Using dpo model config: {config}")

    model_builder = config["model_builder"]
    tokenizer_builder = config["tokenizer_builder"]

    # check datatype from argparse
    if args.torch_dtype == torch.bfloat16:
        logger.warning("Loading weights directly as bfloat16 for PyTorch dtype")
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float16

    assert args.model != args.ref_model, "policy and reference model should be different"
    # load chat template
    chat_template = args.chat_template
    conv = get_conv_template(chat_template)

    # define reference free
    if args.ref_model is None:
        ref_free = True
        logger.info("Running reference free DPO - no reference model provided")
    else:
        ref_free = False
        logger.info(f"Running DPO with reference model {args.ref_model}")

    ############################
    # Load dataset
    ############################

    logger.info("*** Load dataset ***")
    tokenizer_path = args.tokenizer if args.tokenizer else args.model
    tokenizer = tokenizer_builder(tokenizer_path, trust_remote_code=args.trust_remote_code)
    tokenizer.pad_token = tokenizer.eos_token
    # if no BOS token, set as pad token, e.g. QWEN models
    if tokenizer.bos_token is None:
        tokenizer.bos_token_id = tokenizer.eos_token_id
        tokenizer.pad_token_id = tokenizer.eos_token_id

    prompt_filepath = os.path.join(args.datapath, f"{args.task}_filtered_prompt_disturbance.json")
    response_filepath = os.path.join(args.datapath, f"{args.task}_filtered_response_disturbance.json")

    logger.info(f"Task '{args.task}': Preparing to process prompt data from {prompt_filepath}")
    logger.info(f"Task '{args.task}': Preparing to process response data from {response_filepath}")

    # raw_dataset_list_prompt = convert_robust_dataset_to_preference_dataset_list(prompt_filepath)
    # raw_dataset_list_response = convert_robust_dataset_to_preference_dataset_list(response_filepath)

    if (
        ("llama-3" in args.model)
        or ("Llama3" in args.model)
        or ("Llama-3" in args.model)
        or ("LLaMA3" in args.model)
        or args.not_quantized
    ):
        model_kwargs = {
            "device_map": "auto",
            "torch_dtype": torch_dtype if torch.cuda.is_available() else None,
        }
        model_kwargs_ref = {
            "device_map": "auto",
            "torch_dtype": torch_dtype if torch.cuda.is_available() else None,
        }
    else:
        model_kwargs = {
            "load_in_8bit": True,
            "device_map": "auto",
            "torch_dtype": torch_dtype if torch.cuda.is_available() else None,
        }
        model_kwargs_ref = {
            "load_in_8bit": True,
            "device_map": "auto",
            "torch_dtype": torch_dtype if torch.cuda.is_available() else None,
        }

    model = model_builder(
        args.model,
        trust_remote_code=args.trust_remote_code,
        attn_implementation="sdpa",
        **model_kwargs,
    )

    if ref_free:
        ref_model = None
    else:
        ref_model = model_builder(
            args.ref_model,
            trust_remote_code=args.trust_remote_code,
            **model_kwargs_ref,
        )

    # use internal inference functions in BetaDPO trainer
    dpo = BetaDPOInference(
        model,
        ref_model,
        tokenizer=tokenizer,
        accelerator=accelerator,
        ref_free_norm=args.ref_free_type,
        # norm is norm, avg is average, sum is sum
    )
    # score_original = []
    
    with open(prompt_filepath, 'r', encoding='utf-8') as f:
        original_prompt_data = json.load(f)
    if not original_prompt_data:
        raise ValueError("Prompt dataset is empty!")
    
    raw_dataset_prompt_list = []
    num_pairs_prompt = len(original_prompt_data[0]['prompt'])
    for idx in range(num_pairs_prompt):
        para_corp_dataset = Dataset.from_dict({
            "id": [unit['id'] for unit in original_prompt_data],
            "subset": ['subset' for unit in original_prompt_data],
            "prompt": [unit['prompt'][idx] for unit in original_prompt_data],
            "chosen": [unit['chosen'] for unit in original_prompt_data],
            "chosen_model": ["chosen" for _ in original_prompt_data],
            "rejected": [unit['rejected'] for unit in original_prompt_data],
            "rejected_model": ["rejected" for _ in original_prompt_data],
        })
        raw_dataset_prompt_list.append(para_corp_dataset)

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




    # --- Loop 1: Inference for prompt disturbances ---
    logger.info("Data restructuring complete. Starting prompt evaluation loops.")
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

        print(f"GPU memory allocated (prompt): {torch.cuda.memory_allocated() / 1024 / 1024 / 1024:.2f} GB")
        
        hf_dataset = Dataset.from_list(raw_dataset)
        # breakpoint()
        dataset, subsets = load_eval_dataset(
            hf_dataset,
            core_set=not args.pref_sets,
            conv=conv,
            tokenizer=tokenizer,
            logger=logger,
            keep_columns=["text_chosen", "text_rejected", "id", "prompt"],
        )

        dataset = dataset.remove_columns("id")
        # debug: use only 10 examples
        if args.debug:
            dataset = dataset.select(range(10))
            subsets = subsets[:10]

        BATCH_SIZE = args.batch_size

        # tokenize dataset
        column_names = list(dataset.features)

        tokenized_dataset_prompt = dataset.map(dpo.tokenize_row, remove_columns=column_names)

        dataloader = torch.utils.data.DataLoader(
            tokenized_dataset_prompt,
            batch_size=BATCH_SIZE,
            collate_fn=DPODataCollatorWithPadding(
                pad_token_id=tokenizer.pad_token_id,
                label_pad_token_id=dpo.label_pad_token_id,
                is_encoder_decoder=dpo.is_encoder_decoder,
            ),
            # collate_fn = lambda x: x, # fix weird batching error
            shuffle=False,
            drop_last=False,
        )
        results_prompt = []
        scores_chosen_prompt = []
        scores_rejected_prompt = []

        for step, batch in enumerate(tqdm(dataloader, desc="RM batch steps")):
            logger.info(f"RM inference step {step}/{len(dataloader)}")

            rewards_chosen_prompt, rewards_rejected_prompt = dpo.inference_step(batch, ref_free=ref_free)

            # for each item in batch, record 1 if chosen > rejected
            # extra score from dict within batched results (e.g. logits)
            # [{'label': 'LABEL_1', 'score': 0.6826171875},... ]
            if isinstance(rewards_chosen_prompt[0], dict):
                scores_chosen_batch_prompt = [results_prompt["score"] for result in rewards_chosen_prompt]
                scores_rejected_batch_prompt = [results_prompt["score"] for result in rewards_rejected_prompt]
            # for classes that directly output scores (custom code)
            else:
                scores_chosen_batch_prompt = rewards_chosen_prompt.float().cpu().numpy().tolist()  # convert to float for bfloat16 case
                scores_rejected_batch_prompt = rewards_rejected_prompt.float().cpu().numpy().tolist()

            [
                results_prompt.append(1) if chosen > rejected else results_prompt.append(0)
                for chosen, rejected in zip(scores_chosen_batch_prompt, scores_rejected_batch_prompt)
            ]
            scores_chosen_prompt += scores_chosen_batch_prompt
            scores_rejected_prompt += scores_rejected_batch_prompt


        score_chosen_prompt.append(scores_chosen_prompt)
        score_rejected_prompt.append(scores_rejected_prompt)




    # --- Loop 2: Inference for response disturbances ---
    logger.info("Response evaluation complete. Starting response evaluation loops.")
    score_chosen_response = []
    score_rejected_response = []
    
    for dataset_idx, raw_dataset in enumerate(raw_dataset_response_list):
        
        dataset = None
        dataloader = None
        torch.cuda.synchronize()
        del dataset
        del dataloader
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        print(f"GPU memory allocated (response): {torch.cuda.memory_allocated() / 1024 / 1024 / 1024:.2f} GB")
        
        hf_dataset = Dataset.from_list(raw_dataset)
        # breakpoint()
        dataset, subsets = load_eval_dataset(
            hf_dataset,
            core_set=not args.pref_sets,
            conv=conv,
            tokenizer=tokenizer,
            logger=logger,
            keep_columns=["text_chosen", "text_rejected", "id", "prompt"],
        )

        dataset = dataset.remove_columns("id")
        # debug: use only 10 examples
        if args.debug:
            dataset = dataset.select(range(10))
            subsets = subsets[:10]

        BATCH_SIZE = args.batch_size

        # tokenize dataset
        column_names = list(dataset.features)

        tokenized_dataset_response = dataset.map(dpo.tokenize_row, remove_columns=column_names)

        dataloader = torch.utils.data.DataLoader(
            tokenized_dataset_response,
            batch_size=BATCH_SIZE,
            collate_fn=DPODataCollatorWithPadding(
                pad_token_id=tokenizer.pad_token_id,
                label_pad_token_id=dpo.label_pad_token_id,
                is_encoder_decoder=dpo.is_encoder_decoder,
            ),
            # collate_fn = lambda x: x, # fix weird batching error
            shuffle=False,
            drop_last=False,
        )
        results_response = []
        scores_chosen_response = []
        scores_rejected_response = []

        for step, batch in enumerate(tqdm(dataloader, desc="RM batch steps")):
            logger.info(f"RM inference step {step}/{len(dataloader)}")

            rewards_chosen_response, rewards_rejected_response = dpo.inference_step(batch, ref_free=ref_free)

            # for each item in batch, record 1 if chosen > rejected
            # extra score from dict within batched results (e.g. logits)
            # [{'label': 'LABEL_1', 'score': 0.6826171875},... ]
            if isinstance(rewards_chosen_prompt[0], dict):
                scores_chosen_batch_response = [results_response["score"] for result in rewards_chosen_response]
                scores_rejected_batch_response = [results_response["score"] for result in rewards_rejected_response]
            # for classes that directly output scores (custom code)
            else:
                scores_chosen_batch_response = rewards_chosen_response.float().cpu().numpy().tolist()  # convert to float for bfloat16 case
                scores_rejected_batch_response = rewards_rejected_response.float().cpu().numpy().tolist()

            [
                results_response.append(1) if chosen > rejected else results_response.append(0)
                for chosen, rejected in zip(scores_chosen_batch_response, scores_rejected_batch_response)
            ]
            scores_chosen_response += scores_chosen_batch_response
            scores_rejected_response += scores_rejected_batch_response


        score_chosen_response.append(scores_chosen_response)
        score_rejected_response.append(scores_rejected_response)

    ############################
    # Save results
    ############################
    
    # HACK: load the dataset from the file
    dataset_json_prompt: list = json.load(open(prompt_filepath))
    dataset_json_response: list = json.load(open(response_filepath))

    print(f"Type of score_chosen_prompt: {type(score_chosen_prompt[0])}")
    print(f"Lenght of score_chosen: {len(score_chosen_prompt[0])}")
    # print(score_chosen[0])
    print(f"Type of score_rejected_prompt: {type(score_rejected_prompt[0])}")
    print(f"Lenght of score_rejected_prompt: {len(score_rejected_prompt[0])}")
    # print(score_rejected[0])

    if args.debug:
        dataset_json_prompt = dataset_json_prompt[:10]
        dataset_json_response = dataset_json_response[:10]
    
    for idx, unit in enumerate(dataset_json_prompt):
        unit['score_chosen'] = [
            score_list[idx] for score_list in score_chosen_prompt
        ]
        unit['score_rejected'] = [
            score_list[idx] for score_list in score_rejected_prompt
        ]
    
    for idx, unit in enumerate(dataset_json_response):
        unit['score_chosen'] = [
            score_list[idx] for score_list in score_chosen_response
        ]
        unit['score_rejected'] = [
            score_list[idx] for score_list in score_rejected_response
        ]
    
    
    # Audit combined results
    audit_results = auditing_reward_model(
        dataset_json_prompt,
        dataset_json_response,
        chosen_key='score_chosen',
        rejected_key='score_rejected'
    )

    from datetime import datetime
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
    audit_stats_output_path = os.path.join(final_output_dir, args.result_output_dir)
    print(f"Saving final audit stats to: {audit_stats_output_path}")
    save_audit_results(audit_results, audit_stats_output_path)

    headers = list(audit_results.keys())

    # save effect_size and significance states to csv
    effect_sizes = [audit_results[key]['effect_size'] for key in headers]
    significances = [audit_results[key]['significance'] for key in headers]
    df_effect_size = pd.DataFrame([effect_sizes], columns=headers)
    df_effect_size.insert(0, 'model', offical_model_name) 
    df_significance = pd.DataFrame([significances], columns=headers)
    df_significance.insert(0, 'model', offical_model_name) 
    effect_size_csv_path = os.path.join(final_output_dir, "effect_size.csv")
    significance_csv_path = os.path.join(final_output_dir, "significance.csv")
    df_effect_size.to_csv(effect_size_csv_path, index=False)
    df_significance.to_csv(significance_csv_path, index=False)

    print(f"Effect size data saved to: {effect_size_csv_path}")
    print(f"Significance data saved to: {significance_csv_path}")

    def map_significance_to_symbol(sig_string: str) -> str:
        """
        将统计显著性字符串映射到常用的星号符号。
        - 'p < 0.001' -> '***'
        - 'p < 0.01'  -> '**'
        - 'p < 0.05'  -> '*'
        - 'ns'        -> '' (空字符串)
        """
        if 'p <= 0.0001' in sig_string or '****' in sig_string:
            return '***'
        if 'p <= 0.001' in sig_string or '***' in sig_string:
            return '***'
        elif 'p <= 0.01' in sig_string or '**' in sig_string:
            return '**'
        elif 'p <= 0.05' in sig_string or "*" in sig_string:
            return '*'
        elif 'ns' in sig_string:
            return ''
        return '' 

    headers = list(audit_results.keys())
    combined_data = []
    combined_data_normality = []
    for key in headers:
        item = audit_results[key]
        effect_size = item['effect_size']
        significance = item['significance']
        formatted_effect_size = f"{effect_size:.3f}"
        symbol = map_significance_to_symbol(significance)

        final_value = f"{formatted_effect_size}{symbol}"
        combined_data.append(final_value)

        normality_statistic = item['normality_statistic']
        normality_normality_significance = item['normality_significance']
        formatted_normality_statistic = f"{normality_statistic:.3f}"
        normality_symbol = map_significance_to_symbol(normality_normality_significance)
        normality_test_report = f"{formatted_normality_statistic}{normality_symbol}"
        combined_data_normality.append(normality_test_report)

    df_suitability = pd.DataFrame([combined_data], columns=headers)
    df_suitability.insert(0, 'model', args.model) 

    suitability_risk_path = os.path.join(final_output_dir, "suitability_risk.csv")
    df_suitability.to_csv(suitability_risk_path, index=False)

    df_normality = pd.DataFrame([combined_data_normality], columns=headers)
    df_normality.insert(0, 'model', args.model) 
    normality_test_path = os.path.join(final_output_dir, "normality_test.csv")
    df_normality.to_csv(normality_test_path, index=False)

    for key in headers:
        item = audit_results[key]
        normality_statistic = item['normality_statistic']
        normality_normality_significance = item['normality_significance']
        formatted_normality_statistic = f"{normality_statistic:.3f}"
        normality_symbol = map_significance_to_symbol(normality_normality_significance)
        normality_test_report = f"{formatted_normality_statistic}{normality_symbol}"
        

    # compute decrease accuracy
    if args.compute_decrease_accuracy:
        decrease_accuracy_results = compute_elementwise_accuracy(
            dataset_json_response, 
            dataset_json_prompt)
        print(decrease_accuracy_results)
        decrease_accuracy_csv_path = os.path.join(final_output_dir, "decrease_accuracy.csv")
        df_decrease_accuracy = pd.DataFrame([decrease_accuracy_results])
        df_decrease_accuracy.to_csv(decrease_accuracy_csv_path, index=False)
        print(f"Decrease accuracy data saved to: {decrease_accuracy_csv_path}")

if __name__ == "__main__":
    main()