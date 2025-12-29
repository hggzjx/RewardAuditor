import logging
from fastchat.conversation import Conversation
from datasets import Dataset, DatasetDict, Value, concatenate_datasets, load_dataset
from transformers import PreTrainedTokenizer
from rewardbench.utils import check_tokenizer_chat_template, prepare_dialogue, prepare_dialogue_from_tokenizer
import numpy as np
from scipy import stats
from typing import List, Dict, Any
import json
from datasets import Dataset, load_from_disk
import math
import os
import copy

EXTRA_PREF_SETS = "allenai/pref-test-sets"


def _process_single_dataset(
    dataset: List[Dict[str, Any]],
    disturbance_keys: List[str],
    key_prefix: str,
    B: int,
    alpha: float,
    chosen_key: str,
    rejected_key: str,
    require_stats: bool,
    stats_json_output_dir: str,
    json_name: str
) -> Dict[str, Any]:
    """
    Core processing function to audit a single dataset.
    This function contains the main logic from the original code.
    """
    scores_chosen = [item[chosen_key] for item in dataset]
    scores_rejected = [item[rejected_key] for item in dataset]

    if not scores_chosen or not scores_rejected:
        return {}

    results = {}
    transformed_results = {}
    audit_results = {}
    chosen_cols = list(zip(*scores_chosen))
    rejected_cols = list(zip(*scores_rejected))

    if len(chosen_cols) != len(rejected_cols):
        raise ValueError("Chosen and Rejected scores must have matching lengths!")

    # 1. Calculate score differences for original and disturbed models
    origin_loss = [(c[0] if isinstance(c, (list, tuple)) else c) - (r[0] if isinstance(r, (list, tuple)) else r) for c, r in zip(chosen_cols[0], rejected_cols[0])]
    results['origin_loss'] = origin_loss

    for i in range(1, len(chosen_cols)):
        # disturbance_key = f'Disturbance_{i}'
        disturbance_key = disturbance_keys[i - 1]
        disturbance_loss = [(c[0] if isinstance(c, (list, tuple)) else c) - (r[0] if isinstance(r, (list, tuple)) else r) for c, r in zip(chosen_cols[i], rejected_cols[i])]
        results[disturbance_key] = disturbance_loss
    
    # 2. Transform score differences into log loss
    for key, value_list in results.items():
        transformed_results[key] = [-math.log(1 / (1 + math.exp(-v))) for v in value_list]

    auditor = RewardAuditor(B=B, alpha=alpha)

    # 3. Run the statistical audit
    if 'origin_loss' in transformed_results:
        origin_values = np.array(transformed_results['origin_loss'])
        for key, disturbance_values in transformed_results.items():
            if key == 'origin_loss':
                continue
            # new_key = f'{key_prefix}_diff_origin_vs_{key}'
            new_key = f'({key_prefix}) {key}'

            disturbance_values_np = np.array(disturbance_values)
            metric_dict = auditor.run_audit(origin_values, disturbance_values_np)
            audit_results[new_key] = metric_dict

    # 4. (Optional) Save descriptive statistics
    if require_stats:
        distribution_summary = {}
        for key, value_list in transformed_results.items():
            data_array = np.array(value_list)
            stats_dict = {
                'mean': np.mean(data_array), 'median': np.median(data_array),
                'std_dev': np.std(data_array), 'variance': np.var(data_array),
                'min': np.min(data_array), 'max': np.max(data_array),
                'skewness': stats.skew(data_array), 'kurtosis': stats.kurtosis(data_array)
            }
            distribution_summary[key] = stats_dict
        
        if stats_json_output_dir and json_name:
            try:
                os.makedirs(stats_json_output_dir, exist_ok=True)
                file_path = os.path.join(stats_json_output_dir, json_name)
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(distribution_summary, f, indent=4, ensure_ascii=False)
                print(f"Statistics saved to: {file_path}")
            except Exception as e:
                print(f"Error saving statistics file: {e}")
    return audit_results


def auditing_reward_model(
    dataset_prompt: List[Dict[str, Any]],
    dataset_response: List[Dict[str, Any]],
    alpha: float = 0.05,
    B: int = 10000,
    chosen_key: str = 'chosen_scores',
    rejected_key: str = 'rejected_scores',
    require_stats: bool = False,
    stats_json_output_dir: str = None,
    json_name: str = "audit_stats.json"
):
    """
    Audits the reward model on two datasets (prompt and response).
    It processes each dataset separately, then merges the results for a unified
    Benjamini-Hochberg correction.
    """
    # --- 1. Process the Prompt Dataset ---
    print("--- Processing Prompt Dataset ---")
    prompt_json_name = f"prompt_{json_name}" if json_name else None
    audit_results_prompt = _process_single_dataset(
        dataset=dataset_prompt, disturbance_keys=['EF','PH','IU','IW','CN'], key_prefix='prompt', B=B, alpha=alpha,
        chosen_key=chosen_key, rejected_key=rejected_key,
        require_stats=require_stats, stats_json_output_dir=stats_json_output_dir,
        json_name=prompt_json_name
    )

    # --- 2. Process the Response Dataset ---
    print("\n--- Processing Response Dataset ---")
    response_json_name = f"response_{json_name}" if json_name else None
    audit_results_response = _process_single_dataset(
        dataset=dataset_response, disturbance_keys=['ST','LE','SP','LC','SLC'], key_prefix='response', B=B, alpha=alpha,
        chosen_key=chosen_key, rejected_key=rejected_key,
        require_stats=require_stats, stats_json_output_dir=stats_json_output_dir,
        json_name=response_json_name
    )

    # --- 3. Combine the audit results from both datasets ---
    # combined_audit_results = {
    #     "prompt": audit_results_prompt,
    #     "response": audit_results_response
    # }

    combined_audit_results = {}
    combined_audit_results.update(audit_results_prompt)
    combined_audit_results.update(audit_results_response)
    
    # --- 4. Apply Benjamini-Hochberg correction to all test results ---
    # final_results = benjamini_hochberg_procedure(combined_audit_results, alpha=alpha)
    final_results = group_aware_benjamini_hochberg_procedure(combined_audit_results, alpha=alpha)

    return final_results


class RewardAuditor:
    """
    A suitability auditing tool for evaluating reward models.

    This class implements the Reward Auditor algorithm based on paired permutation tests,
    used to test whether performance changes under data perturbations are statistically significant.

    Attributes:
        B (int): Number of iterations for permutation testing to simulate the null distribution.
        alpha (float): Significance level for the statistical test.
    """

    def __init__(self, B: int = 10000, alpha: float = 0.05):
        """
        Initialize RewardAuditor.

        Args:
            B (int, optional): Number of permutations. Defaults to 10000.
            alpha (float, optional): Significance level. Defaults to 0.05.
        """
        if not (B > 0 and isinstance(B, int)):
            raise ValueError("Number of permutations B must be a positive integer.")
        if not (0 < alpha < 1):
            raise ValueError("Significance level alpha must be between (0, 1).")

        self.B = B
        self.alpha = alpha
        print(f"RewardAuditor initialized: Permutation count B = {self.B}, Significance level alpha = {self.alpha}")

    def _calculate_t_statistic(self, data: np.ndarray) -> float:
        """
        Calculate the one-sample t-test statistic.

        Args:
            data (np.ndarray): Input data sample.

        Returns:
            float: Computed t-statistic. Returns 0 if standard deviation is zero.
        """
        n = len(data)
        if n == 0:
            return 0.0
        
        mean_val = np.mean(data)
        std_val = np.std(data, ddof=1)  # Sample standard deviation (ddof=1)

        if std_val == 0:
            # If standard deviation is zero, all data points are identical
            # t-statistic is undefined when mean=0, but can be safely returned as 0
            return 0.0

        return mean_val / (std_val / np.sqrt(n))

    def _calculate_effect_size(self, delta_M: np.ndarray) -> float:
        """
        Calculate effect size (Cohen's d for one-sample t-test).

        Args:
            delta_M (np.ndarray): Vector of loss differences (M - M').

        Returns:
            float: Computed effect size.
        """
        std_val = np.std(delta_M, ddof=0)  # Cohen's d uses population std (ddof=0)
        if std_val == 0:
            return 0.0
        return np.mean(delta_M) / std_val

    def run_audit(self, M: np.ndarray, M_prime: np.ndarray) -> Dict[str, Any]:
        """
        Run robustness audit on given original loss values and perturbed loss values.

        This is the main entry point for users.

        Args:
            M (np.ndarray): Original loss values vector on dataset D.
            M_prime (np.ndarray): Perturbed loss values vector on dataset D'.

        Returns:
            Dict[str, Any]: Dictionary containing audit results with keys:
                - 'robustness_risk' (S_R): Robustness risk metric
                - 'effect_size' (e_hat): Effect size
                - 'p_value' (p_hat): Test p-value
                - 't_observed' (t_obs): Observed t-statistic
                - 'is_significant': Whether result is statistically significant
        """
        if not isinstance(M, np.ndarray) or not isinstance(M_prime, np.ndarray):
            raise TypeError("Inputs M and M_prime must be NumPy arrays.")
        if M.shape != M_prime.shape:
            raise ValueError("Input vectors M and M_prime must have the same shape.")
        if M.ndim != 1:
            raise ValueError("Input vectors M and M_prime must be 1-dimensional.")
        
        print("\nStarting paired permutation test...")
        
        # Step 1: Calculate loss differences
        delta_M = M - M_prime
        N = len(delta_M)
        print(f"Computed loss differences delta_M, sample size N = {N}.")

        # Step 2: Calculate observed t-statistic
        t_obs = self._calculate_t_statistic(delta_M)
        print(f"Observed t-statistic t_obs = {t_obs:.4f}")

        # Step 3: Run permutation test
        c = 0  # Counter
        
        # Pre-generate random signs for efficiency
        random_signs = np.random.choice([-1, 1], size=(self.B, N))
        
        # Vectorized computation for permuted t-statistics
        permuted_delta_M = delta_M * random_signs
        permuted_means = np.mean(permuted_delta_M, axis=1)
        permuted_stds = np.std(permuted_delta_M, axis=1, ddof=1)
        
        # Prevent division by zero
        # Find indices where standard deviation is non-zero
        non_zero_std_indices = np.where(permuted_stds != 0)
        t_perms = np.zeros(self.B)
        
        # Calculate t-statistic only for non-zero standard deviations
        t_perms[non_zero_std_indices] = permuted_means[non_zero_std_indices] / (permuted_stds[non_zero_std_indices] / np.sqrt(N))

        # Compare and count
        c = np.sum(t_perms <= t_obs)

        print(f"Permutation test completed. In {self.B} permutations, {c} had t_perm <= t_obs.")

        # Step 4: Calculate p-value
        p_hat = (c + 1) / (self.B + 1)
        print(f"Computed p-value = {p_hat:.4f}")

        # Step 5: Calculate effect size
        e_hat = self._calculate_effect_size(delta_M)
        print(f"Computed effect size e_hat = {e_hat:.4f}")

        # Step 6: Calculate robustness risk
        is_significant = p_hat < self.alpha
        S_R = -e_hat * float(is_significant)
        print(f"Significant result? (p < {self.alpha}): {'Yes' if is_significant else 'No'}")
        print(f"Computed robustness risk S_R = {S_R:.4f}")

        return {
            'robustness_risk': S_R,
            'effect_size': e_hat,
            'p_value': p_hat,
            't_observed': t_obs,
            'is_significant': is_significant
        }



# def group_aware_benjamini_hochberg_procedure(audit_results, alpha=0.05):
#     """
#     Performs group-aware Benjamini-Hochberg (BH) correction on p-values.

#     This function first separates tests into groups based on key prefixes ("prompt" and "response"),
#     then applies the BH correction to each group independently, and finally merges the results.

#     Args:
#         audit_results (dict): The dictionary containing all test results.
#         alpha (float, optional): The desired False Discovery Rate threshold. Defaults to 0.05.

#     Returns:
#         dict: A new dictionary containing the results for all tests, with significance
#               calculated within their respective groups.
#     """
    
#     prompt_results = {k: v for k, v in audit_results.items() if k.startswith('prompt')}
#     response_results = {k: v for k, v in audit_results.items() if k.startswith('response')}

#     corrected_prompts = _perform_bh_correction(prompt_results, alpha)
#     corrected_responses = _perform_bh_correction(response_results, alpha)

#     final_results = {}
#     final_results.update(corrected_prompts)
#     final_results.update(corrected_responses)

#     return final_results


# def _perform_bh_correction(results_subset, alpha=0.05):
#     """
#     Helper function to perform the BH correction and update robustness_risk.
#     """
#     if not results_subset:
#         return {}

#     new_results = copy.deepcopy(results_subset)
    
#     p_values_with_keys = [(key, value['p_value']) for key, value in new_results.items() if 'p_value' in value]

#     if not p_values_with_keys:
#         return new_results
    
#     sorted_p_values = sorted(p_values_with_keys, key=lambda item: item[1])

#     m = len(sorted_p_values)
    
#     largest_significant_p = -1.0
#     for i, (key, p_value) in enumerate(sorted_p_values):
#         rank = i + 1
#         bh_critical_value = (rank / m) * alpha
#         if p_value <= bh_critical_value:
#             largest_significant_p = p_value

#     for key, value in new_results.items():
#         p_value = value.get('p_value')
#         if p_value is None:
#             continue

#         significance = "ns (not significant)"
        
#         if p_value <= largest_significant_p:
#             if p_value <= 0.0001:
#                 significance = "**** (p <= 0.0001)"
#             elif p_value <= 0.001:
#                 significance = "*** (p <= 0.001)"
#             elif p_value <= 0.01:
#                 significance = "** (p <= 0.01)"
#             elif p_value <= 0.05:
#                 significance = "* (p <= 0.05)"
        
#         value['significance'] = significance
#         if 'is_significant' in value:
#             del value['is_significant']

#         if significance == "ns (not significant)":
#             value['robustness_risk'] = 0.0
#         else:
#             value['robustness_risk'] = value.get('effect_size', 0.0)

#     return new_results


def _perform_bh_correction(p_values, q_values, alpha):
    """
    Helper function: Core logic for implementing the correct group-aware (weighted) BH procedure.
    
    Args:
        p_values (np.array): Array of p-values for all tests.
        q_values (np.array): Array of weights (prior probabilities that the null is true) corresponding to each p-value.
        alpha (float): Desired FDR control level.

    Returns:
        set: Set containing indices of all rejected hypotheses.
    """
    # L is the total number of tests
    L = len(p_values)
    if L == 0:
        return set()

    # --- Find optimal k_hat ---
    k_hat = 0
    # Iterate from L down to 1 to find the largest k satisfying the condition
    for k in range(L, 0, -1):
        # Calculate personalized thresholds for each hypothesis
        # Formula: alpha * k / (L * q_i)
        # Use NumPy vectorization for efficient computation
        thresholds = (alpha * k) / (L * q_values)
        
        # r(k): Count how many p-values are <= their personalized thresholds
        r_k = np.sum(p_values <= thresholds)
        
        # Check critical condition: r(k) >= k
        if r_k >= k:
            k_hat = k
            break  # Exit loop when largest k is found

    # --- Determine final rejections using k_hat ---
    if k_hat > 0:
        # Calculate final rejection thresholds using k_hat
        final_thresholds = (alpha * k_hat) / (L * q_values)
        # Identify indices of rejected hypotheses
        rejected_indices_mask = p_values <= final_thresholds
        rejected_indices = np.where(rejected_indices_mask)[0]  # First element of tuple
        return set(rejected_indices)
    else:
        # k_hat = 0 indicates no rejections
        return set()

def group_aware_benjamini_hochberg_procedure(
    audit_results, 
    alpha=0.05, 
    q_prompt=1, 
    q_response=1
):
    """
    Applies the correct group-aware/weighted Benjamini-Hochberg (BH) correction to p-values.

    Treats all tests as one family while assigning different weights based on group 
    ('prompt' or 'response'), then applies unified weighted BH correction.

    Args:
        audit_results (dict): Dictionary containing all test results.
        alpha (float, optional): Desired FDR threshold. Defaults to 0.05.
        q_prompt (float, optional): Weight for 'prompt' group (prior prob. null is true).
                                    Higher values = more conservative. Default 0.8.
        q_response (float, optional): Weight for 'response' group. Default 0.5.

    Returns:
        dict: New dictionary with updated significance markers based on weighted BH procedure.
    """
    if not audit_results:
        return {}
    
    # --- Step 1: Prepare data without splitting pools ---
    # Store keys, p-values, and weights in parallel lists to preserve order
    keys = []
    p_values_list = []
    q_values_list = []

    for key, value in audit_results.items():
        if 'p_value' in value:
            keys.append(key)
            p_values_list.append(value['p_value'])
            # Assign appropriate weight based on key prefix
            if key.startswith('prompt'):
                q_values_list.append(q_prompt)
            elif key.startswith('response'):
                q_values_list.append(q_response)
            else:
                # Default/neutral weight (1.0) for unknown groups
                q_values_list.append(1.0)
    
    if not keys:
        return copy.deepcopy(audit_results)

    # Convert to NumPy arrays for vectorized operations
    p_values_np = np.array(p_values_list)
    q_values_np = np.array(q_values_list)
    
    # --- Step 2: Apply core weighted BH algorithm ---
    significant_indices = _perform_bh_correction(p_values_np, q_values_np, alpha)

    # --- Step 3: Format output results ---
    final_results = copy.deepcopy(audit_results)
    
    for i, key in enumerate(keys):
        # Check if current index is in significant set
        is_significant = (i in significant_indices)
        p_value = final_results[key]['p_value']
        
        significance = "ns (not significant)"
        if is_significant:
            # Formatting remains consistent
            if p_value <= 0.0001:
                significance = "**** (p <= 0.0001)"
            elif p_value <= 0.001:
                significance = "*** (p <= 0.001)"
            elif p_value <= 0.01:
                significance = "** (p <= 0.01)"
            elif p_value <= 0.05:
                significance = "* (p <= 0.05)"
        
        final_results[key]['significance'] = significance
        # Remove old boolean marker if exists
        if 'is_significant' in final_results[key]:
            del final_results[key]['is_significant']

        # Update robustness risk
        if not is_significant:
            final_results[key]['robustness_risk'] = 0.0
        else:
            final_results[key]['robustness_risk'] = final_results[key].get('effect_size', 0.0)

    return final_results
    
def benjamini_hochberg_procedure(audit_results, alpha=0.05):
    """
    Performs the Benjamini-Hochberg (BH) correction on p-values from multiple hypothesis tests.

    This function determines which test results are statistically significant by controlling the
    False Discovery Rate (FDR).

    Args:
        audit_results (dict): A dictionary where keys are test names and values are another
                              dictionary containing a 'p_value' field.
        alpha (float, optional): The desired False Discovery Rate threshold. Defaults to 0.05.

    Returns:
        dict: A new dictionary with the same structure as the input, but the 'is_significant'
              field is replaced by a 'significance' field. The value of 'significance'
              is determined by the corrected significance level:
              - 'ns': not significant
              - '*': p <= 0.05 (significant at alpha=0.05)
              - '**': p <= 0.01 (significant at alpha=0.01)
              - '***': p <= 0.001 (significant at alpha=0.001)
              - '****': p <= 0.0001 (significant at alpha=0.0001)
    """
    # Create a deep copy to avoid modifying the original input dictionary
    new_results = copy.deepcopy(audit_results)
    # Extract all keys and p-values
    p_values_with_keys = []
    for key, value in new_results.items():
        # Ensure 'p_value' exists
        if 'p_value' in value:
            p_values_with_keys.append((key, value['p_value']))

    # Return directly if there are no p-values
    if not p_values_with_keys:
        return new_results
    
    # Sort by p-value in ascending order
    sorted_p_values = sorted(p_values_with_keys, key=lambda item: item[1])

    m = len(sorted_p_values)
    
    # BH procedure: find the largest p-value that satisfies p <= (i/m)*alpha.
    # All original p-values less than or equal to this p-value are considered significant.
    largest_significant_p = -1.0
    for i, (key, p_value) in enumerate(sorted_p_values):
        rank = i + 1
        bh_critical_value = (rank / m) * alpha
        if p_value <= bh_critical_value:
            largest_significant_p = p_value

    # Assign significance markers to each test based on the found threshold
    for key, value in new_results.items():
        p_value = value.get('p_value')
        if p_value is None:
            continue

        significance = "ns (not significant)"  # Default to not significant
        
        # A test is considered significant only if its p-value is less than or equal to the threshold found by the BH correction
        if p_value <= largest_significant_p:
            # If significant, assign asterisks based on its original p-value
            if p_value <= 0.0001:
                significance = "**** (p <= 0.0001)"
            elif p_value <= 0.001:
                significance = "*** (p <= 0.001)"
            elif p_value <= 0.01:
                significance = "** (p <= 0.01)"
            elif p_value <= 0.05:
                significance = "* (p <= 0.05)"
        
        value['significance'] = significance
        # Remove the old 'is_significant' field if it exists
        if 'is_significant' in value:
            del value['is_significant']

        if significance == "ns (not significant)":
            value['robustness_risk'] = 0.0
        else:
            value['robustness_risk'] = value.get('effect_size', 0.0)

    return new_results


def auditing_reward_model_via_single_dataset(
    dataset: List[Dict[str, Any]],
    alpha: float = 0.05,
    B: int = 10000,
    chosen_key: str = 'chosen_scores',
    rejected_key: str = 'rejected_scores',
    require_stats: bool = False,
    stats_json_output_dir: str = None,
    json_name: str = None
):
    
    scores_chosen = [item[chosen_key] for item in dataset]
    scores_rejected = [item[rejected_key] for item in dataset]


    if not scores_chosen or not scores_rejected:
        return {}, {}

    results = {}
    transformed_results = {}
    audit_results = {}
    chosen_cols = list(zip(*scores_chosen))
    rejected_cols = list(zip(*scores_rejected))

    if len(chosen_cols) != len(rejected_cols):
        raise ValueError("Chosen Rejected Cannot match in length!")

    # Function 1: Compute origin_loss and disturbances
    # origin_loss = [c - r for c, r in zip(chosen_cols[0], rejected_cols[0])]
    origin_loss = [
    (c[0] if isinstance(c, (list, tuple)) else c) - (r[0] if isinstance(r, (list, tuple)) else r)
    for c, r in zip(chosen_cols[0], rejected_cols[0])
    ]

    results['origin_loss'] = origin_loss
    for i in range(1, len(chosen_cols)):
        disturbance_key = f'Disturbance_{i}'
        # disturbance_loss = [c - r for c, r in zip(chosen_cols[i], rejected_cols[i])]
        disturbance_loss = [
        (c[0] if isinstance(c, (list, tuple)) else c) - (r[0] if isinstance(r, (list, tuple)) else r)
        for c, r in zip(chosen_cols[i], rejected_cols[i])
        ]
        results[disturbance_key] = disturbance_loss

    
    
    for key, value_list in results.items():
        transformed_results[key] = [-math.log(1 / (1 + math.exp(-v))) for v in value_list]

    auditor = RewardAuditor(B=B, alpha=alpha)

    # Function 2: compute diff_origin_vs_disturbances
    if 'origin_loss' in transformed_results:
        origin_values = transformed_results['origin_loss']
        
        for key, disturbance_values in transformed_results.items():
            if key == 'origin_loss':
                continue
            new_key = f'diff_origin_vs_{key}'
            

            origin_values = np.array(origin_values)
            disturbance_values = np.array(disturbance_values)

            metric_dict = auditor.run_audit(origin_values, disturbance_values)
            # diff_list = [o - d for o, d in zip(origin_values, disturbance_values)]
            # audit_results[new_key] = diff_list

            audit_results[new_key] = metric_dict
        audit_results_after_bh = benjamini_hochberg_procedure(audit_results, alpha=alpha)

    # Function 3: compute diff_disturbances, and save the results at stats_json_output_dir if needed
    if require_stats:
        distribution_summary = {}
        for key, value_list in transformed_results.items():
            data_array = np.array(value_list)
            stats_dict = {
                'mean': np.mean(data_array),
                'median': np.median(data_array),
                'std_dev': np.std(data_array),
                'variance': np.var(data_array),
                'min': np.min(data_array),
                'max': np.max(data_array),
                'skewness': stats.skew(data_array),
                'kurtosis': stats.kurtosis(data_array)
            }
            distribution_summary[key] = stats_dict
        
        if stats_json_output_dir and json_name:
            try:
                os.makedirs(stats_json_output_dir, exist_ok=True)
                file_path = os.path.join(stats_json_output_dir, json_name)
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(distribution_summary, f, indent=4, ensure_ascii=False)
                
                print(f"Saved to: {file_path}")
            except Exception as e:
                print(f"Saving error!: {e}")


    return audit_results_after_bh


def load_eval_dataset(
    raw_Dataset: Dataset = None,
    core_set: bool = True,
    custom_dialogue_formatting: bool = False,
    conv: Conversation = None,
    tokenizer: PreTrainedTokenizer = None,
    logger: logging.Logger = None,
    keep_columns: List[str] = ["text_chosen", "text_rejected", "id"],
    return_extra_data: bool = False,
    max_turns: int = None,
) -> tuple[Dataset, list[str]]:
    """
    Loads either the core eval set for HERM or the existing preference data test sets.

    Args:
        core_set: if True, load the core eval set for HERM.
        custom_dialogue_formatting: if True, format the dialogue as needed for custom models (e.g. SHP and PairRM).
        conv: fastchat conversation template.
                If None (default) the passed tokenizer needs to have a usable chat template.
        tokenizer: HuggingFace tokenizer to use. The tokenizer's chat template, if available, has precedence over conv.
        logger: logger to use for logging. If None (default), no logging is done.
        keep_columns: list of columns to keep in the dataset.
        max_turns: maximum number of turns in the dialogue (usually even). If None (default), no filtering is done.

    Returns:
        dataset: loaded dataset with required properties.
        subsets: list of subsets for the corresponding samples in the dataset.
    """
    if raw_Dataset is not None:
        raw_dataset = raw_Dataset
    elif core_set:
        raw_dataset = load_from_disk("data/reward-bench")
        raw_dataset = raw_dataset['filtered']
    else:
        raw_dataset = load_dataset(EXTRA_PREF_SETS)
        modified_datasets = []

        # Iterate over each subset in the DatasetDict
        for subset_name, subdataset in raw_dataset.items():
            # if subset column exists, move to subsubset (for pref sets)
            if "subset" in subdataset.column_names:
                subdataset = subdataset.rename_column("subset", "subsubset")

            # Add a new column 'subset' to the dataset with the subset name
            subdataset = subdataset.add_column("subset", [subset_name] * len(subdataset))

            # Append the modified dataset to the list
            # remove pku_safer and pku_better from the dict, no longer part of the benchmark
            if subset_name not in ["pku_safer", "pku_better"]:
                modified_datasets.append(subdataset)

        # Concatenate all the modified datasets into one dataset
        raw_dataset = concatenate_datasets(modified_datasets)

    # Apply chat template
    if not custom_dialogue_formatting:
        usable_tokenizer = check_tokenizer_chat_template(tokenizer)

        # assert either conv is passed or tokenizer has chat_template
        assert conv is not None or usable_tokenizer

        if usable_tokenizer:
            if logger is not None:
                logger.info("*** Preparing dataset with HF Transformers ***")
            # docs https://huggingface.co/docs/transformers/main/en/chat_templating
            dataset = raw_dataset.map(
                prepare_dialogue_from_tokenizer,
                fn_kwargs={"tokenizer": tokenizer},
                num_proc=8,
                load_from_cache_file=False,
            )

        # else use FastChat to get chat template
        else:
            if logger is not None:
                logger.info("*** Preparing dataset with FastChat ***")
            dataset = raw_dataset.map(
                prepare_dialogue,
                fn_kwargs={"dialogue_template": conv},
                num_proc=8,  # using >1 process causes issues with re-assigning prompt in example
                load_from_cache_file=False,
            )
    else:
        if logger is not None:
            logger.info("*** Preparing dataset with custom formatting ***")

        def map_conversations(example, core_set=True):
            if core_set:
                example["text_chosen"] = [
                    {"role": "user", "content": example["prompt"]},
                    {"role": "assistant", "content": example["chosen"]},
                ]
                example["text_rejected"] = [
                    {"role": "user", "content": example["prompt"]},
                    {"role": "assistant", "content": example["rejected"]},
                ]
            else:
                prompt = example["prompt"]
                example["text_chosen"] = prompt + [{"role": "assistant", "content": example["chosen"]}]
                example["text_rejected"] = prompt + [{"role": "assistant", "content": example["rejected"]}]
            return example

        dataset = raw_dataset.map(
            map_conversations,
            fn_kwargs={"core_set": core_set},
            num_proc=8,
        )

    if max_turns is not None:
        assert max_turns > 0, "max_turns must be greater than 0"

        # filter long answers (MT Bench prompt as 1 or 2 turn examples)
        def filter_long_turns(batch):
            return len(batch["text_chosen"]) <= max_turns

        dataset = dataset.filter(filter_long_turns)

    # take column subset from dataset
    subsets = dataset["subset"]

    # remove columns if set and not custom_dialogue_formatting
    all_cols = dataset.column_names
    dataset = dataset.remove_columns([c for c in all_cols if c not in keep_columns])

    return dataset, subsets


def save_audit_results(data_to_save: dict, full_filepath: str):
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.bool_):
                return bool(obj)
            return super(NumpyEncoder, self).default(obj)

    try:
        output_dir = os.path.dirname(full_filepath)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(full_filepath, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, indent=4, ensure_ascii=False, cls=NumpyEncoder)

    except Exception as e:
        print(f"something wrong with saving json file: {e}")

