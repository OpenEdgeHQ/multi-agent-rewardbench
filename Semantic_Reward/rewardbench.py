from .prime_math import compute_score
# Import necessary libraries
import logging
import os
import sys
import re
import math
from dataclasses import dataclass, field
from typing import List, Optional
import json
import time
import requests
from nltk.translate import bleu_score
from rouge_score import rouge_scorer
# Import PyTorch and Hugging Face Transformers
import torch

# Import math-related utilities
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify

def accuracy_reward(completion: str, solution: str, **kwargs):
    """
    Reward function to check if the model's response is mathematically 
    equivalent to the ground truth solution.
    Uses latex2sympy2 for parsing and math_verify for validation.
    """
    
    answer_parsed = parse(
        completion,
        extraction_config=[
            LatexExtractionConfig(
                normalization_config=NormalizationConfig(
                    nits=False,
                    malformed_operators=False,
                    basic_latex=True,
#                         equations=True,
                    boxed="all",
                    units=True,
                ),
                boxed_match_priority=0,
                try_extract_without_anchor=False,
            )
        ],
        extraction_mode="first_match",
    )

    # Reward 1.0 if correct, 0.0 if incorrect
    reward = float(verify(answer_parsed, solution))
    if reward > 0.5:
        print(f"Correct answer: {answer_parsed}, Ground truth: {solution}")
    return reward

# Implement Format Reward Function
def format_reward(completion: str, **kwargs):
    """
    Reward function to check if the completion has the correct format:
    <think>...</think> <answer>...</answer>.
    """
    # Define the regex pattern for the desired format
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"

    # Check if completion matches the pattern
    match = re.match(pattern, completion, re.DOTALL | re.MULTILINE)

    # Reward 1.0 for correct format, 0.0 otherwise
    return 1.0 if match else 0.0

def reasoning_steps_reward(completion: str, **kwargs):
    r"""
    Reward function to encourage clear step-by-step reasoning.
    It looks for patterns like "Step 1:", numbered lists, bullet points,
    and transition words.
    """
    # Regex pattern to find indicators of reasoning steps
    pattern = r"(Step \d+:|^\d+\.|\n-|\n\*|First,|Second,|Next,|Finally,)"

    # Count the number of reasoning step indicators in the completion
    matches = len(re.findall(pattern, completion, re.MULTILINE))

    # Reward is proportional to the number of reasoning steps, maxing out at 1.0
    # We're using a "magic number" 3 here - encourage at least 3 steps for full reward
    return min(1.0, matches / 3)

# Implement Cosine Scaled Reward Function
def get_cosine_scaled_reward(
    min_value_wrong: float = -0.5,
    max_value_wrong: float = -0.1,
    min_value_correct: float = 0.8,
    max_value_correct: float = 1.0,
    max_len: int = 1000,
):
    """
    Returns a cosine scaled reward function. This function scales the accuracy reward
    based on completion length. Shorter correct solutions get higher rewards,
    longer incorrect solutions get less penalty.
    """
    def cosine_scaled_reward(completion: str, solution: str, accuracy_reward: float, **kwargs):
        """
        Cosine scaled reward function that adjusts accuracy rewards based on completion length.
        """
        gen_len = len(completion)  # Length of the generated answer
        progress = gen_len / max_len # How far we are to max length
        cosine = math.cos(progress * math.pi) # Cosine value based on progress

        if accuracy_reward > 0.5: # Assuming accuracy_reward gives ~1.0 for correct answers
            min_value = min_value_correct
            max_value = max_value_correct
        else: # Incorrect answer
            min_value = max_value_wrong  # Note the swap!
            max_value = min_value_wrong

        # Cosine scaling formula!
        reward = min_value + 0.5 * (max_value - min_value) * (1.0 + cosine)
        return float(reward)
    return cosine_scaled_reward

def get_repetition_penalty_reward(ngram_size: int = 3, max_penalty: float = -0.1):
    """
    Returns a repetition penalty reward function. Penalizes repetitions of n-grams
    in the generated text.
    """
    if max_penalty > 0:
        raise ValueError(f"max_penalty {max_penalty} should not be positive")

    def zipngram(text: str, ngram_size: int):
        """Helper function to generate n-grams from text."""
        words = text.lower().split() # Lowercase and split into words
        return zip(*[words[i:] for i in range(ngram_size)]) # Create n-grams

    def repetition_penalty_reward(completion: str, **kwargs) -> float:
        """
        Repetition penalty reward function.
        """
        if completion == "": # No penalty for empty completions
            return 0.0
        if len(completion.split()) < ngram_size: # No penalty for short completions
            return 0.0

        ngrams = set() # Use a set to store unique n-grams
        total = 0
        for ng in zipngram(completion, ngram_size): # Generate n-grams
            ngrams.add(ng) # Add n-gram to the set (duplicates are ignored)
            total += 1 # Count total n-grams

        # Calculate scaling factor: more repetition -> higher scaling
        scaling = 1 - len(ngrams) / total
        reward = scaling * max_penalty # Apply penalty based on scaling
        return reward
    return repetition_penalty_reward

def compute_score_ours(model_output: str, ground_truth: str):
    """
    Aggregate function that combines all reward functions with equal weights.
    Returns the average of all individual rewards.
    """
    # Initialize reward functions
    cosine_scaled_reward = get_cosine_scaled_reward()
    repetition_penalty_reward = get_repetition_penalty_reward()
    semantic_score=compute_semantic_score(ground_truth,model_output)
    # Calculate individual rewards
    acc_reward = accuracy_reward(model_output, ground_truth)
    format_reward_val = format_reward(model_output)
    reasoning_reward = reasoning_steps_reward(model_output)
    cosine_reward = cosine_scaled_reward(model_output, ground_truth, acc_reward)
    repetition_reward = repetition_penalty_reward(model_output)
    
    # Calculate equal-weighted average
    total_reward = (acc_reward + format_reward_val + reasoning_reward + 
                   cosine_reward + repetition_reward + 2*semantic_score) / 6.0
    
    return total_reward 


def compute_semantic_score(target_text, generated_text, rerank_score=1.0,
                           bleu=None, rouge1=None, rouge2=None, rougeL=None, ngram=None,
                           max_retries=100, base_backoff=0):
    """
    max_retries: 最多重试次数（包含第一次请求）
    base_backoff: 退避基数（秒），第 i 次失败后 sleep = base_backoff * i
    """

    # 1) 调用 vLLM Reranker API 获取的 rerank_score
    url = "http://127.0.0.1:7000/score"
    payload = {
        "text_1": target_text,
        "text_2": generated_text,
        "model": "Qwen3-Reranker-0.6B"  # 必须和 --served-model-name 一致
    }
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json",
        "Connection": "close",  
    }

    last_err_text = None
    success = False
    for i in range(1, max_retries + 1):
        try:
            resp = requests.post(
                url,
                headers=headers,
                data=json.dumps(payload),
                proxies={"http": None, "https": None},  # 禁用代理
                timeout=30,
            )
            if resp.status_code == 200:
                try:
                    rerank_score = resp.json()["data"][0]["score"]
                    success = True
                    break
                except Exception as je:
                    last_err_text = f"JSON parse error: {je} | body={resp.text[:300]}"
            else:
                last_err_text = f"HTTP {resp.status_code}: {resp.text[:300]}"
        except Exception as e:
            last_err_text = f"Exception: {e}"

        # 失败则退避等待
        if i < max_retries:
            time.sleep(base_backoff * i)

    if not success:
        print(f"[Reranker] Failed after {max_retries} tries. Using fallback score={rerank_score}. Last error: {last_err_text}")

    final_score = float(rerank_score)

    # 2) 计算 BLEU
    if bleu is not None:
        # 可加平滑：SmoothingFunction().method3
        bleu_score_value = bleu_score.sentence_bleu(
            [target_text.split()],
            generated_text.split()
        )
        final_score += bleu * bleu_score_value

    # 3) 计算 ROUGE
    if rouge1 is not None or rouge2 is not None or rougeL is not None:
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(target_text, generated_text)

        if rouge1 is not None:
            final_score += rouge1 * scores["rouge1"].fmeasure
        if rouge2 is not None:
            final_score += rouge2 * scores["rouge2"].fmeasure
        if rougeL is not None:
            final_score += rougeL * scores["rougeL"].fmeasure

    # 4) 计算 N-Gram
    if ngram is not None:
        n, weight = ngram
        ngram_score = n_gram_similarity(target_text, generated_text, n)
        final_score += weight * ngram_score

    return final_score


def n_gram_similarity(a: str, b: str, n: int) -> float:
    def get_ngrams(text, n):
        tokens = text.split()
        return set(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))
    A, B = get_ngrams(a, n), get_ngrams(b, n)
    return len(A & B) / max(1, len(A | B))