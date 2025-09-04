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
import numpy as np
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
                   cosine_reward + repetition_reward + semantic_score) / 6.0
    
    return total_reward 


EMBED_URL = "http://127.0.0.1:7000/v1/embeddings"
EMBED_MODEL = "Qwen3-Embedding-0.6B"  # 必须与 --served-model-name 一致
def _post_with_retry(url, payload, headers, max_retries=10000, base_backoff=0):
    last_err = None
    for i in range(1, max_retries + 1):
        try:
            resp = requests.post(
                url,
                headers=headers,
                data=json.dumps(payload),
                proxies={"http": None, "https": None},
                timeout=30,
            )
            if resp.status_code == 200:
                data = resp.json()
                if "data" in data and len(data["data"]) > 0:
                    return data
                else:
                    last_err = "Empty 'data' field, retrying..."
            else:
                last_err = f"HTTP {resp.status_code}: {resp.text[:300]}"
        except Exception as e:
            last_err = f"Exception: {e}"
        
        # 等待后重试
        if i < max_retries and base_backoff > 0:
            time.sleep(base_backoff * i)

    raise RuntimeError(f"Request failed after {max_retries} tries. last_err={last_err}")

def _get_embeddings(text_list, max_retries=10000, base_backoff=0):
    """一次性取多条文本的embedding，避免两次HTTP开销。"""
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json",
        "Connection": "close",
    }
    payload = {
        "model": EMBED_MODEL,
        "input": text_list
    }
    data = _post_with_retry(EMBED_URL, payload, headers, max_retries, base_backoff)
    # vLLM 兼容 OpenAI：data["data"] 为列表，按输入顺序返回
    vecs = [np.array(item["embedding"], dtype=np.float32) for item in data["data"]]
    return vecs

def _cosine_sim(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    # 若模型已归一化，这里等同点积；若未归一化，这里是标准 cosine
    denom = (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if denom == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)

def n_gram_similarity(a: str, b: str, n: int) -> float:
    def get_ngrams(text, n):
        tokens = text.split()
        return set(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))
    A, B = get_ngrams(a, n), get_ngrams(b, n)
    return len(A & B) / max(1, len(A | B))

def compute_semantic_score(
    target_text: str,
    generated_text: str,
    bleu: float | None = None,
    rouge1: float | None = None,
    rouge2: float | None = None,
    rougeL: float | None = None,
    ngram: tuple[int, float] | None = None,
    max_retries: int = 10000,
    base_backoff: float = 0
) -> float:
    """
    语义相似度由 Qwen3-Embedding 余弦相似度为主，其余指标加权叠加。
    max_retries: 最多重试次数
    base_backoff: 退避基数（秒），第 i 次失败后 sleep = base_backoff * i
    """
    # 1) 取 embedding（批量）
    vec_target, vec_gen = _get_embeddings(
        [target_text, generated_text],
        max_retries=max_retries,
        base_backoff=base_backoff
    )

    # 2) 余弦相似度（主分）
    final_score = _cosine_sim(vec_target, vec_gen)

    # 3) BLEU（可选）
    if bleu is not None:
        bleu_score_value = bleu_score.sentence_bleu(
            [target_text.split()],
            generated_text.split()
        )
        final_score += bleu * bleu_score_value

    # 4) ROUGE（可选）
    if rouge1 is not None or rouge2 is not None or rougeL is not None:
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(target_text, generated_text)
        if rouge1 is not None:
            final_score += rouge1 * scores["rouge1"].fmeasure
        if rouge2 is not None:
            final_score += rouge2 * scores["rouge2"].fmeasure
        if rougeL is not None:
            final_score += rougeL * scores["rougeL"].fmeasure

    # 5) N-gram Jaccard（可选）
    if ngram is not None:
        n, weight = ngram
        final_score += weight * n_gram_similarity(target_text, generated_text, n)

    return float(final_score)
