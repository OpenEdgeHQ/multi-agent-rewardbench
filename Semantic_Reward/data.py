import json
import os
import pandas as pd
from math_verify import parse

def extract_ground_truth(solution: str) -> str:
    """
    用 Math-Verify 从 solution 中提取最终答案
    """
    try:
        parsed = parse(solution)
        if parsed:
            return str(parsed[0])  # 返回第一个解析成功的表达式
        else:
            return ""
    except Exception as e:
        print(f"[WARN] Extraction failed: {e}")
        return ""

def convert_record(record: dict) -> dict:
    """
    把单条 {problem, solution} 转换为目标格式
    """
    problem = record.get("problem", "")
    solution = record.get("solution", "")

    return {
        "data_source": "Rewardbench",
        "ability": "math",
        "prompt": [
            {
                "content": (
                    "A conversation between User and Assistant. "
                    "The user asks a question, and the Assistant solves it. "
                    "The assistant first thinks about the reasoning process in the mind "
                    "and then provides the user with the answer. "
                    "The reasoning process and answer are enclosed within <think> </think> "
                    "and <answer> </answer> tags, respectively, i.e., "
                    "<think> reasoning process here </think><answer> answer here </answer>"
                ),
                "role": "system"
            },
            {
                "content": problem,
                "role": "user"
            }
        ],
        "reward_model": {
            "ground_truth": extract_ground_truth(solution),
            "style": "rule"
        },
        "problem": problem
    }

def convert_file(input_file: str, output_file: str):
    ext = os.path.splitext(input_file)[-1].lower()

    if ext == ".jsonl":
        with open(input_file, "r", encoding="utf-8") as fin, \
             open(output_file, "w", encoding="utf-8") as fout:
            for line in fin:
                if not line.strip():
                    continue
                data = json.loads(line.strip())
                fout.write(json.dumps(convert_record(data), ensure_ascii=False) + "\n")

    elif ext == ".parquet":
        df = pd.read_parquet(input_file)
        with open(output_file, "w", encoding="utf-8") as fout:
            for _, row in df.iterrows():
                data = {"problem": row.get("problem", ""), "solution": row.get("solution", "")}
                fout.write(json.dumps(convert_record(data), ensure_ascii=False) + "\n")

    else:
        raise ValueError(f"Unsupported file type: {ext}")

if __name__ == "__main__":
    # 输入文件可以是 .jsonl 或 .parquet
    convert_file("input.jsonl", "output.jsonl")
    # convert_file("input.parquet", "output.jsonl")
