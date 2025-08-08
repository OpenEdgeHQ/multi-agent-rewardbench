
import json
from typing import Dict, List
from multi_llm.agents.llm_agents.claude import handle
from multi_llm.agents.teacher_agent.teacher_reasoning_prompt import SYSTEM_PROMPT as REASONING_SYSTEM_PROMPT, \
    USER_PROMPT as REASONING_USER_PROMPT


def evaluate_and_teach(problem: str, model_output: str, ground_truth: str, model_name: str="claude-sonnet-4-20250514") -> Dict:
    """
    评估模型输出并通过多轮对话提升accuracy
    """
    current_output = model_output
    conversation_history = []


    user_prompt = REASONING_USER_PROMPT.format(
        problem=problem,
        answer=ground_truth,
        model_output=current_output
    )

    # 调用accuracy teacher
    response = handle(user_prompt, REASONING_SYSTEM_PROMPT, model=model_name)
    return response
