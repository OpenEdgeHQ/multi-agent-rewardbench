import os
import json
import asyncio
from typing import Any, Dict, List
from datetime import datetime

from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_ext.models.anthropic import AnthropicChatCompletionClient
from datasets import load_dataset
from tqdm.asyncio import tqdm_asyncio  # tqdm 的 asyncio 版本

# ---------- 全局配置 ----------
os.environ["ANTHROPIC_API_KEY"] = "XXX"
OUTPUT_DIR = "optimized_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)
RESULT_FILE = f"{OUTPUT_DIR}/optimized_results_{datetime.now().strftime('%Y%m%d')}.jsonl"

# ---------- 全局复用 Claude 客户端 ----------
global_client: AnthropicChatCompletionClient | None = None


def build_agents(model_client: AnthropicChatCompletionClient):
    """一次性创建并复用 Agents"""
    data_analyzer = AssistantAgent(
        name="data_analyzer",
        model_client=model_client,
        system_message=(
            "You are a professional data analyst. Your tasks are:\n"
            "1. Analyze the structure and content of input data\n"
            "2. Identify reasoning chains and logical relationships in the data\n"
            "3. Discover potential problems or inconsistencies in the data\n"
            "4. Propose data optimization suggestions\n"
            "Please output your analysis results in a structured manner, and say "
            "\"Data analysis complete\" when finished."
        ),
    )

    data_optimizer = AssistantAgent(
        name="data_optimizer",
        model_client=model_client,
        system_message=(
            "You are a data optimization expert. Your tasks are:\n"
            "1. Optimize original data based on analysis results\n"
            "2. Enhance data reasoning logic and completeness\n"
            "3. Verify that optimized data satisfies the specific requirements for GRPO reinforcement learning tasks.\n"
            "4. Maintain the original semantic meaning of the data unchanged\n"
            "Please output the final optimized data in JSON format, and say "
            "\"Data optimization complete\" when finished."
        ),
    )

    quality_assessor = AssistantAgent(
        name="quality_assessor",
        model_client=model_client,
        system_message=(
            "You are a data quality assessment expert. Your tasks are:\n"
            "1. Assess quality differences between data before and after optimization\n"
            "2. Verify whether data meets specific task requirements\n"
            "3. Check logical consistency and completeness of data\n"
            "4. Provide quality scores and improvement suggestions\n"
            "Please provide a detailed assessment report, and say "
            "\"Quality assessment complete\" when finished."
        ),
    )

    data_synthesizer = AssistantAgent(
        name="data_synthesizer",
        model_client=model_client,
        system_message=(
            "You are a data integration expert. Your tasks are:\n"
            "1. Synthesize analysis results and suggestions from all previous agents\n"
            "2. Re-integrate complex optimized data into a concise format\n"
            "3. Ensure output data structure is completely consistent with the original raw_data fields\n"
            "4. Retain all important optimized content but present it in a more concise form\n"
            "5. Output format must strictly match the field names and structure of the original data\n"
            "6. Preserve the original mathematical expression format\n"
            "7. Please return only valid JSON starting with '{' and end with 'OPTIMIZATION_COMPLETE'"
        ),
    )

    user_proxy = UserProxyAgent(name="user_proxy")

    return [data_analyzer, data_optimizer, quality_assessor, data_synthesizer, user_proxy]


async def optimize_single_data(raw_data: Dict[str, Any], agents: List[Any]) -> Dict[str, Any]:
    """对单条数据进行优化并提取结果"""
    termination = TextMentionTermination("OPTIMIZATION_COMPLETE")
    team = RoundRobinGroupChat(agents, termination_condition=termination)

    task_prompt = (
        "Please optimize the following data to make it more suitable for GRPO "
        "(Group Relative Policy Optimization) training:\n\n"
        f"Original data:\n{raw_data}\n\n"
        "Optimization requirements:\n"
        "1. Enhance data reasoning logic chains\n"
        "2. Ensure data is suitable for reinforcement learning training\n"
        "3. Optimize question-answer pair quality\n"
        "4. If it's a math or code problem, ensure the solution process is complete and verifiable\n\n"
        "Workflow:\n"
        "1. data_analyzer → 2. data_optimizer → 3. quality_assessor → 4. data_synthesizer\n"
        "Begin now."
    )

    result = await team.run(task=task_prompt)
    return extract_optimized_data(result)


def extract_optimized_data(team_result) -> Dict[str, Any]:
    """抽取 data_synthesizer 或 data_optimizer 的 JSON 输出"""
    import re

    messages = getattr(team_result, "messages", [])
    json_pattern = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"

    # 优先 data_synthesizer
    for m in reversed(messages):
        if "data_synthesizer" in m.source:
            matches = re.findall(json_pattern, m.content)
            for s in matches:
                try:
                    return json.loads(s)
                except json.JSONDecodeError:
                    continue

    # 回退 data_optimizer
    for m in reversed(messages):
        if "data_optimizer" in m.source:
            matches = re.findall(json_pattern, m.content)
            for s in matches:
                try:
                    return json.loads(s)
                except json.JSONDecodeError:
                    continue

    raise ValueError("Failed to extract optimized JSON")


async def process_dataset(concurrency: int = 4):
    """主入口：加载数据并并发优化"""
    global global_client
    global_client = AnthropicChatCompletionClient(
        model="claude-3-5-sonnet-20241022",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
    )
    agents = build_agents(global_client)

    # 只取部分数据做演示
    ds = load_dataset("AI-MO/NuminaMath-TIR", name="default", split="train")
    items = ds.select(range(2538, 10000))  # 示例 500 条

    sem = asyncio.Semaphore(concurrency)
    tasks = []

    async def worker(idx, row):
        raw_data = {"problem": row["problem"], "solution": row["solution"]}
        async with sem:
            try:
                if len(str(raw_data)) <= 10000:
                    optimized = await optimize_single_data(raw_data, agents)
            except Exception as exc:
                optimized = raw_data  # 失败时写回原始数据
                print(f"[{idx}] failed: {exc}")

            async with aiofiles.open(RESULT_FILE, "a", encoding="utf-8") as f:
                await f.write(json.dumps(optimized, ensure_ascii=False) + "\n")

    import aiofiles  # 文件异步写入

    for i, r in enumerate(items):
        tasks.append(asyncio.create_task(worker(i, r)))

    await tqdm_asyncio.gather(*tasks, desc="Optimizing")

    await global_client.close()


if __name__ == "__main__":
    asyncio.run(process_dataset(concurrency=2))
