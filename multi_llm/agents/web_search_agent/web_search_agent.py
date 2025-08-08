import json
import logging
import asyncio
import re
import concurrent.futures
from functools import partial
from multi_llm.agents.llm_agents.gpt import handle as gpt_handle
from typing import List
from multi_llm.agents.tools.fetcher import fetch
from multi_llm.agents.tools.searcher import search
from multi_llm.agents.tools.cleaner import clean_html
from utils.load_config import load_yaml, get_random_key

from multi_llm.agents.web_search_agent.web_search_prompt import (
    SYSTEM_PROMPT,
    USER_PROMPT,
USER_PROMPT_URL,
SYSTEM_PROMPT_URL,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration and logging
config = load_yaml("web_search.yaml")
api_keys = config['serpapi']['api-key']
logger.info("Web search agent starting up")
MAX_CONCURRENT_TASKS = config['service']['MAX_CONCURRENT_TASKS']
TIMEOUT_SEC = config['service']['TIMEOUT_SEC']

# Create WebFetcher instance


async def fetch_page_with_timeout(url: str) -> str:
    """Fetch page with timeout"""
    try:
        html = await asyncio.wait_for(
            fetch(url),
            timeout=TIMEOUT_SEC
        )
        return html
    except asyncio.TimeoutError:
        logger.warning(f"Page fetch timeout: {url}")
        return None
    except Exception as e:
        logger.warning(f"Page fetch failed {url}: {str(e)}")
        return None

async def select_search_result(search_res: list, model: str) -> str:
    field_list = ["url","displayed_url","description", "position", "title", "domain"]
    search_res = [{field: original_dict[field] for field in field_list} for original_dict in search_res]
    user_prompt = USER_PROMPT_URL.format(
        search_res=search_res,

    )


async def extract_page_content(url: list, html: str, target: dict, model: str) -> dict:
    """Second sub-agent: Extract structured and unstructured page content"""
    if not html:
        return {}

    # Limit HTML length
    max_html_length = 170000
    if len(html) > max_html_length:
        html = html[:max_html_length]

    formatted_prompt = USER_PROMPT.format(
        url=url,
        html=html,
        target=target,
    )

    try:
        response = await asyncio.wait_for(
            asyncio.to_thread(gpt_handle, formatted_prompt, SYSTEM_PROMPT, model),
            timeout=TIMEOUT_SEC
        )

        try:
            result = json.loads(response)
        except:
            pattern = re.compile(r'```json\s*(\{[\s\S]*\})\s*```')
            match = pattern.search(response)
            if match:
                result = json.loads(match.group(1))
            else:
                result = {"structured_data": "", "unstructured_data": ""}
        return result

    except asyncio.TimeoutError:
        logger.error(f"Page content extraction timeout: {url}")
        return {"structured_data": "", "unstructured_data": ""}
    except Exception as e:
        logger.error(f"Page content extraction failed {url}: {e}")
        return {"structured_data": "", "unstructured_data": ""}


def fetch_and_analyze_page_sync(url: str, target: dict, extraction_model: str) -> dict:
    """同步版本的页面分析函数，用于线程池"""
    return asyncio.run(fetch_and_analyze_page(url, target, extraction_model))


async def fetch_and_analyze_page(url: str, target: dict, extraction_model: str) -> dict:
    """Enhanced page analysis: Using two sub-agents"""
    # Fetch page content
    html = await fetch_page_with_timeout(url)

    html = clean_html(html)

    if not html:
        logger.warning(f"Failed to fetch content from {url}")
        return None

    content_result = await extract_page_content(url, html, target, extraction_model)

    result = content_result

    logger.info(f"Successfully analyzed page: {url}")
    return result


async def agent_run(target: dict, search_query: str = None,
                    extraction_model: str = "gpt-4.1-mini"
                    ) -> List:
    if not target:
        logger.error("target is required")
        return []

    logger.info("Web search agent started.")
    time0 = time.time()


    # Search mode
    logger.info(f"Searching and analyzing pages for: {target}")
    search_results = await search(search_query)



    if not search_results:
        return ""

    # Extract search result URLs and analyze using ThreadPoolExecutor
    search_urls = [result["url"] for result in search_results
                   if not result["url"].endswith('.pdf') and "zhihu" not in result["url"]][:10]

    # search_urls = [search_results[0]["url"]]
    logger.info(f"Search URLs: {search_urls}")

    # 使用线程池处理多个URL
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(MAX_CONCURRENT_TASKS, len(search_urls))) as executor:
        # 创建部分应用函数，固定除URL外的其他参数
        process_func = partial(fetch_and_analyze_page_sync,
                               target=target,
                               extraction_model=extraction_model)

        # 提交所有任务到线程池
        future_to_url = {executor.submit(process_func, url): url for url in search_urls}
        analyzed_pages = []

        # 获取结果
        for future in concurrent.futures.as_completed(future_to_url):
            url = future_to_url[future]
            try:
                result = future.result()
                if result:
                    analyzed_pages.append(result)
            except Exception as exc:
                logger.error(f"Failed to analyze {url}: {exc}")

        # Filter valid results
        valid_results = [result for result in analyzed_pages if result is not None]

        logger.info(f"Total time: {time.time() - time0:.1f}s")
        return valid_results


if __name__ == "__main__":
    # 测试搜索模式
    import time

    for i in range(5):
        time_0 = time.time()
        result1 = asyncio.run(agent_run(
            search_query="低俗小说 1994 电影 评分",
            target={'name': '电影评分', 'description': '低俗小说 1994 在各大平台上的电影评分'}
        ))
        time_1 = time.time()
        print(time_1 - time_0)
