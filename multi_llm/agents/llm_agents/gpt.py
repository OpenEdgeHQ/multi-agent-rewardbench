from openai import OpenAI
from utils.load_config import load_yaml, get_random_key

cfg = load_yaml("llm_api_key.yaml", "openai")


def handle(prompt, role_content, model, shared_memory=None):
    key_group = cfg["models"][model]  # e.g. "shared"
    api_keys = cfg["api-keys"][key_group]
    client = OpenAI(api_key=get_random_key(api_keys))
    messages = [
        {"role": "system", "content": role_content},
        {"role": "user", "content": prompt}
    ]
    res = client.chat.completions.create(model=model, messages=messages)
    result = res.choices[0].message.content.strip()
    return result


def handle_stream(prompt, role_content, model, on_token=None):
    """
    流式处理函数，支持实时输出

    Args:
        prompt: 用户输入
        role_content: 系统角色内容
        model: 模型名称
        on_token: 令牌回调函数

    Yields:
        str: 流式输出的文本片段
    """
    key_group = cfg["models"][model]
    api_keys = cfg["api-keys"][key_group]
    client = OpenAI(api_key=get_random_key(api_keys))
    messages = [
        {"role": "system", "content": role_content},
        {"role": "user", "content": prompt}
    ]

    # 开启 stream=True 进行流式输出
    try:
        for chunk in client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True
        ):
            # 检查是否有内容
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                content = delta.content if hasattr(delta, 'content') else None

                if content:
                    # 如果提供了回调函数，则调用
                    if on_token:
                        on_token(content)
                    yield content
    except Exception as e:
        # 错误处理
        error_msg = f"流式输出错误: {str(e)}"
        if on_token:
            on_token(error_msg)
        yield error_msg


def handle_image(user_prompt, role_content, model, shared_memory=None):
    """
    处理图像相关的请求

    Args:
        user_prompt: 用户输入
        role_content: 角色内容/指令
        model: 模型名称
        shared_memory: 共享内存（可选）

    Returns:
        str: 处理结果
    """
    key_group = cfg["models"][model]  # e.g. "shared"
    api_keys = cfg["api-keys"][key_group]
    client = OpenAI(api_key=get_random_key(api_keys))

    # 构建消息格式
    messages = [
        {"role": "system", "content": role_content},
        {"role": "user", "content": user_prompt}
    ]

    try:
        # 使用 chat completions API 处理图像请求
        response = client.chat.completions.create(
            model=model,
            messages=messages
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"图像处理错误: {str(e)}"


def handle_batch(prompts, role_content, model, shared_memory=None):
    """
    批量处理多个请求

    Args:
        prompts: 用户输入列表
        role_content: 系统角色内容
        model: 模型名称
        shared_memory: 共享内存（可选）

    Returns:
        list: 处理结果列表
    """
    results = []
    for prompt in prompts:
        try:
            result = handle(prompt, role_content, model, shared_memory)
            results.append(result)
        except Exception as e:
            results.append(f"处理错误: {str(e)}")
    return results
