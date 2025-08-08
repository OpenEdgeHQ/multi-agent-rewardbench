import anthropic
from utils.load_config import load_yaml, get_random_key

# 加载 Anthropic 配置
cfg = load_yaml("llm_api_key.yaml", "claude")


def handle(prompt, role_content, model, shared_memory=None):
    """
    调用 Claude 文本对话接口
    """
    # 根据模型选择 API key 分组
    key_group = cfg["models"][model]
    api_keys = cfg["api-keys"][key_group]
    # 初始化 Anthropic 客户端
    client = anthropic.Anthropic(api_key=get_random_key(api_keys))
    # 构造消息列表
    # 发起对话请求
    res = client.messages.create(
        model=model,
        system=role_content,  # 将 system 内容放在顶层
        messages=[  # messages 列表只包含 user/assistant
            {"role": "user", "content": prompt}
        ],
        # thinking={
        #     "type": "enabled",
        #     "budget_tokens": 2000
        # },
        max_tokens=15000,
    )
    text_idx = [i.type for i in res.content].index('text')
    # 提取并返回回答
    return res.content[text_idx].text


def handle_stream(prompt, role_content, model, on_token=None):
    # 根据模型选择 API key 分组
    key_group = cfg["models"][model]
    api_keys = cfg["api-keys"][key_group]
    # 初始化 Anthropic 客户端
    client = anthropic.Anthropic(api_key=get_random_key(api_keys))

    # 发起流式对话请求
    with client.messages.stream(
            model=model,
            system=role_content,
            messages=[
                {"role": "user", "content": prompt}
            ],
            # thinking={
            #     "type": "enabled",
            #     "budget_tokens": 2000
            # },
            max_tokens=50000,
    ) as stream:
        # 定义一个全局变量用于跟踪当前正在处理的内容类型
        current_content_type = None

        for text in stream.text_stream:
            # 只处理 text 类型的内容
            if current_content_type is None or current_content_type == "text":
                current_content_type = "text"

                # 如果有回调函数，则调用
                if on_token:
                    on_token(text)

                # 返回当前 token
                yield text
