"""
图片描述生成模块：使用 qwen-vl-plus 多模态模型为图片生成自然语言描述
描述结果用于嵌入向量检索，不存储图片本身
"""
import base64
from typing import Optional

from langchain_openai import ChatOpenAI

from config import DASHSCOPE_API_KEY


_PROMPT_TEMPLATE = """\
请用中文详细描述这张图片的内容，这是一个售后客服知识库中的图片。

描述要求：
1. 描述图片中显示的产品、零件、操作流程或故障现象
2. 如果是产品图，指出外观特征、标注的名称或型号
3. 如果是流程图/示意图，按步骤描述流程
4. 如果是故障/问题图片，描述故障现象和可能位置
5. 描述要便于通过文字搜索找到这张图片

图片在文档中的上下文说明：{context}"""


def describe_image(image_bytes: bytes, caption: str = "", context: str = "") -> str:
    """
    用 qwen-vl-plus 为图片生成详细描述

    Args:
        image_bytes: 图片原始字节（PNG/JPEG 均可）
        caption:     图片标题（来自文档，可能为空）
        context:     图片在文档中的上下文信息（如所在章节标题）

    Returns:
        图片的自然语言描述字符串
    """
    ctx = caption or context or "无"
    prompt = _PROMPT_TEMPLATE.format(context=ctx)

    b64 = base64.b64encode(image_bytes).decode("utf-8")

    llm = ChatOpenAI(
        model="qwen-vl-plus",
        api_key=DASHSCOPE_API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=0,
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    try:
        response = llm.invoke(messages)
        return response.content.strip()
    except Exception as e:
        fallback = caption.strip() if caption else ""
        return f"图片描述生成失败: {e}" + (f"（标题：{fallback}）" if fallback else "")