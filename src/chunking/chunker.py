"""
切块与摘要生成模块
- 文本：结构感知递归切块（中文友好）
- 表格：用 LLM 生成摘要，原表格整体保留
- 图片：基于 caption 生成描述
"""
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter


_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "！", "？", ".", " "],
)


def split_texts(texts: List[str]) -> List[str]:
    """将多段文本合并后切块"""
    combined = "\n\n".join(t for t in texts if t.strip())
    return _text_splitter.split_text(combined)


def summarize_table(table_content: str) -> str:
    """用 LLM 为表格生成自然语言摘要（用于语义检索）"""
    from langchain_openai import ChatOpenAI
    from config import DASHSCOPE_API_KEY, LLM_MODEL

    llm = ChatOpenAI(
        model=LLM_MODEL,
        api_key=DASHSCOPE_API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=0,
    )
    prompt = (
        "请用中文简洁概括以下表格的内容，包括：\n"
        "1. 这个表格是关于什么的\n"
        "2. 包含哪些关键字段/列\n"
        "3. 关键数据要点\n\n"
        f"表格内容：\n{table_content}"
    )
    response = llm.invoke(prompt)
    return response.content


def describe_image(image_info: dict) -> str:
    """根据图片 caption 生成描述（可扩展为多模态 LLM）"""
    caption = image_info.get("caption", "").strip()
    return f"图片: {caption}" if caption else "图片内容（无文字描述）"
