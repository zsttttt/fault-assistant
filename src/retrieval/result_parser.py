"""
检索结果解析：将 KnowledgeRetriever 返回的混合 context 列表
按类型分离为文本上下文、图片引用、视频引用
"""
import json
from typing import List


def parse_retrieved_results(context: List[dict]) -> dict:
    """
    Args:
        context: KnowledgeRetriever.retrieve() 返回的第一个元素

    Returns:
        {
            "text_contexts":  [{"title":..., "content":..., "error_code":...}],
            "image_refs":     [{"url":..., "description":..., "filename":...}],
            "video_refs":     [{"url":..., "description":..., "filename":..., "frame_urls":[...]}],
        }
    """
    parsed = {"text_contexts": [], "image_refs": [], "video_refs": [], "table_image_refs": []}

    for item in context:
        item_type = item.get("type", "knowledge_entry")

        if item_type == "image":
            ref = _parse_media_json(item, expected_type="image")
            if ref:
                parsed["image_refs"].append(ref)

        elif item_type == "video":
            ref = _parse_media_json(item, expected_type="video")
            if ref:
                parsed["video_refs"].append(ref)

        else:
            parsed["text_contexts"].append(item)
            if item_type == "table" and item.get("table_image_url"):
                parsed["table_image_refs"].append({
                    "url": item["table_image_url"],
                    "description": item.get("title", "参考表格"),
                })

    return parsed


def _parse_media_json(item: dict, expected_type: str) -> dict:
    """从 content 字段解析媒体 JSON，失败时用 media_url 降级"""
    content = item.get("content", "")
    media_url = item.get("media_url", "")

    try:
        data = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        data = {}

    url = data.get("url") or media_url
    if not url:
        return {}

    if expected_type == "image":
        return {
            "url": url,
            "description": data.get("description", ""),
            "filename": data.get("filename", ""),
        }

    return {
        "url": url,
        "description": data.get("description", ""),
        "filename": data.get("filename", ""),
        "frame_urls": data.get("frame_urls", []),
    }