"""
多模态回答生成模块
- 把文本/表格/图片/视频统一组装进 LLM prompt
- LLM 在回答中以 [图片N] / [视频N] 占位符引用媒体
- 生成后扫描占位符，回传被引用的媒体 URL 列表
"""
import re
import json
import httpx
from typing import AsyncGenerator, List, Optional

from langchain_openai import ChatOpenAI

from config import DASHSCOPE_API_KEY, LLM_MODEL, REDIS_URL

_SYSTEM_PROMPT = """\
你是一个专业的加油机设备故障诊断助手。

## 工作原则：
1. 根据提供的知识库内容回答用户问题
2. 如果知识库有明确答案，直接给出解决步骤
3. 如果信息不完全匹配，可基于相关信息推理，但须说明"以下建议基于相似故障经验，请谨慎操作"
4. 如果完全没有相关信息，诚实告知，并提供通用排查建议
5. 涉及高压电、燃油等危险操作时，务必提醒安全注意事项

## 版本知识检索规则：
你收到的参考资料中，每条内容前可能有来源标记，格式为：
[版本:XXX | 文档类型:XXX | 优先级:XXX | 来源:XXX]

请遵守以下规则：
1. 优先级 "high" 的内容来自用户当前使用的版本，优先级 "low" 的内容来自更早的版本（可能包含多个历史版本）。
2. 当高优先级和低优先级内容描述同一功能但存在差异时，以高优先级内容为准。
3. 低优先级内容中若有多个不同版本，版本号更新（数字更大）的内容优先于版本号更旧的内容。
4. 当只有低优先级内容涉及某功能时，正常使用该内容回答。
5. 回答时，如果当前版本的内容与旧版本存在差异，请简要说明（如"该功能在您的版本中已有更新"）。
6. 不要向用户暴露"优先级"、"high/low"等内部标记术语。

## 媒体引用规则（重要）：
- 只有当参考资料中明确列出了【相关图片资料】或【相关视频资料】时，才可在回答中用 [图片1]、[视频1] 等方式引用
- 若参考资料中没有提供图片或视频，严禁自行生成 [图片N] / [视频N] 占位符
- 只引用确实相关的媒体，不要强行引用

## 内联图片规则：
- 参考资料的【文本知识库内容】中如包含 Markdown 图片（如 `![图示](https://...)`），
  请在回答对应步骤处原样保留该图片 Markdown，不要修改 URL 或省略

## 回答格式：
- 先简述可能原因（1-2 句）
- 再给出具体操作步骤（分步骤）
- 最后提醒注意事项（如有必要）"""


def _build_prompt(
    question: str,
    parsed: dict,
    confidence: str,
    device_model: Optional[str],
) -> str:
    parts = []

    if device_model:
        parts.append(f"【当前设备型号】{device_model}")

    text_contexts = parsed.get("text_contexts", [])
    image_refs = parsed.get("image_refs", [])
    video_refs = parsed.get("video_refs", [])

    if text_contexts:
        parts.append("【文本知识库内容】")
        for i, item in enumerate(text_contexts, 1):
            parts.append(f"\n--- 参考资料 {i} ---")
            label_parts = []
            if item.get("version"):
                label_parts.append(f"版本:{item['version']}")
            if item.get("doc_type"):
                label_parts.append(f"文档类型:{item['doc_type']}")
            if item.get("priority"):
                label_parts.append(f"优先级:{item['priority']}")
            if item.get("source"):
                label_parts.append(f"来源:{item['source']}")
            if label_parts:
                parts.append(f"[{' | '.join(label_parts)}]")
            if item.get("error_code"):
                parts.append(f"错误码: {item['error_code']}")
            if item.get("title"):
                parts.append(f"标题: {item['title']}")
            parts.append(f"内容: {item['content']}")
    else:
        parts.append("【文本知识库内容】未找到直接相关的故障记录")

    if image_refs:
        parts.append("\n【相关图片资料】（回答中可用 [图片N] 引用）")
        for i, img in enumerate(image_refs, 1):
            parts.append(f"[图片{i}] {img.get('description', '无描述')}  （文件：{img.get('filename', '')}）")

    if video_refs:
        parts.append("\n【相关视频资料】（回答中可用 [视频N] 引用）")
        for i, vid in enumerate(video_refs, 1):
            parts.append(f"[视频{i}] {vid.get('description', '无描述')}  （文件：{vid.get('filename', '')}）")

    confidence_hints = {
        "high": "知识库中有明确解答，请直接回答。",
        "medium": "知识库中有部分相关信息，请基于这些信息回答，并建议用户如未解决可联系售后。",
        "low": "知识库中暂无详细资料，请提供通用排查建议，并建议联系售后。",
    }
    parts.append(f"\n【检索置信度】{confidence}")
    parts.append(f"提示: {confidence_hints.get(confidence, confidence_hints['low'])}")
    parts.append(f"\n【用户问题】{question}")

    return "\n".join(parts)


def _extract_media_refs(text: str, image_refs: list, video_refs: list) -> dict:
    """扫描回答文本中出现的 [图片N] / [视频N] 占位符，返回被引用的媒体列表"""
    referenced_images = []
    referenced_videos = []

    for m in re.finditer(r"\[图片(\d+)\]", text):
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(image_refs) and image_refs[idx] not in referenced_images:
            referenced_images.append(image_refs[idx])

    for m in re.finditer(r"\[视频(\d+)\]", text):
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(video_refs) and video_refs[idx] not in referenced_videos:
            referenced_videos.append(video_refs[idx])

    return {"images": referenced_images, "videos": referenced_videos}


def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=LLM_MODEL,
        api_key=DASHSCOPE_API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=0.7,
        max_tokens=4096,
    )


def _get_redis_history(session_id: str):
    if not REDIS_URL:
        return None
    try:
        from src.context.history import get_chat_history
        return get_chat_history(session_id)
    except Exception as e:
        print(f"⚠️ Redis 对话历史不可用: {e}")
        return None


def _build_messages(system: str, history, user_prompt: str) -> list:
    messages = [{"role": "system", "content": system}]
    if history:
        try:
            for msg in history.messages[-20:]:
                role = "user" if msg.type == "human" else "assistant"
                messages.append({"role": role, "content": msg.content})
        except Exception:
            pass
    messages.append({"role": "user", "content": user_prompt})
    return messages


async def generate_multimodal_answer(
    question: str,
    parsed: dict,
    confidence: str,
    device_model: Optional[str] = None,
    session_id: Optional[str] = None,
) -> dict:
    """
    生成多模态回答（非流式）

    Returns:
        {
            "text_answer": str,
            "media": {
                "images": [{"url":..., "description":..., "filename":...}],
                "videos": [{"url":..., "description":..., "filename":..., "frame_urls":[...]}],
            }
        }
    """
    user_prompt = _build_prompt(question, parsed, confidence, device_model)
    history = _get_redis_history(session_id) if session_id else None
    messages = _build_messages(_SYSTEM_PROMPT, history, user_prompt)

    try:
        llm = _get_llm()
        lc_messages = [
            {"role": m["role"], "content": m["content"]} for m in messages
        ]
        response = await llm.ainvoke(lc_messages)
        text_answer = response.content.strip()
    except Exception as e:
        print(f"❌ 多模态 LLM 调用失败: {e}")
        text_answer = _fallback(confidence, parsed)

    media = _extract_media_refs(
        text_answer,
        parsed.get("image_refs", []),
        parsed.get("video_refs", []),
    )
    media["table_images"] = parsed.get("table_image_refs", [])

    if session_id and history:
        try:
            history.add_user_message(question)
            history.add_ai_message(text_answer)
        except Exception as e:
            print(f"⚠️ 保存对话历史失败: {e}")

    return {"text_answer": text_answer, "media": media}


async def generate_multimodal_answer_stream(
    question: str,
    parsed: dict,
    confidence: str,
    device_model: Optional[str] = None,
    session_id: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    """
    流式生成多模态回答。

    yield 规则：
    - 普通文本 chunk → yield str
    - 流结束后 → yield ("__MEDIA__", {images:[...], videos:[...]})
    """
    user_prompt = _build_prompt(question, parsed, confidence, device_model)
    history = _get_redis_history(session_id) if session_id else None
    messages = _build_messages(_SYSTEM_PROMPT, history, user_prompt)

    lc_messages = [{"role": m["role"], "content": m["content"]} for m in messages]

    full_answer = ""
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream(
                "POST",
                "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": LLM_MODEL,
                    "messages": lc_messages,
                    "max_tokens": 4096,
                    "temperature": 0.7,
                    "stream": True,
                },
            ) as response:
                if response.status_code != 200:
                    raise Exception(f"API 错误: {response.status_code}")
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        delta_content = json.loads(data_str)["choices"][0]["delta"].get("content", "")
                        if delta_content:
                            full_answer += delta_content
                            yield delta_content
                    except Exception:
                        continue
    except Exception as e:
        print(f"❌ 流式多模态 LLM 调用失败: {e}")
        if not full_answer:
            fallback = _fallback(confidence, parsed)
            full_answer = fallback
            yield fallback

    media = _extract_media_refs(
        full_answer,
        parsed.get("image_refs", []),
        parsed.get("video_refs", []),
    )
    media["table_images"] = parsed.get("table_image_refs", [])

    if session_id and history and full_answer:
        try:
            history.add_user_message(question)
            history.add_ai_message(full_answer)
        except Exception as e:
            print(f"⚠️ 保存对话历史失败: {e}")

    yield ("__MEDIA__", media)


def _fallback(confidence: str, parsed: dict) -> str:
    text_contexts = parsed.get("text_contexts", [])
    if text_contexts and confidence in ("high", "medium"):
        item = text_contexts[0]
        resp = "关于您的问题，参考以下信息：\n\n"
        if item.get("error_code"):
            resp += f"**错误码 {item['error_code']}**\n\n"
        resp += f"**{item.get('title', '')}**\n\n{item.get('content', '')}"
        if confidence == "medium":
            resp += "\n\n---\n💡 如以上信息未能解决您的问题，请联系售后支持。"
        return resp
    return (
        "抱歉，暂未找到该问题的详细解决方案。\n\n"
        "**通用排查建议：**\n"
        "1. 检查设备是否有明显的物理损坏\n"
        "2. 确认电源连接是否正常\n"
        "3. 尝试重启设备\n"
        "4. 查看设备显示屏上的完整错误信息\n\n"
        "如问题持续，请联系售后支持。"
    )