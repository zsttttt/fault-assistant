"""
LLM 回答生成模块 - 支持通义千问
"""
from typing import List, Optional
import httpx

from config import LLM_PROVIDER, DASHSCOPE_API_KEY, GEMINI_API_KEY, ANTHROPIC_API_KEY


SYSTEM_PROMPT = """你是一个专业的加油机和充电桩设备故障诊断助手。你的任务是帮助用户解决设备故障问题。

## 工作原则：
1. 根据提供的知识库内容回答用户问题
2. 如果知识库有明确答案，直接给出解决步骤
3. 如果知识库信息不完全匹配，可以基于相关信息推理，但要说明"以下建议基于相似故障经验，请谨慎操作"
4. 如果完全没有相关信息，诚实告知"暂无该故障的详细资料"，并提供通用排查建议
5. 涉及高压电、燃油等危险操作时，务必提醒安全注意事项

## 回答格式：
- 先简述可能原因（1-2句话）
- 再给出具体操作步骤（分步骤列出）
- 最后提醒注意事项或安全警告（如有必要）

## 语言风格：
- 使用简洁、易懂的语言
- 避免过于专业的术语，或在使用时给出解释
- 保持友好、耐心的态度"""


def _build_user_prompt(question: str, context: List[dict], confidence: str, device_model: Optional[str]) -> str:
    prompt_parts = []
    
    if device_model:
        prompt_parts.append(f"【当前设备型号】{device_model}")
    
    if context:
        prompt_parts.append("【相关知识库内容】")
        for i, item in enumerate(context, 1):
            prompt_parts.append(f"\n--- 参考资料 {i} ---")
            if item.get('error_code'):
                prompt_parts.append(f"错误码: {item['error_code']}")
            prompt_parts.append(f"标题: {item['title']}")
            prompt_parts.append(f"内容: {item['content']}")
    else:
        prompt_parts.append("【相关知识库内容】未找到直接相关的故障记录")
    
    confidence_hints = {
        "high": "知识库中有该问题的明确解答，请直接回答。",
        "medium": "知识库中有部分相关信息，请基于这些信息回答，并提醒用户如未解决可联系售后。",
        "low": "知识库中暂无该问题的详细资料，请提供通用排查建议，并建议用户联系售后获取专业支持。"
    }
    prompt_parts.append(f"\n【检索置信度】{confidence}")
    prompt_parts.append(f"提示: {confidence_hints.get(confidence, confidence_hints['low'])}")
    prompt_parts.append(f"\n【用户问题】{question}")
    
    return "\n".join(prompt_parts)


async def _call_qwen(user_prompt: str) -> str:
    """调用通义千问 API (OpenAI 兼容格式)"""
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "qwen-turbo",
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                "max_tokens": 1024,
                "temperature": 0.7
            }
        )
        
        if response.status_code != 200:
            raise Exception(f"通义千问 API 错误: {response.status_code} - {response.text}")
        
        result = response.json()
        return result["choices"][0]["message"]["content"]


async def _call_gemini(user_prompt: str) -> str:
    """调用 Gemini API"""
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}",
            headers={"Content-Type": "application/json"},
            json={
                "contents": [{"parts": [{"text": f"{SYSTEM_PROMPT}\n\n{user_prompt}"}]}],
                "generationConfig": {"temperature": 0.7, "maxOutputTokens": 1024}
            }
        )
        
        if response.status_code != 200:
            raise Exception(f"Gemini API 错误: {response.status_code} - {response.text}")
        
        result = response.json()
        return result["candidates"][0]["content"]["parts"][0]["text"]


async def _call_claude(user_prompt: str) -> str:
    """调用 Claude API"""
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            },
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 1024,
                "system": SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": user_prompt}]
            }
        )
        
        if response.status_code != 200:
            raise Exception(f"Claude API 错误: {response.status_code} - {response.text}")
        
        result = response.json()
        return result["content"][0]["text"]


def _fallback_response(question: str, context: List[dict], confidence: str) -> str:
    """备用回答"""
    if context and confidence in ["high", "medium"]:
        item = context[0]
        response = f"关于您的问题，参考以下信息：\n\n"
        if item.get('error_code'):
            response += f"**错误码 {item['error_code']}**\n\n"
        response += f"**{item['title']}**\n\n{item['content']}"
        if confidence == "medium":
            response += "\n\n---\n💡 如以上信息未能解决您的问题，请联系售后支持。"
        return response
    else:
        return """抱歉，暂未找到该问题的详细解决方案。

**通用排查建议：**
1. 检查设备是否有明显的物理损坏
2. 确认电源连接是否正常
3. 尝试重启设备
4. 查看设备显示屏上的完整错误信息

如问题持续，请联系售后支持，我们会尽快为您处理。"""


async def generate_answer(
    question: str, 
    context: List[dict], 
    confidence: str, 
    device_model: Optional[str] = None
) -> str:
    """生成回答"""
    user_prompt = _build_user_prompt(question, context, confidence, device_model)
    
    try:
        if LLM_PROVIDER == "qwen":
            if not DASHSCOPE_API_KEY:
                print("⚠️ 未配置 DASHSCOPE_API_KEY，使用备用回答")
                return _fallback_response(question, context, confidence)
            return await _call_qwen(user_prompt)
        
        elif LLM_PROVIDER == "gemini":
            if not GEMINI_API_KEY:
                print("⚠️ 未配置 GEMINI_API_KEY，使用备用回答")
                return _fallback_response(question, context, confidence)
            return await _call_gemini(user_prompt)
        
        elif LLM_PROVIDER == "claude":
            if not ANTHROPIC_API_KEY:
                print("⚠️ 未配置 ANTHROPIC_API_KEY，使用备用回答")
                return _fallback_response(question, context, confidence)
            return await _call_claude(user_prompt)
        
        else:
            return _fallback_response(question, context, confidence)
    
    except Exception as e:
        print(f"❌ LLM 调用失败: {e}")
        return _fallback_response(question, context, confidence)
