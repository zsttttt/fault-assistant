"""
Redis 对话历史管理
为每个 session_id 维护独立的多轮对话记录
"""
from langchain_community.chat_message_histories import RedisChatMessageHistory
from config import REDIS_URL, REDIS_CHAT_HISTORY_PREFIX, REDIS_TTL


def get_chat_history(session_id: str) -> RedisChatMessageHistory:
    """获取指定会话的 Redis 对话历史（自动按 TTL 过期）"""
    return RedisChatMessageHistory(
        session_id=session_id,
        url=REDIS_URL,
        key_prefix=REDIS_CHAT_HISTORY_PREFIX,
        ttl=REDIS_TTL,
    )