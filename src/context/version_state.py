"""
会话级版本状态管理
每个 session_id 对应一个 current_version，存储在 Redis 中。
若 Redis 不可用则降级为进程内字典（重启后丢失）。
"""
from typing import Optional
from config import REDIS_URL, REDIS_TTL

_VERSION_KEY_PREFIX = "version_state:"
_fallback_store: dict = {}


def _get_redis():
    if not REDIS_URL:
        return None
    try:
        import redis
        return redis.from_url(REDIS_URL, decode_responses=True)
    except Exception as e:
        print(f"⚠️ Redis 版本状态不可用: {e}")
        return None


def get_session_version(session_id: str) -> str:
    """获取会话当前版本号，未设置则返回空字符串"""
    r = _get_redis()
    if r:
        try:
            return r.get(f"{_VERSION_KEY_PREFIX}{session_id}") or ""
        except Exception:
            pass
    return _fallback_store.get(session_id, "")


def set_session_version(session_id: str, version_code: str) -> None:
    """设置会话当前版本号，与对话历史使用相同 TTL"""
    r = _get_redis()
    if r:
        try:
            r.set(f"{_VERSION_KEY_PREFIX}{session_id}", version_code, ex=REDIS_TTL)
            return
        except Exception:
            pass
    _fallback_store[session_id] = version_code


def detect_version_in_text(text: str) -> Optional[str]:
    """
    从用户消息中识别版本号。
    策略1：从已注册版本中查找（按版本号长度降序，避免短码提前匹配长码）。
    策略2：正则兜底，提取版本号格式的字符串（4-8 位数字 + 可选大写字母后缀，
           如 1101、110101、110102V），适用于用户提供了未注册的版本号的情况。
    """
    import re

    # 策略1：已注册版本（按长度降序，防止 "1101" 先于 "110102V" 匹配）
    try:
        from database.version_registry import get_all_versions
        versions = sorted(get_all_versions(), key=lambda v: len(v.get("version_code", "")), reverse=True)
        for v in versions:
            code = v["version_code"]
            if code and code in text:
                return code
    except Exception:
        pass

    # 策略2：正则兜底（匹配 4-8 位数字 + 可选大写字母，如 1101、110101、110102V）
    match = re.search(r'(?<![.\d])(\d{4,8}[A-Z]*)(?![.\d])', text)
    if match:
        return match.group(1)

    return None