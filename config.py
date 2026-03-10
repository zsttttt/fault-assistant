"""
配置管理
"""
import os
from dotenv import load_dotenv

load_dotenv()

# LLM 配置
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "qwen")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3-max")
VLM_MODEL = os.getenv("VLM_MODEL", "qwen-vl-max")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# 服务配置
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# 数据库
DATABASE_PATH = os.getenv("DATABASE_PATH", "fault_assistant.db")

# 向量检索（DashScope 嵌入模型）
EMBEDDING_MODEL = "text-embedding-v4"
EMBEDDING_DIMENSION = 1024
SIMILARITY_THRESHOLD_HIGH = 0.75
SIMILARITY_THRESHOLD_LOW = 0.45

# Qdrant Cloud（向量数据库）
_qdrant_url_raw = os.getenv("QDRANT_URL", "")
# 自动补全 https:// 前缀
if _qdrant_url_raw and not _qdrant_url_raw.startswith(("http://", "https://")):
    _qdrant_url_raw = "https://" + _qdrant_url_raw
QDRANT_URL = _qdrant_url_raw
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "fault_assistant_rag")

# Redis（对话历史）
# 优先使用 REDIS_URL；若未设置则从 REDIS_HOST/PORT/PASSWORD 组装
_redis_url = os.getenv("REDIS_URL", "")
if not _redis_url:
    _redis_host = os.getenv("REDIS_HOST", "localhost")
    _redis_port = os.getenv("REDIS_PORT", "6379")
    _redis_password = os.getenv("REDIS_PASSWORD", "")
    if _redis_password:
        _redis_url = f"redis://:{_redis_password}@{_redis_host}:{_redis_port}"
    else:
        _redis_url = f"redis://{_redis_host}:{_redis_port}"
REDIS_URL = _redis_url
REDIS_TTL = int(os.getenv("REDIS_TTL", "86400"))
REDIS_CHAT_HISTORY_PREFIX = "chat_history:"

# 前台用户登录
USER_USERNAME = os.getenv("USER_USERNAME", "user")
USER_PASSWORD = os.getenv("USER_PASSWORD", "user123")
# 后台管理员登录
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")

# ============================================================
# 对象存储（图片/视频原始文件）
# 开发阶段：Backblaze B2；生产阶段：MinIO（只改下面的值，业务代码不动）
# ============================================================
# B2 endpoint 格式：https://s3.<region>.backblazeb2.com
_storage_endpoint_raw = os.getenv("B2_ENDPOINT", os.getenv("STORAGE_ENDPOINT", ""))
if _storage_endpoint_raw and not _storage_endpoint_raw.startswith(("http://", "https://")):
    _storage_endpoint_raw = "https://" + _storage_endpoint_raw
STORAGE_ENDPOINT = _storage_endpoint_raw
# B2 Key ID（applicationKeyId）
STORAGE_ACCESS_KEY = os.getenv("B2_KEY_ID", os.getenv("STORAGE_ACCESS_KEY", ""))
# B2 Application Key
STORAGE_SECRET_KEY = os.getenv("B2_APP_KEY", os.getenv("STORAGE_SECRET_KEY", ""))
STORAGE_BUCKET = os.getenv("B2_BUCKET_NAME", os.getenv("STORAGE_BUCKET", "after-sales-media"))
# B2 region 格式：us-west-004 / eu-central-003 等（与 endpoint 中保持一致）
STORAGE_REGION = os.getenv("B2_REGION", os.getenv("STORAGE_REGION", "us-west-004"))
