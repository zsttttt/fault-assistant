"""
配置管理
"""
import os
from dotenv import load_dotenv

load_dotenv()

# LLM 配置
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "qwen")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# 服务配置
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# 数据库
DATABASE_PATH = os.getenv("DATABASE_PATH", "fault_assistant.db")

# 向量检索
EMBEDDING_MODEL = "text-embedding-v4"
SIMILARITY_THRESHOLD_HIGH = 0.75
SIMILARITY_THRESHOLD_LOW = 0.45
