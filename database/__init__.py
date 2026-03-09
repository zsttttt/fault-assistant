from .db import init_db, save_conversation, update_feedback, get_unsolved_issues, add_knowledge
from .excel_importer import import_from_excel, preview_excel, get_excel_sheets


def get_all_knowledge():
    """从 Qdrant 获取所有知识条目（用于管理界面）"""
    from src.indexing.indexer import get_all_knowledge_entries
    return get_all_knowledge_entries()
