"""
知识库迁移脚本
将 SQLite 中的现有知识条目迁移到 Qdrant + InMemoryStore

用法：
    python migrate_knowledge.py

前提：
    1. .env 中已配置 QDRANT_URL 和 QDRANT_API_KEY
    2. Qdrant Cloud collection 已创建（运行本脚本会自动创建）
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv()

from config import QDRANT_URL, QDRANT_API_KEY, DATABASE_PATH


def migrate():
    if not QDRANT_URL or QDRANT_URL == "https://your-cluster.cloud.qdrant.io:6333":
        print("❌ 请先在 .env 中配置有效的 QDRANT_URL 和 QDRANT_API_KEY")
        sys.exit(1)

    import sqlite3
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='knowledge'")
    if not cursor.fetchone():
        print("⚠️ SQLite 中没有 knowledge 表，无需迁移")
        conn.close()
        return

    cursor.execute("SELECT COUNT(*) as cnt FROM knowledge")
    total = cursor.fetchone()["cnt"]
    if total == 0:
        print("⚠️ SQLite knowledge 表为空，无需迁移")
        conn.close()
        return

    print(f"📦 发现 {total} 条知识，开始迁移到 Qdrant...")

    from src.indexing.indexer import init_indexer, add_knowledge_entry
    init_indexer()

    cursor.execute("SELECT * FROM knowledge ORDER BY id")
    rows = cursor.fetchall()
    conn.close()

    success = 0
    failed = 0
    for row in rows:
        item = dict(row)
        try:
            add_knowledge_entry(
                error_code=item.get("error_code") or "",
                title=item.get("title") or "",
                content=item.get("content") or "",
                keywords=item.get("keywords") or "",
                device_models=item.get("device_models") or "",
                created_at=item.get("created_at") or "",
            )
            success += 1
            print(f"  ✅ [{success}/{total}] {item.get('title', '无标题')}")
        except Exception as e:
            failed += 1
            print(f"  ❌ 失败: {item.get('title', '无标题')} - {e}")

    print(f"\n迁移完成: 成功 {success} 条，失败 {failed} 条")


if __name__ == "__main__":
    migrate()
