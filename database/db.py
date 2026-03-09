"""
数据库操作模块
"""
import sqlite3
from config import DATABASE_PATH


def get_connection():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS knowledge (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            error_code TEXT,
            keywords TEXT,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            device_models TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            effectiveness_score INTEGER DEFAULT 0
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            device_model TEXT,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            confidence TEXT,
            solved INTEGER DEFAULT NULL,
            feedback_text TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.commit()
    conn.close()
    print("✅ 数据库初始化完成")


def save_conversation(conversation_id: str, device_model: str, question: str, answer: str, confidence: str):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO conversations (id, device_model, question, answer, confidence)
        VALUES (?, ?, ?, ?, ?)
    """, (conversation_id, device_model, question, answer, confidence))
    conn.commit()
    conn.close()


def update_feedback(conversation_id: str, solved: bool, feedback_text: str = None):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE conversations SET solved = ?, feedback_text = ? WHERE id = ?
    """, (1 if solved else 0, feedback_text, conversation_id))
    conn.commit()
    conn.close()


def get_unsolved_issues():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, device_model, question, answer, created_at
        FROM conversations WHERE solved = 0 ORDER BY created_at DESC
    """)
    results = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return results


def add_knowledge(error_code: str, title: str, content: str, keywords: str = "", device_models: str = "") -> str:
    """委托到 Qdrant 索引模块（向量存储 + InMemoryStore）"""
    from src.indexing.indexer import add_knowledge_entry
    return add_knowledge_entry(error_code, title, content, keywords, device_models)
