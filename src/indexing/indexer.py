"""
Qdrant + InMemoryStore 知识索引模块

架构：
- Qdrant Cloud：存储向量嵌入 + payload（含原始内容，用于重建 docstore）
- InMemoryStore：运行时文档存储，启动时从 Qdrant payload 恢复
- MultiVectorRetriever：摘要检索 + 原文生成

知识条目存储格式：
  vectorstore：搜索文本（error_code + title + keywords + content）
  docstore：原始 content（Document 对象）
"""
import uuid
from datetime import datetime
from typing import List, Optional, Tuple

from langchain_core.stores import InMemoryBaseStore
from langchain_core.documents import Document
from langchain_classic.retrievers.multi_vector import MultiVectorRetriever
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, Filter, FieldCondition, MatchValue, FilterSelector

from config import (
    QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION_NAME,
    EMBEDDING_MODEL, EMBEDDING_DIMENSION, DASHSCOPE_API_KEY,
)

ID_KEY = "doc_id"

_qdrant_client: Optional[QdrantClient] = None
_embeddings: Optional[DashScopeEmbeddings] = None
_vectorstore: Optional[QdrantVectorStore] = None
_docstore: InMemoryBaseStore = InMemoryBaseStore()
_retriever: Optional[MultiVectorRetriever] = None


def get_qdrant_client() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    return _qdrant_client


def get_embeddings() -> DashScopeEmbeddings:
    global _embeddings
    if _embeddings is None:
        import os
        os.environ.setdefault("DASHSCOPE_API_KEY", DASHSCOPE_API_KEY)
        _embeddings = DashScopeEmbeddings(model=EMBEDDING_MODEL)
    return _embeddings


def ensure_collection():
    client = get_qdrant_client()
    existing = [c.name for c in client.get_collections().collections]
    if QDRANT_COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_DIMENSION, distance=Distance.COSINE),
        )
        print(f"✅ 已创建 Qdrant collection: {QDRANT_COLLECTION_NAME}")
    client.create_payload_index(
        collection_name=QDRANT_COLLECTION_NAME,
        field_name="metadata.source",
        field_schema="keyword",
    )
    client.create_payload_index(
        collection_name=QDRANT_COLLECTION_NAME,
        field_name="metadata.type",
        field_schema="keyword",
    )
    client.create_payload_index(
        collection_name=QDRANT_COLLECTION_NAME,
        field_name="metadata.doc_id",
        field_schema="keyword",
    )
    client.create_payload_index(
        collection_name=QDRANT_COLLECTION_NAME,
        field_name="metadata.version",
        field_schema="keyword",
    )
    client.create_payload_index(
        collection_name=QDRANT_COLLECTION_NAME,
        field_name="metadata.doc_type",
        field_schema="keyword",
    )


def get_vectorstore() -> QdrantVectorStore:
    global _vectorstore
    if _vectorstore is None:
        ensure_collection()
        _vectorstore = QdrantVectorStore(
            client=get_qdrant_client(),
            collection_name=QDRANT_COLLECTION_NAME,
            embedding=get_embeddings(),
        )
    return _vectorstore


def get_docstore() -> InMemoryBaseStore:
    return _docstore


def get_multi_vector_retriever() -> MultiVectorRetriever:
    global _retriever
    if _retriever is None:
        _retriever = MultiVectorRetriever(
            vectorstore=get_vectorstore(),
            docstore=get_docstore(),
            id_key=ID_KEY,
        )
    return _retriever


def _populate_docstore_from_qdrant():
    """应用启动时，从 Qdrant payload 恢复 InMemoryStore（避免重建嵌入）"""
    client = get_qdrant_client()
    pairs: List[Tuple[str, Document]] = []
    offset = None

    while True:
        points, next_offset = client.scroll(
            collection_name=QDRANT_COLLECTION_NAME,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        for point in points:
            metadata = point.payload.get("metadata", {})
            doc_id = metadata.get(ID_KEY)
            content = metadata.get("original_content", "")
            if doc_id and content:
                doc_meta = {k: v for k, v in metadata.items() if k != "original_content"}
                pairs.append((doc_id, Document(page_content=content, metadata=doc_meta)))

        if next_offset is None:
            break
        offset = next_offset

    if pairs:
        _docstore.mset(pairs)
        print(f"✅ 从 Qdrant 恢复 {len(pairs)} 条文档到内存存储")


def _migrate_from_sqlite():
    """首次运行时，将 SQLite 知识库迁移到 Qdrant（自动执行一次）"""
    try:
        import sqlite3
        from config import DATABASE_PATH

        conn = sqlite3.connect(DATABASE_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='knowledge'")
        if not cursor.fetchone():
            conn.close()
            return

        cursor.execute("SELECT * FROM knowledge ORDER BY id")
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return

        print(f"🔄 正在将 {len(rows)} 条 SQLite 知识迁移到 Qdrant...")
        for row in rows:
            item = dict(row)
            add_knowledge_entry(
                error_code=item.get("error_code") or "",
                title=item.get("title") or "",
                content=item.get("content") or "",
                keywords=item.get("keywords") or "",
                device_models=item.get("device_models") or "",
                created_at=item.get("created_at") or "",
            )
        print(f"✅ 迁移完成：{len(rows)} 条知识已导入 Qdrant")
    except Exception as e:
        print(f"⚠️ SQLite 迁移跳过: {e}")


def init_indexer():
    """初始化索引模块，在应用启动时调用"""
    if not QDRANT_URL:
        print("⚠️ 未配置有效 QDRANT_URL，跳过 Qdrant 初始化（检索功能降级为不可用）")
        return

    print("🔄 正在初始化 Qdrant 连接...")
    try:
        ensure_collection()
        client = get_qdrant_client()
        count = client.count(collection_name=QDRANT_COLLECTION_NAME).count

        if count == 0:
            _migrate_from_sqlite()
        else:
            print(f"✅ Qdrant collection 已有 {count} 条向量")
            _populate_docstore_from_qdrant()

        get_multi_vector_retriever()
        print("✅ Qdrant 索引模块初始化完成")
    except Exception as e:
        print(f"❌ Qdrant 初始化失败: {e}")


def add_knowledge_entry(
    error_code: str,
    title: str,
    content: str,
    keywords: str = "",
    device_models: str = "",
    created_at: str = "",
) -> str:
    """添加知识条目到 Qdrant vectorstore + InMemoryStore"""
    if not QDRANT_URL:
        raise RuntimeError("未配置有效的 QDRANT_URL，无法存储知识")

    doc_id = str(uuid.uuid4())
    ts = created_at or datetime.utcnow().isoformat()
    search_text = " ".join(filter(None, [error_code, title, keywords, content])).strip()

    search_doc = Document(
        page_content=search_text,
        metadata={
            ID_KEY: doc_id,
            "original_content": content,
            "error_code": error_code or "",
            "keywords": keywords or "",
            "title": title,
            "device_models": device_models or "",
            "created_at": ts,
            "type": "knowledge_entry",
        },
    )
    get_vectorstore().add_documents([search_doc])

    original_doc = Document(
        page_content=content,
        metadata={
            ID_KEY: doc_id,
            "error_code": error_code or "",
            "title": title,
            "device_models": device_models or "",
            "created_at": ts,
            "type": "knowledge_entry",
        },
    )
    _docstore.mset([(doc_id, original_doc)])

    return doc_id


def get_all_knowledge_entries() -> List[dict]:
    """从 Qdrant 获取所有知识条目（用于管理界面列表）"""
    if not QDRANT_URL:
        return []

    client = get_qdrant_client()
    results = []
    offset = None

    while True:
        points, next_offset = client.scroll(
            collection_name=QDRANT_COLLECTION_NAME,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        for point in points:
            metadata = point.payload.get("metadata", {})
            if metadata.get("type") == "knowledge_entry":
                results.append({
                    "id": metadata.get(ID_KEY, str(point.id)),
                    "error_code": metadata.get("error_code", ""),
                    "keywords": metadata.get("keywords", ""),
                    "title": metadata.get("title", ""),
                    "content": metadata.get("original_content", ""),
                    "device_models": metadata.get("device_models", ""),
                    "created_at": metadata.get("created_at", ""),
                    "updated_at": metadata.get("created_at", ""),
                    "effectiveness_score": 0,
                })

        if next_offset is None:
            break
        offset = next_offset

    return results


def search_by_error_code(error_code: str) -> List[dict]:
    """按错误码精确过滤（用于检索增强）"""
    if not QDRANT_URL:
        return []

    client = get_qdrant_client()
    try:
        points, _ = client.scroll(
            collection_name=QDRANT_COLLECTION_NAME,
            scroll_filter=Filter(
                must=[FieldCondition(key="metadata.error_code", match=MatchValue(value=error_code))]
            ),
            limit=10,
            with_payload=True,
            with_vectors=False,
        )
        return [
            {"id": p.payload.get("metadata", {}).get(ID_KEY, str(p.id)),
             "error_code": p.payload.get("metadata", {}).get("error_code", "")}
            for p in points
        ]
    except Exception:
        return []


def _delete_by_source(source_file: str):
    """删除 Qdrant 中指定来源文件的所有向量，并同步清理 InMemoryStore"""
    if not source_file:
        return
    client = get_qdrant_client()
    offset = None
    doc_ids_to_delete = []
    while True:
        points, next_offset = client.scroll(
            collection_name=QDRANT_COLLECTION_NAME,
            scroll_filter=Filter(
                must=[FieldCondition(key="metadata.source", match=MatchValue(value=source_file))]
            ),
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            doc_id = point.payload.get("metadata", {}).get(ID_KEY)
            if doc_id:
                doc_ids_to_delete.append(doc_id)
        if next_offset is None:
            break
        offset = next_offset

    if doc_ids_to_delete:
        client.delete(
            collection_name=QDRANT_COLLECTION_NAME,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[FieldCondition(key="metadata.source", match=MatchValue(value=source_file))]
                )
            ),
        )
        _docstore.mdelete(doc_ids_to_delete)
        print(f"🗑️ 已删除来源 '{source_file}' 的 {len(doc_ids_to_delete)} 条旧向量")


def index_document_elements(
    text_chunks: List[str],
    tables: List[str],
    table_summaries: List[str],
    images: List[dict],
    image_descriptions: List[str],
    source_file: str = "",
    table_image_urls: Optional[List[Optional[str]]] = None,
    table_image_object_keys: Optional[List[Optional[str]]] = None,
    version: str = "",
    doc_type: str = "",
):
    """索引 Docling 解析后的文档元素（文本/表格/图片），同名文件自动覆盖旧数据"""
    _delete_by_source(source_file)
    retriever = get_multi_vector_retriever()

    if text_chunks:
        text_ids = [str(uuid.uuid4()) for _ in text_chunks]
        retriever.vectorstore.add_documents([
            Document(
                page_content=chunk,
                metadata={
                    ID_KEY: text_ids[i],
                    "original_content": chunk,
                    "source": source_file,
                    "type": "text_chunk",
                    "version": version,
                    "doc_type": doc_type,
                },
            )
            for i, chunk in enumerate(text_chunks)
        ])
        _docstore.mset([
            (text_ids[i], Document(page_content=chunk, metadata={"source": source_file, "type": "text_chunk", "version": version, "doc_type": doc_type}))
            for i, chunk in enumerate(text_chunks)
        ])

    if tables and table_summaries:
        table_ids = [str(uuid.uuid4()) for _ in tables]
        retriever.vectorstore.add_documents([
            Document(
                page_content=summary,
                metadata={
                    ID_KEY: table_ids[i],
                    "original_content": tables[i],
                    "source": source_file,
                    "type": "table",
                    "table_image_url": (table_image_urls[i] if table_image_urls and i < len(table_image_urls) else None) or "",
                    "table_image_object_key": (table_image_object_keys[i] if table_image_object_keys and i < len(table_image_object_keys) else None) or "",
                    "version": version,
                    "doc_type": doc_type,
                },
            )
            for i, summary in enumerate(table_summaries)
        ])
        _docstore.mset([
            (table_ids[i], Document(page_content=tables[i], metadata={"source": source_file, "type": "table", "version": version, "doc_type": doc_type}))
            for i, _ in enumerate(tables)
        ])

    if images and image_descriptions:
        image_ids = [str(uuid.uuid4()) for _ in images]
        retriever.vectorstore.add_documents([
            Document(
                page_content=desc,
                metadata={
                    ID_KEY: image_ids[i],
                    "original_content": str(images[i]),
                    "source": source_file,
                    "type": "image",
                    "version": version,
                    "doc_type": doc_type,
                },
            )
            for i, desc in enumerate(image_descriptions)
        ])
        _docstore.mset([
            (image_ids[i], Document(page_content=str(img), metadata={"source": source_file, "type": "image", "version": version, "doc_type": doc_type}))
            for i, img in enumerate(images)
        ])


def get_all_documents() -> List[dict]:
    """按来源文件聚合文档分块统计（仅文档上传类型，不含手动知识条目）"""
    if not QDRANT_URL:
        return []

    client = get_qdrant_client()
    counts: dict = {}
    offset = None

    while True:
        points, next_offset = client.scroll(
            collection_name=QDRANT_COLLECTION_NAME,
            limit=200,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            metadata = point.payload.get("metadata", {})
            type_ = metadata.get("type", "")
            source = metadata.get("source", "")
            if type_ not in ("text_chunk", "table", "image") or not source:
                continue
            entry = counts.setdefault(source, {"text_chunks": 0, "tables": 0, "images": 0})
            if type_ == "text_chunk":
                entry["text_chunks"] += 1
            elif type_ == "table":
                entry["tables"] += 1
            elif type_ == "image":
                entry["images"] += 1

        if next_offset is None:
            break
        offset = next_offset

    return [
        {
            "source": src,
            "text_chunks": v["text_chunks"],
            "tables": v["tables"],
            "images": v["images"],
            "total": v["text_chunks"] + v["tables"] + v["images"],
        }
        for src, v in sorted(counts.items())
    ]


def get_image_object_keys_by_source(source_file: str) -> List[str]:
    """获取指定来源文件的所有图片 object_key（用于 B2 删除）"""
    if not QDRANT_URL or not source_file:
        return []

    client = get_qdrant_client()
    object_keys: List[str] = []
    offset = None

    while True:
        points, next_offset = client.scroll(
            collection_name=QDRANT_COLLECTION_NAME,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="metadata.source", match=MatchValue(value=source_file)),
                    FieldCondition(key="metadata.type", match=MatchValue(value="image")),
                ]
            ),
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            key = point.payload.get("metadata", {}).get("object_key")
            if key:
                object_keys.append(key)

        if next_offset is None:
            break
        offset = next_offset

    return object_keys


def delete_knowledge_entry(doc_id: str) -> bool:
    """从 Qdrant 和 InMemoryStore 删除指定知识条目"""
    if not QDRANT_URL:
        raise RuntimeError("未配置有效的 QDRANT_URL")

    client = get_qdrant_client()
    client.delete(
        collection_name=QDRANT_COLLECTION_NAME,
        points_selector=FilterSelector(
            filter=Filter(
                must=[FieldCondition(key=f"metadata.{ID_KEY}", match=MatchValue(value=doc_id))]
            )
        ),
    )
    _docstore.mdelete([doc_id])
    return True
