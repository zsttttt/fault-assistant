"""
知识检索模块 - 基于 Qdrant 向量检索
替换原有的 DashScope API + numpy 内存方案
"""
import re
from typing import List, Tuple, Optional

from langchain_core.documents import Document

from config import SIMILARITY_THRESHOLD_HIGH, SIMILARITY_THRESHOLD_LOW, QDRANT_URL, QDRANT_COLLECTION_NAME

_TEXT_TOP_K = 3
_IMAGE_QUOTA = 2


class KnowledgeRetriever:
    def __init__(self):
        print("🔄 正在初始化 Qdrant 检索器...")
        if not QDRANT_URL:
            print("⚠️ 未配置有效 QDRANT_URL，检索功能降级为不可用")
            self._available = False
            return

        from src.indexing.indexer import get_vectorstore, get_docstore
        self._vectorstore = get_vectorstore()
        self._docstore = get_docstore()
        self._available = True
        print("✅ Qdrant 检索器初始化完成")

    def reload_knowledge(self):
        """Qdrant 持久化存储，无需重新加载嵌入"""
        pass

    def _extract_error_code(self, query: str) -> Optional[str]:
        patterns = [
            r'[Ee][-_]?\d{1,4}',
            r'[Ee][Rr][Rr][-_]?\d{1,4}',
            r'0x[0-9A-Fa-f]{1,4}',
            r'错误码?\s*[:：]?\s*([Ee]?\d{1,4})',
            r'故障码?\s*[:：]?\s*([Ee]?\d{1,4})',
            r'代码\s*[:：]?\s*([Ee]?\d{1,4})',
        ]
        for pattern in patterns:
            match = re.search(pattern, query)
            if match:
                result = match.group(1) if match.lastindex else match.group(0)
                result = result.strip().upper()
                if not result.startswith('E') and result.isdigit():
                    result = 'E' + result
                return result
        return None

    def _keyword_supplement(
        self, query: str, version_code: str, exclude_ids: set, priority: str
    ) -> Tuple[List[dict], List[dict]]:
        """关键词扫描补充：弥补短查询与长文本嵌入余弦相似度偏低导致的召回损失"""
        from src.indexing.indexer import get_qdrant_client, ID_KEY
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        cn_chars = [c for c in query if '\u4e00' <= c <= '\u9fff']
        bigrams = list({cn_chars[i] + cn_chars[i + 1] for i in range(len(cn_chars) - 1)})
        if not bigrams:
            return [], []

        client = get_qdrant_client()
        text_cands: List[dict] = []
        image_cands: List[dict] = []
        offset = None

        while True:
            points, next_offset = client.scroll(
                collection_name=QDRANT_COLLECTION_NAME,
                scroll_filter=Filter(should=[
                    FieldCondition(key="metadata.version", match=MatchValue(value=version_code)),
                    FieldCondition(key="metadata.version", match=MatchValue(value="")),
                ]),
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                meta = point.payload.get("metadata", {})
                doc_id = meta.get(ID_KEY)
                if not doc_id or doc_id in exclude_ids:
                    continue
                page_content = point.payload.get("page_content", "")
                matched = sum(1 for bg in bigrams if bg in page_content)
                if matched == 0:
                    continue
                score = min(0.5 + 0.1 * matched, 0.9)
                entry = {"doc_id": doc_id, "score": score, "meta": meta, "priority": priority}
                if meta.get("type") == "image":
                    image_cands.append(entry)
                else:
                    text_cands.append(entry)

            if next_offset is None:
                break
            offset = next_offset

        print(f"[KEYWORD] version={version_code!r}  text={len(text_cands)}  image={len(image_cands)}", flush=True)
        return text_cands, image_cands

    def _expand_image_group(self, group_id: str, exclude_ids: set) -> List[dict]:
        """从 Qdrant 获取同 group_id 的所有兄弟图片，排除已收录的 ID"""
        from src.indexing.indexer import get_qdrant_client, ID_KEY
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        client = get_qdrant_client()
        try:
            points, _ = client.scroll(
                collection_name=QDRANT_COLLECTION_NAME,
                scroll_filter=Filter(must=[
                    FieldCondition(key="metadata.group_id", match=MatchValue(value=group_id)),
                    FieldCondition(key="metadata.type", match=MatchValue(value="image")),
                ]),
                limit=20,
                with_payload=True,
                with_vectors=False,
            )
        except Exception:
            return []

        siblings = []
        for point in points:
            meta = point.payload.get("metadata", {})
            doc_id = meta.get(ID_KEY)
            if not doc_id or doc_id in exclude_ids:
                continue
            siblings.append({
                "id": doc_id,
                "error_code": "",
                "keywords": "",
                "title": "",
                "content": meta.get("original_content", ""),
                "device_models": "",
                "similarity_score": 0.0,
                "type": "image",
                "media_url": meta.get("media_url", ""),
                "object_key": meta.get("object_key", ""),
            })
        return siblings

    def _search_with_version_filter(
        self, query: str, version_code: str, k: int, priority: str
    ) -> Tuple[List[dict], List[dict]]:
        """带版本过滤的向量检索，复用 vectorstore 内置嵌入实例。
        返回 (text_cands, image_cands)。
        """
        from qdrant_client.models import Filter, FieldCondition, MatchValue, SearchParams

        from qdrant_client.models import Should
        version_filter = Filter(
            should=[
                FieldCondition(key="metadata.version", match=MatchValue(value=version_code)),
                FieldCondition(key="metadata.version", match=MatchValue(value="")),
            ]
        )
        print(f"[FILTER] version_code={version_code!r}  k={k}", flush=True)
        try:
            docs_with_scores = self._vectorstore.similarity_search_with_relevance_scores(
                query, k=k, filter=version_filter,
                search_params=SearchParams(exact=True),
            )
        except Exception as e:
            print(f"[FILTER] ERROR: {e}", flush=True)
            return [], []

        print(f"[FILTER] found={len(docs_with_scores)}"
              + (f"  top_score={docs_with_scores[0][1]:.3f}" if docs_with_scores else "  (empty)"),
              flush=True)

        text_cands: List[dict] = []
        image_cands: List[dict] = []
        for doc, score in docs_with_scores:
            doc_id = doc.metadata.get("doc_id")
            if not doc_id:
                continue
            entry = {
                "doc_id": doc_id,
                "score": float(score),
                "meta": doc.metadata,
                "priority": priority,
            }
            if doc.metadata.get("type") == "image":
                image_cands.append(entry)
            else:
                text_cands.append(entry)

        # 关键词补充检索，弥补短查询对长文本嵌入相似度偏低的召回损失
        kw_seen = {c["doc_id"] for c in text_cands + image_cands}
        kw_text, kw_img = self._keyword_supplement(query, version_code, kw_seen, priority)
        text_cands.extend(kw_text)
        image_cands.extend(kw_img)

        text_cands.sort(key=lambda x: x["score"], reverse=True)
        image_cands.sort(key=lambda x: x["score"], reverse=True)

        return text_cands, image_cands

    def retrieve(self, query: str, top_k: int = _TEXT_TOP_K, version_code: str = "") -> Tuple[List[dict], str]:
        print(f"[RETRIEVE] query={query!r}  version_code={version_code!r}", flush=True)
        if not self._available:
            return [], "low"

        error_code = self._extract_error_code(query)
        chain: List[str] = []

        # 版本感知检索
        if version_code:
            from database.version_registry import get_version_chain
            chain = get_version_chain(version_code)
            # chain 从新到旧：chain[0] 是用户当前版本（高优先级），后续祖先版本依次补充
            seen_ids: set = set()
            text_cands: List[dict] = []
            image_cands: List[dict] = []
            for i, vc in enumerate(chain):
                priority = "high" if i == 0 else "low"
                t, img = self._search_with_version_filter(
                    query, vc, (top_k + _IMAGE_QUOTA) * 2, priority
                )
                for c in t:
                    if c["doc_id"] not in seen_ids:
                        seen_ids.add(c["doc_id"])
                        text_cands.append(c)
                for c in img:
                    if c["doc_id"] not in seen_ids:
                        seen_ids.add(c["doc_id"])
                        image_cands.append(c)
        else:
            # 无版本号：原有全量检索逻辑
            try:
                docs_with_scores = self._vectorstore.similarity_search_with_relevance_scores(
                    query, k=(top_k + _IMAGE_QUOTA) * 2
                )
            except Exception as e:
                print(f"❌ Qdrant 检索失败: {e}")
                return [], "low"

            seen_ids = set()
            text_cands = []
            image_cands = []
            for doc, score in docs_with_scores:
                doc_id = doc.metadata.get("doc_id")
                if not doc_id or doc_id in seen_ids:
                    continue
                seen_ids.add(doc_id)
                entry = {"doc_id": doc_id, "score": float(score), "meta": doc.metadata, "priority": ""}
                if doc.metadata.get("type") == "image":
                    image_cands.append(entry)
                else:
                    text_cands.append(entry)

        if error_code:
            try:
                from src.indexing.indexer import search_by_error_code
                exact_matches = search_by_error_code(error_code)
                exact_ids = {m["id"] for m in exact_matches}
                for c in text_cands:
                    if c["doc_id"] in exact_ids:
                        c["score"] = max(c["score"], 0.95)
                text_cands.sort(key=lambda x: x["score"], reverse=True)
            except Exception:
                pass

        selected = text_cands[:top_k] + image_cands[:_IMAGE_QUOTA]

        ids = [c["doc_id"] for c in selected]
        raw_docs = self._docstore.mget(ids)

        results: List[dict] = []
        result_ids: set = set()
        for i, c in enumerate(selected):
            if c["score"] < SIMILARITY_THRESHOLD_LOW:
                continue
            raw = raw_docs[i]
            if raw is None:
                content = c["meta"].get("original_content", "")
            elif isinstance(raw, Document):
                content = raw.page_content
            else:
                content = str(raw)

            result_ids.add(c["doc_id"])
            results.append({
                "id": c["doc_id"],
                "error_code": c["meta"].get("error_code", ""),
                "keywords": c["meta"].get("keywords", ""),
                "title": c["meta"].get("title", ""),
                "content": content,
                "device_models": c["meta"].get("device_models", ""),
                "similarity_score": c["score"],
                "type": c["meta"].get("type", "knowledge_entry"),
                "media_url": c["meta"].get("media_url", ""),
                "object_key": c["meta"].get("object_key", ""),
                "table_image_url": c["meta"].get("table_image_url", ""),
                "table_image_object_key": c["meta"].get("table_image_object_key", ""),
                "version": c["meta"].get("version", ""),
                "doc_type": c["meta"].get("doc_type", ""),
                "priority": c.get("priority", ""),
            })

            if c["meta"].get("type") == "image":
                gid = c["meta"].get("group_id")
                if gid:
                    for sib in self._expand_image_group(gid, result_ids):
                        result_ids.add(sib["id"])
                        results.append(sib)

        # 优先级修正：当前版本没有命中任何内容时，结果全部来自祖先版本。
        # 此时不能简单清空标签（多个祖先版本之间仍需区分新旧），
        # 而是按版本链顺序重新分配：链上最新的祖先版本升为 high，更旧的保留 low。
        # 若只有一个祖先版本有内容（无版本间冲突），则统一清空为中性标签。
        if results and chain and not any(r["priority"] == "high" for r in results):
            version_rank = {vc: i for i, vc in enumerate(chain)}
            versioned = [r for r in results if r.get("version") and r["version"] in version_rank]
            if versioned:
                min_rank = min(version_rank[r["version"]] for r in versioned)
                has_older = any(version_rank[r["version"]] > min_rank for r in versioned)
                for r in results:
                    v = r.get("version", "")
                    if not v or v not in version_rank:
                        r["priority"] = ""
                    elif version_rank[v] == min_rank:
                        r["priority"] = "high" if has_older else ""
                    else:
                        r["priority"] = "low"
            else:
                for r in results:
                    r["priority"] = ""

        if not results:
            confidence = "low"
        elif results[0]["similarity_score"] >= SIMILARITY_THRESHOLD_HIGH:
            confidence = "high"
        else:
            confidence = "medium"

        return results, confidence


_retriever_instance: Optional[KnowledgeRetriever] = None


def get_retriever() -> KnowledgeRetriever:
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = KnowledgeRetriever()
    return _retriever_instance


def reload_retriever():
    global _retriever_instance
    if _retriever_instance is not None:
        _retriever_instance.reload_knowledge()
