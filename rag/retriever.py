"""
知识检索模块
"""
import re
import numpy as np
from typing import List, Tuple, Optional
import dashscope
from dashscope import TextEmbedding

from config import EMBEDDING_MODEL, SIMILARITY_THRESHOLD_HIGH, SIMILARITY_THRESHOLD_LOW, DASHSCOPE_API_KEY
from database.db import get_all_knowledge, search_by_error_code


class KnowledgeRetriever:
    def __init__(self):
        print("🔄 正在初始化千问向量模型...")
        dashscope.api_key = DASHSCOPE_API_KEY
        self.model_name = EMBEDDING_MODEL
        print("✅ 向量模型配置完成")
        self.knowledge_items = []
        self.embeddings = None
        self.reload_knowledge()
    
    def reload_knowledge(self):
        self.knowledge_items = get_all_knowledge()
        if not self.knowledge_items:
            print("⚠️ 知识库为空，请先添加知识条目")
            self.embeddings = None
            return

        texts = []
        for item in self.knowledge_items:
            search_text = f"{item['error_code'] or ''} {item['title']} {item['keywords'] or ''} {item['content']}"
            texts.append(search_text)

        self.embeddings = self._get_embeddings(texts)
        print(f"✅ 已加载 {len(self.knowledge_items)} 条知识 (包含完整语义信息)")

    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        print(f"🔄 调用千问向量API，文本数量: {len(texts)}")

        batch_size = 10
        all_embeddings = []
        total_tokens = 0

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            print(f"   处理批次 {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")

            resp = TextEmbedding.call(
                model=self.model_name,
                input=batch
            )

            if resp.status_code == 200:
                usage = resp.usage
                total_tokens += usage['total_tokens']
                embeddings = [item['embedding'] for item in resp.output['embeddings']]
                all_embeddings.extend(embeddings)
            else:
                raise Exception(f"向量化失败: {resp.code} - {resp.message}")

        print(f"✅ API调用成功 - 总token: {total_tokens}")
        embeddings_array = np.array(all_embeddings)
        norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
        return embeddings_array / norms
    
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
    
    def retrieve(self, query: str, top_k: int = 3) -> Tuple[List[dict], str]:
        if self.embeddings is None or len(self.knowledge_items) == 0:
            return [], "low"

        error_code = self._extract_error_code(query)

        query_embedding = self._get_embeddings([query])[0]
        similarities = np.dot(self.embeddings, query_embedding)

        if error_code:
            exact_matches = search_by_error_code(error_code)
            if exact_matches:
                exact_ids = {item['id'] for item in exact_matches}
                for idx, item in enumerate(self.knowledge_items):
                    if item['id'] in exact_ids:
                        similarities[idx] = max(similarities[idx], 0.95)

        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score > SIMILARITY_THRESHOLD_LOW:
                item = self.knowledge_items[idx].copy()
                item['similarity_score'] = score
                results.append(item)

        if not results:
            confidence = "low"
        elif results[0].get('similarity_score', 0) >= SIMILARITY_THRESHOLD_HIGH:
            confidence = "high"
        else:
            confidence = "medium"

        return results, confidence


_retriever_instance = None

def get_retriever() -> KnowledgeRetriever:
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = KnowledgeRetriever()
    return _retriever_instance

def reload_retriever():
    global _retriever_instance
    if _retriever_instance is not None:
        _retriever_instance.reload_knowledge()
