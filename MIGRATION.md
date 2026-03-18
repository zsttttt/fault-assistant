# 售后客服 RAG 系统技术栈迁移指令

## 项目背景

我正在开发一个**售后客服机器人**，使用 RAG（检索增强生成）架构。当前项目需要迁移到以下技术栈：**Docling + LangChain + Qdrant Cloud + Redis**。

我的文档特点：
- 格式复杂且不规则（PDF、DOCX、PPTX 等混合）
- 包含**大量表格**（含嵌套表头、合并单元格、跨页表格）
- 包含**大量图片**（产品图、流程图、示意图等）
- 包含中文内容

---

## 目标架构

请将我的项目迁移为以下完整 RAG 管线：

```
文档解析 (Docling) 
    → 元素分离 (文本/表格/图片) 
    → 切块 + 摘要生成 
    → 向量化存储 (Qdrant Cloud 存嵌入, Redis 存原文) 
    → 混合检索 (BM25 + 向量) 
    → Rerank 
    → LLM 答案生成 (带 Redis 对话历史的多轮对话)
```

---

## 第一步：安装依赖

请在项目中安装以下依赖包：

```bash
# 核心文档解析
pip install docling

# LangChain 生态
pip install langchain langchain-openai langchain-community langchain-qdrant

# Docling 与 LangChain 集成
pip install docling-langchain

# 向量数据库：Qdrant（使用云端服务）
pip install qdrant-client

# 上下文/对话历史/文档存储：Redis
pip install redis langchain-redis

# OCR 支持（如果文档有扫描页）
# 系统级安装：apt-get install tesseract-ocr (Linux) 或 brew install tesseract (macOS)

# 可选：混合检索
pip install rank-bm25
```

---

## 第二步：文档解析模块

使用 Docling 解析文档，将文档分离为**文本、表格、图片**三类元素。

```python
from docling.document_converter import DocumentConverter
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from docling.document_converter import PdfFormatOption

# 配置管线：开启 OCR 和表格结构识别
pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = True  # 如有扫描页
pipeline_options.do_table_structure = True  # 表格结构识别

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)

def parse_document(file_path: str):
    """解析单个文档，返回结构化结果"""
    result = converter.convert(file_path)
    doc = result.document
    
    # 导出为 Markdown（文本+表格）
    markdown_output = doc.export_to_markdown()
    
    # 导出为 JSON（完整结构信息，含坐标、类型标签等）
    json_output = doc.export_to_dict()
    
    return doc, markdown_output, json_output
```

### 元素分离逻辑

请实现以下分离逻辑，将 Docling 解析结果拆分为三类：

```python
def separate_elements(doc):
    """
    从 Docling 文档中分离出文本、表格、图片三类元素
    """
    texts = []      # 纯文本段落
    tables = []     # 表格（保留 HTML/Markdown 结构）
    images = []     # 图片（保留路径或 base64）
    
    # 遍历 doc 的元素，按类型分类
    # Docling 的 DoclingDocument 提供了 iterate_items() 等方法
    # 文本元素 → texts
    # 表格元素 → tables（导出为 HTML 或 Markdown 表格格式）
    # 图片元素 → images（保存图片文件，记录路径）
    
    return texts, tables, images
```

---

## 第三步：切块策略

### 文本切块
使用**结构感知切块**（先按标题/章节分，再递归分割）：

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,       # 约 512 tokens
    chunk_overlap=50,     # 10% 重叠
    separators=["\n\n", "\n", "。", "！", "？", ".", " "]  # 中文友好
)

text_chunks = text_splitter.split_text(markdown_text)
```

### 表格处理
**不要对表格做文本切块！** 表格应作为独立元素处理：
1. 用 LLM 为每个表格生成**自然语言摘要**（用于检索）
2. 保留表格**原始 HTML/Markdown 结构**（用于答案生成）

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def summarize_table(table_content: str) -> str:
    """用 LLM 为表格生成摘要"""
    prompt = f"""请用中文简洁概括以下表格的内容，包括：
    1. 这个表格是关于什么的
    2. 包含哪些关键字段/列
    3. 关键数据要点
    
    表格内容：
    {table_content}
    """
    response = llm.invoke(prompt)
    return response.content
```

### 图片处理
用多模态 LLM 为图片生成文字描述：

```python
def describe_image(image_path: str) -> str:
    """用多模态 LLM 为图片生成描述"""
    # 使用 GPT-4o 或其他多模态模型
    # 将图片转为 base64 后发送给 LLM
    # 返回图片的文字描述
    pass
```

---

## 第四步：向量化与存储

使用 **LangChain 的 MultiVectorRetriever** 实现"摘要检索 + 原文生成"模式：

```python
import uuid
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_redis import RedisStore
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.schema import Document
import redis

# ========== 配置（从 config/settings.py 或环境变量读取） ==========
QDRANT_URL = "https://xxx.cloud.qdrant.io:6333"   # 你的 Qdrant Cloud 地址
QDRANT_API_KEY = "你的_qdrant_api_key"              # 已配置好，可直接使用
REDIS_URL = "redis://localhost:6379"                 # Redis 连接地址
COLLECTION_NAME = "after_sales_rag"

# ========== 嵌入模型 ==========
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# ========== 向量数据库：Qdrant Cloud（存摘要的嵌入，用于检索） ==========
qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

vectorstore = QdrantVectorStore(
    client=qdrant_client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings,
)
# 注意：首次使用需确保 collection 已创建，可在 Qdrant Cloud 控制台创建
# 或通过代码创建：
# from qdrant_client.models import Distance, VectorParams
# qdrant_client.create_collection(
#     collection_name=COLLECTION_NAME,
#     vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
# )

# ========== 文档存储：Redis（存原始内容，用于答案生成） ==========
# Redis 同时也用于管理对话上下文/历史
redis_client = redis.from_url(REDIS_URL)

docstore = RedisStore(
    client=redis_client,
    namespace="docstore"  # 键前缀，避免与其他数据冲突
)

# ========== 对话历史管理：Redis ==========
from langchain_community.chat_message_histories import RedisChatMessageHistory

def get_chat_history(session_id: str):
    """获取指定会话的对话历史"""
    return RedisChatMessageHistory(
        session_id=session_id,
        url=REDIS_URL,
        key_prefix="chat_history:"
    )

# ========== 构建 MultiVectorRetriever ==========
id_key = "doc_id"

retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    docstore=docstore,
    id_key=id_key,
)

def index_documents(text_chunks, text_summaries,
                    tables, table_summaries,
                    images, image_descriptions):
    """
    将所有元素索引到 MultiVectorRetriever 中
    - vectorstore 中存放：文本块、表格摘要、图片描述（用于语义检索）
    - docstore 中存放：原始文本块、原始表格、原始图片路径（用于答案生成）
    """
    
    # 索引文本块（文本块本身既是摘要也是原文）
    text_ids = [str(uuid.uuid4()) for _ in text_chunks]
    text_docs = [
        Document(page_content=chunk, metadata={id_key: text_ids[i]})
        for i, chunk in enumerate(text_chunks)
    ]
    retriever.vectorstore.add_documents(text_docs)
    retriever.docstore.mset(list(zip(text_ids, text_chunks)))
    
    # 索引表格（摘要用于检索，原始表格用于生成）
    table_ids = [str(uuid.uuid4()) for _ in tables]
    table_summary_docs = [
        Document(page_content=summary, metadata={id_key: table_ids[i]})
        for i, summary in enumerate(table_summaries)
    ]
    retriever.vectorstore.add_documents(table_summary_docs)
    retriever.docstore.mset(list(zip(table_ids, tables)))
    
    # 索引图片（描述用于检索，图片路径/原文用于生成）
    image_ids = [str(uuid.uuid4()) for _ in images]
    image_desc_docs = [
        Document(page_content=desc, metadata={id_key: image_ids[i]})
        for i, desc in enumerate(image_descriptions)
    ]
    retriever.vectorstore.add_documents(image_desc_docs)
    retriever.docstore.mset(list(zip(image_ids, images)))
```

---

## 第五步：检索与生成

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ========== 带对话历史的 Prompt ==========
prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个专业的售后客服助手。请根据以下参考资料回答用户的问题。

参考资料：
{context}

要求：
1. 仅基于参考资料回答，不要编造信息
2. 如果参考资料中没有相关信息，请明确告知用户
3. 如果涉及表格数据，请准确引用
4. 回答简洁专业，使用中文
5. 结合对话历史理解用户的追问和上下文"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

# ========== 基础 Chain ==========
base_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ========== 包装为带对话历史的 Chain（使用 Redis） ==========
chain_with_history = RunnableWithMessageHistory(
    base_chain,
    get_session_history=get_chat_history,  # 第四步中定义的 Redis 历史函数
    input_messages_key="question",
    history_messages_key="chat_history",
)

# ========== 使用（每次调用需传入 session_id） ==========
answer = chain_with_history.invoke(
    "用户的售后问题",
    config={"configurable": {"session_id": "user_123_session_001"}}
)
# session_id 用于区分不同用户/不同会话，Redis 会自动持久化对话历史
```

---

## 第六步（可选增强）：混合检索 + Rerank

如果纯向量检索效果不够好，添加 BM25 混合检索和 Rerank：

```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

# BM25 检索器（关键词匹配，对专业术语、型号等有帮助）
bm25_retriever = BM25Retriever.from_documents(all_documents)
bm25_retriever.k = 5

# 向量检索器
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# 混合检索（各 50% 权重，可调节）
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.4, 0.6]
)
```

---

## 项目结构建议

```
project/
├── config/
│   └── settings.py          # API keys、Qdrant/Redis 连接配置、模型配置
├── data/
│   ├── raw_docs/            # 原始文档存放
│   └── processed/           # 解析后的中间文件
├── src/
│   ├── parser/
│   │   └── docling_parser.py    # Docling 文档解析模块
│   ├── chunking/
│   │   └── chunker.py           # 切块逻辑（文本/表格/图片分别处理）
│   ├── indexing/
│   │   └── indexer.py           # 向量化与索引模块（Qdrant + Redis）
│   ├── retrieval/
│   │   └── retriever.py         # 检索模块（MultiVectorRetriever）
│   ├── context/
│   │   └── history.py           # 对话历史管理（Redis）
│   ├── generation/
│   │   └── generator.py         # LLM 答案生成模块
│   └── pipeline.py              # 完整 RAG 管线编排
├── requirements.txt
└── main.py                       # 入口文件
```

### config/settings.py 参考结构

```python
import os

# Qdrant Cloud 配置（已配置好，可直接使用）
QDRANT_URL = os.getenv("QDRANT_URL", "https://xxx.cloud.qdrant.io:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "你的key")
QDRANT_COLLECTION_NAME = "after_sales_rag"

# Redis 配置
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
REDIS_DOCSTORE_NAMESPACE = "docstore"        # 原始文档存储的键前缀
REDIS_CHAT_HISTORY_PREFIX = "chat_history:"  # 对话历史的键前缀

# LLM 配置
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
```

---

## 关键设计原则

1. **检索用摘要，生成用原文** — 表格和图片的摘要用于语义检索，但传给 LLM 生成答案时要用原始内容
2. **表格不切块** — 表格作为独立元素完整保留，不要用文本分割器切割表格
3. **中文分隔符** — 切块时注意中文句号、问号等分隔符
4. **元数据保留** — 每个 chunk 应携带来源文档名、页码、元素类型等元数据，方便溯源
5. **渐进式增强** — 先跑通基础管线，再逐步加入 BM25 混合检索、Rerank 等优化
6. **存储职责分离** — Qdrant Cloud 只存向量嵌入（用于语义检索），Redis 存原始文档内容（docstore）和对话历史（chat_history），各司其职
7. **会话管理** — 每个用户/会话使用唯一的 session_id，Redis 自动持久化对话历史，支持多轮追问和上下文理解

---

## 迁移注意事项

- 请先检查我当前代码中已有的文档解析、切块、向量化逻辑，理解现有实现
- 保留现有的业务逻辑和接口不变，只替换底层技术实现
- 如果当前项目中有自定义的 prompt 模板、对话历史管理等逻辑，请保留并适配
- 迁移完成后，请确保所有原有功能正常运行

### 关于 Qdrant Cloud
- Qdrant Cloud 已配置好，API Key 可直接使用，不需要本地部署
- 首次运行时需要创建 collection（向量维度 1536 对应 text-embedding-3-small，距离函数用 Cosine）
- 如果当前项目使用了其他向量数据库（如 ChromaDB、FAISS、Milvus 等），请替换为 Qdrant
- Qdrant 的 payload 字段可用于存储元数据（文档来源、页码、元素类型等），支持过滤检索

### 关于 Redis
- Redis 在本项目中承担两个职责：
  1. **docstore** — 存储原始文档内容（文本块、表格原文、图片路径），供 MultiVectorRetriever 的答案生成阶段使用
  2. **对话历史** — 存储每个用户会话的多轮对话记录，支持上下文追问
- 如果当前项目使用了 InMemoryStore 或其他内存存储做 docstore，请替换为 Redis
- 如果当前项目使用了其他方式管理对话历史（如数据库、文件等），请统一迁移到 Redis
- 注意给不同用途设置不同的 key namespace/前缀，避免数据冲突