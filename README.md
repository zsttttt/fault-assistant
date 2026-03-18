# 故障助手后端服务

加油机/充电桩设备故障诊断 AI 助手后端服务。

## 技术栈

| 组件 | 技术 | 说明 |
|------|------|------|
| Web 框架 | FastAPI | 异步 REST API |
| LLM | 通义千问 qwen3-max | 文字生成，DashScope API |
| 视觉模型 | qwen-vl-max | 图片/视频理解 |
| 嵌入模型 | text-embedding-v4（1024 维） | 语义向量，DashScope API |
| 向量数据库 | Qdrant | 语义检索，本地部署 |
| 对话历史 | Redis | 多轮对话记忆 |
| 对象存储 | MinIO | 图片/视频文件存储 |
| 文档解析 | Docling | PDF/DOCX/PPTX 解析 |
| 关系数据库 | SQLite | 对话记录存储 |

---

## 快速开始（本地开发）

### 1. 启动依赖服务（Docker）

本地部署需要先启动 Qdrant、Redis、MinIO。

```bash
# Qdrant 向量数据库（端口 6333）
docker run -d --name qdrant \
  -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/data/qdrant:/qdrant/storage \
  qdrant/qdrant

# Redis（端口 6379）
docker run -d --name redis \
  -p 6379:6379 \
  redis:7-alpine

# MinIO 对象存储（API 端口 9000，控制台端口 9001）
docker run -d --name minio \
  -p 9000:9000 -p 9001:9001 \
  -v $(pwd)/data/minio:/data \
  -e MINIO_ROOT_USER=minioadmin \
  -e MINIO_ROOT_PASSWORD=minioadmin \
  minio/minio server /data --console-address ":9001"
```

MinIO 首次使用需创建 Bucket：访问 `http://localhost:9001`，登录后新建名为 `fault-assistant` 的 Bucket。

### 2. 安装 Python 依赖

```bash
python -m venv venv

# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate

pip install -r requirements.txt
```

### 3. 下载 Docling 模型

首次部署需下载文档解析模型（约 1-2 GB），国内需要配置镜像：

```bash
# 设置 HuggingFace 镜像（国内）
export HF_ENDPOINT=https://hf-mirror.com

python download_docling_models.py
```

### 4. 配置环境变量

复制并编辑 `.env` 文件：

```bash
cp .env.example .env
```

本地部署关键配置项：

```ini
# LLM（通义千问，必填）
DASHSCOPE_API_KEY=your_api_key_here

# Qdrant 本地地址（无需 API Key）
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION_NAME=fault_assistant_rag

# Redis 本地地址（无密码）
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_TTL=86400

# MinIO 本地对象存储
STORAGE_ENDPOINT=http://localhost:9000
STORAGE_ACCESS_KEY=minioadmin
STORAGE_SECRET_KEY=minioadmin
STORAGE_BUCKET=fault-assistant
STORAGE_REGION=us-east-1

# Docling 模型目录
DOCLING_ARTIFACTS_PATH=./models/docling
```

### 5. 启动服务

```bash
python main.py
```

服务启动后：
- 服务地址：`http://localhost:8000`
- 前台聊天界面：`http://localhost:8000`
- 后台管理界面：`http://localhost:8000/admin`
- API 文档：`http://localhost:8000/docs`

---

## 项目结构

```
fault-assistant/
├── main.py                      # FastAPI 入口，所有路由定义
├── config.py                    # 环境变量读取与配置管理
├── requirements.txt             # Python 依赖
├── .env                         # 环境变量（不提交 git）
├── download_docling_models.py   # 预下载 Docling 模型脚本
├── migrate_knowledge.py         # SQLite 知识迁移至 Qdrant 脚本
├── database/
│   ├── db.py                    # SQLite 操作（对话记录、知识条目）
│   └── version_registry.py      # 版本注册表
├── rag/
│   ├── retriever.py             # Qdrant 语义检索
│   └── generator.py             # LLM 回答生成
├── src/
│   ├── parser/
│   │   ├── docling_parser.py    # Docling 文档解析（文本/表格/图片）
│   │   ├── image_describer.py   # VLM 图片描述生成
│   │   ├── video_processor.py   # 视频关键帧提取
│   │   └── media_extractor.py   # 媒体文件处理
│   ├── chunking/
│   │   └── chunker.py           # 文本分块 + 表格摘要
│   ├── indexing/
│   │   ├── indexer.py           # Qdrant 向量索引核心模块
│   │   ├── image_indexer.py     # 图片向量索引
│   │   └── video_indexer.py     # 视频向量索引
│   ├── retrieval/
│   │   └── result_parser.py     # 检索结果解析
│   ├── generation/
│   │   └── multimodal_generator.py  # 多模态回答生成（流式/非流式）
│   ├── storage/
│   │   └── object_store.py      # 对象存储抽象（MinIO/B2）
│   ├── context/
│   │   ├── history.py           # Redis 对话历史
│   │   └── version_state.py     # 会话版本状态管理
│   └── pipeline/
│       └── realtime_media.py    # 用户上传媒体实时处理流水线
├── static/
│   ├── index.html               # 前台聊天界面
│   └── admin.html               # 后台管理界面
├── models/
│   └── docling/                 # Docling 本地模型文件（运行下载脚本后生成）
└── data/
    ├── qdrant/                  # Qdrant 持久化数据
    └── minio/                   # MinIO 持久化数据
```

---

## API 接口

### 对话接口

**POST /api/chat** - 单次问答

```json
// 请求
{
  "question": "加油枪不出油怎么办",
  "device_model": "XX-2000",
  "session_id": "uuid-xxx",
  "version_code": "V1.2.3"
}

// 响应
{
  "answer": "关于加油枪不出油的问题...",
  "conversation_id": "uuid-yyy",
  "confidence": "high",
  "media": {"images": [], "videos": []}
}
```

**POST /api/chat/stream** - 流式问答（SSE）

请求体同上，响应为 Server-Sent Events 流。

**POST /api/chat/media** - 携带图片/视频问答（multipart/form-data）

| 字段 | 类型 | 说明 |
|------|------|------|
| question | string | 问题文本 |
| file | file | 图片（jpg/png/webp 等）或视频（mp4/mov 等） |
| device_model | string | 设备型号（可选） |
| session_id | string | 会话 ID（可选） |

### 反馈接口

**POST /api/feedback**

```json
{
  "conversation_id": "uuid-yyy",
  "solved": true,
  "feedback_text": "按步骤操作后解决了"
}
```

### 管理接口

| 接口 | 方法 | 说明 | 认证 |
|------|------|------|------|
| /api/admin/knowledge | GET | 获取所有知识条目 | 需要 |
| /api/admin/knowledge | POST | 添加文本知识条目 | 需要 |
| /api/admin/knowledge/{id} | DELETE | 删除单个知识条目 | 需要 |
| /api/admin/knowledge/batch-delete | POST | 批量删除知识条目 | 需要 |
| /api/admin/knowledge/import | POST | 从 Excel 导入知识 | 需要 |
| /api/admin/knowledge/import/preview | POST | 预览 Excel 文件 | 需要 |
| /api/admin/knowledge/import/document | POST | 从文档解析导入（PDF/DOCX/PPTX） | 需要 |
| /api/admin/documents | GET | 获取已上传文档列表 | 需要 |
| /api/admin/documents/{filename} | DELETE | 删除文档及其所有索引 | 需要 |
| /api/admin/unsolved | GET | 获取未解决问题列表 | 需要 |
| /api/admin/reload | POST | 重新加载知识库 | 需要 |
| /api/admin/versions | GET/POST | 版本列表 / 新增版本 | 需要 |
| /api/admin/versions/{code} | GET/PUT/DELETE | 版本详情 / 修改 / 删除 | 需要 |

管理接口认证方式：请求头加 `X-Admin-Password: <ADMIN_PASSWORD>`。

---

## 环境变量说明

| 变量 | 默认值 | 说明 |
|------|--------|------|
| DASHSCOPE_API_KEY | - | 通义千问 API Key（必填） |
| LLM_MODEL | qwen3-max | LLM 模型名 |
| VLM_MODEL | qwen-vl-max | 视觉模型名 |
| HOST | 0.0.0.0 | 服务监听地址 |
| PORT | 8000 | 服务端口 |
| DEBUG | false | 是否开启热重载 |
| QDRANT_URL | - | Qdrant 地址，本地填 `http://localhost:6333` |
| QDRANT_API_KEY | - | Qdrant API Key，本地部署留空 |
| QDRANT_COLLECTION_NAME | fault_assistant_rag | Collection 名称 |
| REDIS_URL | - | Redis 完整连接串（优先级高于下面三项） |
| REDIS_HOST | localhost | Redis 主机 |
| REDIS_PORT | 6379 | Redis 端口 |
| REDIS_PASSWORD | - | Redis 密码，本地部署留空 |
| REDIS_TTL | 86400 | 对话历史过期时间（秒） |
| STORAGE_ENDPOINT | - | MinIO 端点，本地填 `http://localhost:9000` |
| STORAGE_ACCESS_KEY | - | MinIO Access Key |
| STORAGE_SECRET_KEY | - | MinIO Secret Key |
| STORAGE_BUCKET | after-sales-media | MinIO Bucket 名称 |
| STORAGE_REGION | us-east-1 | MinIO Region（本地填任意值） |
| ADMIN_PASSWORD | admin123 | 后台管理员密码 |
| USER_USERNAME | user | 前台用户名 |
| USER_PASSWORD | user123 | 前台密码 |
| DOCLING_ARTIFACTS_PATH | - | Docling 模型本地目录，设为 `./models/docling` |
| HF_ENDPOINT | - | HuggingFace 镜像地址，国内填 `https://hf-mirror.com` |
| DATABASE_PATH | fault_assistant.db | SQLite 数据库文件路径 |

---

## 常见问题

**Q: 启动时报 Qdrant 连接失败？**
A: 确认 Docker 容器已运行：`docker ps | grep qdrant`。首次启动若 collection 为空，系统会自动从 SQLite 迁移现有知识。

**Q: 首次文档解析很慢？**
A: Docling 首次解析需加载 AI 模型到显存/内存，建议提前运行 `python download_docling_models.py` 预下载模型，并设置 `DOCLING_ARTIFACTS_PATH`。

**Q: 图片/视频上传后无法访问？**
A: 确认 MinIO 已启动且 Bucket 已创建。检查 `STORAGE_ENDPOINT`、`STORAGE_ACCESS_KEY`、`STORAGE_SECRET_KEY` 配置是否正确。

**Q: 多轮对话不生效？**
A: 前端每次请求需传入相同的 `session_id`。检查 Redis 是否正常运行：`docker ps | grep redis`。

**Q: 如何迁移旧的 SQLite 知识数据到 Qdrant？**
A: 运行迁移脚本：`python migrate_knowledge.py`。