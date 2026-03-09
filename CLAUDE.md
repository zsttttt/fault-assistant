## 应做和不应做的事
- Check for existing implementations first
- Prefer editing existing files
- Don't add comments unless requested
- Don't create a document unless I have a specific request
- 阅读迁移文档时先说执行计划，等我确认后再写代码
- 每次只执行迁移文档中的 1-2 个步骤，不要一次性全部执行

## 项目背景
售后客服 RAG 系统，技术栈：
- 文档解析：Docling
- 编排框架：LangChain
- 向量数据库：Qdrant Cloud
- 文档存储 & 对话历史：Redis
- 对象存储：Backblaze B2（开发），MinIO（生产）
- 所有配置通过环境变量 / config/settings.py 管理
- 中文注释，PEP 8 风格