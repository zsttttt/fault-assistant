# 故障助手后端服务

加油机/充电桩设备故障诊断 AI 助手后端服务。

## 🚀 快速开始（本地开发）

### 1. 安装依赖

```bash
# 创建虚拟环境（推荐）
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
# 复制配置文件
cp .env .env

# 编辑 .env 文件，填入你的 API Key
```

**LLM 选择：**
- 国内推荐使用**通义千问**（访问稳定，价格便宜）
- 注册地址：https://dashscope.console.aliyun.com/
- 海外推荐使用 **Claude**

### 3. 初始化数据

```bash
# 初始化数据库并添加示例数据
python init_data.py
```

### 4. 启动服务

```bash
python main.py
```

服务启动后：
- 服务地址：http://localhost:8000
- API 文档：http://localhost:8000/docs（可在线测试接口）

---

## 📖 API 接口说明

### 对话接口

**POST /api/chat**

```json
// 请求
{
  "question": "加油枪不出油怎么办",
  "device_model": "XX-2000"  // 可选
}

// 响应
{
  "answer": "关于加油枪不出油的问题...",
  "conversation_id": "uuid-xxx",
  "confidence": "high"  // high/medium/low
}
```

### 反馈接口

**POST /api/feedback**

```json
// 请求
{
  "conversation_id": "uuid-xxx",
  "solved": true,
  "feedback_text": "按步骤操作后解决了"  // 可选
}
```

### 管理接口

| 接口 | 方法 | 说明 |
|------|------|------|
| /api/admin/knowledge | GET | 获取所有知识条目 |
| /api/admin/knowledge | POST | 添加新知识 |
| /api/admin/unsolved | GET | 获取未解决问题列表 |
| /api/admin/reload | POST | 重新加载知识库 |

---

## 📁 项目结构

```
fault-assistant/
├── main.py              # FastAPI 入口
├── config.py            # 配置管理
├── init_data.py         # 初始化脚本
├── requirements.txt     # 依赖列表
├── .env                 # 环境变量（不要提交到 git）
├── .env.example         # 环境变量示例
├── database/
│   ├── __init__.py
│   └── db.py           # 数据库操作
├── rag/
│   ├── __init__.py
│   ├── retriever.py    # 知识检索
│   └── generator.py    # LLM 生成
└── fault_assistant.db  # SQLite 数据库（运行后生成）
```

---

## 🔧 添加知识库数据

### 方式一：通过 API

```bash
curl -X POST http://localhost:8000/api/admin/knowledge \
  -H "Content-Type: application/json" \
  -d '{
    "error_code": "E04",
    "title": "油泵异响",
    "content": "可能原因：...\n解决步骤：...",
    "keywords": "油泵 异响 噪音",
    "device_models": "XX-2000, XX-3000"
  }'
```

### 方式二：直接编辑数据库

使用 SQLite 客户端（如 DB Browser for SQLite）打开 `fault_assistant.db`，
在 `knowledge` 表中添加记录。

### 方式三：批量导入（待开发）

可以开发一个脚本，从 Excel/CSV 批量导入知识数据。

---

## 🌐 部署到服务器

### 部署到 AWS EC2（和 C# 后端相同方式）

1. 在 EC2 上安装 Python 3.9+
2. 上传代码到服务器
3. 安装依赖：`pip install -r requirements.txt`
4. 配置环境变量
5. 使用 systemd 或 supervisor 管理进程
6. 配置 Nginx 反向代理

详细部署文档见：[部署指南](./DEPLOY.md)（待补充）

---

## ❓ 常见问题

**Q: 首次启动很慢？**
A: 首次运行需要下载向量模型（约 500MB），请耐心等待。

**Q: 没有 API Key 能用吗？**
A: 可以，系统会使用备用回答模式（直接返回知识库内容，不经过 LLM 润色）。

**Q: 如何查看未解决的问题？**
A: 访问 http://localhost:8000/api/admin/unsolved

---

## 📞 技术支持

如有问题，请联系开发者。
