"""
故障助手后端服务
"""
import sys
import io
import uuid
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tempfile
import os

from config import HOST, PORT, DEBUG
from database import init_db, save_conversation, update_feedback, get_unsolved_issues, add_knowledge, get_all_knowledge, import_from_excel, preview_excel, get_excel_sheets
from rag import get_retriever, reload_retriever, generate_answer

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


@asynccontextmanager
async def lifespan(app: FastAPI):
    """服务生命周期管理"""
    print("🚀 故障助手服务启动中...")
    init_db()
    get_retriever()
    print(f"✅ 服务已启动: http://{HOST}:{PORT}")
    print(f"📖 API 文档: http://{HOST}:{PORT}/docs")
    yield
    print("服务关闭")


app = FastAPI(
    title="故障助手 API",
    description="加油机/充电桩设备故障诊断服务",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== 数据模型 ====================

class ChatRequest(BaseModel):
    question: str
    device_model: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    conversation_id: str
    confidence: str

class FeedbackRequest(BaseModel):
    conversation_id: str
    solved: bool
    feedback_text: Optional[str] = None

class KnowledgeRequest(BaseModel):
    error_code: Optional[str] = ""
    title: str
    content: str
    keywords: Optional[str] = ""
    device_models: Optional[str] = ""


# ==================== API 接口 ====================

@app.get("/")
async def root():
    """健康检查"""
    return {"status": "ok", "service": "故障助手"}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """对话接口"""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")
    
    retriever = get_retriever()
    context, confidence = retriever.retrieve(req.question)
    
    answer = await generate_answer(
        question=req.question,
        context=context,
        confidence=confidence,
        device_model=req.device_model
    )
    
    conversation_id = str(uuid.uuid4())
    save_conversation(
        conversation_id=conversation_id,
        device_model=req.device_model or "",
        question=req.question,
        answer=answer,
        confidence=confidence
    )
    
    return ChatResponse(
        answer=answer,
        conversation_id=conversation_id,
        confidence=confidence
    )


@app.post("/api/feedback")
async def feedback(req: FeedbackRequest):
    """反馈接口"""
    update_feedback(
        conversation_id=req.conversation_id,
        solved=req.solved,
        feedback_text=req.feedback_text
    )
    return {"status": "ok", "message": "感谢您的反馈"}


@app.get("/api/admin/unsolved")
async def get_unsolved():
    """获取未解决问题列表"""
    issues = get_unsolved_issues()
    return {"issues": issues, "count": len(issues)}


@app.get("/api/admin/knowledge")
async def list_knowledge():
    """获取所有知识条目"""
    items = get_all_knowledge()
    return {"items": items, "count": len(items)}


@app.post("/api/admin/knowledge")
async def create_knowledge(req: KnowledgeRequest):
    """添加知识条目"""
    knowledge_id = add_knowledge(
        error_code=req.error_code,
        title=req.title,
        content=req.content,
        keywords=req.keywords,
        device_models=req.device_models
    )
    reload_retriever()
    return {"status": "ok", "id": knowledge_id, "message": "知识添加成功"}


@app.post("/api/admin/reload")
async def reload_knowledge():
    """重新加载知识库"""
    reload_retriever()
    return {"status": "ok", "message": "知识库已重新加载"}


@app.post("/api/admin/knowledge/import/preview")
async def preview_excel_file(file: UploadFile = File(...)):
    """预览Excel文件内容"""
    if not file.filename.endswith(('.xlsx', '.xls')):
        raise HTTPException(status_code=400, detail="仅支持Excel文件(.xlsx, .xls)")

    with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_path = tmp_file.name

    try:
        sheets = get_excel_sheets(tmp_path)
        preview = preview_excel(tmp_path, rows=5)
        return {
            "status": "ok",
            "filename": file.filename,
            "sheets": sheets,
            "preview": preview
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"预览失败: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/api/admin/knowledge/import")
async def import_excel_file(file: UploadFile = File(...), sheet_name: str = None):
    """从Excel文件导入知识数据"""
    if not file.filename.endswith(('.xlsx', '.xls')):
        raise HTTPException(status_code=400, detail="仅支持Excel文件(.xlsx, .xls)")

    with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_path = tmp_file.name

    try:
        results = import_from_excel(tmp_path, sheet_name=sheet_name)

        if results["success"] > 0:
            reload_retriever()

        return {
            "status": "ok",
            "message": f"导入完成: 成功 {results['success']} 条, 失败 {results['failed']} 条",
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"导入失败: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=HOST, port=PORT, reload=DEBUG)
