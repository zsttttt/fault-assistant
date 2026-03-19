"""
故障助手后端服务
"""
import sys
import io
import uuid
import json
import asyncio
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import tempfile
import os

from config import HOST, PORT, DEBUG, ADMIN_PASSWORD, USER_USERNAME, USER_PASSWORD
from src.context.version_state import get_session_version, set_session_version, detect_version_in_text
from database import init_db, save_conversation, update_feedback, get_unsolved_issues, add_knowledge, get_all_knowledge, import_from_excel, preview_excel, get_excel_sheets
from rag import get_retriever, reload_retriever
from src.retrieval.result_parser import parse_retrieved_results
from src.generation.multimodal_generator import generate_multimodal_answer, generate_multimodal_answer_stream

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


def _replace_inline_placeholders(text: str, mapping: dict) -> str:
    for pid, replacement in mapping.items():
        text = text.replace(f"[[IMG:{pid}]]", replacement)
    return text


def _clean_search_query(question: str, version_code: str) -> str:
    """从用户问题中移除版本号描述，避免版本信息污染语义检索向量。
    版本过滤由 Qdrant Filter 负责，检索向量只需包含实际问题内容。
    """
    import re
    q = question
    if version_code:
        # 移除 "版本号为/是 XXXX"、"程序版本XXXX" 等句式及版本号本身
        escaped = re.escape(version_code)
        q = re.sub(
            r'(程序|软件|固件|产品)?\s*(版本号?|version)\s*[为是：:\s]\s*' + escaped,
            '', q, flags=re.IGNORECASE
        )
        # 兜底：直接移除孤立的版本号字符串
        q = re.sub(r'(?<![.\d])' + escaped + r'(?![.\d])', '', q)
    # 清理首尾多余标点
    q = re.sub(r'^[\s，。！？,.\-]+|[\s，。！？,.\-]+$', '', q)
    return q if q.strip() else question


def _refresh_media_urls(context: list, media_store) -> list:
    """用 object_key 为每条检索结果重新生成预签名 URL，避免 7 天过期失效"""
    if not media_store:
        return context
    for item in context:
        try:
            if item.get("type") == "image" and item.get("object_key"):
                fresh = media_store.refresh_url(item["object_key"])
                item["media_url"] = fresh
                try:
                    data = json.loads(item.get("content", "{}"))
                    if "url" in data:
                        data["url"] = fresh
                        item["content"] = json.dumps(data, ensure_ascii=False)
                except Exception:
                    pass
            if item.get("table_image_object_key"):
                item["table_image_url"] = media_store.refresh_url(item["table_image_object_key"])
        except Exception as e:
            print(f"⚠️ URL 刷新失败 [{item.get('id', '')}]: {e}")
    return context


@asynccontextmanager
async def lifespan(app: FastAPI):
    """服务生命周期管理"""
    print("🚀 故障助手服务启动中...")
    init_db()

    from src.indexing.indexer import init_indexer
    init_indexer()

    get_retriever()
    print(f"✅ 服务已启动: http://{HOST}:{PORT}")
    print(f"📖 API 文档: http://{HOST}:{PORT}/docs")
    yield
    print("服务关闭")


app = FastAPI(
    title="故障助手 API",
    description="加油机/充电桩设备故障诊断服务",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


# ==================== 数据模型 ====================

class ChatRequest(BaseModel):
    question: str
    device_model: Optional[str] = None
    session_id: Optional[str] = None  # 多轮对话会话 ID，不传则为单轮
    version_code: Optional[str] = None  # 用户当前版本号，不传则从会话状态读取

class ChatResponse(BaseModel):
    answer: str
    conversation_id: str
    confidence: str
    media: Optional[dict] = None

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

class UserAuthRequest(BaseModel):
    username: str
    password: str

class AdminAuthRequest(BaseModel):
    password: str

class BatchDeleteRequest(BaseModel):
    ids: List[str]

class VersionCreateRequest(BaseModel):
    version_code: str
    version_name: Optional[str] = ""
    is_base: bool
    base_version_code: Optional[str] = None
    doc_type_label: Optional[str] = ""

class VersionUpdateRequest(BaseModel):
    version_name: Optional[str] = None
    is_base: Optional[bool] = None
    base_version_code: Optional[str] = None
    doc_type_label: Optional[str] = None


# ==================== Auth 依赖 ====================

def require_admin(x_admin_password: str = Header(...)):
    if x_admin_password != ADMIN_PASSWORD:
        raise HTTPException(status_code=401, detail="密码错误")


# ==================== API 接口 ====================

@app.get("/", response_class=HTMLResponse)
async def chat_page():
    """聊天主界面"""
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/admin", response_class=HTMLResponse)
async def admin_page():
    """后台管理界面"""
    with open("static/admin.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.post("/api/user/auth")
async def user_auth(req: UserAuthRequest):
    """前台用户登录验证"""
    if req.username != USER_USERNAME or req.password != USER_PASSWORD:
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    return {"status": "ok"}


@app.post("/api/admin/auth")
async def admin_auth(req: AdminAuthRequest):
    """后台管理员密码验证"""
    if req.password != ADMIN_PASSWORD:
        raise HTTPException(status_code=401, detail="密码错误")
    return {"status": "ok"}


@app.get("/health")
async def health():
    return {"status": "ok", "service": "故障助手"}


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    """流式对话接口（SSE），实时返回 LLM 生成内容"""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")

    async def event_stream():
        try:
            # 版本确认前置
            effective_version = req.version_code or ""
            if req.session_id:
                detected = detect_version_in_text(req.question)
                if detected:
                    from database.version_registry import get_version as _gv_check
                    if _gv_check(detected) is not None:
                        # 已注册的版本号：更新会话
                        set_session_version(req.session_id, detected)
                        effective_version = detected
                    else:
                        # 未注册的数字串（可能是参数代码等）：有现有版本则保留，无则保留 detected 触发"未录入"提示
                        existing = effective_version or get_session_version(req.session_id)
                        effective_version = existing or detected
                elif not effective_version:
                    effective_version = get_session_version(req.session_id)

            if req.session_id and not effective_version:
                from database.version_registry import get_all_versions
                version_list = [v["version_code"] for v in get_all_versions()]
                hint = "请问您使用的是哪个版本？"
                if version_list:
                    hint += f"可选版本：{', '.join(version_list)}"
                yield f"data: {json.dumps({'content': hint}, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps({'done': True, 'conversation_id': str(uuid.uuid4()), 'confidence': 'low', 'media': {'images': [], 'videos': []}}, ensure_ascii=False)}\n\n"
                return

            if req.session_id and effective_version:
                from database.version_registry import get_version as _gv, get_all_versions as _gav
                if _gv(effective_version) is None:
                    version_list = [v["version_code"] for v in _gav()]
                    hint = f"版本「{effective_version}」暂未录入系统，请确认版本号是否正确。"
                    if version_list:
                        hint += f"当前已录入版本：{', '.join(version_list)}"
                    set_session_version(req.session_id, "")
                    yield f"data: {json.dumps({'content': hint}, ensure_ascii=False)}\n\n"
                    yield f"data: {json.dumps({'done': True, 'conversation_id': str(uuid.uuid4()), 'confidence': 'low', 'media': {'images': [], 'videos': []}}, ensure_ascii=False)}\n\n"
                    return
                set_session_version(req.session_id, effective_version)

            retriever = get_retriever()
            search_query = _clean_search_query(req.question, effective_version)
            print(f"[CHAT/STREAM] effective_version={effective_version!r}  search_query={search_query!r}", flush=True)
            context, confidence = await asyncio.to_thread(
                lambda: retriever.retrieve(search_query, version_code=effective_version)
            )
            from src.storage.object_store import get_media_store as _gms
            context = _refresh_media_urls(context, _gms())
            parsed = parse_retrieved_results(context)

            full_answer = ""
            media = {"images": [], "videos": []}
            async for chunk in generate_multimodal_answer_stream(
                question=req.question,
                parsed=parsed,
                confidence=confidence,
                device_model=req.device_model,
                session_id=req.session_id,
            ):
                if isinstance(chunk, tuple) and chunk[0] == "__MEDIA__":
                    media = chunk[1]
                else:
                    full_answer += chunk
                    yield f"data: {json.dumps({'content': chunk}, ensure_ascii=False)}\n\n"

            conversation_id = str(uuid.uuid4())
            await asyncio.to_thread(
                save_conversation,
                conversation_id=conversation_id,
                device_model=req.device_model or "",
                question=req.question,
                answer=full_answer,
                confidence=confidence,
            )
            yield f"data: {json.dumps({'done': True, 'conversation_id': conversation_id, 'confidence': confidence, 'media': media}, ensure_ascii=False)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """对话接口（支持多轮对话，传入 session_id 即可）"""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")

    # 版本确认前置
    effective_version = req.version_code or ""
    if req.session_id:
        detected = detect_version_in_text(req.question)
        if detected:
            from database.version_registry import get_version as _gv_check
            if _gv_check(detected) is not None:
                set_session_version(req.session_id, detected)
                effective_version = detected
            else:
                existing = effective_version or get_session_version(req.session_id)
                effective_version = existing or detected
        elif not effective_version:
            effective_version = get_session_version(req.session_id)

    if req.session_id and not effective_version:
        from database.version_registry import get_all_versions
        version_list = [v["version_code"] for v in get_all_versions()]
        hint = "请问您使用的是哪个版本？"
        if version_list:
            hint += f"可选版本：{', '.join(version_list)}"
        return ChatResponse(answer=hint, conversation_id=str(uuid.uuid4()), confidence="low", media=None)

    if req.session_id and effective_version:
        from database.version_registry import get_version as _gv, get_all_versions as _gav
        if _gv(effective_version) is None:
            version_list = [v["version_code"] for v in _gav()]
            hint = f"版本「{effective_version}」暂未录入系统，请确认版本号是否正确。"
            if version_list:
                hint += f"当前已录入版本：{', '.join(version_list)}"
            set_session_version(req.session_id, "")
            return ChatResponse(answer=hint, conversation_id=str(uuid.uuid4()), confidence="low", media=None)
        set_session_version(req.session_id, effective_version)

    retriever = get_retriever()
    search_query = _clean_search_query(req.question, effective_version)
    context, confidence = await asyncio.to_thread(
        lambda: retriever.retrieve(search_query, version_code=effective_version)
    )
    from src.storage.object_store import get_media_store as _gms
    context = _refresh_media_urls(context, _gms())
    parsed = parse_retrieved_results(context)

    result = await generate_multimodal_answer(
        question=req.question,
        parsed=parsed,
        confidence=confidence,
        device_model=req.device_model,
        session_id=req.session_id,
    )

    conversation_id = str(uuid.uuid4())
    save_conversation(
        conversation_id=conversation_id,
        device_model=req.device_model or "",
        question=req.question,
        answer=result["text_answer"],
        confidence=confidence
    )

    return ChatResponse(
        answer=result["text_answer"],
        conversation_id=conversation_id,
        confidence=confidence,
        media=result["media"],
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
async def get_unsolved(_=Depends(require_admin)):
    """获取未解决问题列表"""
    issues = get_unsolved_issues()
    return {"issues": issues, "count": len(issues)}


@app.get("/api/admin/knowledge")
async def list_knowledge(_=Depends(require_admin)):
    """获取所有知识条目"""
    items = get_all_knowledge()
    return {"items": items, "count": len(items)}


@app.post("/api/admin/knowledge")
async def create_knowledge(req: KnowledgeRequest, _=Depends(require_admin)):
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


@app.post("/api/admin/knowledge/batch-delete")
async def batch_delete_knowledge(req: BatchDeleteRequest, _=Depends(require_admin)):
    """批量删除知识条目"""
    from src.indexing.indexer import delete_knowledge_entry
    failed = []
    for doc_id in req.ids:
        try:
            delete_knowledge_entry(doc_id)
        except Exception as e:
            print(f"❌ 批量删除失败 [{doc_id}]: {e}")
            failed.append(doc_id)
    if failed:
        raise HTTPException(status_code=500, detail=f"{len(failed)} 条删除失败，其余已完成")
    return {"status": "ok", "deleted": len(req.ids)}


@app.delete("/api/admin/knowledge/{doc_id}")
async def delete_knowledge(doc_id: str, _=Depends(require_admin)):
    """删除知识条目"""
    from src.indexing.indexer import delete_knowledge_entry
    try:
        delete_knowledge_entry(doc_id)
    except Exception as e:
        print(f"❌ 删除知识条目失败 [{doc_id}]: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    return {"status": "ok"}


@app.get("/api/admin/documents")
async def list_documents(_=Depends(require_admin)):
    """获取已上传文档列表及各文件分块统计"""
    from src.indexing.indexer import get_all_documents
    docs = get_all_documents()
    return {"items": docs, "count": len(docs)}


@app.delete("/api/admin/documents/{filename:path}")
async def delete_document(filename: str, _=Depends(require_admin)):
    """删除指定文档的所有向量及 B2 图片"""
    from src.indexing.indexer import get_image_object_keys_by_source, _delete_by_source
    from src.storage.object_store import get_media_store

    media_store = get_media_store()
    if media_store:
        object_keys = get_image_object_keys_by_source(filename)
        for key in object_keys:
            try:
                media_store.delete_file(key)
            except Exception as e:
                print(f"⚠️ B2 删除失败 [{key}]: {e}")

    _delete_by_source(filename)
    return {"status": "ok", "message": f"已删除文件 {filename} 的所有索引数据"}


@app.post("/api/admin/reload")
async def reload_knowledge(_=Depends(require_admin)):
    """重新加载知识库"""
    reload_retriever()
    return {"status": "ok", "message": "知识库已重新加载"}


@app.post("/api/admin/knowledge/import/preview")
async def preview_excel_file(file: UploadFile = File(...), _=Depends(require_admin)):
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
async def import_excel_file(file: UploadFile = File(...), sheet_name: str = None, _=Depends(require_admin)):
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


@app.post("/api/admin/knowledge/import/document")
async def import_document_file(file: UploadFile = File(...), version: str = Form(""), doc_type: str = Form(""), _=Depends(require_admin)):
    """
    从文档文件解析并导入知识（支持 PDF、DOCX、PPTX）
    使用 Docling 解析，自动分离文本/表格/图片并向量化入库
    """
    allowed_suffixes = ('.pdf', '.docx', '.pptx', '.doc', '.ppt')
    if not any(file.filename.lower().endswith(s) for s in allowed_suffixes):
        raise HTTPException(status_code=400, detail="支持的格式: PDF、DOCX、PPTX")

    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_path = tmp_file.name

    try:
        from src.parser.docling_parser import parse_document_with_images
        from src.chunking.chunker import split_texts, summarize_table
        from src.indexing.indexer import index_document_elements
        from src.indexing.image_indexer import index_images
        from src.storage.object_store import get_media_store

        print(f"📄 开始解析文档: {file.filename}  version={version!r}  doc_type={doc_type!r}", flush=True)
        texts, tables, images, table_images = await asyncio.to_thread(
            parse_document_with_images, tmp_path
        )
        print(f"✅ 文档解析完成: 文本段={len(texts)}  表格={len(tables)}  图片={len(images)}", flush=True)

        inline_imgs  = [img for img in images if img.get("inline")]
        regular_imgs = [img for img in images if not img.get("inline")]

        media_store = get_media_store()

        placeholder_map: dict = {}
        if inline_imgs and media_store:
            for img in inline_imgs:
                try:
                    upload_result = media_store.upload_bytes(img["bytes"], img["filename"])
                    alt = img.get("caption") or "图示"
                    placeholder_map[img["placeholder_id"]] = f"![{alt}]({upload_result['url']})"
                except Exception as e:
                    print(f"⚠️ 内联图片上传失败 [{img['filename']}]: {e}")
                    placeholder_map[img["placeholder_id"]] = f"[{img.get('caption') or '图示'}]"

        table_image_urls = []
        table_image_object_keys = []
        if table_images and media_store:
            for i, img_bytes in enumerate(table_images):
                if img_bytes:
                    try:
                        upload_result = media_store.upload_bytes(img_bytes, f"table_{i+1:03d}.png")
                        table_image_urls.append(upload_result["url"])
                        table_image_object_keys.append(upload_result["object_key"])
                    except Exception as e:
                        print(f"⚠️ 表格图片上传失败 [table_{i+1:03d}]: {e}")
                        table_image_urls.append(None)
                        table_image_object_keys.append(None)
                else:
                    table_image_urls.append(None)
                    table_image_object_keys.append(None)
        else:
            table_image_urls = [None] * len(table_images) if table_images else []
            table_image_object_keys = [None] * len(table_images) if table_images else []

        text_chunks = split_texts(texts)
        if placeholder_map:
            text_chunks = [_replace_inline_placeholders(chunk, placeholder_map) for chunk in text_chunks]

        print(f"⚙️  生成表格摘要: {len(tables)} 个表格", flush=True)
        table_summaries = await asyncio.to_thread(
            lambda: [summarize_table(t) for t in tables]
        )

        print(f"📥 写入 Qdrant: {len(text_chunks)} 段文本 + {len(tables)} 个表格", flush=True)
        index_document_elements(
            text_chunks=text_chunks,
            tables=tables,
            table_summaries=table_summaries,
            table_image_urls=table_image_urls,
            table_image_object_keys=table_image_object_keys,
            images=[],
            image_descriptions=[],
            source_file=file.filename,
            version=version,
            doc_type=doc_type,
        )

        print(f"✅ Qdrant 写入完成", flush=True)
        indexed_image_ids = []
        if regular_imgs and media_store:
            print(f"🖼️  上传图片到对象存储: {len(regular_imgs)} 张", flush=True)
            indexed_image_ids = await asyncio.to_thread(
                index_images, regular_imgs, media_store, file.filename, version, doc_type
            )
        elif regular_imgs and not media_store:
            print("⚠️ 未配置对象存储，跳过图片索引")

        return {
            "status": "ok",
            "filename": file.filename,
            "indexed": {
                "text_chunks": len(text_chunks),
                "tables": len(tables),
                "images": len(indexed_image_ids),
            },
            "message": f"文档解析完成，已导入 {len(text_chunks)} 段文本、{len(tables)} 个表格、{len(indexed_image_ids)} 张图片"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文档解析失败: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/api/chat/media")
async def chat_with_media(
    question: str = Form(...),
    file: UploadFile = File(...),
    device_model: Optional[str] = Form(None),
    session_id: Optional[str] = Form(None),
):
    """
    用户在对话中上传图片或视频时调用
    multipart/form-data：question（表单字段）+ file（文件）
    """
    allowed_exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp",
                    ".mp4", ".avi", ".mov", ".mkv", ".webm"}
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in allowed_exts:
        raise HTTPException(status_code=400, detail=f"不支持的文件类型: {ext}")

    suffix = ext or ".tmp"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        from src.pipeline.realtime_media import handle_user_uploaded_media
        result = await handle_user_uploaded_media(
            file_path=tmp_path,
            question=question,
            device_model=device_model,
            session_id=session_id,
        )

        conversation_id = str(uuid.uuid4())
        save_conversation(
            conversation_id=conversation_id,
            device_model=device_model or "",
            question=question,
            answer=result["text_answer"],
            confidence="medium",
        )

        return ChatResponse(
            answer=result["text_answer"],
            conversation_id=conversation_id,
            confidence="medium",
            media=result["media"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"媒体处理失败: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ==================== 版本管理接口 ====================

@app.get("/api/admin/versions")
async def list_versions(_=Depends(require_admin)):
    """获取所有版本列表"""
    from database.version_registry import get_all_versions
    return {"items": get_all_versions()}


@app.get("/api/admin/versions/{version_code}")
async def get_version(version_code: str, _=Depends(require_admin)):
    """查询单个版本"""
    from database.version_registry import get_version as _get_version
    record = _get_version(version_code)
    if record is None:
        raise HTTPException(status_code=404, detail=f"版本 {version_code} 不存在")
    return record


@app.post("/api/admin/versions")
async def create_version(req: VersionCreateRequest, _=Depends(require_admin)):
    """新增版本"""
    from database.version_registry import create_version as _create_version
    try:
        record = _create_version(
            version_code=req.version_code,
            is_base=req.is_base,
            version_name=req.version_name or "",
            base_version_code=req.base_version_code,
            doc_type_label=req.doc_type_label or "",
        )
        return {"status": "ok", "item": record}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/admin/versions/{version_code}")
async def update_version(version_code: str, req: VersionUpdateRequest, _=Depends(require_admin)):
    """修改版本信息"""
    from database.version_registry import update_version as _update_version
    try:
        record = _update_version(
            version_code=version_code,
            version_name=req.version_name,
            is_base=req.is_base,
            base_version_code=req.base_version_code,
            doc_type_label=req.doc_type_label,
        )
        if record is None:
            raise HTTPException(status_code=404, detail=f"版本 {version_code} 不存在")
        return {"status": "ok", "item": record}
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/admin/versions/{version_code}")
async def delete_version(version_code: str, _=Depends(require_admin)):
    """删除版本记录"""
    from database.version_registry import delete_version as _delete_version
    deleted = _delete_version(version_code)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"版本 {version_code} 不存在")
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=HOST, port=PORT, reload=DEBUG)
