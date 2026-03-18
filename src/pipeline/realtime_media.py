"""
用户即时上传媒体处理管线

区别于"文档导入索引"流程，本模块处理用户在对话中直接发送的图片或视频：
  图片 → VLM 理解 → 增强查询 → 检索知识库 → 生成回答
  视频 → 关键帧 VLM 描述 → 增强查询 → 检索知识库 → 生成回答
        （视频不做永久索引，仅用于本次对话）
"""
import asyncio
import os
import tempfile
from pathlib import Path
from typing import Optional

from src.parser.image_describer import describe_image
from src.parser.video_processor import extract_keyframes, extract_audio_bytes
from src.retrieval.result_parser import parse_retrieved_results
from src.generation.multimodal_generator import generate_multimodal_answer

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

_MAX_FRAMES_REALTIME = 5


async def handle_user_uploaded_media(
    file_path: str,
    question: str,
    device_model: Optional[str] = None,
    session_id: Optional[str] = None,
) -> dict:
    """
    处理用户在对话中直接上传的图片或视频

    Args:
        file_path:    已保存到本地临时路径的媒体文件
        question:     用户提问文字
        device_model: 可选设备型号
        session_id:   可选会话 ID（多轮对话）

    Returns:
        {
            "text_answer": str,
            "media": {"images": [...], "videos": [...]},
            "media_description": str,  # VLM 对用户上传媒体的理解（调试用）
        }
    """
    ext = Path(file_path).suffix.lower()

    if ext in _IMAGE_EXTS:
        media_description = await _describe_image_async(file_path, question)
    elif ext in _VIDEO_EXTS:
        media_description = await _describe_video_async(file_path, question)
    else:
        media_description = ""

    enhanced_query = question
    if media_description:
        enhanced_query = f"{question}\n\n【用户上传媒体的内容描述】\n{media_description}"

    from rag.retriever import get_retriever
    retriever = get_retriever()
    context, confidence = await asyncio.to_thread(retriever.retrieve, enhanced_query)
    parsed = parse_retrieved_results(context)

    result = await generate_multimodal_answer(
        question=enhanced_query,
        parsed=parsed,
        confidence=confidence,
        device_model=device_model,
        session_id=session_id,
    )

    return {
        "text_answer": result["text_answer"],
        "media": result["media"],
        "media_description": media_description,
    }


async def _describe_image_async(file_path: str, context: str) -> str:
    """读取图片字节并调用 VLM 描述（在线程池中执行阻塞 I/O）"""
    def _read_and_describe():
        with open(file_path, "rb") as f:
            img_bytes = f.read()
        return describe_image(image_bytes=img_bytes, context=context)

    try:
        return await asyncio.to_thread(_read_and_describe)
    except Exception as e:
        print(f"⚠️ 用户图片 VLM 描述失败: {e}")
        return ""


async def _describe_video_async(file_path: str, context: str) -> str:
    """
    提取视频关键帧 → VLM 描述 → 合并为文本摘要
    不做音频转文字（实时场景，无需上传音频至 B2）
    """
    def _process():
        tmp_dir = tempfile.mkdtemp(prefix="realtime_frames_")
        try:
            frames = extract_keyframes(
                video_path=file_path,
                output_dir=tmp_dir,
                max_frames=_MAX_FRAMES_REALTIME,
            )
            descriptions = []
            for frame_info in frames:
                try:
                    with open(frame_info["frame_path"], "rb") as f:
                        frame_bytes = f.read()
                    desc = describe_image(
                        image_bytes=frame_bytes,
                        context=f"{context}（{frame_info['timestamp']} 秒处画面）",
                    )
                    descriptions.append(f"[{frame_info['timestamp']}s] {desc}")
                except Exception as e:
                    print(f"⚠️ 帧描述失败 [{frame_info['timestamp']}s]: {e}")
                finally:
                    if os.path.exists(frame_info["frame_path"]):
                        os.remove(frame_info["frame_path"])
            return "\n".join(descriptions)
        finally:
            try:
                os.rmdir(tmp_dir)
            except Exception:
                pass

    try:
        return await asyncio.to_thread(_process)
    except Exception as e:
        print(f"⚠️ 用户视频处理失败: {e}")
        return ""