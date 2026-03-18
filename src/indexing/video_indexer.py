"""
视频索引模块：完整管线
  1. 上传原始视频到对象存储 → 获得持久化 URL
  2. 提取关键帧 → 用 qwen-vl-plus 描述每帧
  3. 提取音频 → 临时上传 B2 → DashScope paraformer 转文字 → 删除临时音频
  4. 合并为综合摘要 → 向量化存 Qdrant（metadata 含视频 URL）
  5. 完整媒体信息（JSON）存 InMemoryStore
"""
import json
import os
import time
import uuid
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document

from src.storage.object_store import MediaStore
from src.parser.video_processor import extract_keyframes, extract_audio_bytes
from src.parser.image_describer import describe_image
from src.indexing.indexer import get_vectorstore, get_docstore, ID_KEY
from config import DASHSCOPE_API_KEY


def process_and_index_video(
    video_path: str,
    media_store: MediaStore,
    source_file: str = "",
    max_frames: int = 10,
) -> Optional[str]:
    """
    视频完整处理流程

    Args:
        video_path:   视频本地文件路径
        media_store:  MediaStore 实例（已配置 B2 或 MinIO）
        source_file:  原始文档文件名（记录来源）
        max_frames:   最多描述帧数

    Returns:
        成功则返回 doc_id，失败返回 None
    """
    video_filename = Path(video_path).name

    upload_result = media_store.upload_file(video_path, media_type="video")
    video_url = upload_result["url"]

    frames = extract_keyframes(video_path, max_frames=max_frames)
    frame_descriptions: List[dict] = []
    frame_urls: List[dict] = []

    for frame_info in frames:
        try:
            with open(frame_info["frame_path"], "rb") as f:
                frame_bytes = f.read()

            desc = describe_image(
                image_bytes=frame_bytes,
                context=f"这是视频《{video_filename}》在 {frame_info['timestamp']} 秒处的画面",
            )
            frame_descriptions.append({
                "timestamp": frame_info["timestamp"],
                "description": desc,
            })

            frame_upload = media_store.upload_bytes(
                data=frame_bytes,
                filename=f"frame_{frame_info['timestamp']}s.jpg",
                media_type="image",
            )
            frame_urls.append({
                "timestamp": frame_info["timestamp"],
                "url": frame_upload["url"],
            })
        except Exception as e:
            print(f"⚠️ 关键帧处理失败 [{frame_info['timestamp']}s]: {e}")
        finally:
            if os.path.exists(frame_info["frame_path"]):
                os.remove(frame_info["frame_path"])

    audio_text = _extract_and_transcribe(video_path, media_store)

    combined_summary = _build_summary(
        video_filename=video_filename,
        frame_descriptions=frame_descriptions,
        audio_text=audio_text,
    )

    doc_id = str(uuid.uuid4())

    media_payload = json.dumps(
        {
            "type": "video",
            "url": video_url,
            "description": combined_summary,
            "filename": video_filename,
            "frame_urls": frame_urls,
            "audio_text": audio_text,
        },
        ensure_ascii=False,
    )

    get_vectorstore().add_documents([
        Document(
            page_content=combined_summary,
            metadata={
                ID_KEY: doc_id,
                "original_content": media_payload,
                "media_type": "video",
                "media_url": video_url,
                "source": source_file,
                "type": "video",
            },
        )
    ])

    get_docstore().mset([
        (
            doc_id,
            Document(
                page_content=media_payload,
                metadata={
                    "media_type": "video",
                    "media_url": video_url,
                    "source": source_file,
                    "type": "video",
                },
            ),
        )
    ])

    return doc_id


def _extract_and_transcribe(video_path: str, media_store: MediaStore) -> str:
    """
    提取音频字节 → 临时上传 B2 → DashScope paraformer 转文字 → 删除临时音频
    任意步骤失败均静默返回空字符串
    """
    audio_bytes = extract_audio_bytes(video_path)
    if not audio_bytes:
        return ""

    temp_key: Optional[str] = None
    try:
        upload_result = media_store.upload_bytes(
            data=audio_bytes,
            filename="temp_audio.mp3",
            media_type="video",
        )
        temp_key = upload_result["object_key"]
        audio_url = upload_result["url"]

        text = _dashscope_transcribe(audio_url)
        return text
    except Exception as e:
        print(f"⚠️ 音频转文字失败: {e}")
        return ""
    finally:
        if temp_key:
            try:
                media_store.delete_file(temp_key)
            except Exception:
                pass


def _dashscope_transcribe(audio_url: str) -> str:
    """
    调用 DashScope paraformer-v2 批量转写
    流程：async_call 提交 → 轮询 fetch → 解析转写结果 URL → 拼接文本
    """
    import requests
    from dashscope.audio.asr import Transcription

    response = Transcription.async_call(
        model="paraformer-v2",
        file_urls=[audio_url],
        language_hints=["zh"],
        api_key=DASHSCOPE_API_KEY,
    )

    if not (response and response.output):
        return ""

    task_id = response.output.task_id

    for _ in range(60):
        time.sleep(3)
        poll = Transcription.fetch(task_id=task_id, api_key=DASHSCOPE_API_KEY)
        status = getattr(getattr(poll, "output", None), "task_status", "")
        if status in ("SUCCEEDED", "FAILED"):
            response = poll
            break

    if getattr(getattr(response, "output", None), "task_status", "") != "SUCCEEDED":
        return ""

    results = getattr(response.output, "results", []) or []
    if not results:
        return ""

    transcript_url = results[0].get("transcription_url", "")
    if not transcript_url:
        return ""

    data = requests.get(transcript_url, timeout=15).json()
    sentences = (data.get("transcripts") or [{}])[0].get("sentences", [])
    return " ".join(s.get("text", "") for s in sentences if s.get("text"))


def _build_summary(
    video_filename: str,
    frame_descriptions: List[dict],
    audio_text: str,
) -> str:
    frames_text = "\n".join(
        f"[{fd['timestamp']}s] {fd['description']}" for fd in frame_descriptions
    )
    audio_section = audio_text if audio_text else "（无语音内容）"
    return (
        f"【视频内容摘要】\n"
        f"文件名：{video_filename}\n\n"
        f"画面描述：\n{frames_text}\n\n"
        f"音频内容：\n{audio_section}"
    )