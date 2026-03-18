"""
图片索引模块：完整管线
  1. 上传图片字节到对象存储（B2 / MinIO）→ 获得持久化 URL
  2. 用 qwen-vl-plus 生成图片描述（用于语义检索）
  3. 描述向量化后存入 Qdrant（metadata 含 URL）
  4. 完整媒体信息（JSON）存入 InMemoryStore
"""
import json
import uuid
from typing import List

from langchain_core.documents import Document

from src.storage.object_store import MediaStore
from src.parser.image_describer import describe_image
from src.indexing.indexer import get_vectorstore, get_docstore, ID_KEY


def index_images(
    images: List[dict],
    media_store: MediaStore,
    source_file: str = "",
) -> List[str]:
    """
    批量索引图片

    Args:
        images:       analyze_document_with_images 返回的图片列表
                      每项含 bytes / filename / caption / page_number / width / height / group_id
        media_store:  MediaStore 实例（已配置 B2 或 MinIO）
        source_file:  原始文档文件名（记录来源）

    Returns:
        已成功索引的 doc_id 列表
    """
    vectorstore = get_vectorstore()
    docstore = get_docstore()
    indexed_ids: List[str] = []

    for img_info in images:
        try:
            doc_id = _index_single_image(
                img_info=img_info,
                media_store=media_store,
                vectorstore=vectorstore,
                docstore=docstore,
                source_file=source_file,
            )
            indexed_ids.append(doc_id)
        except Exception as e:
            fname = img_info.get("filename", "unknown")
            print(f"⚠️ 图片索引失败 [{fname}]: {e}")

    return indexed_ids


def _index_single_image(
    img_info: dict,
    media_store: MediaStore,
    vectorstore,
    docstore,
    source_file: str,
) -> str:
    img_bytes = img_info["bytes"]
    filename = img_info.get("filename", "image.png")
    caption = img_info.get("caption", "")
    page_number = img_info.get("page_number")
    group_id = img_info.get("group_id")

    upload_result = media_store.upload_bytes(
        data=img_bytes,
        filename=filename,
        media_type="image",
    )
    media_url = upload_result["url"]
    object_key = upload_result["object_key"]

    description = describe_image(
        image_bytes=img_bytes,
        caption=caption,
    )

    doc_id = str(uuid.uuid4())

    media_payload = json.dumps(
        {
            "type": "image",
            "url": media_url,
            "description": description,
            "filename": filename,
            "group_id": group_id,
        },
        ensure_ascii=False,
    )

    vectorstore.add_documents([
        Document(
            page_content=description,
            metadata={
                ID_KEY: doc_id,
                "original_content": media_payload,
                "media_type": "image",
                "media_url": media_url,
                "object_key": object_key,
                "source": source_file,
                "type": "image",
                "page_number": page_number,
                "group_id": group_id,
            },
        )
    ])

    docstore.mset([
        (
            doc_id,
            Document(
                page_content=media_payload,
                metadata={
                    "media_type": "image",
                    "media_url": media_url,
                    "source": source_file,
                    "type": "image",
                    "group_id": group_id,
                },
            ),
        )
    ])

    return doc_id