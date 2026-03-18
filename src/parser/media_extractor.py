"""
从 Docling 解析结果中提取图片，返回图片字节 + 上下文信息
"""
import io
from typing import List


def extract_images_from_docling(doc) -> List[dict]:
    """
    从 Docling DoclingDocument 中提取所有图片

    Args:
        doc: Docling DoclingDocument 对象（parse_document 返回的 result.document）

    Returns:
        [
            {
                "bytes": bytes,          # 图片原始字节（PNG）
                "filename": str,         # 自动命名，如 "image_001.png"
                "caption": str,          # 图片标题（可能为空）
                "page_number": int|None, # 所在页码（PDF 有效）
            },
            ...
        ]
    """
    from docling.datamodel.document import PictureItem

    images = []

    for item, _ in doc.iterate_items():
        if not isinstance(item, PictureItem):
            continue

        pil_img = None
        try:
            pil_img = item.get_image(doc)
        except Exception:
            pass

        if pil_img is None:
            continue

        width, height = pil_img.size

        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        img_bytes = buf.getvalue()

        caption = ""
        try:
            caption = item.caption_text(doc) or ""
        except Exception:
            pass

        page_number = None
        try:
            if item.prov and len(item.prov) > 0:
                page_number = item.prov[0].page_no
        except Exception:
            pass

        idx = len(images) + 1
        images.append({
            "bytes": img_bytes,
            "filename": f"image_{idx:03d}.png",
            "caption": caption.strip(),
            "page_number": page_number,
            "width": width,
            "height": height,
        })

    return images
