"""
Docling 文档解析模块
支持 PDF、DOCX、PPTX 等格式，分离文本、表格、图片三类元素
"""
import gc
import io
import uuid
from typing import List, Optional, Tuple

_PDF_CHUNK_SIZE = 15
_INLINE_TEXT_THRESHOLD = 5


def _parse_md_col_count(md: str) -> int:
    """解析 Markdown 表格的列数（取第一行非分隔行）"""
    for line in md.splitlines():
        stripped = line.strip()
        if not (stripped.startswith('|') and stripped.endswith('|')):
            continue
        inner = stripped[1:-1]
        if all(c in '|-: ' for c in inner):
            continue
        return stripped.count('|') - 1
    return 0


def _get_md_data_rows(md: str) -> List[str]:
    """提取 Markdown 表格的数据行（跳过表头行和分隔行）"""
    lines = md.splitlines()
    sep_found = False
    data_rows = []
    for line in lines:
        stripped = line.strip()
        if not sep_found:
            if stripped.startswith('|') and stripped.endswith('|'):
                inner = stripped[1:-1]
                if all(c in '|-: ' for c in inner):
                    sep_found = True
        elif stripped:
            data_rows.append(line)
    return data_rows


def _merge_md_tables(md_list: List[str]) -> str:
    """将多个 Markdown 表格合并为一个（保留第一个的表头，追加后续的数据行）"""
    if not md_list:
        return ""
    if len(md_list) == 1:
        return md_list[0]
    result = md_list[0].rstrip()
    for md in md_list[1:]:
        rows = _get_md_data_rows(md)
        if rows:
            result += "\n" + "\n".join(rows)
    return result


def _build_table_merge_groups(doc):
    """
    分析 doc 中所有 TableItem，识别跨页分割的同一张表格。

    判断条件（三条同时满足）：
    1. 相邻 TableItem 页码差恰好为 1
    2. 两表之间（文档顺序）无有意义的 TextItem
       （忽略：page_header / page_footer、长度 ≤ 6 的纯数字/短横线文本页码、
              以"续表"/"（续）"/"(续)"开头的短标注）
    3. 两表列数相同

    Returns:
        groups:   List[List[TableItem]]，每组属于同一逻辑表格
        md_cache: {id(item): markdown}，供调用方复用，避免重复导出
    """
    from docling.datamodel.document import TextItem, TableItem

    IGNORED_LABELS = {"page_header", "page_footer"}

    all_items = list(doc.iterate_items())
    table_positions = []

    for idx, (item, _) in enumerate(all_items):
        if isinstance(item, TableItem):
            page_no = item.prov[0].page_no if item.prov else None
            table_positions.append((idx, item, page_no))

    if not table_positions:
        return [], {}

    md_cache: dict = {}

    def get_md(item):
        key = id(item)
        if key not in md_cache:
            md_cache[key] = item.export_to_markdown(doc)
        return md_cache[key]

    groups: List[List] = [[table_positions[0][1]]]

    for i in range(1, len(table_positions)):
        prev_idx, prev_item, prev_page = table_positions[i - 1]
        curr_idx, curr_item, curr_page = table_positions[i]

        if prev_page is None or curr_page is None or curr_page != prev_page + 1:
            groups.append([curr_item])
            continue

        has_significant = False
        for j in range(prev_idx + 1, curr_idx):
            mid_item, _ = all_items[j]
            if not isinstance(mid_item, TextItem):
                continue
            label = getattr(mid_item, "label", None)
            if label in IGNORED_LABELS:
                continue
            text = mid_item.text.strip()
            if not text:
                continue
            if len(text) <= 6 and all(c in '0123456789- \t' for c in text):
                continue
            if text.startswith(("续表", "（续）", "(续)")):
                continue
            has_significant = True
            break

        if has_significant:
            groups.append([curr_item])
            continue

        if _parse_md_col_count(get_md(prev_item)) != _parse_md_col_count(get_md(curr_item)):
            groups.append([curr_item])
            continue

        groups[-1].append(curr_item)

    return groups, md_cache


def _make_converter(extract_images: bool = False, table_structure: bool = True):
    import os
    from pathlib import Path
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.datamodel.base_models import InputFormat

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = table_structure
    pipeline_options.generate_picture_images = extract_images

    artifacts_path = os.getenv("DOCLING_ARTIFACTS_PATH", "")
    if artifacts_path:
        pipeline_options.artifacts_path = Path(artifacts_path)
        print(f"🔧 Docling 使用本地模型: {artifacts_path}  table_structure={table_structure}  extract_images={extract_images}", flush=True)
    else:
        print(f"⚠️  DOCLING_ARTIFACTS_PATH 未设置，Docling 将从 HuggingFace 下载模型（首次可能需要数分钟）", flush=True)

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )


def _get_pdf_page_count(file_path: str) -> int:
    import pypdfium2 as pdfium
    pdf = pdfium.PdfDocument(file_path)
    count = len(pdf)
    pdf.close()
    return count


def capture_table_images(pdf_path: str, doc) -> List[Optional[bytes]]:
    """
    对 doc 中的每张表格，用 pypdfium2 裁剪 PDF 页面区域渲染为 PNG。

    处理三种复杂情况：
    1. 同页多 prov（Docling ML 误合并相邻表格）→ 每页只保留面积最大的 prov
    2. 跨页表格（连续页码）→ 竖向拼接
    3. 非连续页码引用 → 只取面积最大的单页

    坐标系自动识别：
    - bbox.t >= bbox.b → BOTTOMLEFT，y_pixel = (page_h - bbox_y) * scale
    - bbox.t <  bbox.b → TOPLEFT，  y_pixel = bbox_y * scale
    """
    from PIL import Image as PILImage

    def _crop_prov(pdf, prov, scale: float):
        try:
            page_h  = pdf[prov.page_no - 1].get_height()
            bbox    = prov.bbox
            pil_img = pdf[prov.page_no - 1].render(scale=scale).to_pil()

            if bbox.t >= bbox.b:        # BOTTOMLEFT（PDF 标准）
                raw_y0 = (page_h - bbox.t) * scale
                raw_y1 = (page_h - bbox.b) * scale
            else:                        # TOPLEFT
                raw_y0 = bbox.t * scale
                raw_y1 = bbox.b * scale

            pad = 8
            x0 = max(0,              int(bbox.l * scale) - pad)
            y0 = max(0,              int(raw_y0) - pad)
            x1 = min(pil_img.width,  int(bbox.r * scale) + pad)
            y1 = min(pil_img.height, int(raw_y1) + pad)

            if x0 >= x1 or y0 >= y1:
                return None
            return pil_img.crop((x0, y0, x1, y1))
        except Exception as e:
            print(f"⚠️ prov 裁剪失败 (页{prov.page_no}): {e}")
            return None

    groups, _ = _build_table_merge_groups(doc)

    result: List[Optional[bytes]] = []
    try:
        import pypdfium2 as pdfium
        pdf = pdfium.PdfDocument(pdf_path)
    except Exception as e:
        print(f"⚠️ 打开 PDF 失败，跳过表格图片捕获: {e}")
        return [None] * len(groups)

    scale = 2.0
    try:
        for group in groups:
            all_provs = []
            for item in group:
                all_provs.extend(item.prov or [])

            if not all_provs:
                result.append(None)
                continue

            try:
                # 每页只保留面积最大的 prov（去除同页重复/ML误合并）
                page_best: dict = {}
                for p in all_provs:
                    area = abs((p.bbox.r - p.bbox.l) * (p.bbox.t - p.bbox.b))
                    if p.page_no not in page_best or area > page_best[p.page_no][1]:
                        page_best[p.page_no] = (p, area)

                sorted_pages = sorted(page_best.keys())

                # 非连续页码 → 只取面积最大的单页
                if len(sorted_pages) > 1:
                    all_consecutive = all(
                        sorted_pages[i + 1] == sorted_pages[i] + 1
                        for i in range(len(sorted_pages) - 1)
                    )
                    if not all_consecutive:
                        best_pno = max(page_best, key=lambda k: page_best[k][1])
                        sorted_pages = [best_pno]

                # 裁剪每页区域（含坐标系自动识别），连续多页竖向拼接
                parts = []
                for pno in sorted_pages:
                    crop = _crop_prov(pdf, page_best[pno][0], scale)
                    if crop is not None:
                        parts.append(crop)

                if not parts:
                    result.append(None)
                    continue

                if len(parts) == 1:
                    final_img = parts[0]
                else:
                    gap = 6
                    total_h = sum(p.height for p in parts) + gap * (len(parts) - 1)
                    max_w   = max(p.width  for p in parts)
                    final_img = PILImage.new("RGB", (max_w, total_h), (255, 255, 255))
                    y_off = 0
                    for part in parts:
                        final_img.paste(part, (0, y_off))
                        y_off += part.height + gap

                buf = io.BytesIO()
                final_img.save(buf, format="PNG")
                result.append(buf.getvalue())
            except Exception as e:
                print(f"⚠️ 表格图片处理失败: {e}")
                result.append(None)
    finally:
        pdf.close()

    return result


def _convert_chunked(file_path: str, with_images: bool):
    total = _get_pdf_page_count(file_path)
    all_texts, all_tables, all_images, all_table_images = [], [], [], []

    if not with_images:
        # 不需要图片：文本 + 表格一遍完成
        conv_text = _make_converter(extract_images=False, table_structure=True)
        for start in range(1, total + 1, _PDF_CHUNK_SIZE):
            end = min(start + _PDF_CHUNK_SIZE - 1, total)
            result = conv_text.convert(file_path, page_range=(start, end))
            texts, tables, _ = separate_elements(result.document)
            all_texts.extend(texts)
            all_tables.extend(tables)
            del result
            gc.collect()
        return all_texts, all_tables, [], []

    # 第一遍：图片渲染（关闭表格 ML），采集文本（含内联图片占位符）+ 所有图片
    conv_img = _make_converter(extract_images=True, table_structure=False)
    for start in range(1, total + 1, _PDF_CHUNK_SIZE):
        end = min(start + _PDF_CHUNK_SIZE - 1, total)
        result = conv_img.convert(file_path, page_range=(start, end))
        texts, _, imgs = analyze_document_with_images(result.document, img_offset=len(all_images))
        all_texts.extend(texts)
        all_images.extend(imgs)
        del result
        gc.collect()

    # 第二遍：表格结构 ML（关闭图片渲染），采集表格 + 渲染表格图片
    conv_text = _make_converter(extract_images=False, table_structure=True)
    for start in range(1, total + 1, _PDF_CHUNK_SIZE):
        end = min(start + _PDF_CHUNK_SIZE - 1, total)
        result = conv_text.convert(file_path, page_range=(start, end))
        _, tables, _ = separate_elements(result.document)
        all_tables.extend(tables)
        all_table_images.extend(capture_table_images(file_path, result.document))
        del result
        gc.collect()

    return all_texts, all_tables, all_images, all_table_images


def parse_document(file_path: str) -> Tuple[List[str], List[str], List[dict]]:
    """
    解析单个文档，返回 (文本列表, 表格列表, 图片列表)
    """
    if file_path.lower().endswith(".pdf") and _get_pdf_page_count(file_path) > _PDF_CHUNK_SIZE:
        texts, tables, images, _ = _convert_chunked(file_path, with_images=False)
        return texts, tables, images

    converter = _make_converter()
    result = converter.convert(file_path)
    return separate_elements(result.document)


def parse_document_with_images(file_path: str):
    """
    解析文档并提取实际图片字节（用于 B2 上传 + VLM 描述管线）

    PDF 采用两遍处理（文本/表格遍 + 图片遍），分离内存峰值避免 std::bad_alloc。
    DOCX/PPTX 单遍处理（无 PDF 表格 ML 内存问题）。

    Returns:
        (texts, tables, images_with_bytes)
        images_with_bytes 每项含 bytes/filename/caption/page_number/width/height/group_id
    """
    if file_path.lower().endswith(".pdf"):
        if _get_pdf_page_count(file_path) > _PDF_CHUNK_SIZE:
            return _convert_chunked(file_path, with_images=True)

        # 小 PDF：同样两遍处理，图片遍先，表格遍后
        conv_img = _make_converter(extract_images=True, table_structure=False)
        result = conv_img.convert(file_path)
        texts, _, images = analyze_document_with_images(result.document)
        del result
        gc.collect()

        conv_text = _make_converter(extract_images=False, table_structure=True)
        result = conv_text.convert(file_path)
        _, tables, _ = separate_elements(result.document)
        table_images = capture_table_images(file_path, result.document)
        return texts, tables, images, table_images

    # DOCX / PPTX：单遍处理（无法用 pypdfium2 裁剪，表格图片留空）
    converter = _make_converter(extract_images=True)
    result = converter.convert(file_path)
    texts, tables, images = analyze_document_with_images(result.document)
    return texts, tables, images, [None] * len(tables)


def _collect_inline_bboxes(doc) -> dict:
    """
    前置扫描：收集所有内联图片（尺寸阈值内）的包围框。
    只读 prov.bbox，不渲染图片，开销极低。
    返回 {page_no: [bbox, ...]}
    """
    from docling.datamodel.document import PictureItem

    inline_bboxes: dict = {}
    for item, _ in doc.iterate_items():
        if not isinstance(item, PictureItem):
            continue
        if not item.prov:
            continue
        try:
            bbox = item.prov[0].bbox
            pno = item.prov[0].page_no
            w = abs(bbox.r - bbox.l)
            h = abs(bbox.t - bbox.b)
            # 用稍宽松的点坐标阈值（PDF points ≈ pixels @72dpi）
            if w < 350 and h < 250:
                inline_bboxes.setdefault(pno, []).append(bbox)
        except Exception:
            pass
    return inline_bboxes


def _text_in_inline_bbox(item, inline_bboxes: dict) -> bool:
    """
    判断 TextItem 的中心点是否落在任意内联图片包围框内。
    使用中心点而非全框，避免误杀与图片相邻的正文。
    """
    if not item.prov:
        return False
    try:
        prov = item.prov[0]
        pno = prov.page_no
        bboxes = inline_bboxes.get(pno)
        if not bboxes:
            return False
        tb = prov.bbox
        cx = (tb.l + tb.r) / 2
        cy = (tb.t + tb.b) / 2
        for ib in bboxes:
            if (min(ib.l, ib.r) <= cx <= max(ib.l, ib.r) and
                    min(ib.t, ib.b) <= cy <= max(ib.t, ib.b)):
                return True
    except Exception:
        pass
    return False


def analyze_document_with_images(doc, img_offset: int = 0) -> Tuple[List[str], List[str], List[dict]]:
    """
    单次遍历文档，分离文本/表格/图片并完成：
    - 内联图片识别（宽<300 且 高<200）→ 将图注嵌入文本，不单独索引
    - 顺序图片分组（相邻图片之间文字 < _INLINE_TEXT_THRESHOLD 个字符）→ 共享 group_id
    每张独立图片含 bytes/filename/caption/page_number/width/height/group_id
    """
    from docling.datamodel.document import TextItem, TableItem, PictureItem

    # 前置扫描：收集全部内联图片 bbox，用于过滤图内文字
    # 预先收集可解决 PDF 文字层乱序导致的图文绑定错乱问题
    inline_bboxes = _collect_inline_bboxes(doc)

    groups, md_cache = _build_table_merge_groups(doc)
    skip_ids: set = set()
    first_md: dict = {}
    for group in groups:
        md_list = [md_cache.get(id(item), item.export_to_markdown(doc)) for item in group]
        first_md[id(group[0])] = _merge_md_tables(md_list)
        for item in group[1:]:
            skip_ids.add(id(item))

    texts: List[str] = []
    tables: List[str] = []
    images: List[dict] = []
    img_group: List[dict] = []
    img_counter = [img_offset]

    def finalize_group():
        if not img_group:
            return
        gid = str(uuid.uuid4())
        for entry in img_group:
            entry["group_id"] = gid
            images.append(entry)
        img_group.clear()

    for item, _ in doc.iterate_items():
        if isinstance(item, TableItem):
            if id(item) in skip_ids:
                continue
            finalize_group()
            tables.append(first_md.get(id(item), item.export_to_markdown(doc)))

        elif isinstance(item, PictureItem):
            pil_img = None
            try:
                pil_img = item.get_image(doc)
            except Exception:
                pass
            if pil_img is None:
                continue

            width, height = pil_img.size

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

            if width < 300 and height < 200:
                finalize_group()
                pid = str(uuid.uuid4())
                buf = io.BytesIO()
                pil_img.save(buf, format="PNG")
                img_counter[0] += 1
                texts.append(f"[[IMG:{pid}]]")
                img_group.append({
                    "bytes": buf.getvalue(),
                    "filename": f"inline_{img_counter[0]:03d}.png",
                    "caption": caption.strip(),
                    "page_number": page_number,
                    "width": width,
                    "height": height,
                    "group_id": None,
                    "inline": True,
                    "placeholder_id": pid,
                })
                finalize_group()
            else:
                buf = io.BytesIO()
                pil_img.save(buf, format="PNG")
                img_counter[0] += 1
                img_group.append({
                    "bytes": buf.getvalue(),
                    "filename": f"image_{img_counter[0]:03d}.png",
                    "caption": caption.strip(),
                    "page_number": page_number,
                    "width": width,
                    "height": height,
                    "group_id": None,
                })

        elif isinstance(item, TextItem):
            if getattr(item, "label", None) in ("page_header", "page_footer"):
                continue
            text = item.text.strip()
            if not text:
                continue
            if _text_in_inline_bbox(item, inline_bboxes):
                continue
            if len(text) >= _INLINE_TEXT_THRESHOLD:
                finalize_group()
            texts.append(text)

    finalize_group()
    return texts, tables, images


def separate_elements(doc) -> Tuple[List[str], List[str], List[dict]]:
    """
    从 Docling DoclingDocument 中分离文本、表格、图片三类元素
    """
    from docling.datamodel.document import TextItem, TableItem, PictureItem

    groups, md_cache = _build_table_merge_groups(doc)
    skip_ids: set = set()
    first_md: dict = {}
    for group in groups:
        md_list = [md_cache.get(id(item), item.export_to_markdown(doc)) for item in group]
        first_md[id(group[0])] = _merge_md_tables(md_list)
        for item in group[1:]:
            skip_ids.add(id(item))

    texts: List[str] = []
    tables: List[str] = []
    images: List[dict] = []

    for item, _ in doc.iterate_items():
        if isinstance(item, TableItem):
            if id(item) in skip_ids:
                continue
            tables.append(first_md.get(id(item), item.export_to_markdown(doc)))
        elif isinstance(item, PictureItem):
            caption = ""
            if hasattr(item, "caption_text"):
                try:
                    caption = item.caption_text(doc)
                except Exception:
                    pass
            images.append({"caption": caption, "ref": str(item.get_ref()) if hasattr(item, "get_ref") else ""})
        elif isinstance(item, TextItem):
            if getattr(item, "label", None) in ("page_header", "page_footer"):
                continue
            if item.text.strip():
                texts.append(item.text)

    return texts, tables, images
