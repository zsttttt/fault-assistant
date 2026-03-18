"""
一次性脚本：将 Docling PDF 解析所需模型下载到本地 models/docling/ 目录。
使用 local_dir 模式下载，不创建符号链接，绕过 Windows WinError 1314 限制。

运行方式：
    python download_docling_models.py
"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

OUTPUT_DIR = Path("models/docling")

print(f"目标目录: {OUTPUT_DIR.resolve()}")
print(f"HF 镜像: {os.environ.get('HF_ENDPOINT', '未设置（使用官方地址）')}")
print()

from docling.utils.model_downloader import download_models

download_models(
    output_dir=OUTPUT_DIR,
    with_layout=True,
    with_tableformer=True,
    with_picture_classifier=True,
    with_code_formula=False,
    with_rapidocr=True,
    with_easyocr=False,
    progress=True,
)

print()
print(f"模型下载完成，路径：{OUTPUT_DIR.resolve()}")
print(f"请确认 .env 中已设置：DOCLING_ARTIFACTS_PATH=./models/docling")
