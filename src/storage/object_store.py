"""
对象存储管理：统一封装 Backblaze B2 和 MinIO（S3 兼容协议）
开发阶段用 B2，生产阶段切 MinIO，只需改 config.py，业务代码无需修改
"""
import io
import os
import uuid
from pathlib import Path

import boto3
from botocore.config import Config


from typing import Optional

_CONTENT_TYPE_MAP = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".mp4": "video/mp4",
    ".avi": "video/x-msvideo",
    ".mov": "video/quicktime",
    ".mkv": "video/x-matroska",
    ".webm": "video/webm",
}

_PRESIGNED_EXPIRES = 7 * 24 * 3600
_media_store_instance: Optional["MediaStore"] = None


class MediaStore:
    """
    图片和视频文件的对象存储管理器
    兼容 Backblaze B2（开发）和 MinIO（生产）
    统一使用预签名 URL，不依赖公开 Bucket
    """

    def __init__(
        self,
        endpoint_url: str,
        access_key_id: str,
        secret_access_key: str,
        bucket_name: str,
        region: str = "us-west-004",
    ):
        """
        Args:
            endpoint_url:      S3 兼容端点
                B2:    https://s3.<region>.backblazeb2.com
                MinIO: http://minio.internal:9000
            access_key_id:     B2 Key ID / MinIO Access Key
            secret_access_key: B2 Application Key / MinIO Secret Key
            bucket_name:       Bucket 名称
            region:            S3 region（B2 如 "us-west-004"，MinIO 用 "us-east-1"）
        """
        self.bucket_name = bucket_name

        self.s3 = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            config=Config(
                region_name=region,
                signature_version="s3v4",
            ),
        )

    def upload_file(self, file_path: str, media_type: str = "image") -> dict:
        """
        上传本地文件到对象存储

        Returns:
            {
                "object_key": "images/abc123.png",
                "url": "https://... (预签名，7天有效)",
                "media_type": "image",
                "original_filename": "产品图1.png"
            }
        """
        ext = Path(file_path).suffix.lower()
        object_key = f"{media_type}s/{uuid.uuid4().hex}{ext}"
        content_type = _CONTENT_TYPE_MAP.get(ext, "application/octet-stream")

        self.s3.upload_file(
            file_path,
            self.bucket_name,
            object_key,
            ExtraArgs={"ContentType": content_type},
        )

        return {
            "object_key": object_key,
            "url": self._presigned_url(object_key),
            "media_type": media_type,
            "original_filename": os.path.basename(file_path),
        }

    def upload_bytes(self, data: bytes, filename: str, media_type: str = "image") -> dict:
        """
        从内存字节数据上传（用于 Docling 提取的内嵌图片）
        """
        ext = Path(filename).suffix.lower() or ".png"
        object_key = f"{media_type}s/{uuid.uuid4().hex}{ext}"
        content_type = _CONTENT_TYPE_MAP.get(ext, "application/octet-stream")

        self.s3.upload_fileobj(
            io.BytesIO(data),
            self.bucket_name,
            object_key,
            ExtraArgs={"ContentType": content_type},
        )

        return {
            "object_key": object_key,
            "url": self._presigned_url(object_key),
            "media_type": media_type,
            "original_filename": filename,
        }

    def delete_file(self, object_key: str):
        """删除对象存储中的文件"""
        self.s3.delete_object(Bucket=self.bucket_name, Key=object_key)

    def refresh_url(self, object_key: str, expires_in: int = _PRESIGNED_EXPIRES) -> str:
        """
        为已存储的文件重新生成预签名 URL（用于 URL 过期后刷新）

        Args:
            object_key: upload_file / upload_bytes 返回的 object_key
            expires_in: 有效期（秒），默认 7 天
        """
        return self._presigned_url(object_key, expires_in)

    def _presigned_url(self, object_key: str, expires_in: int = _PRESIGNED_EXPIRES) -> str:
        return self.s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket_name, "Key": object_key},
            ExpiresIn=expires_in,
        )


def get_media_store() -> Optional["MediaStore"]:
    """
    返回 MediaStore 单例（延迟初始化）
    未配置 B2/存储环境变量时返回 None（图片/视频功能降级跳过）
    """
    global _media_store_instance
    if _media_store_instance is None:
        from config import (
            STORAGE_ENDPOINT, STORAGE_ACCESS_KEY,
            STORAGE_SECRET_KEY, STORAGE_BUCKET, STORAGE_REGION,
        )
        if STORAGE_ENDPOINT and STORAGE_ACCESS_KEY and STORAGE_SECRET_KEY:
            _media_store_instance = MediaStore(
                endpoint_url=STORAGE_ENDPOINT,
                access_key_id=STORAGE_ACCESS_KEY,
                secret_access_key=STORAGE_SECRET_KEY,
                bucket_name=STORAGE_BUCKET,
                region=STORAGE_REGION,
            )
        else:
            return None
    return _media_store_instance