# 售后客服 RAG — 多模态增强：图片与视频的输入输出

## 背景

在前一次迁移（Docling + LangChain + Qdrant + Redis）的基础上，本次需要增加**图片和视频的完整处理能力**，包括：
1. **输入端**：用户上传的文档中包含图片和视频，需要被理解和索引
2. **输出端**：回答用户问题时，需要将相关的图片和视频原样返回给用户展示

---

## 新增架构总览

```
用户上传文档（含文本/表格/图片/视频）
    ↓
Docling 解析 + 自定义媒体提取
    ├── 文本 → 切块 → 嵌入 → Qdrant
    ├── 表格 → LLM 摘要 → 嵌入 → Qdrant（原表格存 Redis）
    ├── 图片 → 存 Cloudflare R2(得到URL) → VLM 生成描述 → 嵌入 → Qdrant（metadata 含 URL）
    └── 视频 → 存 Cloudflare R2(得到URL) → 抽关键帧描述 + 音频转文字 → 嵌入 → Qdrant（metadata 含 URL）

用户提问
    ↓
检索 → 取回文本 + 表格 + 图片URL + 视频URL
    ↓
LLM 生成文字回答，结构化输出包含引用的媒体资源
    ↓
前端渲染：文字回答 + 内嵌图片展示 + 视频播放器
```

---

## 第一步：新增依赖

在现有依赖基础上新增：

```bash
# 对象存储：使用 Cloudflare R2（S3 兼容，通过 boto3 访问）
pip install boto3

# 视频处理
pip install opencv-python  # 视频关键帧提取
pip install moviepy        # 视频音频分离

# 音频转文字（视频中的语音）
pip install openai-whisper  # 本地 Whisper 模型
# 或者直接用 OpenAI Whisper API（无需额外安装）

# 图片处理
pip install Pillow          # 图片格式转换和压缩
```

---

## 第二步：对象存储模块（Cloudflare R2）

图片和视频的原始文件需要存储在对象存储中，以便通过 URL 返回给前端展示。

### 为什么不用 base64 存 Redis？
- 图片 base64 体积膨胀约 33%，大图片占用 Redis 内存严重
- 视频文件通常几十 MB 以上，绝对不能存 Redis
- URL 方式对前端最友好，可直接 `<img src="url">` 和 `<video src="url">` 渲染

### 为什么选 Cloudflare R2？
- **永久免费层**：10 GB 存储 + 每月 100 万次写入 + 1000 万次读取
- **零出流量费**：用户查看图片/视频时不产生任何下载费用（这是最大优势）
- **S3 兼容 API**：用 boto3 即可操作，以后迁移到 AWS S3 或 MinIO 只改配置即可
- 开发阶段 10 GB 足够存储数千张图片和数十个视频

### Cloudflare R2 配置步骤

1. 注册 Cloudflare 账号：https://dash.cloudflare.com/sign-up
2. 进入 R2 Object Storage → 创建 Bucket（如 `after-sales-media`）
3. 在 R2 设置中创建 API Token（选择 "Admin Read & Write" 权限），获得：
   - `Account ID`
   - `Access Key ID`
   - `Secret Access Key`
4. 如果需要公开访问图片/视频 URL：
   - 方式一：在 Bucket Settings 中启用 "Public Access"（绑定自定义域名或使用 r2.dev 子域名）
   - 方式二：使用预签名 URL（Presigned URL），有时效性，更安全

### 实现参考

```python
# src/storage/object_store.py

import boto3
from botocore.config import Config
import uuid
import os
from pathlib import Path

class MediaStore:
    """Cloudflare R2 对象存储管理器，负责图片和视频文件的存储与 URL 生成"""
    
    def __init__(self, account_id, access_key_id, secret_access_key, 
                 bucket_name, public_domain=None):
        """
        Args:
            account_id: Cloudflare Account ID
            access_key_id: R2 API Token 的 Access Key ID
            secret_access_key: R2 API Token 的 Secret Access Key
            bucket_name: R2 Bucket 名称
            public_domain: 公开访问域名（如 "media.example.com" 或 "xxx.r2.dev"）
                           如果不设置，则使用预签名 URL
        """
        self.bucket_name = bucket_name
        self.public_domain = public_domain
        
        # Cloudflare R2 的 S3 兼容 endpoint
        endpoint_url = f"https://{account_id}.r2.cloudflarestorage.com"
        
        self.s3_client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            config=Config(
                region_name="auto",
                signature_version="s3v4",
            ),
        )
    
    def upload_file(self, file_path: str, media_type: str = "image") -> dict:
        """
        上传文件到 Cloudflare R2
        
        Args:
            file_path: 本地文件路径
            media_type: "image" 或 "video"
        
        Returns:
            {
                "object_key": "images/xxx-xxx.png",
                "url": "https://media.example.com/images/xxx-xxx.png",
                "media_type": "image",
                "original_filename": "产品图1.png"
            }
        """
        ext = Path(file_path).suffix.lower()
        object_key = f"{media_type}s/{uuid.uuid4().hex}{ext}"
        
        # 设置 Content-Type
        content_type_map = {
            ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".png": "image/png", ".gif": "image/gif",
            ".webp": "image/webp", ".bmp": "image/bmp",
            ".mp4": "video/mp4", ".avi": "video/x-msvideo",
            ".mov": "video/quicktime", ".mkv": "video/x-matroska",
            ".webm": "video/webm",
        }
        content_type = content_type_map.get(ext, "application/octet-stream")
        
        # 上传到 R2
        self.s3_client.upload_file(
            file_path,
            self.bucket_name,
            object_key,
            ExtraArgs={"ContentType": content_type},
        )
        
        # 生成访问 URL
        url = self._get_url(object_key)
        
        return {
            "object_key": object_key,
            "url": url,
            "media_type": media_type,
            "original_filename": os.path.basename(file_path),
        }
    
    def upload_bytes(self, data: bytes, filename: str, media_type: str = "image") -> dict:
        """从内存中的字节数据上传（用于 Docling 提取的内嵌图片）"""
        import io
        ext = Path(filename).suffix.lower() or ".png"
        object_key = f"{media_type}s/{uuid.uuid4().hex}{ext}"
        
        self.s3_client.upload_fileobj(
            io.BytesIO(data),
            self.bucket_name,
            object_key,
        )
        
        url = self._get_url(object_key)
        
        return {
            "object_key": object_key,
            "url": url,
            "media_type": media_type,
            "original_filename": filename,
        }
    
    def _get_url(self, object_key: str) -> str:
        """生成文件的访问 URL"""
        if self.public_domain:
            # 方式一：公开访问域名（推荐开发阶段使用 r2.dev 子域名）
            return f"https://{self.public_domain}/{object_key}"
        else:
            # 方式二：预签名 URL（有效期 7 天，更安全）
            return self.s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket_name, "Key": object_key},
                ExpiresIn=7 * 24 * 3600,  # 7 天
            )
    
    def delete_file(self, object_key: str):
        """删除文件"""
        self.s3_client.delete_object(Bucket=self.bucket_name, Key=object_key)
```

### config/settings.py 新增配置

```python
# Cloudflare R2 对象存储配置
R2_ACCOUNT_ID = os.getenv("R2_ACCOUNT_ID", "你的_cloudflare_account_id")
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID", "你的_r2_access_key_id")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY", "你的_r2_secret_access_key")
R2_BUCKET_NAME = os.getenv("R2_BUCKET_NAME", "after-sales-media")

# 公开访问域名（在 R2 Bucket Settings 中开启 Public Access 后获得）
# 开发阶段可使用 r2.dev 子域名，如 "pub-xxxxx.r2.dev"
# 生产环境建议绑定自定义域名，如 "media.yourcompany.com"
R2_PUBLIC_DOMAIN = os.getenv("R2_PUBLIC_DOMAIN", "pub-xxxxx.r2.dev")
```

### 初始化 MediaStore

```python
from src.storage.object_store import MediaStore
from config.settings import *

media_store = MediaStore(
    account_id=R2_ACCOUNT_ID,
    access_key_id=R2_ACCESS_KEY_ID,
    secret_access_key=R2_SECRET_ACCESS_KEY,
    bucket_name=R2_BUCKET_NAME,
    public_domain=R2_PUBLIC_DOMAIN,  # 设为 None 则使用预签名 URL
)
```

### 生产环境本地部署：切换到 MinIO

公司内部资料不适合放云端，生产环境可以用 **MinIO** 本地部署。MinIO 是最流行的自建对象存储，S3 完全兼容，Docker 一行命令启动。

**MediaStore 代码完全不需要改**，只需要修改 config/settings.py 中的配置：

```python
# ============================================================
# 开发阶段：Cloudflare R2（当前使用）
# ============================================================
STORAGE_ENDPOINT = f"https://{os.getenv('R2_ACCOUNT_ID')}.r2.cloudflarestorage.com"
STORAGE_ACCESS_KEY = os.getenv("R2_ACCESS_KEY_ID")
STORAGE_SECRET_KEY = os.getenv("R2_SECRET_ACCESS_KEY")
STORAGE_BUCKET = os.getenv("R2_BUCKET_NAME", "after-sales-media")
STORAGE_PUBLIC_DOMAIN = os.getenv("STORAGE_PUBLIC_DOMAIN", "pub-xxxxx.r2.dev")
STORAGE_REGION = "auto"

# ============================================================
# 生产阶段：MinIO 本地部署（切换时取消注释，注释掉上面的）
# ============================================================
# STORAGE_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://minio.internal:9000")
# STORAGE_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
# STORAGE_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
# STORAGE_BUCKET = os.getenv("MINIO_BUCKET", "after-sales-media")
# STORAGE_PUBLIC_DOMAIN = os.getenv("STORAGE_PUBLIC_DOMAIN", "minio.internal:9000/after-sales-media")
# STORAGE_REGION = "us-east-1"  # MinIO 默认 region
```

**对应地，MediaStore 的初始化改为读取统一配置：**

```python
# 统一初始化方式（开发/生产通用，只靠 config 切换）
media_store = MediaStore(
    account_id=None,           # MinIO 不需要 account_id
    access_key_id=STORAGE_ACCESS_KEY,
    secret_access_key=STORAGE_SECRET_KEY,
    bucket_name=STORAGE_BUCKET,
    public_domain=STORAGE_PUBLIC_DOMAIN,
    endpoint_url=STORAGE_ENDPOINT,  # 直接传入 endpoint
)
```

> 注意：为了支持这种统一初始化，需要小改一下 MediaStore 的 `__init__`，让它接受 `endpoint_url` 参数而不是从 `account_id` 拼接。改动很小，见下方。

**MediaStore.__init__ 改造（兼容 R2 和 MinIO）：**

```python
def __init__(self, endpoint_url, access_key_id, secret_access_key,
             bucket_name, public_domain=None, region="auto",
             account_id=None):
    """
    统一初始化，同时兼容 Cloudflare R2 和 MinIO
    
    开发阶段（R2）：
        endpoint_url = "https://{account_id}.r2.cloudflarestorage.com"
        或者直接传 account_id，自动拼接
    
    生产阶段（MinIO）：
        endpoint_url = "http://minio.internal:9000"
    """
    self.bucket_name = bucket_name
    self.public_domain = public_domain
    
    if account_id and not endpoint_url:
        endpoint_url = f"https://{account_id}.r2.cloudflarestorage.com"
    
    self.s3_client = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
        config=Config(
            region_name=region,
            signature_version="s3v4",
        ),
    )
```

**MinIO 部署只需一行 Docker 命令：**

```bash
# 启动 MinIO（数据持久化到 ./minio-data 目录）
docker run -d \
  --name minio \
  -p 9000:9000 \
  -p 9001:9001 \
  -v ./minio-data:/data \
  -e MINIO_ROOT_USER=minioadmin \
  -e MINIO_ROOT_PASSWORD=your_secure_password \
  minio/minio server /data --console-address ":9001"

# 访问 MinIO 控制台：http://localhost:9001
# 在控制台中创建 Bucket: after-sales-media
# 设置 Bucket Policy 为 public（或使用预签名 URL）
```

**迁移路径总结：**

```
开发阶段                              生产阶段
─────────────                    ──────────────
Cloudflare R2 (云端)       →     MinIO (本地/内网)
Qdrant Cloud (云端)        →     Qdrant 本地部署 (Docker)
Redis (本地/云端)          →     Redis (本地)

需要改的：只有 config/settings.py 中的连接地址和密钥
不需要改的：所有业务代码、MediaStore、索引逻辑、检索逻辑
```

---

## 第三步：图片处理管线

### 3.1 从文档中提取图片

```python
# src/parser/media_extractor.py

import base64
from pathlib import Path
from PIL import Image
import io

def extract_images_from_docling(doc, output_dir: str = "./temp_images") -> list:
    """
    从 Docling 解析结果中提取所有图片
    
    Returns:
        [
            {
                "local_path": "/tmp/images/img_001.png",
                "page_number": 3,
                "context": "图片上方的标题或说明文字（如果有）"
            },
            ...
        ]
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    images = []
    
    # Docling 的 DoclingDocument 提供图片提取能力
    # 遍历文档中的图片元素，保存到本地临时目录
    # 同时记录图片所在的页码和上下文信息
    
    # 请根据 Docling 最新 API 实现具体提取逻辑
    # doc.export_to_dict() 中的 pictures/images 字段包含图片信息
    
    return images
```

### 3.2 处理用户直接上传的图片文件

```python
def process_uploaded_images(file_paths: list) -> list:
    """
    处理用户单独上传的图片文件（非嵌入在文档中的）
    
    Args:
        file_paths: 图片文件路径列表
    
    Returns:
        同上格式的图片信息列表
    """
    images = []
    for path in file_paths:
        # 验证是否为有效图片
        try:
            img = Image.open(path)
            img.verify()
            images.append({
                "local_path": path,
                "page_number": None,
                "context": f"用户上传的图片: {Path(path).name}"
            })
        except Exception:
            continue
    return images
```

### 3.3 图片描述生成（用于检索索引）

```python
# src/parser/image_describer.py

import base64
from langchain_openai import ChatOpenAI

def encode_image_to_base64(image_path: str) -> str:
    """将图片转为 base64 字符串"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def describe_image(image_path: str, context: str = "") -> str:
    """
    用多模态 LLM 为图片生成详细描述
    
    Args:
        image_path: 图片本地路径
        context: 图片在文档中的上下文（标题、说明文字等）
    
    Returns:
        图片的自然语言描述
    """
    base64_image = encode_image_to_base64(image_path)
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"""请用中文详细描述这张图片的内容。这是一个售后客服知识库中的图片。
                    
描述要求：
1. 描述图片中显示的产品、零件、流程或现象
2. 如果是产品图，指出产品型号、外观特征
3. 如果是流程图/示意图，描述流程步骤
4. 如果是故障/问题图片，描述故障现象
5. 便于通过文字搜索找到这张图片

图片的文档上下文：{context if context else '无'}"""
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        }
    ]
    
    response = llm.invoke(messages)
    return response.content
```

### 3.4 图片索引（上传存储 + 向量化）

```python
# src/indexing/image_indexer.py

import uuid
from langchain.schema import Document

def index_images(images: list, media_store, retriever, describe_fn):
    """
    完整的图片索引流程：
    1. 上传图片到对象存储，获得 URL
    2. 用 VLM 生成描述
    3. 描述做嵌入存入 Qdrant（metadata 中存 URL、类型、原始文件名）
    4. 原始描述 + URL 存入 Redis docstore
    
    Args:
        images: extract_images 返回的图片列表
        media_store: MediaStore 实例
        retriever: MultiVectorRetriever 实例
        describe_fn: 图片描述函数
    """
    id_key = "doc_id"
    
    for img_info in images:
        # 1. 上传到对象存储
        upload_result = media_store.upload_file(
            img_info["local_path"], 
            media_type="image"
        )
        
        # 2. 生成描述
        description = describe_fn(
            img_info["local_path"],
            context=img_info.get("context", "")
        )
        
        # 3. 构建文档并索引
        doc_id = str(uuid.uuid4())
        
        # 嵌入到 Qdrant 的文档（用描述做检索）
        summary_doc = Document(
            page_content=description,
            metadata={
                id_key: doc_id,
                "media_type": "image",
                "media_url": upload_result["url"],
                "original_filename": upload_result["original_filename"],
                "page_number": img_info.get("page_number"),
            }
        )
        retriever.vectorstore.add_documents([summary_doc])
        
        # 4. 存入 Redis docstore（检索命中后返回的完整信息）
        # 这里存一个结构化的 dict 字符串，包含描述、URL、类型
        import json
        original_content = json.dumps({
            "type": "image",
            "description": description,
            "url": upload_result["url"],
            "filename": upload_result["original_filename"],
        }, ensure_ascii=False)
        
        retriever.docstore.mset([(doc_id, original_content)])
```

---

## 第四步：视频处理管线

### 4.1 视频关键帧提取

```python
# src/parser/video_processor.py

import cv2
import os
from pathlib import Path
import numpy as np

def extract_keyframes(video_path: str, output_dir: str = "./temp_frames",
                      max_frames: int = 10, method: str = "uniform") -> list:
    """
    从视频中提取关键帧
    
    Args:
        video_path: 视频文件路径
        output_dir: 关键帧输出目录
        max_frames: 最多提取帧数
        method: 提取方法
            - "uniform": 均匀采样（简单可靠）
            - "scene_change": 基于场景变化检测（更智能）
    
    Returns:
        [
            {"frame_path": "/tmp/frames/frame_001.jpg", "timestamp": 5.0},
            ...
        ]
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    
    frames = []
    
    if method == "uniform":
        # 均匀采样
        interval = max(1, total_frames // max_frames)
        for i in range(0, total_frames, interval):
            if len(frames) >= max_frames:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                timestamp = i / fps if fps > 0 else 0
                frame_path = os.path.join(output_dir, f"frame_{len(frames):03d}.jpg")
                cv2.imwrite(frame_path, frame)
                frames.append({
                    "frame_path": frame_path,
                    "timestamp": round(timestamp, 2),
                })
    
    elif method == "scene_change":
        # 基于帧差异检测场景变化
        prev_frame = None
        threshold = 30.0  # 帧差异阈值，可调节
        
        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if prev_frame is not None:
                diff = cv2.absdiff(prev_frame, gray)
                mean_diff = np.mean(diff)
                
                if mean_diff > threshold:
                    timestamp = i / fps if fps > 0 else 0
                    frame_path = os.path.join(output_dir, f"frame_{len(frames):03d}.jpg")
                    cv2.imwrite(frame_path, frame)
                    frames.append({
                        "frame_path": frame_path,
                        "timestamp": round(timestamp, 2),
                    })
                    
                    if len(frames) >= max_frames:
                        break
            
            prev_frame = gray
    
    cap.release()
    return frames
```

### 4.2 视频音频转文字

```python
def extract_audio_text(video_path: str) -> str:
    """
    从视频中提取音频并转为文字
    
    方案一：使用 OpenAI Whisper API（推荐，简单快速）
    方案二：使用本地 Whisper 模型（离线场景）
    """
    from moviepy.editor import VideoFileClip
    import tempfile
    
    # 提取音频
    video = VideoFileClip(video_path)
    if video.audio is None:
        return ""
    
    audio_path = tempfile.mktemp(suffix=".mp3")
    video.audio.write_audiofile(audio_path, verbose=False, logger=None)
    video.close()
    
    # 方案一：OpenAI Whisper API
    from openai import OpenAI
    client = OpenAI()
    
    with open(audio_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            language="zh",  # 中文
        )
    
    os.remove(audio_path)
    return transcript.text
```

### 4.3 视频综合处理与索引

```python
# src/indexing/video_indexer.py

import json
import uuid
from langchain.schema import Document

def process_and_index_video(video_path: str, media_store, retriever, 
                            describe_fn, max_frames=10):
    """
    视频完整处理流程：
    1. 上传原始视频到对象存储，获得 URL
    2. 提取关键帧 → 用 VLM 描述每帧
    3. 提取音频 → 转文字
    4. 合并为综合摘要 → 嵌入 → Qdrant
    5. 完整信息存入 Redis docstore
    """
    id_key = "doc_id"
    
    # 1. 上传原始视频到对象存储
    upload_result = media_store.upload_file(video_path, media_type="video")
    video_url = upload_result["url"]
    
    # 2. 提取关键帧并描述
    frames = extract_keyframes(video_path, max_frames=max_frames)
    frame_descriptions = []
    frame_urls = []
    
    for frame_info in frames:
        # 描述关键帧
        desc = describe_fn(
            frame_info["frame_path"],
            context=f"这是视频 {Path(video_path).name} 在 {frame_info['timestamp']} 秒处的画面"
        )
        frame_descriptions.append({
            "timestamp": frame_info["timestamp"],
            "description": desc,
        })
        
        # 可选：也把关键帧上传到对象存储（方便前端做视频预览缩略图）
        frame_upload = media_store.upload_file(frame_info["frame_path"], media_type="image")
        frame_urls.append({
            "timestamp": frame_info["timestamp"],
            "url": frame_upload["url"],
        })
    
    # 3. 提取音频转文字
    audio_text = extract_audio_text(video_path)
    
    # 4. 合并为综合摘要
    frames_text = "\n".join([
        f"[{fd['timestamp']}s] {fd['description']}" 
        for fd in frame_descriptions
    ])
    
    combined_summary = f"""【视频内容摘要】
文件名：{Path(video_path).name}

画面描述：
{frames_text}

音频内容：
{audio_text if audio_text else '（无语音内容）'}
"""
    
    # 5. 索引到 Qdrant + Redis
    doc_id = str(uuid.uuid4())
    
    summary_doc = Document(
        page_content=combined_summary,
        metadata={
            id_key: doc_id,
            "media_type": "video",
            "media_url": video_url,
            "original_filename": upload_result["original_filename"],
            "duration_seconds": len(frames) * (frames[-1]["timestamp"] / len(frames)) if frames else 0,
        }
    )
    retriever.vectorstore.add_documents([summary_doc])
    
    original_content = json.dumps({
        "type": "video",
        "description": combined_summary,
        "url": video_url,
        "filename": upload_result["original_filename"],
        "frame_urls": frame_urls,
        "audio_text": audio_text,
    }, ensure_ascii=False)
    
    retriever.docstore.mset([(doc_id, original_content)])
```

---

## 第五步：输出端 — 回答时返回图片和视频

### 5.1 检索结果解析

```python
# src/retrieval/result_parser.py

import json

def parse_retrieved_results(results: list) -> dict:
    """
    将 MultiVectorRetriever 返回的结果解析为结构化的多模态回答素材
    
    Returns:
        {
            "text_contexts": ["文本块1", "文本块2"],
            "table_contexts": ["表格HTML1"],
            "image_refs": [
                {"url": "https://...", "description": "...", "filename": "..."}
            ],
            "video_refs": [
                {"url": "https://...", "description": "...", "filename": "...", "frame_urls": [...]}
            ]
        }
    """
    parsed = {
        "text_contexts": [],
        "table_contexts": [],
        "image_refs": [],
        "video_refs": [],
    }
    
    for result in results:
        content = result if isinstance(result, str) else result.page_content
        
        # 尝试解析为 JSON（图片/视频的 docstore 内容是 JSON 字符串）
        try:
            data = json.loads(content)
            if data.get("type") == "image":
                parsed["image_refs"].append({
                    "url": data["url"],
                    "description": data["description"],
                    "filename": data.get("filename", ""),
                })
            elif data.get("type") == "video":
                parsed["video_refs"].append({
                    "url": data["url"],
                    "description": data["description"],
                    "filename": data.get("filename", ""),
                    "frame_urls": data.get("frame_urls", []),
                })
            else:
                # 表格等其他 JSON 结构
                parsed["table_contexts"].append(content)
        except (json.JSONDecodeError, TypeError):
            # 普通文本
            parsed["text_contexts"].append(content)
    
    return parsed
```

### 5.2 构建多模态回答

```python
# src/generation/multimodal_generator.py

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

def generate_multimodal_answer(question: str, parsed_results: dict, 
                                chat_history=None) -> dict:
    """
    生成包含文字 + 图片引用 + 视频引用的多模态回答
    
    Returns:
        {
            "text_answer": "文字回答内容...",
            "referenced_images": [
                {"url": "https://...", "description": "...", "relevance": "说明为什么引用这张图"}
            ],
            "referenced_videos": [
                {"url": "https://...", "description": "...", "relevance": "..."}
            ]
        }
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # 构建上下文
    context_parts = []
    
    if parsed_results["text_contexts"]:
        context_parts.append("【文本资料】\n" + "\n---\n".join(parsed_results["text_contexts"]))
    
    if parsed_results["table_contexts"]:
        context_parts.append("【表格资料】\n" + "\n---\n".join(parsed_results["table_contexts"]))
    
    if parsed_results["image_refs"]:
        image_texts = []
        for i, img in enumerate(parsed_results["image_refs"]):
            image_texts.append(f"[图片{i+1}] {img['description']} (文件: {img['filename']})")
        context_parts.append("【相关图片】\n" + "\n".join(image_texts))
    
    if parsed_results["video_refs"]:
        video_texts = []
        for i, vid in enumerate(parsed_results["video_refs"]):
            video_texts.append(f"[视频{i+1}] {vid['description']} (文件: {vid['filename']})")
        context_parts.append("【相关视频】\n" + "\n".join(video_texts))
    
    full_context = "\n\n".join(context_parts)
    
    prompt = f"""你是一个专业的售后客服助手。请根据以下参考资料回答用户的问题。

参考资料：
{full_context}

要求：
1. 仅基于参考资料回答，不要编造信息
2. 回答简洁专业，使用中文
3. **如果参考资料中有相关的图片或视频，请在回答中明确指出**，例如：
   - "请参考 [图片1] 中的产品示意图"
   - "详细操作步骤请观看 [视频1]"
4. 在回答末尾，用 JSON 格式列出你引用的图片和视频编号，格式如下：
   {{"referenced_images": [1, 2], "referenced_videos": [1]}}
   如果没有引用任何媒体，返回空数组。

用户问题：{question}
"""
    
    response = llm.invoke(prompt)
    answer_text = response.content
    
    # 解析 LLM 回答中引用的媒体编号
    referenced_images = []
    referenced_videos = []
    
    try:
        # 尝试从回答末尾提取 JSON
        import re
        json_match = re.search(r'\{[^{}]*"referenced_images"[^{}]*\}', answer_text)
        if json_match:
            refs = json.loads(json_match.group())
            
            for idx in refs.get("referenced_images", []):
                if 0 < idx <= len(parsed_results["image_refs"]):
                    img = parsed_results["image_refs"][idx - 1]
                    referenced_images.append(img)
            
            for idx in refs.get("referenced_videos", []):
                if 0 < idx <= len(parsed_results["video_refs"]):
                    vid = parsed_results["video_refs"][idx - 1]
                    referenced_videos.append(vid)
            
            # 从回答文本中移除 JSON 部分
            answer_text = answer_text[:json_match.start()].strip()
    except Exception:
        pass
    
    return {
        "text_answer": answer_text,
        "referenced_images": referenced_images,
        "referenced_videos": referenced_videos,
    }
```

### 5.3 前端 API 响应格式

```python
# 最终给前端的 API 响应结构

{
    "answer": "根据您描述的问题，这是水泵密封圈老化导致的漏水。请参考 [图片1] 中标注的密封圈位置，按照 [视频1] 中的步骤进行更换。",
    "media": {
        "images": [
            {
                "url": "https://pub-xxxxx.r2.dev/images/abc123.png",
                "description": "水泵密封圈位置示意图",
                "filename": "水泵结构图.png"
            }
        ],
        "videos": [
            {
                "url": "https://pub-xxxxx.r2.dev/videos/def456.mp4",
                "description": "密封圈更换操作视频",
                "filename": "维修教程.mp4",
                "thumbnail": "https://pub-xxxxx.r2.dev/images/frame_001.jpg"
            }
        ]
    },
    "sources": ["产品维修手册v3.pdf - 第12页"]
}
```

前端拿到这个结构后：
- `answer` 字段渲染为文字（可以用 Markdown 渲染器）
- `media.images` 中的 URL 渲染为 `<img>` 标签
- `media.videos` 中的 URL 渲染为 `<video>` 播放器或者视频卡片

---

## 第六步：处理用户直接上传的图片/视频（非文档内嵌）

用户在对话中可能直接上传图片（如故障照片）或视频（如故障现象录像），这种情况不走文档解析流程，而是**即时理解 + 结合知识库检索**。

```python
# src/pipeline/realtime_media.py

def handle_user_uploaded_media(file_path: str, question: str, 
                                retriever, media_store, describe_fn):
    """
    处理用户在对话中直接上传的图片/视频
    
    流程：
    1. 用 VLM 理解图片/视频内容
    2. 将理解结果作为查询，检索知识库
    3. 结合用户问题 + 媒体理解 + 检索结果，生成回答
    """
    file_ext = Path(file_path).suffix.lower()
    
    if file_ext in [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"]:
        # 图片：直接用 VLM 理解
        media_description = describe_fn(file_path, context=question)
        
    elif file_ext in [".mp4", ".avi", ".mov", ".mkv", ".webm"]:
        # 视频：抽帧 + 音频转文字
        frames = extract_keyframes(file_path, max_frames=5)
        frame_descs = [describe_fn(f["frame_path"], context=question) for f in frames]
        audio_text = extract_audio_text(file_path)
        media_description = "视频画面：" + " | ".join(frame_descs)
        if audio_text:
            media_description += f"\n视频语音：{audio_text}"
    else:
        media_description = ""
    
    # 将媒体理解结果 + 用户问题 组合为增强查询
    enhanced_query = f"{question}\n\n用户上传的媒体内容描述：{media_description}"
    
    # 检索知识库
    results = retriever.invoke(enhanced_query)
    parsed = parse_retrieved_results(results)
    
    # 生成回答
    answer = generate_multimodal_answer(enhanced_query, parsed)
    
    return answer
```

---

## 项目结构更新

在现有结构基础上新增：

```
project/
├── src/
│   ├── storage/
│   │   └── object_store.py      # 【新增】对象存储管理（R2/MinIO 通用）
│   ├── parser/
│   │   ├── docling_parser.py
│   │   ├── media_extractor.py   # 【新增】从文档中提取图片
│   │   ├── image_describer.py   # 【新增】图片描述生成（VLM）
│   │   └── video_processor.py   # 【新增】视频处理（关键帧+音频）
│   ├── indexing/
│   │   ├── indexer.py
│   │   ├── image_indexer.py     # 【新增】图片索引
│   │   └── video_indexer.py     # 【新增】视频索引
│   ├── retrieval/
│   │   ├── retriever.py
│   │   └── result_parser.py     # 【新增】检索结果解析（区分文本/图片/视频）
│   ├── generation/
│   │   └── multimodal_generator.py  # 【修改】支持多模态输出
│   ├── context/
│   │   └── history.py
│   ├── pipeline/
│   │   ├── pipeline.py
│   │   └── realtime_media.py    # 【新增】用户即时上传媒体处理
│   ...
```

---

## 关键设计原则（本次新增）

1. **媒体文件不进向量库和 Redis** — 原始图片/视频存对象存储（开发用 R2，生产用 MinIO），向量库和 Redis 中只存 URL 引用和文字描述
2. **描述做检索，URL 做输出** — 多模态 LLM 生成的描述用于语义匹配，命中后通过 URL 将原始媒体返回给前端
3. **视频 = 关键帧 + 音频** — 视频不做整体嵌入，而是拆解为关键帧图片描述 + 音频转文字，合并为文本摘要进行检索
4. **区分两种上传场景** — 文档内嵌的图片/视频走"解析→索引"流程；用户对话中直接上传的走"即时理解→检索→回答"流程
5. **前端 API 统一格式** — 回答统一返回 `{answer, media: {images, videos}, sources}` 结构，前端根据类型渲染
6. **存储与代码解耦** — 所有对象存储操作通过 MediaStore 封装，底层是 S3 协议。开发/生产只切配置（R2 → MinIO），业务代码零修改

---

## 迁移注意事项

- 本文档是在已完成的 Docling + LangChain + Qdrant + Redis 基础上的增量修改
- 不需要改动已有的文本和表格处理逻辑
- 主要新增：对象存储模块、图片/视频处理模块、多模态输出格式
- **开发阶段**：使用 Cloudflare R2 云端存储（免费 10GB + 零出流量费），请先完成 R2 账号注册和 Bucket 创建
- **生产阶段**：切换到 MinIO 本地部署（Docker 一行启动），公司内部资料全部存本地内网，只需改 config/settings.py 中的连接配置
- R2 和 MinIO 都使用 S3 兼容 API（boto3），MediaStore 代码无需任何改动
- VLM 调用（图片描述、视频帧描述）会产生较多 API 费用，建议对每个文档的处理结果做缓存，避免重复处理
- 视频处理较耗时，建议用异步任务队列（如 Celery）处理大视频文件

### 完整的本地化部署清单（生产阶段）

当需要将整套系统从云端迁移到本地内网时，涉及以下组件：

| 组件 | 开发阶段（云端） | 生产阶段（本地） | 切换方式 |
|------|-----------------|-----------------|---------|
| 对象存储 | Cloudflare R2 | MinIO (Docker) | 改 config endpoint + key |
| 向量数据库 | Qdrant Cloud | Qdrant (Docker) | 改 config URL + key |
| 文档/上下文存储 | Redis | Redis (Docker) | 改 config URL |
| LLM / VLM | OpenAI API | 本地模型或私有 API | 改 config base_url + key |

所有组件都通过 config/settings.py 中的环境变量控制，可以用 `.env` 文件区分环境：

```bash
# .env.dev（开发环境）
STORAGE_ENDPOINT=https://xxx.r2.cloudflarestorage.com
QDRANT_URL=https://xxx.cloud.qdrant.io:6333
REDIS_URL=redis://localhost:6379

# .env.prod（生产环境）
STORAGE_ENDPOINT=http://minio.internal:9000
QDRANT_URL=http://qdrant.internal:6333
REDIS_URL=redis://redis.internal:6379
```