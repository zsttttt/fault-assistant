"""
视频本地处理模块（无网络依赖）
- 关键帧提取：均匀采样 / 场景变化检测
- 音频提取：moviepy 2.x 分离出 MP3 字节
"""
import os
import tempfile
from pathlib import Path
from typing import List, Optional


def extract_keyframes(
    video_path: str,
    output_dir: Optional[str] = None,
    max_frames: int = 10,
    method: str = "uniform",
) -> List[dict]:
    """
    从视频中提取关键帧，保存为 JPEG 文件

    Args:
        video_path:  视频文件路径
        output_dir:  帧输出目录；None 则自动创建临时目录
        max_frames:  最多提取帧数
        method:      "uniform"（均匀采样）/ "scene_change"（场景变化检测）

    Returns:
        [{"frame_path": str, "timestamp": float}, ...]
    """
    import cv2
    import numpy as np

    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="frames_")
    else:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 1.0
    frames: List[dict] = []

    if method == "uniform":
        interval = max(1, total_frames // max_frames)
        for i in range(0, total_frames, interval):
            if len(frames) >= max_frames:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                continue
            frame_path = os.path.join(output_dir, f"frame_{len(frames):03d}.jpg")
            cv2.imwrite(frame_path, frame)
            frames.append({
                "frame_path": frame_path,
                "timestamp": round(i / fps, 2),
            })

    elif method == "scene_change":
        threshold = 30.0
        prev_gray = None
        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                diff = cv2.absdiff(prev_gray, gray)
                if np.mean(diff) > threshold:
                    frame_path = os.path.join(output_dir, f"frame_{len(frames):03d}.jpg")
                    cv2.imwrite(frame_path, frame)
                    frames.append({
                        "frame_path": frame_path,
                        "timestamp": round(i / fps, 2),
                    })
                    if len(frames) >= max_frames:
                        break
            prev_gray = gray

    cap.release()
    return frames


def extract_audio_bytes(video_path: str) -> Optional[bytes]:
    """
    从视频中提取音频，返回 MP3 字节数据

    Args:
        video_path: 视频文件路径

    Returns:
        MP3 字节，无音轨或失败时返回 None
    """
    from moviepy import VideoFileClip

    try:
        clip = VideoFileClip(video_path)
    except Exception:
        return None

    if clip.audio is None:
        clip.close()
        return None

    tmp_audio = tempfile.mktemp(suffix=".mp3")
    try:
        clip.audio.write_audiofile(tmp_audio, logger=None)
        clip.close()
        with open(tmp_audio, "rb") as f:
            return f.read()
    except Exception:
        return None
    finally:
        clip.close()
        if os.path.exists(tmp_audio):
            os.remove(tmp_audio)