"""
Excel导入模块
"""
import pandas as pd
from typing import List, Dict, Optional
from pathlib import Path
from .db import add_knowledge, get_connection


def import_from_excel(file_path: str, sheet_name: Optional[str] = None) -> Dict:
    """
    从Excel文件导入知识数据到数据库

    参数:
        file_path: Excel文件路径
        sheet_name: 工作表名称,默认读取第一个sheet

    返回:
        包含导入统计信息的字典
    """
    if not Path(file_path).exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name or 0)
    except Exception as e:
        raise ValueError(f"读取Excel文件失败: {str(e)}")

    if df.empty:
        return {"success": 0, "failed": 0, "total": 0, "errors": []}

    column_mapping = detect_columns(df)

    results = {
        "success": 0,
        "failed": 0,
        "total": len(df),
        "errors": []
    }

    for idx, row in df.iterrows():
        try:
            knowledge_data = extract_knowledge_data(row, column_mapping)

            if not knowledge_data.get("title") or not knowledge_data.get("content"):
                results["failed"] += 1
                results["errors"].append(f"第{idx+2}行: 缺少必填字段(标题或内容)")
                continue

            add_knowledge(
                error_code=knowledge_data.get("error_code", ""),
                title=knowledge_data["title"],
                content=knowledge_data["content"],
                keywords=knowledge_data.get("keywords", ""),
                device_models=knowledge_data.get("device_models", "")
            )
            results["success"] += 1

        except Exception as e:
            results["failed"] += 1
            results["errors"].append(f"第{idx+2}行导入失败: {str(e)}")

    return results


def detect_columns(df: pd.DataFrame) -> Dict[str, str]:
    """
    自动检测Excel列名并映射到数据库字段

    支持的列名变体:
    - error_code: 故障代码/错误代码/error_code/编号
    - title: 标题/问题/故障名称/title
    - content: 内容/解决方案/处理方法/content/详情
    - keywords: 关键词/关键字/keywords
    - device_models: 设备型号/适用设备/device_models/型号
    """
    columns = df.columns.tolist()
    mapping = {}

    error_code_variants = ["故障代码", "错误代码", "error_code", "编号", "代码", "error code", "提示代码"]
    title_variants = ["标题", "问题", "故障名称", "title", "名称", "故障", "问题描述", "提示信息", "原因", "信息"]
    content_variants = ["内容", "解决方案", "处理方法", "content", "详情", "说明", "解决办法", "处理步骤", "恢复处理", "恢复方法"]
    keywords_variants = ["关键词", "关键字", "keywords", "keyword", "标签"]
    device_variants = ["设备型号", "适用设备", "device_models", "型号", "设备", "device model"]

    for col in columns:
        col_lower = str(col).lower().strip()

        if not mapping.get("error_code") and any(v.lower() in col_lower for v in error_code_variants):
            mapping["error_code"] = col
        elif not mapping.get("title") and any(v.lower() in col_lower for v in title_variants):
            mapping["title"] = col
        elif not mapping.get("content") and any(v.lower() in col_lower for v in content_variants):
            mapping["content"] = col
        elif not mapping.get("keywords") and any(v.lower() in col_lower for v in keywords_variants):
            mapping["keywords"] = col
        elif not mapping.get("device_models") and any(v.lower() in col_lower for v in device_variants):
            mapping["device_models"] = col

    return mapping


def extract_knowledge_data(row: pd.Series, column_mapping: Dict[str, str]) -> Dict[str, str]:
    """
    从DataFrame行中提取知识数据
    """
    data = {}

    for field, col_name in column_mapping.items():
        value = row.get(col_name, "")
        if pd.isna(value):
            value = ""
        data[field] = str(value).strip()

    return data


def preview_excel(file_path: str, sheet_name: Optional[str] = None, rows: int = 5) -> Dict:
    """
    预览Excel文件内容

    参数:
        file_path: Excel文件路径
        sheet_name: 工作表名称
        rows: 预览行数

    返回:
        包含预览数据的字典
    """
    if not Path(file_path).exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name or 0, nrows=rows)
        column_mapping = detect_columns(df)

        preview_data = []
        for _, row in df.iterrows():
            preview_data.append(extract_knowledge_data(row, column_mapping))

        return {
            "total_columns": len(df.columns),
            "columns": df.columns.tolist(),
            "detected_mapping": column_mapping,
            "preview_rows": preview_data,
            "preview_count": len(preview_data)
        }
    except Exception as e:
        raise ValueError(f"预览Excel文件失败: {str(e)}")


def get_excel_sheets(file_path: str) -> List[str]:
    """
    获取Excel文件中的所有工作表名称
    """
    if not Path(file_path).exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    try:
        excel_file = pd.ExcelFile(file_path)
        return excel_file.sheet_names
    except Exception as e:
        raise ValueError(f"读取Excel工作表失败: {str(e)}")
