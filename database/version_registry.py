"""
版本注册表 CRUD 模块
管理产品版本与基础版本的映射关系，供售后人员维护
"""
from typing import Optional
from .db import get_connection


def create_version(
    version_code: str,
    is_base: bool,
    version_name: str = "",
    base_version_code: Optional[str] = None,
    doc_type_label: str = "",
) -> dict:
    """新增版本记录"""
    if is_base and base_version_code:
        raise ValueError("基础版本的 base_version_code 必须为空")
    if not is_base and not base_version_code:
        raise ValueError("特殊版本必须指定 base_version_code")

    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO version_registry
                (version_code, version_name, is_base, base_version_code, doc_type_label)
            VALUES (?, ?, ?, ?, ?)
            """,
            (version_code, version_name, 1 if is_base else 0, base_version_code, doc_type_label),
        )
        conn.commit()
        return get_version(version_code)
    finally:
        conn.close()


def get_version(version_code: str) -> Optional[dict]:
    """根据版本号获取版本记录，不存在返回 None"""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM version_registry WHERE version_code = ?", (version_code,)
        )
        row = cursor.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_all_versions() -> list:
    """获取所有版本列表（按创建时间升序）"""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM version_registry ORDER BY created_at ASC")
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


def update_version(
    version_code: str,
    version_name: Optional[str] = None,
    is_base: Optional[bool] = None,
    base_version_code: Optional[str] = None,
    doc_type_label: Optional[str] = None,
) -> Optional[dict]:
    """更新版本记录，只更新传入的非 None 字段"""
    existing = get_version(version_code)
    if existing is None:
        return None

    new_is_base = is_base if is_base is not None else bool(existing["is_base"])
    new_base = base_version_code if base_version_code is not None else existing["base_version_code"]
    new_name = version_name if version_name is not None else existing["version_name"]
    new_doc_type = doc_type_label if doc_type_label is not None else existing["doc_type_label"]

    if new_is_base and new_base:
        raise ValueError("基础版本的 base_version_code 必须为空")
    if not new_is_base and not new_base:
        raise ValueError("特殊版本必须指定 base_version_code")

    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE version_registry
            SET version_name = ?, is_base = ?, base_version_code = ?,
                doc_type_label = ?, updated_at = CURRENT_TIMESTAMP
            WHERE version_code = ?
            """,
            (new_name, 1 if new_is_base else 0, new_base if not new_is_base else None, new_doc_type, version_code),
        )
        conn.commit()
        return get_version(version_code)
    finally:
        conn.close()


def delete_version(version_code: str) -> bool:
    """删除版本记录，返回是否删除成功"""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM version_registry WHERE version_code = ?", (version_code,)
        )
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def get_base_version(version_code: str) -> Optional[str]:
    """
    返回特殊版本对应的基础版本号。
    - 输入特殊版本 → 返回其 base_version_code
    - 输入基础版本 → 返回 None（自身即基础）
    - 输入不存在的版本号 → 返回 None
    """
    record = get_version(version_code)
    if record is None:
        return None
    if record["is_base"]:
        return None
    return record["base_version_code"]


def is_base_version(version_code: str) -> bool:
    """判断版本号是否为基础版本，不存在时返回 False"""
    record = get_version(version_code)
    if record is None:
        return False
    return bool(record["is_base"])


def get_version_chain(version_code: str) -> list:
    """
    返回从当前版本到根版本的完整版本链，顺序为从新到旧。

    示例：
        get_version_chain("110103V") → ["110103V", "110102V", "1101"]
        get_version_chain("1101")    → ["1101"]
        get_version_chain("1102")    → ["1102", "1101"]  （1102 的 base_version_code = "1101"）

    规则：
    - 当前版本排在最前（优先级最高）
    - 沿 base_version_code 逐级向上，直到 is_base=True 或 base_version_code 为空
    - 有防循环保护（最多 20 层）
    - 版本不存在时返回仅含自身的单元素列表
    """
    chain = []
    current = version_code
    visited: set = set()
    max_depth = 20

    while current and len(chain) < max_depth:
        if current in visited:
            break
        visited.add(current)
        chain.append(current)
        record = get_version(current)
        if record is None or record["is_base"] or not record["base_version_code"]:
            break
        current = record["base_version_code"]

    return chain if chain else [version_code]