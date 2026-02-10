"""
快速查看数据库内容
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from database import get_all_knowledge
from database.db import get_connection

print("=" * 80)
print("知识库数据 (knowledge表)")
print("=" * 80)

knowledge = get_all_knowledge()
print(f"\n总计: {len(knowledge)} 条\n")

for i, item in enumerate(knowledge, 1):
    print(f"{i}. [{item['error_code']}] {item['title']}")
    print(f"   内容: {item['content'][:50]}...")
    print(f"   创建时间: {item['created_at']}")
    print()

print("\n" + "=" * 80)
print("对话记录 (conversations表)")
print("=" * 80)

conn = get_connection()
cursor = conn.cursor()
cursor.execute("SELECT COUNT(*) FROM conversations")
conv_count = cursor.fetchone()[0]
print(f"\n总对话数: {conv_count}")

cursor.execute("""
    SELECT question, confidence, solved, created_at
    FROM conversations
    ORDER BY created_at DESC
    LIMIT 10
""")
recent = cursor.fetchall()

if recent:
    print("\n最近10条对话:")
    for i, conv in enumerate(recent, 1):
        question = conv[0][:40] + "..." if len(conv[0]) > 40 else conv[0]
        print(f"{i}. {question}")
        print(f"   置信度: {conv[1]}, 已解决: {conv[2]}, 时间: {conv[3]}")

conn.close()

print("\n" + "=" * 80)
print("数据库统计")
print("=" * 80)

conn = get_connection()
cursor = conn.cursor()

# 按错误代码统计
cursor.execute("""
    SELECT error_code, COUNT(*) as count
    FROM knowledge
    WHERE error_code != ''
    GROUP BY error_code
    ORDER BY error_code
""")
stats = cursor.fetchall()

print(f"\n错误代码分布:")
for code, count in stats[:20]:
    print(f"  {code}: {count}条")

if len(stats) > 20:
    print(f"  ... 还有 {len(stats) - 20} 个错误代码")

conn.close()
