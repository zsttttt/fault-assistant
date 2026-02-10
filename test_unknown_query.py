"""
测试系统对未知问题的处理
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from rag import get_retriever
import asyncio
from rag.generator import generate_answer

# 测试用例 - 数据库中完全不存在的问题
unknown_queries = [
    "充电桩的WiFi密码忘记了怎么办",  # 完全不相关
    "加油机可以用来充电吗",  # 荒谬的问题
    "设备屏幕突然变成紫色了",  # 数据库中没有的故障
    "E99错误代码",  # 不存在的错误代码
]

print("=" * 80)
print("未知问题处理测试")
print("=" * 80)

retriever = get_retriever()

async def test_unknown_query(query):
    print(f"\n问题: {query}")
    print("-" * 80)

    # 1. 向量检索阶段
    results, confidence = retriever.retrieve(query, top_k=3)

    print(f"向量检索结果:")
    print(f"  置信度: {confidence}")
    print(f"  匹配数量: {len(results)}")

    if results:
        print(f"  最佳匹配:")
        best = results[0]
        print(f"    [{best['error_code']}] {best['title']}")
        print(f"    相似度: {best.get('similarity_score', 'N/A'):.4f}")
    else:
        print(f"  无匹配结果")

    # 2. LLM生成阶段
    print(f"\nLLM处理:")
    answer = await generate_answer(query, results, confidence)
    print(f"  回答预览: {answer[:150]}...")

    print("=" * 80)

async def main():
    for query in unknown_queries:
        await test_unknown_query(query)

if __name__ == "__main__":
    asyncio.run(main())
