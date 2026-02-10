"""
测试语义理解能力
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from rag import get_retriever

# 测试用例 - 不包含错误代码,只有语义描述
test_cases = [
    "比例系数超出了正常范围",
    "传感器检测不到了",
    "显示屏检测有问题",
    "流速好像不正常",
    "开机后一直没有交易",
    "密度数值超标了",
    "与后台无法通讯",
]

print("=" * 60)
print("语义理解能力测试")
print("=" * 60)

retriever = get_retriever()

for query in test_cases:
    print(f"\n查询: {query}")
    results, confidence = retriever.retrieve(query, top_k=1)

    if results:
        result = results[0]
        print(f"✓ 匹配: [{result['error_code']}] {result['title']}")
        print(f"  相似度: {result['similarity_score']:.4f}")
        print(f"  置信度: {confidence}")
    else:
        print(f"✗ 未找到匹配")
    print("-" * 60)
