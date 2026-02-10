"""
Excel导入命令行工具
"""
import sys
from database.excel_importer import import_from_excel, preview_excel, get_excel_sheets


def main():
    if len(sys.argv) < 2:
        print("使用方法:")
        print("  python import_excel.py <excel文件路径> [工作表名称]")
        print("\n示例:")
        print("  python import_excel.py data/knowledge.xlsx")
        print("  python import_excel.py data/knowledge.xlsx Sheet1")
        sys.exit(1)

    file_path = sys.argv[1]
    sheet_name = sys.argv[2] if len(sys.argv) > 2 else None

    print(f"📂 正在读取文件: {file_path}")

    try:
        sheets = get_excel_sheets(file_path)
        print(f"📋 可用的工作表: {', '.join(sheets)}")

        print("\n👀 预览数据...")
        preview = preview_excel(file_path, sheet_name=sheet_name, rows=3)
        print(f"   列数: {preview['total_columns']}")
        print(f"   列名: {', '.join(preview['columns'])}")
        print(f"\n   检测到的字段映射:")
        for field, col in preview['detected_mapping'].items():
            print(f"     {field} <- {col}")

        print(f"\n   前{preview['preview_count']}行数据:")
        for i, row in enumerate(preview['preview_rows'], 1):
            print(f"     第{i}行: {row.get('title', '(无标题)')[:30]}...")

        confirm = input("\n是否继续导入? (y/n): ").strip().lower()
        if confirm != 'y':
            print("❌ 已取消导入")
            sys.exit(0)

        print("\n🔄 正在导入数据...")
        results = import_from_excel(file_path, sheet_name=sheet_name)

        print("\n✅ 导入完成!")
        print(f"   总计: {results['total']} 条")
        print(f"   成功: {results['success']} 条")
        print(f"   失败: {results['failed']} 条")

        if results['errors']:
            print("\n⚠️ 错误详情:")
            for error in results['errors'][:10]:
                print(f"   {error}")
            if len(results['errors']) > 10:
                print(f"   ... 还有 {len(results['errors']) - 10} 个错误")

    except FileNotFoundError as e:
        print(f"❌ 错误: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 导入失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
