# Excel导入功能使用指南

## 功能概述

系统支持从Excel文件批量导入故障知识数据,可以自动识别列名并映射到数据库字段。

## Excel文件格式要求

### 支持的文件格式
- `.xlsx` (Excel 2007及以上)
- `.xls` (Excel 97-2003)

### 必填字段
- **标题** - 故障或问题的名称
- **内容** - 详细的解决方案和说明

### 可选字段
- **故障代码** - 错误代码或编号
- **关键词** - 用于搜索的关键词(多个关键词用空格分隔)
- **设备型号** - 适用的设备型号

### 支持的列名变体

系统会自动识别以下列名(不区分大小写):

| 字段 | 支持的列名 |
|------|-----------|
| 故障代码 | 故障代码、错误代码、error_code、编号、代码 |
| 标题 | 标题、问题、故障名称、title、名称、故障、问题描述 |
| 内容 | 内容、解决方案、处理方法、content、详情、说明、解决办法 |
| 关键词 | 关键词、关键字、keywords、keyword、标签 |
| 设备型号 | 设备型号、适用设备、device_models、型号、设备 |

### Excel文件示例

| 故障代码 | 标题 | 内容 | 关键词 | 设备型号 |
|---------|------|------|--------|---------|
| E01 | 加油枪无法出油 | **可能原因：**<br>1. 油枪气阻<br>2. 油泵故障<br><br>**解决步骤：**<br>1. 检查油枪... | 加油枪 不出油 无油 | 通用 |
| E02 | 显示屏不亮 | **可能原因：**<br>1. 电源故障<br>2. 连接线松动 | 显示屏 黑屏 不亮 | 通用 |

## 使用方法

### 方法1: 命令行导入

1. 安装依赖:
```bash
pip install pandas openpyxl
```

2. 执行导入:
```bash
# 导入Excel文件(默认使用第一个工作表)
python import_excel.py your_file.xlsx

# 指定工作表名称
python import_excel.py your_file.xlsx Sheet1
```

### 方法2: API接口导入

#### 2.1 预览文件

在正式导入前,可以先预览文件内容:

```bash
curl -X POST "http://localhost:8000/api/admin/knowledge/import/preview" \
  -F "file=@your_file.xlsx"
```

返回示例:
```json
{
  "status": "ok",
  "filename": "your_file.xlsx",
  "sheets": ["Sheet1", "Sheet2"],
  "preview": {
    "total_columns": 5,
    "columns": ["故障代码", "标题", "内容", "关键词", "设备型号"],
    "detected_mapping": {
      "error_code": "故障代码",
      "title": "标题",
      "content": "内容",
      "keywords": "关键词",
      "device_models": "设备型号"
    },
    "preview_rows": [...]
  }
}
```

#### 2.2 正式导入

```bash
# 导入第一个工作表
curl -X POST "http://localhost:8000/api/admin/knowledge/import" \
  -F "file=@your_file.xlsx"

# 指定工作表
curl -X POST "http://localhost:8000/api/admin/knowledge/import?sheet_name=Sheet1" \
  -F "file=@your_file.xlsx"
```

返回示例:
```json
{
  "status": "ok",
  "message": "导入完成: 成功 10 条, 失败 0 条",
  "results": {
    "success": 10,
    "failed": 0,
    "total": 10,
    "errors": []
  }
}
```

### 方法3: 在代码中使用

```python
from database.excel_importer import import_from_excel, preview_excel

# 预览文件
preview = preview_excel("data/knowledge.xlsx", rows=5)
print(preview)

# 导入数据
results = import_from_excel("data/knowledge.xlsx")
print(f"成功导入 {results['success']} 条数据")
```

## 注意事项

1. **数据验证**
   - 标题和内容为必填字段,缺失会导致该行导入失败
   - 其他字段为可选,留空时会保存为空字符串

2. **列名识别**
   - 系统会自动识别常见的列名变体
   - 建议使用示例中的标准列名以确保正确识别
   - 列名不区分大小写

3. **批量导入**
   - 导入失败的行不会影响其他行
   - 导入完成后会显示详细的错误信息

4. **数据格式**
   - 内容支持Markdown格式
   - 关键词用空格分隔
   - Excel中的换行会被保留

5. **导入后操作**
   - 导入成功后系统会自动重新加载知识库
   - 新数据立即可用于检索

## 常见问题

### Q: 如何将表格传给系统?
**A:** 有以下方式:
1. 将Excel文件放到项目目录下,使用命令行导入
2. 通过API上传文件
3. 将文件路径告诉我,我帮你导入

### Q: 列名不标准怎么办?
**A:** 系统支持多种列名变体,参考上面的"支持的列名变体"表格。如果仍无法识别,建议修改Excel列名为标准格式。

### Q: 导入失败怎么办?
**A:** 查看错误信息,通常是:
- 文件格式不正确
- 必填字段(标题/内容)缺失
- 文件路径错误

### Q: 可以更新已有数据吗?
**A:** 当前版本只支持新增数据,不支持更新。如需更新,请先删除旧数据再导入新数据。

## 下一步

创建好Excel文件后,你可以:
1. 将文件放到项目目录的 `data/` 文件夹(需要先创建)
2. 告诉我文件路径,我帮你导入
3. 或者自己运行 `python import_excel.py <文件路径>`
