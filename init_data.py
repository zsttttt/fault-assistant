"""
初始化示例数据
"""
from database.db import init_db, add_knowledge, get_all_knowledge

SAMPLE_KNOWLEDGE = [
    {
        "error_code": "E01",
        "title": "加油枪无法出油",
        "keywords": "加油枪 不出油 无油 油枪故障",
        "content": """**可能原因：**
1. 油枪气阻
2. 油泵故障
3. 电磁阀未开启
4. 油路堵塞

**解决步骤：**
1. 首先检查油枪是否正确插入油箱口
2. 检查加油机显示屏是否正常工作
3. 尝试松开油枪把手后重新按压
4. 检查是否已正确选择油品
5. 如以上步骤无效，请检查电磁阀指示灯是否亮起

**安全提示：** 请勿自行拆卸油枪或油路部件。""",
        "device_models": "通用"
    },
    {
        "error_code": "E02",
        "title": "加油机显示屏不亮",
        "keywords": "显示屏 黑屏 不亮 无显示",
        "content": """**可能原因：**
1. 电源故障
2. 显示屏连接线松动
3. 显示屏模块损坏

**解决步骤：**
1. 检查加油机电源开关是否打开
2. 检查加油站总电源是否正常
3. 查看加油机背面的电源指示灯
4. 尝试重启加油机（关闭电源等待30秒后重新开启）

**注意事项：** 请勿自行打开机箱检修。""",
        "device_models": "通用"
    },
    {
        "error_code": "E03",
        "title": "计量不准确",
        "keywords": "计量 数字 不准 误差 油量",
        "content": """**可能原因：**
1. 流量计故障
2. 编码器信号异常
3. 气阻影响计量

**解决步骤：**
1. 观察是否每次加油都有偏差
2. 记录偏差数值和规律
3. 检查是否有气泡进入油路
4. 此问题需要专业人员校准

**重要提示：** 计量器具属于法定检定设备，请联系有资质的维修单位。""",
        "device_models": "通用"
    },
    {
        "error_code": "E10",
        "title": "充电桩无法启动充电",
        "keywords": "充电桩 无法充电 不充电 启动失败",
        "content": """**可能原因：**
1. 充电枪未正确插入
2. 车辆端充电口异常
3. 充电桩通信故障
4. 账户余额不足

**解决步骤：**
1. 确认充电枪已完全插入车辆充电口
2. 检查充电枪和车辆充电口是否有异物或损坏
3. 在APP或屏幕上确认账户状态和余额
4. 尝试拔出充电枪，等待10秒后重新插入

**安全提示：** 请勿在雨天湿手操作充电设备。""",
        "device_models": "通用充电桩"
    },
    {
        "error_code": "E11",
        "title": "充电过程中自动停止",
        "keywords": "充电中断 自动停止 充电停止 中途断开",
        "content": """**可能原因：**
1. 车辆电池已充满
2. 车辆BMS保护
3. 充电桩过温保护
4. 电网电压波动

**解决步骤：**
1. 检查车辆仪表盘显示的电量是否已满
2. 查看充电桩屏幕上的停止原因代码
3. 检查充电枪连接是否松动
4. 等待5分钟后尝试重新启动充电

**注意：** 如充电桩有异常发热，请立即停止使用。""",
        "device_models": "通用充电桩"
    }
]


def init_sample_data():
    print("🔄 正在初始化数据库...")
    init_db()
    
    existing = get_all_knowledge()
    if existing:
        print(f"ℹ️ 数据库已有 {len(existing)} 条知识，跳过初始化")
        return
    
    print("📝 正在添加示例知识数据...")
    for item in SAMPLE_KNOWLEDGE:
        add_knowledge(
            error_code=item["error_code"],
            title=item["title"],
            content=item["content"],
            keywords=item["keywords"],
            device_models=item["device_models"]
        )
        print(f"  ✓ 已添加: {item['error_code']} - {item['title']}")
    
    print(f"✅ 初始化完成，共添加 {len(SAMPLE_KNOWLEDGE)} 条知识")


if __name__ == "__main__":
    init_sample_data()
