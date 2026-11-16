import json
import os
import time
import threading
from flask import Flask, jsonify, render_template, request

# ============ 1. 导入算法包 ============
import sys
PKG_DIR = os.path.join(os.path.dirname(__file__), 'algorithm_package')
if PKG_DIR not in sys.path:
    sys.path.append(PKG_DIR)

import algorithm

app = Flask(__name__)

INTERVAL = 6
JSON_PATH = os.path.join(os.path.dirname(__file__), 'risk_result.json')

# ============ 2. 字段适配函数（必须提前定义）============
def adapt_bin(bin_dict):
    """字段适配：兼容队友返回的数字 risk 和时间对象，带防御"""
    # 防御：如果队友返回的不是字典（比如字符串），包装成最小可用对象
    if not isinstance(bin_dict, dict):
        print(f"⚠️ adapt_bin 收到非字典数据: {type(bin_dict)} = {bin_dict}")
        return {
            "id": str(bin_dict),
            "location": None,
            "fillRate": 0,
            "risk": "low",
            "riskLevel": "low",
            "isFull": False,
            "lastUpdated": None,
            "growthRatePerHour": 0.0,
        }

    # 处理 risk：队友返回的是数字(0.0)，映射成 low/medium/high
    risk_val = bin_dict.get("risk", 0.0)
    if isinstance(risk_val, (int, float)):
        risk_str = "low" if risk_val < 0.5 else ("medium" if risk_val < 0.8 else "high")
    else:
        risk_str = risk_val

    # 处理时间对象：datetime 不能直接 json，转成字符串
    last_updated = bin_dict.get("lastUpdated")
    if hasattr(last_updated, 'isoformat'):
        last_updated = last_updated.isoformat()

    return {
        "id": bin_dict.get("bin_id") or bin_dict.get("id"),
        "location": bin_dict.get("location"),
        "fillRate": bin_dict.get("fillRate", 0),
        "risk": risk_str,
        "riskLevel": bin_dict.get("riskLevel", "low"),
        "isFull": bin_dict.get("isFull", False),
        "lastUpdated": last_updated,
        "growthRatePerHour": bin_dict.get("growthRatePerHour", 0.0),
    }

# ============ 3. 初始化点位 ============
# 队友返回的是 {id: {...}, ...} 嵌套字典，需 .values() 转列表
RAW_BINS = algorithm.init_bins(initial_fill=0, start_time="2026-07-01 00:00:00")
DATA = {
    "bins": [adapt_bin(v) for v in RAW_BINS.values()],
    "tasks": []
}
CLEAN_FLAG = {}

def save_json():
    """同步一份到文件，方便查看/对接"""
    try:
        with open(JSON_PATH, 'w', encoding='utf-8') as f:
            json.dump(DATA, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print("⚠️ 写文件失败:", e)

def background_loop():
    """后台推理线程"""
    next_run = time.monotonic() + INTERVAL
    while True:
        clean_flags = dict(CLEAN_FLAG)   # 取当前清运标记
        CLEAN_FLAG.clear()               # 清空，避免重复清运

        # tick 需要用原始嵌套字典格式，传 RAW_BINS
        algorithm.tick(
            bins=RAW_BINS,
            clean_flags=clean_flags,
            elapsed_minutes=INTERVAL
        )

        # 每轮重新从 RAW_BINS 适配成前端需要的列表
        DATA["bins"] = [adapt_bin(v) for v in RAW_BINS.values() if isinstance(v, dict)]
        DATA["tasks"] = simulate_generate_tasks(DATA["bins"])
        save_json()
        print(f"📊 {[(b['id'], b['fillRate'], b['risk']) for b in DATA['bins']]} | 卡片: {len(DATA['tasks'])}", flush=True)

        next_run += INTERVAL
        sleep_time = next_run - time.monotonic()
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            next_run = time.monotonic() + INTERVAL

# ============ 4. 任务生成 ============
def simulate_generate_tasks(bins):
    """沿用你原 generate_tasks 逻辑，适配队友字段"""
    tasks = []
    for b in bins:
        if b.get("fillRate", 0) >= 80 or b.get("risk") in ("high", "medium"):
            tasks.append({
                "binId": b["id"],
                "location": b.get("location"),
                "fillRate": b.get("fillRate"),
                "risk": b.get("risk"),
                "riskLevel": b.get("riskLevel"),
                "action": "清运" if b.get("fillRate", 0) >= 80 else "巡检"
            })
    return tasks

# ===== 路由 =====
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/risk')
def api_risk():
    return jsonify(DATA)

@app.route('/api/clean', methods=['POST'])
def api_clean():
    bin_id = (request.json or {}).get('binId')
    if not bin_id:
        return jsonify({"ok": False, "msg": "缺少binId"}), 400

    CLEAN_FLAG[bin_id] = True
    # 立即在内存归零（前端下次轮询就看到绿，不用等6秒）
    for b in DATA["bins"]:
        if b["id"] == bin_id:
            b["fillRate"] = 0
            b["risk"] = "low"
            b["riskLevel"] = "low"
            b["isFull"] = False
    DATA["tasks"] = simulate_generate_tasks(DATA["bins"])
    save_json()
    print(f"✅ {bin_id} 清运完成，填充率归零")
    return jsonify({"ok": True, "bin_id": bin_id, "message": f"{bin_id} 清理完成"})

# ===== 启动 =====
if __name__ == '__main__':
    print("=" * 50)
    print("🚀 智桶潮汐 · 后端启动中 (algorithm 模式)")
    print("📍 访问地址: http://127.0.0.1:5000")
    print("=" * 50)
    threading.Thread(target=background_loop, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)


