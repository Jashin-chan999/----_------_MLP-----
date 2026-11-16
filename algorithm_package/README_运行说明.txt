智桶潮汐：实时点位状态更新 + 已训练增量 MLP 推理版

本包做什么
- algorithm.py 不再内置之前 3.375x 的固定逐小时曲线。
- 无传感器读数、无外部增量输入时，tick() 会加载并调用随包提供的 PyTorch 增量 MLP，预测该时段的正向填充增量。
- 传感器实测、清运事件、上游增量输入优先级高于模型预测。
- 所有点位共用模型、特征构造方法和默认倍率 gain=1.0；不做每日强制归一化，不强制日终 100%。

文件
- algorithm.py：集成状态管理、模型加载、特征生成、模型推理、风险分数计算。
- melbourne_hourly_increment_mlp_seed_20261025_fold_1.pt：真实训练得到的模型权重。
- demo.py：示例，打印各点位预测的小时增量，并模拟推进一个小时。
- test_algorithm.py：测试。

依赖
Python 3.10+，torch、numpy。检查点元数据使用 pandas Series 序列化格式；加载时会由 torch 反序列化，因此建议同时安装 pandas。
安装：pip install torch numpy pandas

运行
python demo.py
python -m unittest -v test_algorithm.py

核心接口
from algorithm import init_bins, tick, snapshot_bins, predict_hourly_increment
bins = init_bins(initial_fill=0, start_time="2026-07-01 00:00:00")
# 可以独立查询某小时预测增量；hour_ending=01:00 表示预测 00:00-01:00
inc = predict_hourly_increment("A001", "2026-07-01 01:00:00")
# 无实测时，tick 默认用已训练 MLP 预测增量
state = tick(bins, elapsed_minutes=5)
rows = snapshot_bins(bins)
# 有传感器实测时，实测优先
state = tick(bins, sensor_readings={"A001": 48.5}, elapsed_minutes=5)
# 发生清运时，传入 clean_flags
state = tick(bins, clean_flags={"A001": True}, elapsed_minutes=5)

输入和输出
输入：点位 ID、时间特征（由代码生成）、固定 POI 特征；运行时可另传实测 fillRate、清运事件或外部新增百分点。
输出：当前 fillRate（0-100%）、risk（0-100规则分数）、riskLevel、growthRatePerHour、lastIncrement、isFull、lastUpdated。

重要限制（务必保留）
1. 这个版本确实调用了 AI/MLP 权重，不再依靠硬编码预测曲线；但它只是“集成了真实推理调用”，不等于模型已具备可靠预测精度。
2. 模型由墨尔本小时传感器数据训练，迁移到温州五个点位；目前没有温州实测标签进行校准。
3. 此次模型在留出测试数据上的逐小时 MAE 比零增量基线差，因此不应将输出宣传为已验证的温州实时预测。对外展示宜标注“跨域试验估计”。
4. 模型只使用点位 POI 与时间日历特征（未使用实时天气、人流或当前填充水平）。它输出的是下一小时正向填充增量估计，不是溢满概率。
5. risk/riskLevel 使用可解释的手工规则，risk=80 不表示有 80% 溢满概率。
6. gain 默认 1.0，没有再用 3.375 倍情景放大。未经实测校准不要为了曲线好看手动提高 gain。
