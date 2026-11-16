# -*- coding: utf-8 -*-
from datetime import datetime
from algorithm import init_bins, tick, snapshot_bins, predict_hourly_increment

# Demonstrate one hourly model prediction and a short real inference loop.
start = datetime.fromisoformat("2026-07-01 00:00:00")
print("MLP-predicted increment over 00:00-01:00 (percentage points):")
for bid in ("A001", "A002", "A003", "A004", "A005"):
    print(bid, round(predict_hourly_increment(bid, datetime(2026, 7, 1, 1)), 4))

bins = init_bins(initial_fill=0, start_time=start)
# Simulate 12 five-minute ticks to advance one hour. Each tick calls the trained MLP.
for _ in range(12):
    tick(bins, elapsed_minutes=5)
print("\nState after the first hour:")
for row in snapshot_bins(bins):
    print(row)
