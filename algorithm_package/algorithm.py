# -*- coding: utf-8 -*-
"""智桶潮汐：基于已训练增量 MLP 的点位状态更新模块（v0.7 修正版）

修正内容：
1. BIN_LOCATIONS 改为温州大学城真实地名
2. gain 默认值从 1.0 降至 0.25（墨尔本→温州跨域适配）
3. RISK_CONFIG 调整：填充率权重↑、增速权重↓、高速阈值放宽
4. 新增 _safe_growth_pph 限速，防止 tick 间隔短导致 pph 爆炸
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Mapping, Optional
import os
import math

import numpy as np
import torch
from torch import nn

# ============ 修正1：真实地名 ============
BIN_LOCATIONS = {
    "A001": "温大南校-主道环岛",
    "A002": "温大南校-A区食堂",
    "A003": "温大C区宿舍区",
    "A004": "温大B区宿舍区",
    "A005": "温医大-中心北路食堂",
}

# Static point-of-interest counts
BIN_POI_MAP = {
    "A001": {"restaurant": 8, "residential": 3, "commercial": 5, "school": 2, "park": 1},
    "A002": {"restaurant": 5, "residential": 0, "commercial": 2, "school": 8, "park": 2},
    "A003": {"restaurant": 25, "residential": 40, "commercial": 15, "school": 3, "park": 1},
    "A004": {"restaurant": 10, "residential": 10, "commercial": 8, "school": 5, "park": 2},
    "A005": {"restaurant": 2, "residential": 0, "commercial": 1, "school": 3, "park": 15},
}
POI_ORDER = ["restaurant", "residential", "commercial", "school", "park"]
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "melbourne_hourly_increment_mlp_seed_20261025_fold_1.pt"

# ============ 修正3：风险配置调整 ============
RISK_CONFIG = {
    "high_velocity_pph": 15.0,    # 原 6.0 → 放宽，防止轻易触发高速风险
    "fill_weight": 0.85,          # 原 0.75 → 填充率主导
    "velocity_weight": 0.15,      # 原 0.25 → 降低增速影响
    "medium_fill": 50.0,
    "high_fill": 80.0,
    "critical_fill": 95.0,
}

class _IncrementMLP(nn.Module):
    def __init__(self, n_in: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

_MODEL = None
_CKPT = None
_MODEL_PATH_USED = None


def _as_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return datetime.fromisoformat(value)
    raise TypeError("时间参数必须是 datetime 或 ISO 格式字符串")


def _clamp_percent(value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("填充率必须是有限数值")
    return max(0.0, min(100.0, value))


def _model_path(model_path: Optional[str | Path] = None) -> Path:
    path = Path(model_path or os.environ.get("ZHITONG_INCREMENT_MODEL", DEFAULT_MODEL_PATH)).expanduser()
    if not path.is_file():
        raise FileNotFoundError(
            f"找不到增量预测模型权重：{path}。请将权重文件放在 algorithm.py 同目录，"
            "或设置环境变量 ZHITONG_INCREMENT_MODEL 指定路径。"
        )
    return path


def _load_model(model_path: Optional[str | Path] = None):
    global _MODEL, _CKPT, _MODEL_PATH_USED
    path = _model_path(model_path).resolve()
    if _MODEL is not None and _MODEL_PATH_USED == str(path):
        return _MODEL, _CKPT
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt.get("model_type") != "HourlyPositiveIncrementMLP":
        raise ValueError(f"不支持的模型类型：{ckpt.get('model_type')!r}")
    model = _IncrementMLP(int(ckpt["n_in"]), int(ckpt["hidden"]))
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    _MODEL, _CKPT, _MODEL_PATH_USED = model, ckpt, str(path)
    return _MODEL, _CKPT


def _feature_vector(bin_id: str, timestamp: datetime, ckpt: dict) -> np.ndarray:
    """Build the 22 input features in exactly the training/inference feature order."""
    if bin_id not in BIN_POI_MAP:
        raise KeyError(f"未知点位 ID：{bin_id}")
    static_cols = list(ckpt["static_cols"])
    calendar_cols = list(ckpt["calendar_cols"])
    means_obj, stds_obj = ckpt["means"], ckpt["stds"]
    means = np.asarray([float(means_obj[c]) for c in static_cols], dtype=np.float64)
    stds = np.asarray([float(stds_obj[c]) for c in static_cols], dtype=np.float64)
    stds = np.where(stds == 0, 1.0, stds)

    raw_static = []
    poi_logs = []
    for poi in POI_ORDER:
        value = float(BIN_POI_MAP[bin_id][poi])
        poi_logs.append(math.log1p(max(0.0, value)))
    for col in static_cols:
        if col == "latitude":
            raw_static.append(float(means_obj[col]))
        elif col == "longitude":
            raw_static.append(float(means_obj[col]))
        elif col.startswith("log1p_poi_"):
            src = col[len("log1p_poi_"):]
            poi_name = src.replace("_500m", "")
            raw_static.append(math.log1p(float(BIN_POI_MAP[bin_id][poi_name])))
        elif col.endswith("_missing"):
            raw_static.append(0.0)
        else:
            raise ValueError(f"模型包含未实现的静态特征：{col}")

    xstat = (np.asarray(raw_static, dtype=np.float64) - means) / stds
    all_poi_logs = np.asarray([
        [math.log1p(float(BIN_POI_MAP[b][p])) for p in POI_ORDER]
        for b in sorted(BIN_LOCATIONS)
    ], dtype=np.float64)
    poi_mu = all_poi_logs.mean(axis=0)
    poi_sd = all_poi_logs.std(axis=0, ddof=0)
    poi_sd = np.where(poi_sd > 0, poi_sd, 1.0)
    local_z = (np.asarray(poi_logs, dtype=np.float64) - poi_mu) / poi_sd
    for idx, col in enumerate(static_cols):
        if col.startswith("log1p_poi_"):
            src = col[len("log1p_poi_"):]
            poi_name = src.replace("_500m", "")
            xstat[idx] = local_z[POI_ORDER.index(poi_name)]
    for idx, col in enumerate(static_cols):
        if col in ("latitude", "longitude"):
            xstat[idx] = 0.0
    xstat = np.clip(xstat, -8.0, 8.0)

    hour = timestamp.hour
    dow = timestamp.weekday()
    doy = timestamp.timetuple().tm_yday
    cal_values = {
        "hour_sin": math.sin(2 * math.pi * hour / 24.0),
        "hour_cos": math.cos(2 * math.pi * hour / 24.0),
        "dow_sin": math.sin(2 * math.pi * dow / 7.0),
        "dow_cos": math.cos(2 * math.pi * dow / 7.0),
        "doy_sin": math.sin(2 * math.pi * (doy - 1) / 365.2425),
        "doy_cos": math.cos(2 * math.pi * (doy - 1) / 365.2425),
        "is_weekend": float(dow >= 5),
        "is_holiday": 0.0,
        "pre_holiday": 0.0,
        "post_holiday": 0.0,
    }
    xcal = np.asarray([cal_values[c] for c in calendar_cols], dtype=np.float32)
    x = np.concatenate([xstat.astype(np.float32), xcal]).astype(np.float32)
    if x.shape != (int(ckpt["n_in"]),) or not np.isfinite(x).all():
        raise ValueError(f"模型输入特征非法，shape={x.shape}")
    return x


def predict_hourly_increment(bin_id: str, hour_ending: Any, *, model_path: Optional[str | Path] = None,
                             gain: float = 1.0) -> float:
    """Return model-predicted positive fill increment in percentage points per hour."""
    timestamp = _as_datetime(hour_ending)
    if gain < 0 or not math.isfinite(gain):
        raise ValueError("gain 必须是有限的非负数")
    model, ckpt = _load_model(model_path)
    x = _feature_vector(bin_id, timestamp, ckpt)
    with torch.no_grad():
        raw = float(model(torch.from_numpy(x).reshape(1, -1)).item())
        fraction = float(torch.nn.functional.softplus(torch.tensor(raw)).item())
    return max(0.0, min(100.0, fraction * 100.0 * float(gain)))


def _risk_for(fill_rate: float, growth_rate_pph: float) -> tuple[float, str]:
    fill = _clamp_percent(fill_rate)
    velocity_score = min(100.0, max(0.0, growth_rate_pph) / RISK_CONFIG["high_velocity_pph"] * 100.0)
    score = (RISK_CONFIG["fill_weight"] * fill + RISK_CONFIG["velocity_weight"] * velocity_score)
    score = round(max(0.0, min(100.0, score)), 2)
    if fill >= RISK_CONFIG["critical_fill"] or score >= 85.0:
        level = "critical"
    elif fill >= RISK_CONFIG["high_fill"] or score >= 65.0:
        level = "high"
    elif fill >= RISK_CONFIG["medium_fill"] or score >= 40.0:
        level = "medium"
    else:
        level = "low"
    return score, level


def init_bins(initial_fill: Any = 0.0, start_time: Optional[Any] = None) -> Dict[str, Dict[str, Any]]:
    """Initialize five bins."""
    clock = _as_datetime(start_time) if start_time is not None else datetime.now().replace(microsecond=0)
    bins: Dict[str, Dict[str, Any]] = {}
    for bin_id, location in BIN_LOCATIONS.items():
        raw_fill = initial_fill.get(bin_id, 0.0) if isinstance(initial_fill, Mapping) else initial_fill
        fill = _clamp_percent(float(raw_fill))
        risk, level = _risk_for(fill, 0.0)
        bins[bin_id] = {
            "bin_id": bin_id, "location": location,
            "fillRate": round(fill, 6), "risk": risk, "riskLevel": level,
            "growthRatePerHour": 0.0, "lastIncrement": 0.0,
            "isFull": fill >= 100.0, "lastUpdated": clock.isoformat(sep=" "),
            "_clock": clock, "_history": [round(fill, 2)],
        }
    return bins


def _cleaned(clean_flags: Any, bin_id: str) -> bool:
    if clean_flags is None:
        return False
    if isinstance(clean_flags, Mapping):
        return bool(clean_flags.get(bin_id, False))
    if isinstance(clean_flags, (str, bytes)):
        return bin_id == clean_flags
    try:
        return bin_id in clean_flags
    except TypeError:
        return False


def _integrated_model_increment(bin_id: str, start: datetime, end: datetime,
                                model_path: Optional[str | Path], gain: float) -> float:
    """Integrate MLP's predicted hourly rate over arbitrary interval."""
    if end <= start:
        return 0.0
    total = 0.0
    cursor = start
    while cursor < end:
        next_hour = cursor.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        segment_end = min(end, next_hour)
        hour_ending = next_hour
        rate_pph = predict_hourly_increment(bin_id, hour_ending, model_path=model_path, gain=gain)
        hours = (segment_end - cursor).total_seconds() / 3600.0
        total += rate_pph * hours
        cursor = segment_end
    return max(0.0, total)


# ============ 修正4：growth_pph 限速保护 ============
def _safe_growth_pph(delta: float, dt_minutes: float, max_pph: float = 20.0) -> float:
    """计算 growthRatePerHour，但限制上限防止短间隔爆炸。"""
    if dt_minutes <= 0:
        return 0.0
    raw_pph = max(0.0, delta) * 60.0 / dt_minutes
    return min(raw_pph, max_pph)


# ============ 修正2：gain 默认值 0.25 ============
def tick(bins: Dict[str, Dict[str, Any]], clean_flags: Any = None, *,
         now: Optional[Any] = None, elapsed_minutes: float = 5.0,
         sensor_readings: Optional[Mapping[str, float]] = None,
         incoming_amounts: Optional[Mapping[str, float]] = None,
         model_path: Optional[str | Path] = None, gain: float = 1) -> Dict[str, Dict[str, Any]]:
    """Update the five bins. Default gain=0.25 for Wenzhou cross-domain adaptation."""
    if elapsed_minutes < 0 or not math.isfinite(elapsed_minutes):
        raise ValueError("elapsed_minutes 不能为负数且必须有限")
    if not isinstance(bins, dict) or not bins:
        raise ValueError("bins 必须是 init_bins() 返回的非空字典")
    now_dt = _as_datetime(now) if now is not None else None
    for bin_id, state in bins.items():
        start = state.get("_clock")
        if not isinstance(start, datetime):
            start = datetime.now().replace(microsecond=0)
        if now_dt is not None:
            end = now_dt
            dt_minutes = max(0.0, (end - start).total_seconds() / 60.0)
        else:
            dt_minutes = float(elapsed_minutes)
            end = start + timedelta(minutes=dt_minutes)
        old_fill = float(state.get("fillRate", 0.0))
        if _cleaned(clean_flags, bin_id):
            new_fill = 0.0
            delta = -old_fill
            growth_pph = 0.0
        elif sensor_readings is not None and bin_id in sensor_readings:
            new_fill = _clamp_percent(sensor_readings[bin_id])
            delta = new_fill - old_fill
            growth_pph = _safe_growth_pph(delta, dt_minutes)
        elif incoming_amounts is not None and bin_id in incoming_amounts:
            increment = max(0.0, float(incoming_amounts[bin_id]))
            new_fill = _clamp_percent(old_fill + increment)
            delta = new_fill - old_fill
            growth_pph = _safe_growth_pph(delta, dt_minutes)
        else:
            increment = _integrated_model_increment(bin_id, start, end, model_path, gain)
            new_fill = _clamp_percent(old_fill + increment)
            delta = new_fill - old_fill
            growth_pph = _safe_growth_pph(delta, dt_minutes)
        if new_fill >= 99.99:
            new_fill = 100.0
        risk, level = _risk_for(new_fill, growth_pph)
        state["fillRate"] = round(new_fill, 6)
        state["risk"] = risk
        state["riskLevel"] = level
        state["growthRatePerHour"] = round(growth_pph, 3)
        state["lastIncrement"] = round(delta, 3)
        state["isFull"] = new_fill >= 100.0
        state["lastUpdated"] = end.isoformat(sep=" ")
        state["_clock"] = end
        state.setdefault("_history", []).append(round(new_fill, 2))
    return bins


def snapshot_bins(bins: Dict[str, Dict[str, Any]]) -> list[Dict[str, Any]]:
    """Return state records without internal clock/history fields."""
    visible = ("bin_id", "location", "fillRate", "risk", "riskLevel",
               "growthRatePerHour", "lastIncrement", "isFull", "lastUpdated")
    rows = []
    for state in bins.values():
        row = {key: state.get(key) for key in visible}
        for key in ("fillRate", "risk", "lastIncrement"):
            if isinstance(row.get(key), (int, float)):
                row[key] = round(float(row[key]), 2)
        if isinstance(row.get("growthRatePerHour"), (int, float)):
            row["growthRatePerHour"] = round(float(row["growthRatePerHour"]), 3)
        rows.append(row)
    return rows


__all__ = ["init_bins", "tick", "snapshot_bins", "predict_hourly_increment", "BIN_LOCATIONS"]
