# coding: utf-8
"""问题3离线探索：全向源、HTTP规则复刻、Q2交会初估+近场闭环追踪。

本文件只生成本地合成案例，不调用正式测试接口。SyntheticArena不读取真实模拟器的隐藏数据；
策略只能通过measure/clear返回值行动，用于验证时间账本和算法边界。
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import uuid
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

ARENA_R = 1800.0
V = 5.0
MEASURE_S = 5.0
SWITCH_S = 1.0
CLEAR_R = 20.0
NEAR_R = 5.0


def norm_angle(deg: float) -> float:
    return deg % 360.0


def angle_diff(a: float, b: float) -> float:
    return (a - b + 180.0) % 360.0 - 180.0


def point_at(p: Tuple[float, float], dist: float, deg: float) -> Tuple[float, float]:
    t = math.radians(deg)
    return (p[0] + dist * math.cos(t), p[1] + dist * math.sin(t))


def ray_intersection(
    p1: Tuple[float, float], a1: float,
    p2: Tuple[float, float], a2: float,
) -> Optional[Tuple[float, float]]:
    """两条带测向误差中心线的前向交点；近平行或背向时返回None。"""
    t1, t2 = math.radians(a1), math.radians(a2)
    d1 = (math.cos(t1), math.sin(t1))
    d2 = (math.cos(t2), math.sin(t2))
    cross = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(cross) < 1e-5:
        return None
    q = (p2[0] - p1[0], p2[1] - p1[1])
    u = (q[0] * d2[1] - q[1] * d2[0]) / cross
    v = (q[0] * d1[1] - q[1] * d1[0]) / cross
    if u < 0.0 or v < 0.0:
        return None
    return (p1[0] + u * d1[0], p1[1] + u * d1[1])


@dataclass
class Source:
    x: float
    y: float
    receive_r: float
    cleared: bool = False


class SyntheticArena:
    """问题3全向规则的黑盒模拟器；仅以接口响应向策略透露信息。"""

    def __init__(self, seed: int, n: Optional[int] = None, keep_log: bool = False):
        self.rng = random.Random(seed)
        self.err_rng = random.Random(seed + 1000003)
        self.n = n if n is not None else self.rng.randint(10, 16)
        channels = self.rng.sample(range(1, 21), self.n)
        self.sources: Dict[int, Source] = {}
        for ch in channels:
            rr = ARENA_R * math.sqrt(self.rng.random())
            aa = self.rng.uniform(0.0, 360.0)
            self.sources[ch] = Source(
                rr * math.cos(math.radians(aa)),
                rr * math.sin(math.radians(aa)),
                self.rng.uniform(1000.0, 1500.0),
            )
        self.x = 0.0
        self.y = 0.0
        self.channel = 1
        self.t = 0.0
        self.cleared_count = 0
        self.log: List[dict] = [] if keep_log else []
        self.keep_log = keep_log

    def _move(self, x: float, y: float) -> float:
        d = math.hypot(x - self.x, y - self.y)
        self.t += d / V
        self.x, self.y = x, y
        return d

    def measure(self, x: float, y: float, channel: int) -> dict:
        dmove = self._move(x, y)
        switched = int(channel != self.channel)
        self.t += switched * SWITCH_S + MEASURE_S
        self.channel = channel
        out = {"accepted": True, "measure_result": "no_signal"}
        src = self.sources.get(channel)
        if src is not None and not src.cleared:
            d = math.hypot(src.x - x, src.y - y)
            if d <= src.receive_r:
                if d <= NEAR_R:
                    out["measure_result"] = "near"
                else:
                    true_a = norm_angle(math.degrees(math.atan2(src.y - y, src.x - x)))
                    err = self.err_rng.uniform(-1.0, 1.0)
                    out["measure_result"] = "direction"
                    out["svd_deg"] = round(norm_angle(true_a + err), 2)
        if self.keep_log:
            self.log.append({"op": "measure", "x": x, "y": y, "channel": channel,
                             "result": out["measure_result"], "time": self.t,
                             "move_m": dmove, "switched": switched})
        return out

    def clear(self, x: float, y: float, channel: int) -> dict:
        dmove = self._move(x, y)
        src = self.sources.get(channel)
        success = src is not None and not src.cleared and math.hypot(src.x - x, src.y - y) <= CLEAR_R
        self.t += 5.0 if success else 3.0
        if success:
            src.cleared = True
            self.cleared_count += 1
        result = "success" if success else "no_target_in_range"
        if self.keep_log:
            self.log.append({"op": "clear", "x": x, "y": y, "channel": channel,
                             "result": result, "time": self.t, "move_m": dmove})
        return {"accepted": True, "clear_result": result}


class Strategy:
    """可直接翻译成HTTP动作的策略状态机。"""

    def __init__(self, arena: SyntheticArena, baseline: float = 800.0, alpha: float = 45.0):
        self.a = arena
        self.baseline = baseline
        self.alpha = alpha
        self.cleared = set()
        self.detected = 0
        self.failed_localizations = 0

    def _mark_clear(self, ch: int, result: dict) -> bool:
        if result.get("clear_result") == "success":
            self.cleared.add(ch)
            return True
        return False

    def pursue(self, ch: int, start: Tuple[float, float], max_steps: int = 140) -> bool:
        """在估计点附近闭环：每15m重测方向，避免±1°累计横向偏差。"""
        x, y = start
        # 先直接尝试，成功时不额外占用一次测向。
        if self._mark_clear(ch, self.a.clear(x, y, ch)):
            return True
        for _ in range(max_steps):
            r = self.a.measure(x, y, ch)
            kind = r.get("measure_result")
            if kind == "near":
                if self._mark_clear(ch, self.a.clear(x, y, ch)):
                    return True
                continue
            if kind != "direction":
                return False
            x, y = point_at((x, y), 15.0, float(r["svd_deg"]))
            if self._mark_clear(ch, self.a.clear(x, y, ch)):
                return True
        return False

    def localize_clear(self, ch: int, s1: Tuple[float, float], theta1: float) -> bool:
        """Q2候选站优先；二站失败时退化为单站近场闭环。"""
        # Q2蝴蝶翼的一个保守中点：b=800m, |alpha|=45°。
        for side in (1.0, -1.0):
            s2 = point_at(s1, self.baseline, theta1 + side * self.alpha)
            r2 = self.a.measure(s2[0], s2[1], ch)
            kind2 = r2.get("measure_result")
            if kind2 == "near":
                if self._mark_clear(ch, self.a.clear(s2[0], s2[1], ch)):
                    return True
                continue
            if kind2 != "direction":
                continue
            guess = ray_intersection(s1, theta1, s2, float(r2["svd_deg"]))
            if guess is None:
                guess = s2
            if self.pursue(ch, guess):
                return True
        # 二站没有返回信号时，S1本身必然仍在覆盖内，直接沿新测向闭环。
        if self.pursue(ch, s1):
            return True
        self.failed_localizations += 1
        return False

    def scan_at(self, p: Tuple[float, float], channels: Iterable[int]) -> None:
        for ch in list(channels):
            if ch in self.cleared:
                continue
            r = self.a.measure(p[0], p[1], ch)
            kind = r.get("measure_result")
            if kind == "near":
                self._mark_clear(ch, self.a.clear(p[0], p[1], ch))
            elif kind == "direction":
                self.detected += 1
                self.localize_clear(ch, p, float(r["svd_deg"]))

    def run(self) -> dict:
        # 第一层：原点20频道普查，119s固定代价。
        self.scan_at((0.0, 0.0), range(1, 21))
        # 第二层：八个外围骨干点。顺序固定，便于HTTP动作复现和日志审计。
        patrol_r = 1150.0
        for i in range(8):
            if len(self.cleared) == 20:
                break
            a = i * 45.0
            p = (patrol_r * math.cos(math.radians(a)), patrol_r * math.sin(math.radians(a)))
            self.scan_at(p, (ch for ch in range(1, 21) if ch not in self.cleared))
        return {
            "sources": self.a.n,
            "cleared": self.a.cleared_count,
            "ratio": self.a.cleared_count / self.a.n,
            "virtual_time_s": self.a.t,
            "avg_time_per_cleared_s": self.a.t / self.a.cleared_count if self.a.cleared_count else None,
            "detected_events": self.detected,
            "failed_localizations": self.failed_localizations,
        }


def benchmark(seeds: int = 100) -> dict:
    rows = []
    failures = []
    for seed in range(seeds):
        arena = SyntheticArena(seed)
        result = Strategy(arena).run()
        result["seed"] = seed
        rows.append(result)
        if result["cleared"] < result["sources"]:
            failures.append(result)
    return {
        "seeds": seeds,
        "clear_ratio_mean": statistics.mean(r["ratio"] for r in rows),
        "clear_ratio_min": min(r["ratio"] for r in rows),
        "virtual_time_mean_s": statistics.mean(r["virtual_time_s"] for r in rows),
        "virtual_time_median_s": statistics.median(r["virtual_time_s"] for r in rows),
        "avg_clear_time_mean_s": statistics.mean(r["avg_time_per_cleared_s"] for r in rows),
        "full_clear_cases": sum(r["cleared"] == r["sources"] for r in rows),
        "failures": failures,
    }


class SimulatorClient:
    """真实模拟器的安全HTTP客户端骨架；默认dry_run，不调用网络。"""

    def __init__(self, base_url: str = "http://127.0.0.1:2026", robot_id: str = "Q3-local", dry_run: bool = True):
        self.base_url = base_url.rstrip("/")
        self.robot_id = robot_id
        self.dry_run = dry_run
        self.current_channel = 1
        self.position = (0.0, 0.0)

    def _body(self, **extra) -> dict:
        body = {"arena_id": "default", "robot_id": self.robot_id,
                "request_id": str(uuid.uuid4())}
        body.update(extra)
        return body

    def _post(self, path: str, body: dict) -> dict:
        if self.dry_run:
            return {"accepted": True, "dry_run": True, "path": path, "body": body}
        import requests
        response = requests.post(self.base_url + path, json=body, timeout=30)
        response.raise_for_status()
        data = response.json()
        if data.get("accepted") is not True:
            raise RuntimeError(f"simulator rejected {path}: {data}")
        return data

    def enter(self) -> dict:
        return self._post("/enter", self._body())

    def measure(self, x: float, y: float, channel: int) -> dict:
        data = self._post("/measure", self._body(position={"x": float(x), "y": float(y)}, channel=int(channel)))
        self.position = (float(x), float(y))
        self.current_channel = int(channel)
        return data

    def clear(self, x: float, y: float, channel: int) -> dict:
        data = self._post("/clear", self._body(position={"x": float(x), "y": float(y)}, channel=int(channel)))
        self.position = (float(x), float(y))
        return data

    def exit(self) -> dict:
        return self._post("/exit", self._body())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    result = benchmark(args.seeds)
    print(json.dumps(result, ensure_ascii=False, indent=2) if args.json else result)


if __name__ == "__main__":
    main()
