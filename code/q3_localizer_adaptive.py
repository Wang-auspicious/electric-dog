# coding: utf-8
"""自适应二站/三站定位子程序（只依赖 measure/clear 黑盒响应）。

与固定 800 m 蝴蝶翼相比，第二站先取 ``100 m 前进+150 m 横移``，第三站
取 ``180 m 前进+350 m 横移``。这两个点在首站检测成功且接收半径为
[1000,1500] m 时仍留在覆盖内（最坏距离约 1.37 km），因此不需要猜测接收
半径。每次测向把 ±1° 扇区和 1500 m 外接圆裁剪，MEC 半径不超过20 m时
直接在MEC中心清除；否则用最小二乘交点后按20 m闭环趋近。

``benchmark`` 使用与 q3_local_explore 相同的本地合成 Arena，绝不连接真实
模拟器。有限步数用于防止异常接口死循环。对正确的 ±1° 接口，在源距大于
20 m时，每次20 m闭环至少减少 20-40 sin(0.5°)=19.65 m；进入20 m近场
后，下一次测向要么报告 near，要么形成近反向双扇区，MEC 收缩后直接清除。
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
Q3_DIR = Path(__file__).resolve().parent
if str(Q3_DIR) not in sys.path:
    sys.path.insert(0, str(Q3_DIR))
from q3_v1_baseline import (  # type: ignore
    ARENA_R,
    CLEAR_R,
    NEAR_R,
    SyntheticArena,
    V,
    angle_diff,
    norm_angle,
    point_at,
)

Point = Tuple[float, float]


def _cross(a: Point, b: Point) -> float:
    return a[0] * b[1] - a[1] * b[0]


def _sub(a: Point, b: Point) -> Point:
    return a[0] - b[0], a[1] - b[1]


def _clip(poly: List[Point], A: float, B: float, C: float, eps: float = 1e-8) -> List[Point]:
    """Sutherland--Hodgman clipping by A*x+B*y <= C."""
    if not poly:
        return []
    out: List[Point] = []
    for p, q in zip(poly, poly[1:] + poly[:1]):
        fp = A * p[0] + B * p[1] - C
        fq = A * q[0] + B * q[1] - C
        inp, inq = fp <= eps, fq <= eps
        if inp:
            out.append(p)
        if inp != inq:
            t = fp / (fp - fq)
            out.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
    return out


def _clip_wedge(poly: List[Point], p: Point, theta: float, delta: float = 1.0) -> List[Point]:
    t = math.radians(theta)
    dl = (math.cos(t - math.radians(delta)), math.sin(t - math.radians(delta)))
    dr = (math.cos(t + math.radians(delta)), math.sin(t + math.radians(delta)))
    # cross(dl, q-p)>=0 and cross(dr, q-p)<=0, plus forward half-plane.
    poly = _clip(poly, dl[1], -dl[0], dl[1] * p[0] - dl[0] * p[1])
    poly = _clip(poly, -dr[1], dr[0], -dr[1] * p[0] + dr[0] * p[1])
    d = (math.cos(t), math.sin(t))
    poly = _clip(poly, -d[0], -d[1], -d[0] * p[0] - d[1] * p[1])
    return poly


def _clip_disk(poly: List[Point], c: Point, radius: float, sides: int = 96) -> List[Point]:
    # Circumscribed regular polygon, hence a conservative outer approximation.
    rr = radius / math.cos(math.pi / sides)
    disk = [(c[0] + rr * math.cos(2 * math.pi * k / sides),
             c[1] + rr * math.sin(2 * math.pi * k / sides)) for k in range(sides)]
    for a, b in zip(disk, disk[1:] + disk[:1]):
        e = _sub(b, a)
        # CCW polygon interior is cross(e,x-a)>=0.
        poly = _clip(poly, e[1], -e[0], e[1] * a[0] - e[0] * a[1])
        if not poly:
            break
    return poly


def feasible_polygon(stations: Sequence[Point], bearings: Sequence[float],
                     rmax: float = 1500.0, delta: float = 1.0) -> List[Point]:
    if not stations:
        return []
    lim = ARENA_R + rmax + 100.0
    poly = [(-lim, -lim), (lim, -lim), (lim, lim), (-lim, lim)]
    for p, th in zip(stations, bearings):
        poly = _clip_wedge(poly, p, th, delta)
        poly = _clip_disk(poly, p, rmax)
        if not poly:
            return []
    return poly


def _circle3(a: Point, b: Point, c: Point) -> Optional[Tuple[Point, float]]:
    d = 2.0 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))
    if abs(d) < 1e-9:
        return None
    aa, bb, cc = a[0] * a[0] + a[1] * a[1], b[0] * b[0] + b[1] * b[1], c[0] * c[0] + c[1] * c[1]
    ux = (aa * (b[1] - c[1]) + bb * (c[1] - a[1]) + cc * (a[1] - b[1])) / d
    uy = (aa * (c[0] - b[0]) + bb * (a[0] - c[0]) + cc * (b[0] - a[0])) / d
    u = (ux, uy)
    return u, math.hypot(ux - a[0], uy - a[1])


def minimum_enclosing_circle(points: Sequence[Point]) -> Tuple[Point, float]:
    """Deterministic incremental MEC, adequate for the <=~100 polygon vertices."""
    pts = list(dict.fromkeys((float(x), float(y)) for x, y in points))
    if not pts:
        return (0.0, 0.0), float("inf")
    c, r = pts[0], 0.0
    for i, p in enumerate(pts):
        if math.hypot(p[0] - c[0], p[1] - c[1]) <= r + 1e-8:
            continue
        c, r = p, 0.0
        for j in range(i):
            q = pts[j]
            if math.hypot(q[0] - c[0], q[1] - c[1]) <= r + 1e-8:
                continue
            c = ((p[0] + q[0]) / 2.0, (p[1] + q[1]) / 2.0)
            r = math.hypot(p[0] - q[0], p[1] - q[1]) / 2.0
            for k in range(j):
                s = pts[k]
                if math.hypot(s[0] - c[0], s[1] - c[1]) <= r + 1e-8:
                    continue
                cc = _circle3(p, q, s)
                if cc is not None:
                    c, r = cc
    return c, r


def line_least_squares(stations: Sequence[Point], bearings: Sequence[float]) -> Optional[Point]:
    if len(stations) < 2:
        return stations[0] if stations else None
    a11 = a12 = a22 = b1 = b2 = 0.0
    for p, th in zip(stations, bearings):
        t = math.radians(th)
        d = (math.cos(t), math.sin(t))
        # Projection onto normal n: n n^T x = n n^T p.
        n = (-d[1], d[0])
        a11 += n[0] * n[0]
        a12 += n[0] * n[1]
        a22 += n[1] * n[1]
        b1 += n[0] * (n[0] * p[0] + n[1] * p[1])
        b2 += n[1] * (n[0] * p[0] + n[1] * p[1])
    det = a11 * a22 - a12 * a12
    if abs(det) < 1e-9:
        return None
    return ((b1 * a22 - b2 * a12) / det, (a11 * b2 - a12 * b1) / det)


class AdaptiveLocalizer:
    def __init__(self, arena: SyntheticArena, delta_deg: float = 1.01):
        # The statement gives ±1°.  The local replica rounds svd_deg to 0.01°;
        # 1.01° keeps that display rounding inside the conservative wedge.
        self.a = arena
        self.delta = delta_deg
        self.failed = 0
        self.mec_direct = 0
        self.measure_steps = 0
        self.move_m = 0.0
        self.clear_attempts = 0

    def _measure(self, p: Point, ch: int) -> dict:
        before = (self.a.x, self.a.y)
        out = self.a.measure(p[0], p[1], ch)
        self.move_m += math.hypot(p[0] - before[0], p[1] - before[1])
        self.measure_steps += 1
        return out

    def _clear(self, p: Point, ch: int) -> bool:
        before = (self.a.x, self.a.y)
        out = self.a.clear(p[0], p[1], ch)
        self.move_m += math.hypot(p[0] - before[0], p[1] - before[1])
        self.clear_attempts += 1
        return out.get("clear_result") == "success"

    def _region_clear(self, stations: List[Point], bearings: List[float], ch: int) -> Optional[bool]:
        poly = feasible_polygon(stations, bearings, delta=self.delta)
        if not poly:
            return None
        c, r = minimum_enclosing_circle(poly)
        if r > CLEAR_R + 1e-6:
            return None
        self.mec_direct += 1
        # Every feasible source is within20m of c, so one clear is sufficient.
        return True if self._clear(c, ch) else None

    def pursue(self, ch: int, start: Point, stations: List[Point], bearings: List[float],
                max_steps: int = 90) -> bool:
        p = start
        last_safe = stations[-1]
        for _ in range(max_steps):
            r = self._measure(p, ch)
            kind = r.get("measure_result")
            if kind == "near":
                return self._clear(p, ch)
            if kind != "direction":
                # An LS guess can be outside the receive disk. Return to a known
                # covered station and spend the remaining finite budget there.
                if p != last_safe:
                    back = self._measure(last_safe, ch)
                    if back.get("measure_result") == "near":
                        return self._clear(last_safe, ch)
                    if back.get("measure_result") == "direction" and max_steps > 1:
                        stations.append(last_safe)
                        bearings.append(float(back["svd_deg"]))
                        q = point_at(last_safe, 20.0, float(back["svd_deg"]))
                        return self.pursue(ch, q, stations, bearings, max_steps=max_steps - 1)
                return False
            th = float(r["svd_deg"])
            stations.append(p)
            bearings.append(th)
            last_safe = p
            got = self._region_clear(stations, bearings, ch)
            if got is not None:
                return got
            # With ±delta error and range>=20m, reduction is at least
            # 20-40*sin(delta/2), which is positive (~19.65m at 1 degree).
            p = point_at(p, 20.0, th)
        # Hard cap prevents a malformed API from causing an infinite loop.
        self.failed += 1
        return False

    def localize_clear(self, ch: int, s1: Point, theta1: float) -> bool:
        stations = [s1]
        bearings = [theta1]
        # If caller supplied a point that was already in the near field, clear now.
        # No measure is repeated here; the caller's first result is authoritative.
        side = 1.0
        t = math.radians(theta1)
        f = (math.cos(t), math.sin(t))
        l = (-math.sin(t), math.cos(t))
        # Guaranteed-safe short baseline followed by a longer adaptive baseline.
        s2 = (s1[0] + 100.0 * f[0] + 150.0 * side * l[0],
              s1[1] + 100.0 * f[1] + 150.0 * side * l[1])
        r2 = self._measure(s2, ch)
        if r2.get("measure_result") == "near":
            return self._clear(s2, ch)
        if r2.get("measure_result") == "direction":
            stations.append(s2); bearings.append(float(r2["svd_deg"]))
            got = self._region_clear(stations, bearings, ch)
            if got is not None:
                return got
        else:
            # This can only happen on an API anomaly because the offset is safe;
            # restart from S1 without trying the opposite 800m wing.
            return self.pursue(ch, s1, [s1], [theta1])
        s3 = (s1[0] + 180.0 * f[0] + 350.0 * side * l[0],
              s1[1] + 180.0 * f[1] + 350.0 * side * l[1])
        r3 = self._measure(s3, ch)
        if r3.get("measure_result") == "near":
            return self._clear(s3, ch)
        if r3.get("measure_result") == "direction":
            stations.append(s3); bearings.append(float(r3["svd_deg"]))
            got = self._region_clear(stations, bearings, ch)
            if got is not None:
                return got
            guess = line_least_squares(stations, bearings) or s3
            if self.pursue(ch, guess, stations, bearings):
                return True
        else:
            if self.pursue(ch, s2, stations, bearings):
                return True
        self.failed += 1
        return False


class FixedLocationArena(SyntheticArena):
    """Same local replica, with the problem's fixed error at each position."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._angle_cache = {}

    def measure(self, x: float, y: float, channel: int) -> dict:
        out = super().measure(x, y, channel)
        if out.get("measure_result") == "direction":
            key = (int(channel), float(x), float(y))
            if key in self._angle_cache:
                out["svd_deg"] = self._angle_cache[key]
            else:
                self._angle_cache[key] = out["svd_deg"]
        return out


def adaptive_localize(arena: SyntheticArena, channel: int,
                      first_station: Point, first_bearing_deg: float) -> bool:
    """One-call adapter for a strategy state machine after its first direction."""
    return AdaptiveLocalizer(arena).localize_clear(
        int(channel), first_station, float(first_bearing_deg)
    )


def benchmark(seeds: int = 100, sources_per_seed: int = 3,
              random_start_distance: bool = False) -> dict:
    """Compare movement of this localizer with fixed-800m two-station geometry.

    For each synthetic source we choose a legal first station 0.7R from the
    source on a random bearing, call one direction measurement, and then run both
    methods on separate identical arenas. This isolates localization movement;
    scanning/patrol costs are deliberately excluded.
    """
    rows = []
    rng = random.Random(20260911)
    from q3_v1_baseline import Strategy, ray_intersection  # type: ignore
    for seed in range(seeds):
        base = SyntheticArena(seed)
        items = list(base.sources.items())[:sources_per_seed]
        for ch, src in items:
            phi = rng.uniform(0.0, 360.0)
            d = (rng.uniform(6.0, src.receive_r) if random_start_distance else
                 min(0.7 * src.receive_r, src.receive_r - 25.0))
            s1 = (src.x - d * math.cos(math.radians(phi)), src.y - d * math.sin(math.radians(phi)))
            # Clone source exactly; error RNG is deterministic by seed.
            def clone(keep_log: bool = False):
                a = FixedLocationArena(seed, n=1, keep_log=keep_log)
                a.sources = {ch: type(src)(src.x, src.y, src.receive_r, False)}
                a.x, a.y, a.channel = s1[0], s1[1], 1
                return a
            aa = clone(); rr = aa.measure(s1[0], s1[1], ch)
            if rr.get("measure_result") != "direction":
                continue
            th = float(rr["svd_deg"])
            # Fixed-800m comparator starts from the same measured S1.
            ab = clone(keep_log=True); ab.measure(s1[0], s1[1], ch)
            sb = Strategy(ab)
            before_b = (ab.x, ab.y)
            sb.localize_clear(ch, s1, th)
            fixed_move = sum(e.get("move_m", 0.0) for e in ab.log)
            # Adaptive on a fresh clone.
            ac = clone(); ac.measure(s1[0], s1[1], ch)
            ad = AdaptiveLocalizer(ac)
            ad.localize_clear(ch, s1, th)
            # ad.move_m starts at S1 and includes only localizer motions; fixed_move
            # includes the initial measure's zero movement as well.
            rows.append({"seed": seed, "channel": ch, "fixed800_move_m": fixed_move,
                         "adaptive_move_m": ad.move_m, "fixed800_success": int(ab.cleared_count == 1),
                         "adaptive_success": int(ac.cleared_count == 1),
                         "fixed800_time_s": ab.t, "adaptive_time_s": ac.t,
                         "adaptive_mec_direct": ad.mec_direct, "adaptive_failed": ad.failed})
    return {
        "seeds": seeds,
        "first_station_distance": "uniform 6m..R" if random_start_distance else "0.7R",
        "cases": len(rows),
        "fixed800_mean_move_m": statistics.mean(r["fixed800_move_m"] for r in rows),
        "adaptive_mean_move_m": statistics.mean(r["adaptive_move_m"] for r in rows),
        "movement_saving_pct": 100.0 * (1.0 - statistics.mean(r["adaptive_move_m"] for r in rows) /
                                          statistics.mean(r["fixed800_move_m"] for r in rows)),
        "fixed800_success_rate": statistics.mean(r["fixed800_success"] for r in rows),
        "adaptive_success_rate": statistics.mean(r["adaptive_success"] for r in rows),
        "fixed800_mean_time_s": statistics.mean(r["fixed800_time_s"] for r in rows),
        "adaptive_mean_time_s": statistics.mean(r["adaptive_time_s"] for r in rows),
        "rows": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--sources-per-seed", type=int, default=3)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--random-start-distance", action="store_true")
    args = ap.parse_args()
    out = benchmark(args.seeds, args.sources_per_seed, args.random_start_distance)
    print(json.dumps(out, ensure_ascii=False, indent=2) if args.json else
          {k: v for k, v in out.items() if k != "rows"})


if __name__ == "__main__":
    main()
