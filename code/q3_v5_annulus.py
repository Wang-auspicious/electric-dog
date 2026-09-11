# coding: utf-8
"""问题3 策略 v5：安努卢斯覆盖证书 + 滚动时域 TSPN。

与公开最强方案（li2396803 v2，严格模式 279.95 s/源）的两点结构差异：

1. 覆盖证书只覆盖「安努卢斯」而不是整个圆盘。
   某频道若在原点没被听到，它要么是空频道，要么其源距原点 > R_G >= 1000 m。
   所以「未被听到的频道」的源只可能落在 1000 < r <= 1800 的环形区里，
   覆盖证书不需要覆盖整个 1800 m 圆盘。

2. 每一个停靠点都同时是「清除点」和「覆盖点」。
   覆盖证书的定义是：对每个尚未清除的频道，在它被测量的所有停靠点里，
   至少有一个落点距离它不超过 1000 m 且当时返回 no_signal。
   因此**去清除一个源的那次停靠，顺便也把该点周围 1000 m 从覆盖残差里扣掉了**。
   于是不需要先绕完固定环再回头收货，只要保证残差最终归零。

正确性（可审计）：只有当
   (a) 已清除源数达到题面上限 16，或
   (b) 残差网格为空（= 对每个未清除频道，其 1000 m 停靠点并集已覆盖整个圆盘）
才允许退出。前者保证不可能还有第 17 个源，后者保证任何未清除频道都没有源。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from q3_v1_baseline import SyntheticArena, point_at, ray_intersection  # noqa: E402

R_ARENA = 1800.0
R_NEAR = 5.0
R_RECV_MIN = 1000.0
COVER_R = 995.0          # 覆盖判据留 5 m 余量（网格 25 m，够用）
GRID_STEP = 25.0


def dir_gap(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def build_grid(step: float = GRID_STEP) -> np.ndarray:
    xs = np.arange(-R_ARENA, R_ARENA + step * 0.5, step)
    X, Y = np.meshgrid(xs, xs)
    m = (X ** 2 + Y ** 2) <= R_ARENA ** 2
    return np.stack([X[m], Y[m]], axis=1).astype(np.float64)


GRID = build_grid()


def build_candidates(step: float = 200.0) -> np.ndarray:
    """候选覆盖落点：粗网格，限制在 r>=700 的环形带（源只可能在这附近被漏掉）。"""
    xs = np.arange(-1550.0, 1550.0 + step * 0.5, step)
    X, Y = np.meshgrid(xs, xs)
    r = np.hypot(X, Y)
    m = (r <= R_ARENA) & (r >= 800.0)
    return np.stack([X[m], Y[m]], axis=1).astype(np.float64)


CAND = build_candidates()


class AnnulusStrategy:
    """黑盒策略：只用 measure / clear 的返回值。"""

    def __init__(self, arena: SyntheticArena,
                 cover_r: float = COVER_R,
                 scan_at_clear_stop: bool = True,
                 beta: float = 700.0):
        self.a = arena
        self.cover_r = cover_r
        self.scan_at_clear = scan_at_clear_stop
        self.beta = beta
        self.bear: Dict[int, List[Tuple[float, float, float]]] = {}
        self.cleared: set = set()
        self.heard: set = set()
        self.giveup: set = set()
        self.residue = np.hypot(GRID[:, 0], GRID[:, 1]) > R_RECV_MIN   # 原点扫描后的安努卢斯
        self.n_scan_stops = 0
        self.n_stops = 0
        self.failed = 0
        self.stats = {"measure": 0, "no_signal": 0, "clear_try": 0, "clear_fail": 0}

    # ------------------------------------------------------------ 基本动作
    def _goto(self, p: Tuple[float, float]) -> None:
        self.a._move(p[0], p[1])

    def _clear(self, ch: int) -> bool:
        self.stats["clear_try"] += 1
        r = self.a.clear(self.a.x, self.a.y, ch)
        if r["clear_result"] == "success":
            self.cleared.add(ch)
            return True
        self.stats["clear_fail"] += 1
        return False

    def _measure(self, ch: int) -> Optional[str]:
        self.stats["measure"] += 1
        r = self.a.measure(self.a.x, self.a.y, ch)
        k = r["measure_result"]
        if k == "direction":
            self.bear.setdefault(ch, []).append((self.a.x, self.a.y, float(r["svd_deg"])))
            self.heard.add(ch)
            return "direction"
        if k == "near":
            self.heard.add(ch)
            return "near"
        self.stats["no_signal"] += 1
        return "no_signal"

    # ------------------------------------------------------------ 覆盖残差
    def _apply_coverage(self, p: Tuple[float, float]) -> None:
        d2 = (GRID[:, 0] - p[0]) ** 2 + (GRID[:, 1] - p[1]) ** 2
        self.residue &= d2 > self.cover_r ** 2

    def _unknown(self) -> List[int]:
        return [c for c in range(1, 21)
                if c not in self.cleared and c not in self.heard]

    def _scan_channels(self, chans: List[int]) -> None:
        for ch in chans:
            if ch in self.cleared:
                continue
            k = self._measure(ch)
            if k == "near":
                self._clear(ch)

    # ------------------------------------------------------------ 估计与追踪
    def estimate(self, ch: int) -> Optional[Tuple[float, float]]:
        bs = self.bear.get(ch, [])
        best = None
        for i in range(len(bs)):
            for j in range(i + 1, len(bs)):
                dth = dir_gap(bs[i][2], bs[j][2])
                if min(dth, 180.0 - dth) < 12.0:
                    continue
                g = ray_intersection((bs[i][0], bs[i][1]), bs[i][2],
                                     (bs[j][0], bs[j][1]), bs[j][2])
                if g is None or math.hypot(g[0], g[1]) > 2300.0:
                    continue
                if best is None:
                    best = g
        if best is not None:
            return best
        if bs:
            x, y, th = bs[-1]
            rho = 650.0 if ch in self.heard else 1400.0
            return point_at((x, y), rho, th)
        return None

    def _ray_sweep(self, ch: int) -> bool:
        bs = self.bear.get(ch)
        if not bs:
            return False
        k = self._measure(ch)
        if k is None or k == "no_signal":
            self._goto((bs[0][0], bs[0][1]))
            if self._measure(ch) in (None, "no_signal"):
                return False
        if self._clear(ch):
            return True
        hop, prev = 240.0, None
        for _ in range(16):
            k = self._measure(ch)
            if k == "near":
                if self._clear(ch):
                    return True
                continue
            if k != "direction":
                return False
            th = self.bear[ch][-1][2]
            if prev is not None and dir_gap(th, prev) > 90.0:
                hop *= 0.5
            prev = th
            if hop < 8.0:
                return False
            self._goto(point_at((self.a.x, self.a.y), hop, th))
            if self._clear(ch):
                return True
        return False

    def chase(self, ch: int) -> bool:
        if len(self.bear.get(ch, [])) < 2:
            return self._ray_sweep(ch)
        est = self.estimate(ch)
        if est is not None:
            self._goto(est)
            if self._clear(ch):
                return True
        step, prev = 300.0, None
        for _ in range(14):
            k = self._measure(ch)
            if k == "near":
                if self._clear(ch):
                    return True
                continue
            if k != "direction":
                return False
            th = self.bear[ch][-1][2]
            if prev is not None and dir_gap(th, prev) > 90.0:
                step *= 0.5
            prev = th
            if step < 8.0:
                return False
            self._goto(point_at((self.a.x, self.a.y), step, th))
            if self._clear(ch):
                return True
        return False

    # ------------------------------------------------------------ 任务选择
    def _best_scan_point(self, cur: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        if not self.residue.any():
            return None
        idx = np.flatnonzero(self.residue)
        pts = GRID[idx]
        best, best_score = None, -1.0
        r2 = self.cover_r ** 2
        for cx, cy in CAND:
            cnt = int((((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2) <= r2).sum())
            if cnt == 0:
                continue
            dist = math.hypot(cx - cur[0], cy - cur[1])
            if dist > 2600.0:
                continue
            score = cnt / (dist + 600.0)
            if score > best_score:
                best, best_score = (float(cx), float(cy)), score
        return best

    def _cover_frac(self, p):
        if not self.residue.any():
            return 0.0
        cnt = int((((GRID[:, 0] - p[0]) ** 2 + (GRID[:, 1] - p[1]) ** 2)
                   <= self.cover_r ** 2)[self.residue].sum())
        return cnt / max(1, int(self.residue.sum()))

    def run(self) -> dict:
        # ---------- 原点普查：20 个频道全扫，免费的覆盖起点 ----------
        self.n_stops += 1
        self._scan_channels(list(range(1, 21)))
        self._apply_coverage((self.a.x, self.a.y))
        # 原点扫描后，未听到频道的源只可能在安努卢斯；heard 的频道用示向线追
        guard = 0
        while guard < 600:
            guard += 1
            if len(self.cleared) >= 16:
                break
            cur = (self.a.x, self.a.y)
            # 待清频道
            pend = [c for c in (set(self.bear) - self.cleared - self.giveup)]
            clear_jobs = []
            for ch in pend:
                e = self.estimate(ch)
                if e is None:
                    continue
                clear_jobs.append((math.hypot(e[0] - cur[0], e[1] - cur[1]) + 5.0, ch, e))
            clear_jobs.sort()
            n_unk = len(self._unknown())
            scan_pt = self._best_scan_point(cur)
            scan_cost = math.inf
            if scan_pt is not None:
                d = math.hypot(scan_pt[0] - cur[0], scan_pt[1] - cur[1])
                cov = int((((GRID[:, 0] - scan_pt[0]) ** 2 +
                            (GRID[:, 1] - scan_pt[1]) ** 2) <= self.cover_r ** 2)[self.residue].sum())
                frac = cov / max(1, int(self.residue.sum()))
                scan_cost = d + 6.0 * n_unk - self.beta * frac

            if not clear_jobs and scan_pt is None:
                break
            take_scan = scan_pt is not None and (not clear_jobs or scan_cost < clear_jobs[0][0])

            if take_scan:
                self._goto(scan_pt)
                self.n_scan_stops += 1
                self._scan_channels(self._unknown())
                self._apply_coverage(scan_pt)
                continue
            _, ch, est = clear_jobs[0]
            # 这次清除停靠本身能盖掉多少残差？够多就顺便把未察觉频道测一遍。
            useful = (self.scan_at_clear and self._unknown()
                      and self._cover_frac(est) >= 0.18)
            if useful:
                self._goto(est)
                self._scan_channels(self._unknown())
                self._apply_coverage(est)
            if not self.chase(ch):
                self.giveup.add(ch)

        # 收尾：残差还在但已无清除任务时，用保证性覆盖把残差清空
        guard = 0
        while self.residue.any() and guard < 40:
            guard += 1
            p = self._best_scan_point((self.a.x, self.a.y))
            if p is None:
                break
            self._goto(p)
            self.n_stops += 1
            self.n_scan_stops += 1
            self._scan_channels(self._unknown())
            self._apply_coverage(p)

        return {
            "sources": self.a.n,
            "cleared": self.a.cleared_count,
            "ratio": self.a.cleared_count / self.a.n,
            "virtual_time_s": self.a.t,
            "avg_time_per_cleared_s": (self.a.t / self.a.cleared_count
                                       if self.a.cleared_count else None),
            "residue_left": int(self.residue.sum()),
            "unknown_left": len(self._unknown()),
            "giveup": len(self.giveup),
            "n_stops": self.n_stops,
            "n_scan_stops": self.n_scan_stops,
            "stats": dict(self.stats),
        }


def benchmark(seeds: int = 200, **kw) -> dict:
    rows, fails = [], []
    for seed in range(seeds):
        arena = SyntheticArena(seed)
        r = AnnulusStrategy(arena, **kw).run()
        r["seed"] = seed
        rows.append(r)
        if r["cleared"] < r["sources"]:
            fails.append((seed, r["cleared"], r["sources"]))
    ok = [r for r in rows]
    return {
        "seeds": seeds,
        "full_clear": sum(r["cleared"] == r["sources"] for r in rows),
        "ratio_mean": statistics.mean(r["ratio"] for r in rows),
        "time_mean_s": statistics.mean(r["virtual_time_s"] for r in rows),
        "time_median_s": statistics.median([r["virtual_time_s"] for r in rows]),
        "per_source_mean_s": statistics.mean([r["avg_time_per_cleared_s"] for r in rows
                                              if r["avg_time_per_cleared_s"]]),
        "per_source_p90_s": sorted([r["avg_time_per_cleared_s"] for r in rows
                                    if r["avg_time_per_cleared_s"]])[int(0.9 * len(ok)) - 1],
        "per_source_max_s": max(r["avg_time_per_cleared_s"] for r in ok
                                if r["avg_time_per_cleared_s"]),
        "mean_stops": statistics.mean(r["n_stops"] for r in rows),
        "mean_scan_stops": statistics.mean(r["n_scan_stops"] for r in rows),
        "mean_measures": statistics.mean(r["stats"]["measure"] for r in rows),
        "fail_seeds": fails[:8],
    }


# =============================================================================
# v5c：解析环 + 全局 2-opt 计划（只在“发现新源”时重规划，而不是每条腿都重规划）
# =============================================================================
def _nn2opt(points, start):
    pts = [np.asarray(p, float) for p in points]
    order, left = [], pts[:]
    cur = np.asarray(start, float)
    while left:
        i = min(range(len(left)), key=lambda k: float(((left[k] - cur) ** 2).sum()))
        order.append(left.pop(i)); cur = order[-1]

    def L(seq):
        s, c = 0.0, np.asarray(start, float)
        for p in seq:
            s += float(np.linalg.norm(p - c)); c = p
        return s

    best, improved = L(order), True
    while improved:
        improved = False
        for i in range(len(order) - 1):
            for j in range(i + 1, len(order)):
                new = order[:i] + order[i:j + 1][::-1] + order[j + 1:]
                v = L(new)
                if v < best - 1e-9:
                    order, best, improved = new, v, True
    return order, best


def _ring(k, rho):
    return [(rho * math.cos(2 * math.pi * i / k), rho * math.sin(2 * math.pi * i / k))
            for i in range(k)]


class PlanStrategy(AnnulusStrategy):
    """解析覆盖环 + 全局 2-opt 任务序列；发现新源才重规划。"""

    def __init__(self, arena, ring_k=7, ring_rho=997.2, **kw):
        super().__init__(arena, **kw)
        self.ring = _ring(ring_k, ring_rho)
        self.used_ring = set()
        self.plan: List[Tuple[str, object]] = []

    def _plan(self):
        cur = (self.a.x, self.a.y)
        tgt = [("scan", self.ring[i], None) for i in range(len(self.ring))
               if i not in self.used_ring]
        for ch in sorted(set(self.bear) - self.cleared - self.giveup):
            e = self.estimate(ch)
            if e is not None:
                tgt.append(("clear", (float(e[0]), float(e[1])), ch))
        if not tgt:
            self.plan = []
            return
        order, _ = _nn2opt([t[1] for t in tgt], cur)
        pool = {}
        for t in tgt:
            pool.setdefault((round(t[1][0], 6), round(t[1][1], 6)), []).append(t)
        rebuilt = []
        for p in order:
            k = (round(float(p[0]), 6), round(float(p[1]), 6))
            if pool.get(k):
                rebuilt.append(pool[k].pop(0))
        self.plan = rebuilt

    def run(self):
        self._scan_channels(list(range(1, 21)))
        self._apply_coverage((self.a.x, self.a.y))
        self._plan()
        guard = 0
        while guard < 400:
            guard += 1
            if len(self.cleared) >= 16:
                break
            if not self.plan:
                self._plan()
                if not self.plan:
                    break
            kind, p, extra = self.plan.pop(0)
            if kind == "scan":
                idx = min(range(len(self.ring)),
                          key=lambda i: (self.ring[i][0] - p[0]) ** 2 + (self.ring[i][1] - p[1]) ** 2)
                if idx in self.used_ring:
                    continue
                self.used_ring.add(idx)
                self._goto(p)
                self.n_scan_stops += 1
                self._scan_channels(self._unknown())
                self._apply_coverage(p)
                self._plan()                      # 新发现 -> 重规划
            else:
                ch = extra
                if ch in self.cleared or ch in self.giveup:
                    continue
                self._goto(p)
                if not self.chase(ch):
                    self.giveup.add(ch)
                self._plan()
        return {
            "sources": self.a.n,
            "cleared": self.a.cleared_count,
            "ratio": self.a.cleared_count / self.a.n,
            "virtual_time_s": self.a.t,
            "avg_time_per_cleared_s": (self.a.t / self.a.cleared_count if self.a.cleared_count else None),
            "residue_left": int(self.residue.sum()),
            "unknown_left": len(self._unknown()),
            "giveup": len(self.giveup),
            "n_stops": self.n_stops,
            "n_scan_stops": self.n_scan_stops,
            "stats": dict(self.stats),
        }


def benchmark_plan(seeds=200, ring_k=7, ring_rho=997.2):
    rows, fails = [], []
    for seed in range(seeds):
        arena = SyntheticArena(seed)
        r = PlanStrategy(arena, ring_k=ring_k, ring_rho=ring_rho).run()
        r["seed"] = seed
        rows.append(r)
        if r["cleared"] < r["sources"]:
            fails.append((seed, r["cleared"], r["sources"]))
    return {
        "seeds": seeds, "ring": [ring_k, ring_rho],
        "full_clear": sum(r["cleared"] == r["sources"] for r in rows),
        "ratio_mean": statistics.mean(r["ratio"] for r in rows),
        "time_mean_s": statistics.mean(r["virtual_time_s"] for r in rows),
        "per_source_mean_s": statistics.mean([r["avg_time_per_cleared_s"] for r in rows if r["avg_time_per_cleared_s"]]),
        "mean_scan_stops": statistics.mean(r["n_scan_stops"] for r in rows),
        "mean_measures": statistics.mean(r["stats"]["measure"] for r in rows),
        "residue_left_mean": statistics.mean(r["residue_left"] for r in rows),
        "fail_seeds": fails[:6],
    }


def _cli_plan():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=200)
    ap.add_argument("--k", type=int, default=7)
    ap.add_argument("--rho", type=float, default=997.2)
    args = ap.parse_args()
    print(json.dumps(benchmark_plan(args.seeds, args.k, args.rho), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=200)
    ap.add_argument("--no-scan-at-clear", action="store_true")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--k", type=int, default=7)
    ap.add_argument("--rho", type=float, default=997.2)
    args = ap.parse_args()
    if args.plan:
        args2 = argparse.ArgumentParser().parse_args([])
        out = benchmark_plan(args.seeds, args.k, args.rho)
    else:
        out = benchmark(args.seeds, scan_at_clear_stop=not args.no_scan_at_clear)
    print(json.dumps(out, ensure_ascii=False, indent=2))
