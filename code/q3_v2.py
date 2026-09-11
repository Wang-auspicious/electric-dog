# coding: utf-8
"""问题3 策略 v2：把 v1 的行程浪费全部砍掉。

v1 诊断（30 seeds 平均）：总 8097 s，travel 7352 s = 90.8%
    origin_scan 5238 m / T_S2out 10427 m / T_pursue_fromS2 8988 m / patrol 12109 m
三处浪费：
  W1 每清完一个源都回到扫描站（origin_scan 5238 m + patrol 回程 ~5000 m）
  W2 每个源都先横移 800 m 去第二站（10427 m）
  W3 8 个骨干点固定顺序、来回重复（12109 m，环周只有 7050 m）

v2 的改法：
  1. 示向线按“站点坐标 + 角度”存档；狗离开站点后不再返回；
  2. 取消 800 m 第二站。探测用的覆盖环本身就把相邻两站间距拉到 1175 m，
     同一条示向线在相邻两站各记一次即可求交，基线完全“顺路白拿”；
  3. 覆盖环 1175 m + 6 点（最大盲区 978.4 m < 1000 m，已证明全覆盖）；
     每站扫完立即把“已可求交”的频道就近清掉，再从当前位置走向下一站。
"""
from __future__ import annotations

import argparse, collections, json, math, os, statistics, sys
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from q3_v1_baseline import (  # noqa: E402
    SyntheticArena, point_at, ray_intersection,
)

PATROL_R = 1175.0
COVER_PTS: List[Tuple[float, float]] = [(0.0, 0.0)] + [
    (PATROL_R * math.cos(math.radians(60.0 * k)), PATROL_R * math.sin(math.radians(60.0 * k)))
    for k in range(6)
]


def _dir_gap(a: float, b: float) -> float:
    """两个示向度是否反向（说明上一跳已经越过目标）。"""
    return abs((a - b + 180.0) % 360.0 - 180.0)   # 0..180，接近 180 表示反向


class StrategyV2:
    """先环巡拿覆盖证书，再顺路清除（贪心最近优先）。"""

    def __init__(self, arena: SyntheticArena, snap: float = 250.0, ring_first: bool = True):
        self.a = arena
        self.snap = snap                  # 估计点与当前位置小于该值才值得立刻去清
        self.ring_first = ring_first
        self.bear: Dict[int, List[Tuple[float, float, float]]] = {}   # ch -> [(x,y,theta)]
        self.cleared: set = set()
        self.heard: set = set()
        self.visited: set = set()
        self.failed = 0
        self.stats = collections.Counter()

    # ------------------------------------------------------------------ 估计
    def estimate(self, ch: int) -> Optional[Tuple[float, float]]:
        """两条以上示向线求交；只有一条时用半径中值兜底。"""
        bs = self.bear.get(ch, [])
        best = None
        for i in range(len(bs)):
            for j in range(i + 1, len(bs)):
                g = ray_intersection((bs[i][0], bs[i][1]), bs[i][2],
                                     (bs[j][0], bs[j][1]), bs[j][2])
                if g is None:
                    continue
                if math.hypot(g[0], g[1]) > 2300.0:
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

    # ------------------------------------------------------------------ 动作
    def _clear(self, ch: int) -> bool:
        r = self.a.clear(self.a.x, self.a.y, ch)
        self.stats["clear_" + r["clear_result"]] += 1
        if r["clear_result"] == "success":
            self.cleared.add(ch)
            return True
        return False

    def _measure(self, ch: int) -> Optional[str]:
        r = self.a.measure(self.a.x, self.a.y, ch)
        k = r["measure_result"]
        self.stats["m_" + k] += 1
        if k == "near":
            self.heard.add(ch)
            return "near"
        if k == "direction":
            self.heard.add(ch)
            self.bear.setdefault(ch, []).append((self.a.x, self.a.y, float(r["svd_deg"])))
            return "direction"
        return "no_signal"

    def _goto(self, p: Tuple[float, float]) -> None:
        self.a._move(p[0], p[1])

    # ------------------------------------------------------------------ 追踪
    def chase(self, ch: int) -> bool:
        """直插估计点；失败则近场重测一次求交修正，仍失败就放弃。"""
        est = self.estimate(ch)
        if est is None:
            return False
        self._goto(est)
        if self._clear(ch):
            return True
        for _ in range(3):
            k = self._measure(ch)
            if k == "near":
                if self._clear(ch):
                    return True
                continue
            if k != "direction":
                break
            x, y, th = self.bear[ch][-1]
            g = None
            for (px, py, pt) in self.bear[ch][:-1]:
                cand = ray_intersection((px, py), pt, (x, y), th)
                if cand is not None and math.hypot(cand[0] - x, cand[1] - y) < 800.0:
                    g = cand
            if g is not None and math.hypot(g[0] - x, g[1] - y) > 20.0:
                self._goto(g)
            else:
                self._goto(point_at((x, y), 70.0, th))
            if self._clear(ch):
                return True
        self.failed += 1
        return False

    # ------------------------------------------------------------------ 扫描
    def unresolved(self) -> List[int]:
        return [c for c in range(1, 21) if c not in self.cleared]

    def scan_and_clear(self, p: Tuple[float, float]) -> None:
        """到 p 点，扫未清除频道；随后就近清除已能求交的频道。"""
        self._goto(p)
        for ch in self.unresolved():
            k = self._measure(ch)
            if k == "near":
                self._clear(ch)
        # 就近清除：反复挑最近的可求交点
        while True:
            cur = (self.a.x, self.a.y)
            cand = []
            for ch in self.unresolved():
                bs = self.bear.get(ch, [])
                if len(bs) < 2:
                    continue
                est = self.estimate(ch)
                if est is None:
                    continue
                cand.append((math.hypot(est[0] - cur[0], est[1] - cur[1]), ch))
            if not cand:
                break
            cand.sort()
            if cand[0][0] > 2100.0:
                break
            if not self.chase(cand[0][1]):
                # 该频道示向度数据可能有误导，删掉最旧的一条避免死循环
                if len(self.bear[cand[0][1]]) > 2:
                    self.bear[cand[0][1]] = self.bear[cand[0][1]][-2:]
                else:
                    break

    # ------------------------------------------------------------------ 主循环
    def run(self) -> dict:
        for k in range(len(COVER_PTS)):
            if len(self.cleared) >= 16:          # 题面已知总数上限 16，清够就可收工
                break
            if k > 0 and not self.unresolved():
                break
            self.scan_and_clear(COVER_PTS[k])
            # 覆盖证书：7 个占坑点的最大盲区 978.4 m < 最小接收半径 1000 m，
            # 因此走完 7 个点后仍未出现的频道必为空频道，不需要再补点搜索。
        # 收尾：凡是有过示向度的频道一定存在源，必须清完（没示向度的才是空频道）
        for _ in range(40):
            pend = [c for c in self.unresolved() if self.bear.get(c)]
            if not pend:
                break
            cur = (self.a.x, self.a.y)
            pend.sort(key=lambda c: math.hypot(self.estimate(c)[0] - cur[0], self.estimate(c)[1] - cur[1]))
            if not self.chase(pend[0]):
                if len(self.bear[pend[0]]) > 2:
                    self.bear[pend[0]] = self.bear[pend[0]][-2:]
                else:
                    break
        return {
            "sources": self.a.n,
            "cleared": self.a.cleared_count,
            "ratio": self.a.cleared_count / self.a.n,
            "virtual_time_s": self.a.t,
            "avg_time_per_cleared_s": self.a.t / self.a.cleared_count if self.a.cleared_count else None,
            "failed": self.failed,
        }


class StrategyV3(StrategyV2):
    """两段式：先环巡拿覆盖证书（只扫、不清），再按最近邻一路清过去。

    与 v2 的区别：v2 每到一个覆盖点就顺手清，清完人被带离环线，
    下一个覆盖点要走很远的回头路；v3 把“发现”和“清除”拆开，
    清除阶段按估计点做最近邻排序，整条清除路线是一条约 9 km 的单向链。
    """

    def __init__(self, arena: SyntheticArena, **kw):
        super().__init__(arena, **kw)
        self.giveup: set = set()

    def run(self) -> dict:
        # ---------------- 第一阶段：覆盖环巡，只测向 ----------------
        for k, node in enumerate(COVER_PTS):
            if len(self.cleared) >= 16:
                break
            self._goto(node)
            for ch in range(1, 21):
                if ch in self.cleared or ch in self.giveup:
                    continue
                if len(self.bear.get(ch, [])) >= 2:      # 已有可靠估计，不必重复扫
                    continue
                r = self._measure(ch)
                if r == "near":
                    self._clear(ch)
        # ---------------- 第二阶段：最近邻清除 ----------------
        for _ in range(40):
            pend = [c for c in range(1, 21)
                    if c not in self.cleared and c not in self.giveup and self.bear.get(c)]
            if not pend:
                break
            cur = (self.a.x, self.a.y)

            def key(c):
                e = self.estimate(c)
                return math.hypot(e[0] - cur[0], e[1] - cur[1])

            pend.sort(key=key)
            ch = pend[0]
            if not self.chase(ch):
                self.giveup.add(ch)                      # 防止同一频道反复空跑
        return {
            "sources": self.a.n,
            "cleared": self.a.cleared_count,
            "ratio": self.a.cleared_count / self.a.n,
            "virtual_time_s": self.a.t,
            "avg_time_per_cleared_s": self.a.t / self.a.cleared_count if self.a.cleared_count else None,
            "failed": self.failed,
        }


def benchmark3(seeds: int = 200, **kw) -> dict:
    rows, fails = [], []
    for seed in range(seeds):
        arena = SyntheticArena(seed)
        r = StrategyV3(arena, **kw).run()
        r["seed"] = seed
        rows.append(r)
        if r["cleared"] < r["sources"]:
            fails.append((seed, r["cleared"], r["sources"]))
    return {
        "seeds": seeds,
        "full_clear": sum(r["cleared"] == r["sources"] for r in rows),
        "ratio_mean": statistics.mean(r["ratio"] for r in rows),
        "time_mean_s": statistics.mean(r["virtual_time_s"] for r in rows),
        "time_median_s": statistics.median(r["virtual_time_s"] for r in rows),
        "per_source_mean_s": statistics.mean([r["avg_time_per_cleared_s"] for r in rows if r["avg_time_per_cleared_s"]]),
        "per_source_p50_s": statistics.median([r["avg_time_per_cleared_s"] for r in rows if r["avg_time_per_cleared_s"]]),
        "per_source_max_s": max(r["avg_time_per_cleared_s"] for r in rows if r["avg_time_per_cleared_s"]),
        "fail_seeds": fails[:10],
    }


class StrategyV4(StrategyV2):
    """覆盖环巡 + 最近邻清除 + 稳健近场收敛。

    三个关键修正：
      A. 发现与清除分离：先沿 1175 m 七点环把示向度全部收集完，再按最近邻一路清过去；
      B. 两站示向线求交得到的估计点直插，失败后用 80/40/20/10 m 递减步长收敛；
      C. 只有一条示向线的频道（占 15.6%）沿该射线做递减步长扫掠 + clear 探测。
         这类源一定在唯一能听到它的那点的 978.4 m 以内，1° 横向偏差 ≤ 17 m < 20 m 清除半径。
    """

    def __init__(self, arena: SyntheticArena, **kw):
        super().__init__(arena, **kw)
        self.giveup: set = set()

    # ---- 求交：只接受离当前位置不太远的交点 ----
    def _near_intersection(self, ch, limit=600.0):
        bs = self.bear.get(ch, [])
        cur = (self.a.x, self.a.y)
        best = None
        for i in range(len(bs)):
            for j in range(i + 1, len(bs)):
                dth = abs((bs[i][2] - bs[j][2] + 180.0) % 360.0 - 180.0)
                if min(dth, 180.0 - dth) < 12.0:      # 近平行，交点不可信
                    continue
                g = ray_intersection((bs[i][0], bs[i][1]), bs[i][2],
                                     (bs[j][0], bs[j][1]), bs[j][2])
                if g is None:
                    continue
                d = math.hypot(g[0] - cur[0], g[1] - cur[1])
                if d < limit and (best is None or d < math.hypot(best[0] - cur[0], best[1] - cur[1])):
                    best = g
        return best

    def _could_be_heard(self, ch, node) -> bool:
        """已知示向线时，只有射线距本站不超过 1000 m 才可能在此听到（必要条件剪枝）。"""
        bs = self.bear.get(ch, [])
        if not bs:
            return True
        for (px, py, th) in bs:
            t = math.radians(th)
            dx, dy = math.cos(t), math.sin(t)
            vx, vy = node[0] - px, node[1] - py
            proj = vx * dx + vy * dy            # 投影到射线方向
            rho = min(1500.0, max(5.0, proj))   # 源可能在的最近点
            if math.hypot(px + rho * dx - node[0], py + rho * dy - node[1]) <= 1500.0:
                return True
        return False

    def chase(self, ch: int) -> bool:
        """多站示向度求交直插；失败后用 300→180→…→递减步长沿最新示向度收敛。

        每一轮都重新测向，所以航向始终指向真源，不会因 1° 误差累积而跑偏；
        step 减半保证从几百米外也能一步步收进 20 m 清除球。
        """
        est = self.estimate(ch)
        if len(self.bear.get(ch, [])) < 2:
            return self._ray_sweep(ch)
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
            if prev is not None and _dir_gap(th, prev) > 90.0:
                step *= 0.5
            prev = th
            if step < 8.0:
                return False
            self._goto(point_at((self.a.x, self.a.y), step, th))
            if self._clear(ch):
                return True
        return False

    def _ray_sweep(self, ch: int) -> bool:
        """只有一条示向线时的兜底：沿该射线跳进，每跳重测一次把航向拉回真视线。

        理论依据：只在唯一一个覆盖点被听到的频道，说明该点就是最近覆盖点，
        源到它不超过 978.4 m；此处 1° 横向偏差 <= 17.1 m < 20 m 清除半径，
        沿射线扫掠一定存在一个落点进入 20 m 球。
        步长只在“示向度反向”（说明已越过目标）时减半，否则保持大步前进。
        """
        bs = self.bear.get(ch)
        if not bs:
            return False
        # 先就在当前位置试一次；测不到再回到“唯一听过它的那个站点”重新起步。
        # 该站必能听到（源不敢超出自己的接收半径），所以这是可靠兜底。
        if self._measure(ch) != "direction":
            self._goto((bs[0][0], bs[0][1]))
            if self._measure(ch) != "direction":
                return False
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
            if prev is not None and _dir_gap(th, prev) > 90.0:
                hop *= 0.5
            prev = th
            if hop < 8.0:
                return False
            self._goto(point_at((self.a.x, self.a.y), hop, th))
            if self._clear(ch):
                return True
        return False

    def run(self) -> dict:
        for k, node in enumerate(COVER_PTS):
            if len(self.cleared) >= 16:
                break
            self._goto(node)
            for ch in range(1, 21):
                if ch in self.cleared or ch in self.giveup:
                    continue
                if len(self.bear.get(ch, [])) >= 2:
                    continue
                if not self._could_be_heard(ch, (node[0], node[1])):
                    continue
                if self._measure(ch) == "near":
                    self._clear(ch)
        for _ in range(40):
            pend = [c for c in range(1, 21)
                    if c not in self.cleared and c not in self.giveup and self.bear.get(c)]
            if not pend:
                break
            cur = (self.a.x, self.a.y)

            def key(c):
                e = self.estimate(c)
                return math.hypot(e[0] - cur[0], e[1] - cur[1])

            pend.sort(key=key)
            ch = pend[0]
            if not self.chase(ch):
                self.giveup.add(ch)
        return {
            "sources": self.a.n,
            "cleared": self.a.cleared_count,
            "ratio": self.a.cleared_count / self.a.n,
            "virtual_time_s": self.a.t,
            "avg_time_per_cleared_s": self.a.t / self.a.cleared_count if self.a.cleared_count else None,
            "failed": self.failed,
        }


def benchmark4(seeds: int = 200, **kw) -> dict:
    rows, fails = [], []
    for seed in range(seeds):
        arena = SyntheticArena(seed)
        r = StrategyV4(arena, **kw).run()
        r["seed"] = seed
        rows.append(r)
        if r["cleared"] < r["sources"]:
            fails.append((seed, r["cleared"], r["sources"]))
    return {
        "seeds": seeds,
        "full_clear": sum(r["cleared"] == r["sources"] for r in rows),
        "ratio_mean": statistics.mean(r["ratio"] for r in rows),
        "time_mean_s": statistics.mean(r["virtual_time_s"] for r in rows),
        "time_median_s": statistics.median(r["virtual_time_s"] for r in rows),
        "per_source_mean_s": statistics.mean([r["avg_time_per_cleared_s"] for r in rows if r["avg_time_per_cleared_s"]]),
        "per_source_p50_s": statistics.median([r["avg_time_per_cleared_s"] for r in rows if r["avg_time_per_cleared_s"]]),
        "per_source_max_s": max(r["avg_time_per_cleared_s"] for r in rows if r["avg_time_per_cleared_s"]),
        "fail_seeds": fails[:8],
    }


def benchmark(seeds: int = 200, **kw) -> dict:
    rows, fails = [], []
    for seed in range(seeds):
        arena = SyntheticArena(seed)
        r = StrategyV2(arena, **kw).run()
        r["seed"] = seed
        rows.append(r)
        if r["cleared"] < r["sources"]:
            fails.append((seed, r["cleared"], r["sources"]))
    return {
        "seeds": seeds,
        "full_clear": sum(r["cleared"] == r["sources"] for r in rows),
        "ratio_mean": statistics.mean(r["ratio"] for r in rows),
        "time_mean_s": statistics.mean(r["virtual_time_s"] for r in rows),
        "time_median_s": statistics.median(r["virtual_time_s"] for r in rows),
        "per_source_mean_s": statistics.mean([r["avg_time_per_cleared_s"] for r in rows if r["avg_time_per_cleared_s"]]),
        "per_source_p50_s": statistics.median([r["avg_time_per_cleared_s"] for r in rows if r["avg_time_per_cleared_s"]]),
        "per_source_max_s": max(r["avg_time_per_cleared_s"] for r in rows if r["avg_time_per_cleared_s"]),
        "fail_seeds": fails[:10],
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=200)
    ap.add_argument("--v3", action="store_true")
    ap.add_argument("--v4", action="store_true")
    args = ap.parse_args()
    fn = benchmark4 if args.v4 else (benchmark3 if args.v3 else benchmark)
    print(json.dumps(fn(args.seeds), ensure_ascii=False, indent=2))
