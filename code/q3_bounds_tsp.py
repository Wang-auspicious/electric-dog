# coding: utf-8
"""问题3理想路线基准和严格下界（仅本地合成案例）。

这里不读取模拟器隐藏数据。对每个随机案例，Held--Karp 求原点出发、末点
自由的开放 Hamilton 路径。``center_route`` 是已知源坐标且在源中心清除时的
精确 oracle 基准；因为实际清除允许距离不超过20 m，它不能冒充物理问题的
严格下界。严格的20 m邻域下界用边权 max(0,|c_i-c_j|-40)（首边
max(0,|c_i|-20)）求开放路径，任何实际选取清除点的路线都不短于该边权和。
MST 只作为更松的交叉检查，不把启发式路线称为下界。
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
Q3_DIR = Path(__file__).resolve().parent
if str(Q3_DIR) not in sys.path:
    sys.path.insert(0, str(Q3_DIR))
from q3_v1_baseline import ARENA_R, SyntheticArena, V  # type: ignore

Point = Tuple[float, float]


def dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def held_karp_open(points: Sequence[Point], edge_floor: float = 0.0,
                   start: Point = (0.0, 0.0)) -> Tuple[float, List[int]]:
    """Exact open TSP from ``start``; path may end at any point.

    ``edge_floor`` is a scalar only for API compatibility; pass transformed edge
    weights through ``edge_fn`` in :func:`held_karp_open_fn`.
    """
    return held_karp_open_fn(points, lambda a, b: max(0.0, dist(a, b) - edge_floor), start)


def held_karp_open_fn(points: Sequence[Point], edge_fn, start: Point = (0.0, 0.0)):
    n = len(points)
    if n == 0:
        return 0.0, []
    size = 1 << n
    inf = float("inf")
    # dp[mask][j] and predecessor; n<=16 in the competition, so this is small.
    dp = [[inf] * n for _ in range(size)]
    pred = [[-1] * n for _ in range(size)]
    for j, p in enumerate(points):
        dp[1 << j][j] = edge_fn(start, p)
    for mask in range(1, size):
        row = dp[mask]
        rem = ((size - 1) ^ mask)
        jbits = mask
        while jbits:
            bitj = jbits & -jbits
            j = bitj.bit_length() - 1
            v = row[j]
            if v < inf:
                nxt = rem
                while nxt:
                    bitk = nxt & -nxt
                    k = bitk.bit_length() - 1
                    cand = v + edge_fn(points[j], points[k])
                    nm = mask | bitk
                    if cand < dp[nm][k]:
                        dp[nm][k] = cand
                        pred[nm][k] = j
                    nxt -= bitk
            jbits -= bitj
    full = size - 1
    end = min(range(n), key=lambda j: dp[full][j])
    best = dp[full][end]
    path = [end]
    mask = full
    while pred[mask][path[-1]] >= 0:
        j = path[-1]
        pj = pred[mask][j]
        mask ^= 1 << j
        path.append(pj)
    path.reverse()
    return best, path


def prim_mst(nodes: Sequence[Point], edge_fn) -> float:
    """MST weight over origin plus all source nodes (a valid, looser LB)."""
    if not nodes:
        return 0.0
    used = [False] * len(nodes)
    key = [float("inf")] * len(nodes)
    key[0] = 0.0
    total = 0.0
    for _ in nodes:
        i = min((j for j, u in enumerate(used) if not u), key=key.__getitem__)
        used[i] = True
        total += key[i]
        for j, u in enumerate(used):
            if not u:
                key[j] = min(key[j], edge_fn(nodes[i], nodes[j]))
    return total


def channel_switch_lower_bound(channels: Iterable[int]) -> int:
    """If clear also required active-channel switching (the local replica does not)."""
    c = list(channels)
    return max(0, len(c) - 1) if 1 in c else len(c)


def case_row(seed: int) -> dict:
    arena = SyntheticArena(seed)
    srcs = list(arena.sources.items())
    points = [(s.x, s.y) for _, s in srcs]
    channels = [ch for ch, _ in srcs]
    center_d, center_order = held_karp_open(points)
    # Any two points chosen in their 20 m clear disks are at least d-40 apart.
    # For the first edge, origin is a point and the first clear disk has radius20.
    # held_karp_open_fn applies the same edge function to origin, so use a custom
    # function that subtracts20 only on the first edge via a dummy start below.
    # Recompute exactly with an explicit first-edge transform.
    n = len(points)
    inf = float("inf")
    dp = [[inf] * n for _ in range(1 << n)]
    for j, p in enumerate(points):
        dp[1 << j][j] = max(0.0, dist((0.0, 0.0), p) - 20.0)
    for mask in range(1, 1 << n):
        for j in range(n):
            v = dp[mask][j]
            if not math.isfinite(v):
                continue
            for k in range(n):
                if mask & (1 << k):
                    continue
                nm = mask | (1 << k)
                cand = v + max(0.0, dist(points[j], points[k]) - 40.0)
                if cand < dp[nm][k]:
                    dp[nm][k] = cand
    nb_d = min(dp[-1])
    # Open-path MST on the same relaxed 20m edge metric.
    all_nodes = [(0.0, 0.0)] + points
    mst20 = prim_mst(
        all_nodes,
        lambda a, b: max(0.0, dist(a, b) - (20.0 if a == (0.0, 0.0) or b == (0.0, 0.0) else 40.0)),
    )
    nsrc = len(points)
    clear_s = 5.0 * nsrc
    sw_s = float(channel_switch_lower_bound(channels))
    return {
        "seed": seed,
        "n": nsrc,
        "center_distance_m": center_d,
        "neighborhood20_lb_distance_m": nb_d,
        "mst20_lb_distance_m": mst20,
        "center_oracle_move_s": center_d / V,
        "center_oracle_total_s": center_d / V + clear_s,
        "strict20_lb_total_s": nb_d / V + clear_s,
        "strict20_lb_with_switch_s": nb_d / V + clear_s + sw_s,
        "center_order_channels": [channels[i] for i in center_order],
        "channel_switch_lb_if_clear_requires_it_s": sw_s,
    }


def benchmark(seeds: int = 100) -> dict:
    rows = [case_row(i) for i in range(seeds)]
    def mean(k): return statistics.mean(r[k] for r in rows)
    def med(k): return statistics.median(r[k] for r in rows)
    return {
        "seeds": seeds,
        "n_range": [min(r["n"] for r in rows), max(r["n"] for r in rows)],
        "mean_n": mean("n"),
        "center_oracle_total_mean_s": mean("center_oracle_total_s"),
        "center_oracle_total_median_s": med("center_oracle_total_s"),
        "center_oracle_per_source_mean_s": statistics.mean(r["center_oracle_total_s"] / r["n"] for r in rows),
        "strict20_lb_total_mean_s": mean("strict20_lb_total_s"),
        "strict20_lb_total_median_s": med("strict20_lb_total_s"),
        "strict20_lb_per_source_mean_s": statistics.mean(r["strict20_lb_total_s"] / r["n"] for r in rows),
        "strict20_lb_with_switch_mean_s": mean("strict20_lb_with_switch_s"),
        "mean_center_move_m": mean("center_distance_m"),
        "mean_neighborhood_lb_move_m": mean("neighborhood20_lb_distance_m"),
        "rows": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    out = benchmark(args.seeds)
    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        print({k: v for k, v in out.items() if k != "rows"})


if __name__ == "__main__":
    main()
