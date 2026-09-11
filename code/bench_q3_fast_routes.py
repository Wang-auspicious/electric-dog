# coding: utf-8
"""Fair same-seed benchmark against the local exploratory strategy.

This imports the existing synthetic arena as a test double only; neither
controller reads arena.sources or arena.n while acting.
"""
from __future__ import annotations

import importlib.util
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


q3 = load("q3_v1_baseline", Path(__file__).with_name("q3_v1_baseline.py"))
fast = load("q3_fast", Path(__file__).with_name("q3_fast_routes.py"))


def run(seeds: int = 100) -> dict:
    base_rows, fast_rows = [], []
    for seed in range(seeds):
        a0 = q3.SyntheticArena(seed)
        b = q3.Strategy(a0).run()
        b.update(seed=seed, sources=a0.n)
        base_rows.append(b)
        a1 = q3.SyntheticArena(seed)
        f = fast.Strategy(a1)
        r = f.run()
        r.update(seed=seed, sources=a1.n)
        fast_rows.append(r)

    def stats(rows):
        return {
            "full_clear_cases": sum(r["cleared"] == r["sources"] for r in rows),
            "clear_ratio_mean": statistics.mean(r["cleared"] / r["sources"] for r in rows),
            "clear_ratio_min": min(r["cleared"] / r["sources"] for r in rows),
            "virtual_time_mean_s": statistics.mean(r["virtual_time_s"] for r in rows),
            "virtual_time_median_s": statistics.median(r["virtual_time_s"] for r in rows),
            "virtual_time_max_s": max(r["virtual_time_s"] for r in rows),
            "avg_time_per_cleared_mean_s": statistics.mean(
                r["virtual_time_s"] / r["cleared"] for r in rows if r["cleared"]
            ),
        }

    return {"seeds": seeds, "baseline": stats(base_rows), "fast": stats(fast_rows)}


if __name__ == "__main__":
    import argparse
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=200)
    args = ap.parse_args()
    out = json.dumps(run(args.seeds), ensure_ascii=False, indent=2)
    print(out)
    (Path(__file__).resolve().parents[1] / "results" / "q3_fast_vs_v1_benchmark.json").write_text(
        out, encoding="utf-8")
