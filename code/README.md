# code/ —— 可复现程序

| 文件 | 作用 | 运行 |
|---|---|---|
| `arena_simulator.py` | 早期离线高保真模拟器（含 Q3/Q4、定向源开关） | `python arena_simulator.py` |
| `q3_v1_baseline.py` | **v1 基线策略** + 合成 arena + HTTP 客户端骨架。含 1000 局统计。 | `python q3_v1_baseline.py --seeds 1000` |
| `q3_fast_routes.py` | **当前最优策略 `FastSharedStrategy`**：共享覆盖环 1124 m×6、多站示向线求交、最近邻+2-opt、示向反向减半收敛。 | 通过 `bench_q3_fast_routes.py` 调用 |
| `bench_q3_fast_routes.py` | 同种子公平对照 v1 与 FastShared，输出写入 `results/`。 | `python bench_q3_fast_routes.py --seeds 200` |
| `q3_v2.py` | 另一条实现路线（`StrategyV4`）：七点覆盖环巡 + 最近邻清除 + 单示向线兜底扫掠。 | `python q3_v2.py --seeds 300 --v4` |
| `q3_bounds_tsp.py` | 问题三时间下界：全知 oracle 与严格 20 m 邻域下界（Held-Karp / 排列枚举）。 | `python q3_bounds_tsp.py` |
| `q3_localizer_adaptive.py` | 定位器方案对比：固定 800 m 二站 vs 自适应基线。 | `python q3_localizer_adaptive.py` |

## 约定

1. 所有控制策略**只使用 `measure` / `clear` 的返回值**，不读取 `arena.sources`、`arena.n`、接收半径等隐藏量。
2. `measure(x, y, ch)` / `clear(x, y, ch)` 的签名与官方 HTTP 接口字段一一对应，移植时替换成 `requests.post` 即可。
3. 移动时间由模拟器按「上一次合法动作的位置」推算 —— 因此**不想去某点就绝不要把那个坐标写进请求**。v1 最大的浪费正是「清完一个源后又把扫描站的坐标写进下一条 `/measure`」。
4. 目标函数的严格性分级：
   - 覆盖不漏检：**严格证明**（扫描点 1000 m 圆盘覆盖 1800 m 圆盘）
   - 时间：**上界 = 实测策略**；**下界 = `q3_bounds_tsp.py` 的 Held-Karp / 排列枚举**。最近邻、固定巡逻点等只能算上界，不得称下界。
