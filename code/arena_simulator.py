# coding: utf-8
"""
2026 CUMCM Problem B Offline High-Fidelity Physics Simulator & Algorithm Benchmark
Strictly implements rules from Problem B text, Appendix 1, 2, 3, 4, and Attachment 1, 2.
"""
import math
import random
import numpy as np

ARENA_RADIUS = 1800.0
SPEED = 5.0 # m/s
MEASURE_TIME = 5.0 # s
SWITCH_TIME = 1.0 # s
CLEAR_SUCCESS_TIME = 5.0 # s (3s optical + 2s laser)
CLEAR_FAIL_TIME = 3.0 # s (3s optical)
CLEAR_RADIUS = 20.0 # m
NEAR_RADIUS = 5.0 # m

class OfflineArena:
    def __init__(self, num_sources=None, is_q4=False, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        if num_sources is None:
            self.num_sources = random.randint(10, 16)
        else:
            self.num_sources = num_sources
            
        self.is_q4 = is_q4
        # Select unique channels
        self.channels = sorted(random.sample(range(1, 21), self.num_sources))
        self.sources = {} # ch -> dict(x, y, r_recv, is_dir, dir_deg, cleared)
        
        for ch in self.channels:
            # Random position within 1800m circle
            r = ARENA_RADIUS * math.sqrt(random.random())
            phi = random.uniform(0, 2 * math.pi)
            x = r * math.cos(phi)
            y = r * math.sin(phi)
            r_recv = random.uniform(1000.0, 1500.0)
            
            is_dir = False
            dir_deg = None
            if is_q4:
                # ~20% directional
                if random.random() < 0.25: # approx 20-30%
                    is_dir = True
                    dir_deg = random.uniform(0, 360.0)
            
            self.sources[ch] = {
                'x': x, 'y': y, 'r_recv': r_recv,
                'is_dir': is_dir, 'dir_deg': dir_deg,
                'cleared': False
            }
            
        self.curr_x = 0.0
        self.curr_y = 0.0
        self.curr_channel = 1
        self.virtual_time = 0.0
        self.cleared_count = 0
        
    def measure(self, x, y, channel):
        # Calculate time
        dist = math.hypot(x - self.curr_x, y - self.curr_y)
        t_move = dist / SPEED
        t_switch = SWITCH_TIME if channel != self.curr_channel else 0.0
        t_measure = MEASURE_TIME
        
        self.virtual_time += (t_move + t_switch + t_measure)
        self.curr_x = x
        self.curr_y = y
        self.curr_channel = channel
        
        if channel not in self.sources:
            return {'result': 'no_signal'}
        
        s = self.sources[channel]
        if s['cleared']:
            return {'result': 'no_signal'}
            
        d_to_s = math.hypot(s['x'] - x, s['y'] - y)
        if d_to_s > s['r_recv']:
            return {'result': 'no_signal'}
            
        # Directional check
        if s['is_dir']:
            # Direction from source to robot
            angle_from_source = math.degrees(math.atan2(y - s['y'], x - s['x'])) % 360.0
            # Difference from dir_deg
            diff = (angle_from_source - s['dir_deg'] + 180.0) % 360.0 - 180.0
            if abs(diff) > 90.0:
                return {'result': 'no_signal'}
                
        if d_to_s <= NEAR_RADIUS:
            return {'result': 'near'}
            
        # True bearing from robot to source
        true_bearing = math.degrees(math.atan2(s['y'] - y, s['x'] - x)) % 360.0
        # Add error [-1, 1]
        err = random.uniform(-1.0, 1.0)
        svd_deg = (true_bearing + err) % 360.0
        return {'result': 'direction', 'svd_deg': round(svd_deg, 2)}
        
    def clear(self, x, y, channel):
        dist = math.hypot(x - self.curr_x, y - self.curr_y)
        t_move = dist / SPEED
        
        self.curr_x = x
        self.curr_y = y
        # /clear does not change测向机channel
        
        if channel not in self.sources:
            self.virtual_time += (t_move + CLEAR_FAIL_TIME)
            return {'result': 'no_target_in_range'}
            
        s = self.sources[channel]
        if s['cleared']:
            self.virtual_time += (t_move + CLEAR_FAIL_TIME)
            return {'result': 'no_target_in_range'}
            
        d_to_s = math.hypot(s['x'] - x, s['y'] - y)
        if d_to_s <= CLEAR_RADIUS:
            s['cleared'] = True
            self.cleared_count += 1
            self.virtual_time += (t_move + CLEAR_SUCCESS_TIME)
            return {'result': 'success'}
        else:
            self.virtual_time += (t_move + CLEAR_FAIL_TIME)
            return {'result': 'no_target_in_range'}

def line_intersection(p1, theta1_deg, p2, theta2_deg):
    """Compute 2D intersection of two bearing rays"""
    r1 = math.radians(theta1_deg)
    r2 = math.radians(theta2_deg)
    d1 = (math.cos(r1), math.sin(r1))
    d2 = (math.cos(r2), math.sin(r2))
    
    # p1 + t1 * d1 = p2 + t2 * d2
    # t1 * d1x - t2 * d2x = p2x - p1x
    # t1 * d1y - t2 * d2y = p2y - p1y
    det = d1[0] * (-d2[1]) - d1[1] * (-d2[0])
    if abs(det) < 1e-4:
        return None
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    t1 = (dx * (-d2[1]) - dy * (-d2[0])) / det
    t2 = (d1[0] * dy - d1[1] * dx) / det
    if t1 < 0 or t2 < 0:
        return None # Rays point away
    ix = p1[0] + t1 * d1[0]
    iy = p1[1] + t1 * d1[1]
    return (ix, iy)

def run_strategy_q3(arena):
    """
    Tier-1: Global scan at origin (0, 0) across all 20 channels.
    Tier-2: For detected channels, navigate to an orthogonal baseline point to get 2nd bearing,
            intersect, sprint to centroid, and clear!
    Tier-3: Peripheral patrol along outer ring (R=1000m) to catch hidden peripheral sources.
    """
    detected_at_origin = {}
    for ch in range(1, 21):
        res = arena.measure(0, 0, ch)
        if res['result'] == 'direction':
            detected_at_origin[ch] = res['svd_deg']
            
    # Process detected channels
    # Sort detected channels by distance/angle to minimize travel
    active_channels = list(detected_at_origin.keys())
    
    for ch in active_channels:
        if arena.sources[ch]['cleared']:
            continue
        theta1 = detected_at_origin[ch]
        # Choose 2nd point S2 orthogonal to S1->G line: offset by 350m perpendicular
        # S1 is (0,0). Line is at angle theta1.
        rad1 = math.radians(theta1)
        # S2 chosen along theta1 + 90 deg at distance ~400m
        s2_dist = 400.0
        s2_x = s2_dist * math.cos(rad1 + math.pi/2)
        s2_y = s2_dist * math.sin(rad1 + math.pi/2)
        # Ensure within arena
        if math.hypot(s2_x, s2_y) > 1700.0:
            s2_x = s2_dist * math.cos(rad1 - math.pi/2)
            s2_y = s2_dist * math.sin(rad1 - math.pi/2)
            
        res2 = arena.measure(s2_x, s2_y, ch)
        if res2['result'] == 'direction':
            theta2 = res2['svd_deg']
            inter = line_intersection((0, 0), theta1, (s2_x, s2_y), theta2)
            if inter:
                tx, ty = inter
                # Check bounds
                if math.hypot(tx, ty) <= 1850:
                    arena.clear(tx, ty, ch)
                    if not arena.sources[ch]['cleared']:
                        # Try near neighbor
                        for offset in [(15, 0), (-15, 0), (0, 15), (0, -15)]:
                            arena.clear(tx + offset[0], ty + offset[1], ch)
                            if arena.sources[ch]['cleared']:
                                break
        elif res2['result'] == 'near':
            arena.clear(s2_x, s2_y, ch)
            
    # Tier-3: Peripheral patrol for remaining channels
    remaining_channels = [ch for ch, s in arena.sources.items() if not s['cleared']]
    if remaining_channels:
        # 8 waypoints at R=1150m provides 100% geometric coverage for any point in 1800m circle with R_recv >= 1000m
        patrol_angles = [i * 45.0 for i in range(8)]
        patrol_r = 1150.0
        for p_deg in patrol_angles:
            rem = [ch for ch in range(1, 21) if ch not in arena.sources or not arena.sources[ch]['cleared']]
            if not rem:
                break
            px = patrol_r * math.cos(math.radians(p_deg))
            py = patrol_r * math.sin(math.radians(p_deg))
            
            for ch in rem:
                res = arena.measure(px, py, ch)
                if res['result'] == 'direction':
                    th1 = res['svd_deg']
                    # Second point for orthogonal intersection
                    p2x = px + 400 * math.cos(math.radians(th1 + 90))
                    p2y = py + 400 * math.sin(math.radians(th1 + 90))
                    if math.hypot(p2x, p2y) > 1750:
                        p2x = px + 400 * math.cos(math.radians(th1 - 90))
                        p2y = py + 400 * math.sin(math.radians(th1 - 90))
                    res_p2 = arena.measure(p2x, p2y, ch)
                    if res_p2['result'] == 'direction':
                        th2 = res_p2['svd_deg']
                        inter = line_intersection((px, py), th1, (p2x, p2y), th2)
                        if inter:
                            arena.clear(inter[0], inter[1], ch)
                            if ch in arena.sources and not arena.sources[ch]['cleared']:
                                for offset in [(18, 0), (-18, 0), (0, 18), (0, -18), (12, 12), (-12, -12)]:
                                    arena.clear(inter[0] + offset[0], inter[1] + offset[1], ch)
                                    if arena.sources[ch]['cleared']:
                                        break
                    elif res_p2['result'] == 'near':
                        arena.clear(p2x, p2y, ch)

def benchmark():
    print("=== Q3 50-Run Benchmark (Omnidirectional) ===")
    q3_times, q3_cleared_ratio, q3_avg_clear_time = [], [], []
    for seed in range(50):
        arena = OfflineArena(seed=seed, is_q4=False)
        run_strategy_q3(arena)
        cleared = arena.cleared_count
        ratio = cleared / arena.num_sources
        avg_t = arena.virtual_time / cleared if cleared > 0 else 0
        q3_times.append(arena.virtual_time)
        q3_cleared_ratio.append(ratio)
        q3_avg_clear_time.append(avg_t)
        
    print(f"Clearance Ratio: {np.mean(q3_cleared_ratio)*100:.2f}% (Min: {np.min(q3_cleared_ratio)*100:.2f}%)")
    print(f"Mean Total Virtual Time: {np.mean(q3_times):.1f} s")
    print(f"Mean Per-Source Clear Time: {np.mean(q3_avg_clear_time):.1f} s")

    print("\n=== Q4 50-Run Benchmark (Mixed Directional ~25%) ===")
    q4_times, q4_cleared_ratio, q4_avg_clear_time = [], [], []
    for seed in range(50):
        arena = OfflineArena(seed=seed + 1000, is_q4=True)
        run_strategy_q3(arena)
        cleared = arena.cleared_count
        ratio = cleared / arena.num_sources
        avg_t = arena.virtual_time / cleared if cleared > 0 else 0
        q4_times.append(arena.virtual_time)
        q4_cleared_ratio.append(ratio)
        q4_avg_clear_time.append(avg_t)
        
    print(f"Clearance Ratio: {np.mean(q4_cleared_ratio)*100:.2f}% (Min: {np.min(q4_cleared_ratio)*100:.2f}%)")
    print(f"Mean Total Virtual Time: {np.mean(q4_times):.1f} s")
    print(f"Mean Per-Source Clear Time: {np.mean(q4_avg_clear_time):.1f} s")

benchmark()
