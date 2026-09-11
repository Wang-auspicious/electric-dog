# coding: utf-8
"""Q3 fast black-box strategy.

Only uses measure(x, y, channel) and clear(x, y, channel) responses. The six
1124 m ring stations plus the origin form a <=1000 m covering of the 1800 m
disk (for every point, one station is at most 999.55 m away), so a source whose
minimum receive radius is 1000 m cannot remain unseen after a complete sweep.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Tuple

ARENA_R = 1800.0
RING_R = 1124.0
STEP_M = 55.0
RING_GUESS_M = 500.0
INTERSECTION_ERROR_MAX_M = 100.0
CHANNELS = tuple(range(1, 21))
Point = Tuple[float, float]


def point_at(p: Point, distance: float, deg: float) -> Point:
    t = math.radians(deg)
    return p[0] + distance * math.cos(t), p[1] + distance * math.sin(t)


def angle_diff(a: float, b: float) -> float:
    return (a - b + 180.0) % 360.0 - 180.0


def ray_intersection(p1: Point, a1: float, p2: Point, a2: float) -> Optional[Point]:
    t1, t2 = math.radians(a1), math.radians(a2)
    d1, d2 = (math.cos(t1), math.sin(t1)), (math.cos(t2), math.sin(t2))
    cross = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(cross) < 1e-6:
        return None
    q = (p2[0] - p1[0], p2[1] - p1[1])
    u = (q[0] * d2[1] - q[1] * d2[0]) / cross
    v = (q[0] * d1[1] - q[1] * d1[0]) / cross
    if u < 0.0 or v < 0.0:
        return None
    return p1[0] + u * d1[0], p1[1] + u * d1[1]


class FastRoutesStrategy:
    """Black-box Q3 controller. ``arena`` may be an HTTP client or test double."""

    def __init__(self, arena, channels: Iterable[int] = CHANNELS,
                 ring_radius: float = RING_R, step_m: float = STEP_M):
        self.a = arena
        self.channels = tuple(int(c) for c in channels)
        self.ring_radius = float(ring_radius)
        self.step_m = float(step_m)
        self.position: Point = (0.0, 0.0)
        self.channel: Optional[int] = 1
        self.cleared: set[int] = set()
        self.detected: set[int] = set()
        self.pending: Dict[int, List[Tuple[Point, float]]] = {}
        self.coverage_stations: List[Point] = []
        self.measure_count = 0
        self.clear_count = 0
        self.failed_localizations = 0
        self.virtual_time_s = 0.0
        self.no_signal_stations: Dict[int, List[Point]] = {
            ch: [] for ch in self.channels
        }

    def _measure(self, p: Point, ch: int) -> dict:
        out = self.a.measure(float(p[0]), float(p[1]), int(ch))
        self.virtual_time_s += math.hypot(
            p[0] - self.position[0], p[1] - self.position[1]) / 5.0
        self.virtual_time_s += 5.0 + float(ch != self.channel)
        self.position = (float(p[0]), float(p[1]))
        self.channel = int(ch)
        self.measure_count += 1
        return out

    def _clear(self, p: Point, ch: int) -> dict:
        out = self.a.clear(float(p[0]), float(p[1]), int(ch))
        self.virtual_time_s += math.hypot(
            p[0] - self.position[0], p[1] - self.position[1]) / 5.0
        self.virtual_time_s += 5.0 if out.get("clear_result") == "success" else 3.0
        self.position = (float(p[0]), float(p[1]))
        self.clear_count += 1
        if out.get("clear_result") == "success":
            self.cleared.add(int(ch))
        return out

    def _pursue_from(self, ch: int, start: Point,
                     first_angle: Optional[float] = None,
                     max_steps: int = 100) -> bool:
        """Move on a measured ray, shrinking steps on bearing reversals."""
        p = (float(start[0]), float(start[1]))
        r = self._measure(p, ch)
        kind = r.get("measure_result")
        if kind == "near":
            return bool(self._clear(p, ch).get("clear_result") == "success")
        if kind != "direction":
            return False
        angle = float(r.get("svd_deg", first_angle if first_angle is not None else 0.0))
        step = self.step_m
        for _ in range(max_steps):
            q = point_at(p, step, angle)
            rr = self._measure(q, ch)
            kind = rr.get("measure_result")
            if kind == "near":
                return bool(self._clear(q, ch).get("clear_result") == "success")
            if kind != "direction":
                return False
            p = q
            new_angle = float(rr["svd_deg"])
            # A fixed step can straddle a source and bounce on opposite
            # sides of the <=5 m near zone.  A large bearing reversal is an
            # observable overshoot signal, so shrink the step adaptively.
            if abs(angle_diff(new_angle, angle)) > 60.0:
                step = max(3.0, step / 2.0)
            angle = new_angle
        return False

    def _localize_channel(self, ch: int,
                          observations: List[Tuple[Point, float]]) -> bool:
        """Localize using a 500 m radial candidate from each ring station.

        A direction result bounds the source distance by 1500 m. Moving 500 m
        along that bearing (error <=1 degree) stays within 1001 m of the source,
        and hence inside every allowed receive disk.
        """
        candidates = list(observations)
        candidates.sort(key=lambda item: math.hypot(
            item[0][0] - self.position[0], item[0][1] - self.position[1]))
        for p, angle in candidates:
            candidate = point_at(p, RING_GUESS_M, angle)
            if self._pursue_from(ch, candidate, angle):
                return True
        self.failed_localizations += 1
        return False

    def _scan(self, p: Point, remaining: set[int]) -> None:
        # Once a direction is found, skip this channel at later stations. The
        # saved station observation is sufficient for a black-box pursuit.
        for ch in self.channels:
            if ch in self.cleared or ch in self.pending or ch not in remaining:
                continue
            r = self._measure(p, ch)
            kind = r.get("measure_result")
            if kind == "near":
                self._clear(p, ch)
                remaining.discard(ch)
            elif kind == "direction":
                self.detected.add(ch)
                self.pending.setdefault(ch, []).append(
                    (p, float(r["svd_deg"])))
                remaining.discard(ch)
            elif kind == "no_signal":
                self.no_signal_stations[ch].append(p)

    def run(self) -> dict:
        origin: Point = (0.0, 0.0)
        unresolved = set(self.channels)
        # Origin directions imply source radius <=1500 m. A candidate at 1000 m
        # on that ray is therefore <=500 m from the source and always in range.
        self._scan(origin, unresolved)
        while self.pending:
            ch = min(self.pending, key=lambda c: math.hypot(
                point_at(origin, 1000.0, self.pending[c][0][1])[0] - self.position[0],
                point_at(origin, 1000.0, self.pending[c][0][1])[1] - self.position[1]))
            obs = self.pending.pop(ch)
            angle = obs[0][1]
            candidate = point_at(origin, 1000.0, angle)
            if not self._pursue_from(ch, candidate, angle):
                self.pending[ch] = obs
                break
            unresolved.discard(ch)

        # Six stations at 60 degrees provide the complete 1000 m coverage
        # certificate for the 1800 m disk.  Visit the nearest unvisited ring
        # station each time, shortening the connector from the last source.
        ring = [point_at(origin, self.ring_radius, i * 60.0) for i in range(6)]
        for _ in range(6):
            p = min(ring, key=lambda z: math.hypot(
                z[0] - self.position[0], z[1] - self.position[1]))
            ring.remove(p)
            self.coverage_stations.append(p)
            self._scan(p, unresolved)
            unresolved.difference_update(self.cleared)

        # Deferred directions are resolved after the sweep, choosing the
        # nearest radial candidate (not merely the observation station) to
        # share travel between channels.
        while self.pending:
            ch = min(self.pending, key=lambda c: math.hypot(
                point_at(self.pending[c][0][0], RING_GUESS_M,
                         self.pending[c][0][1])[0] - self.position[0],
                point_at(self.pending[c][0][0], RING_GUESS_M,
                         self.pending[c][0][1])[1] - self.position[1]))
            obs = self.pending.pop(ch)
            self._localize_channel(ch, obs)

        return {
            "cleared": len(self.cleared),
            "cleared_channels": sorted(self.cleared),
            "current_position": self.position,
            "detected_channels": sorted(self.detected),
            "measure_count": self.measure_count,
            "clear_count": self.clear_count,
            "failed_localizations": self.failed_localizations,
            "coverage_complete": (len(self.coverage_stations) == 6
                                   and self.failed_localizations == 0),
            "coverage_stations": list(self.coverage_stations),
            "virtual_time_s": self.virtual_time_s,
        }


class FastSharedStrategy(FastRoutesStrategy):
    """Share the ring survey as triangulation stations for every channel."""

    def __init__(self, arena, max_count_stop: bool = True, **kwargs):
        super().__init__(arena, **kwargs)
        self.max_count_stop = bool(max_count_stop)
        self.estimates: Dict[int, Point] = {}

    def _best_estimate(self, ch: int) -> Optional[Point]:
        observations = self.pending.get(ch, [])
        best = None
        best_error = float("inf")
        for i in range(len(observations)):
            p1, a1 = observations[i]
            for p2, a2 in observations[i + 1:]:
                separation = abs(math.sin(math.radians(angle_diff(a1, a2))))
                if separation < 0.15:
                    continue
                guess = ray_intersection(p1, a1, p2, a2)
                if guess is None or math.hypot(*guess) > ARENA_R + 100.0:
                    continue
                d1 = math.hypot(guess[0] - p1[0], guess[1] - p1[1])
                d2 = math.hypot(guess[0] - p2[0], guess[1] - p2[1])
                if max(d1, d2) > 1600.0:
                    continue
                error = math.radians(1.01) * (d1 + d2) / separation
                if error < best_error:
                    best, best_error = guess, error
        # If the pair is too nearly parallel, let later shared stations add a
        # better angle instead of trusting an extrapolated intersection.
        return best if best_error <= INTERSECTION_ERROR_MAX_M else None

    def _scan_shared(self, p: Point) -> None:
        for ch in self.channels:
            if ch in self.cleared or ch in self.estimates:
                continue
            r = self._measure(p, ch)
            kind = r.get("measure_result")
            if kind == "near":
                self.detected.add(ch)
                self._clear(p, ch)
            elif kind == "direction":
                self.detected.add(ch)
                self.pending.setdefault(ch, []).append((p, float(r["svd_deg"])))
                estimate = self._best_estimate(ch)
                if estimate is not None:
                    self.estimates[ch] = estimate
            elif kind == "no_signal":
                self.no_signal_stations[ch].append(p)

    def _candidate(self, ch: int) -> Point:
        if ch in self.estimates:
            return self.estimates[ch]
        options = [point_at(p, RING_GUESS_M, angle)
                   for p, angle in self.pending[ch]]
        return min(options, key=lambda q: math.hypot(
            q[0] - self.position[0], q[1] - self.position[1]))

    def _route_pending(self, channels: set[int]) -> List[int]:
        """Nearest-neighbor route with a small 2-opt pass over candidates."""
        left = set(channels)
        order: List[int] = []
        here = self.position
        while left:
            ch = min(left, key=lambda c: math.hypot(
                self._candidate(c)[0] - here[0], self._candidate(c)[1] - here[1]))
            order.append(ch)
            here = self._candidate(ch)
            left.remove(ch)
        def length(seq: List[int]) -> float:
            here2 = self.position
            total = 0.0
            for c in seq:
                q = self._candidate(c)
                total += math.hypot(q[0] - here2[0], q[1] - here2[1])
                here2 = q
            return total
        improved = True
        while improved:
            improved = False
            base = length(order)
            for i in range(len(order) - 1):
                for j in range(i + 1, len(order)):
                    trial = order[:i] + list(reversed(order[i:j + 1])) + order[j + 1:]
                    if length(trial) + 1e-6 < base:
                        order, base, improved = trial, length(trial), True
                        break
                if improved:
                    break
        return order

    def run(self) -> dict:
        origin: Point = (0.0, 0.0)
        self._scan_shared(origin)
        ring = [point_at(origin, self.ring_radius, i * 60.0) for i in range(6)]
        for _ in range(6):
            p = min(ring, key=lambda z: math.hypot(
                z[0] - self.position[0], z[1] - self.position[1]))
            ring.remove(p)
            self.coverage_stations.append(p)
            self._scan_shared(p)
            # The specification caps the total at 16. Once sixteen distinct
            # channels have yielded a signal (all pending or cleared), the
            # upper bound itself is a valid completion certificate; no blind
            # "ten sources" early exit is used.
            if self.max_count_stop and len(self.detected) >= 16:
                break

        remaining = set(self.pending).difference(self.cleared)
        for ch in self._route_pending(remaining):
            if ch in self.cleared:
                continue
            p = self._candidate(ch)
            # Intersections frequently fall inside the optical 20 m circle.
            # A direct optical attempt avoids a needless bearing reading there.
            if ch in self.estimates and self._clear(p, ch).get("clear_result") == "success":
                continue
            if self._pursue_from(ch, p):
                continue
            self._localize_channel(ch, self.pending[ch])

        verified = ((len(self.coverage_stations) == 6
                     or (self.max_count_stop and len(self.detected) >= 16))
                    and self.detected.issubset(self.cleared))
        max_cover_distance = math.sqrt(
            ARENA_R ** 2 + self.ring_radius ** 2
            - 2.0 * ARENA_R * self.ring_radius * math.cos(math.radians(30.0)))
        no_signal_covered = sorted(
            ch for ch in self.channels
            if ch not in self.detected and len(self.no_signal_stations[ch])
            == len(self.coverage_stations) + 1)
        return {
            "cleared": len(self.cleared),
            "cleared_channels": sorted(self.cleared),
            "detected_channels": sorted(self.detected),
            "current_position": self.position,
            "measure_count": self.measure_count,
            "clear_count": self.clear_count,
            "failed_localizations": self.failed_localizations,
            "coverage_complete": verified,
            "coverage_stations": list(self.coverage_stations),
            "virtual_time_s": self.virtual_time_s,
            "avg_time_per_cleared_s": (self.virtual_time_s / len(self.cleared)
                                       if self.cleared else None),
            "intersection_channels": len(self.estimates),
            "termination_reason": ("max16_channels" if self.max_count_stop
                                    and len(self.detected) >= 16
                                    and len(self.coverage_stations) < 6
                                    else "six_ring_coverage"),
            "coverage_max_distance_m": max_cover_distance,
            "no_signal_covered_channels": no_signal_covered,
        }


Strategy = FastSharedStrategy
