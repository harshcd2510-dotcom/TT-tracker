"""Event engine: turns raw ball detections into match events.

Consumes raw detector candidates (which are near-pixel-accurate) rather than
the smoothed Kalman state - a constant-velocity filter lags ~8 frames behind
the violent direction reversals that define hits and bounces.

Emitted events (returned by `step`):
    ("serve", side)                 first hit of a rally
    ("hit",   side, speed_mps)      x-velocity reversal on `side` (0=left,1=right)
    ("bounce", side, zone, xy)      down->up flip near the table surface
    ("rally_end", summary_dict)
"""

from collections import deque

import numpy as np

from . import config

MAX_JUMP_PX = 90          # max plausible ball travel between consecutive frames
FIT_WINDOW = 4            # detections used for the velocity fit


def zone_of(table_x: float) -> str:
    d = abs(table_x - config.NET_X)
    if d < 0.45:
        return "short"
    if d < 0.95:
        return "mid"
    return "long"


class EventEngine:
    def __init__(self, calib=None):
        self.calib = calib
        # current rally
        self.rally_active = False
        self.rally_hits = 0
        self.rally_bounces = 0
        self.rally_start_t = None
        self.misses = 0
        self.still_frames = 0
        # lifetime stats
        self.shots = [0, 0]             # per side
        self.bounces = []               # (table_x, table_y, side)
        self.rallies = []               # summaries
        self.top_speed = 0.0            # m/s
        self.last_speed = 0.0
        self.last_hit_speed = 0.0
        self.serve_count = 0
        # internal
        self._dets = deque(maxlen=10)   # accepted raw detections (x, y, t)
        self._speed_hist = deque(maxlen=5)
        self._vx_sign = 0
        self._flip_count = 0
        self._pending_sign = 0
        self._vy_sign = 0

    # ------------------------------------------------------------------ #
    def _velocity(self):
        """Least-squares px/s velocity over the last FIT_WINDOW detections."""
        d = list(self._dets)[-FIT_WINDOW:]
        if len(d) < 2:
            return None
        ts = np.array([p[2] for p in d]) - d[0][2]
        if ts[-1] - ts[0] < 1e-3:
            return None
        vx = np.polyfit(ts, [p[0] for p in d], 1)[0]
        vy = np.polyfit(ts, [p[1] for p in d], 1)[0]
        return float(vx), float(vy)

    def _end_rally(self, t):
        if not self.rally_active:
            return None
        summary = {
            "hits": self.rally_hits,
            "bounces": self.rally_bounces,
            "duration": (t - self.rally_start_t) if self.rally_start_t else 0.0,
        }
        self.rallies.append(summary)
        self.rally_active = False
        self.rally_hits = 0
        self.rally_bounces = 0
        self.rally_start_t = None
        return ("rally_end", summary)

    # ------------------------------------------------------------------ #
    def step(self, det_pos, t, radius=8.0):
        """det_pos: raw detection (x, y) in image px, or None this frame."""
        events = []

        if det_pos is None:
            self.misses += 1
            if self.misses > 8:
                self._dets.clear()
            if self.rally_active and self.misses > config.RALLY_END_FRAMES:
                ev = self._end_rally(t)
                if ev:
                    events.append(ev)
            return events

        x, y = det_pos
        # reject isolated far jumps (not the ball); a new valid position will
        # rebuild the chain on subsequent frames
        if self._dets:
            lx, ly, lt = self._dets[-1]
            gap_f = max(1.0, (t - lt) * 60.0)
            if np.hypot(x - lx, y - ly) > MAX_JUMP_PX * gap_f:
                self._dets.clear()
        self._dets.append((x, y, t))
        self.misses = 0

        vel = self._velocity()
        if vel is None:
            return events
        vx, vy = vel
        # scale from the ball's apparent size: the homography is only valid on
        # the table plane and wildly overestimates for a ball in the air
        mpp = config.BALL_DIAMETER_M / (2.0 * max(radius, 3.0))
        speed_mps = float(np.hypot(vx, vy) * mpp)
        self.last_speed = speed_mps
        self._speed_hist.append(speed_mps)
        # median-of-3 kills single-frame spikes from fit transients
        med = float(np.median(list(self._speed_hist)[-3:]))
        if self.top_speed < med <= 60.0:  # physical sanity cap
            self.top_speed = med

        table_xy = self.calib.image_to_table(x, y) if self.calib else None
        tx = float(table_xy[0]) if table_xy is not None else None

        # --- still ball (picked up / dead) ------------------------------
        if speed_mps < config.RALLY_END_STILL_MPS:
            self.still_frames += 1
            if self.rally_active and self.still_frames > config.RALLY_END_STILL_FRAMES:
                ev = self._end_rally(t)
                if ev:
                    events.append(ev)
                return events
        else:
            self.still_frames = 0

        # --- hit: x-velocity sign reversal ------------------------------
        vx_mps = vx * mpp
        sign = 1 if vx_mps > 0.05 else (-1 if vx_mps < -0.05 else 0)
        if sign != 0:
            if self._vx_sign != 0 and sign != self._vx_sign:
                self._flip_count += 1
                self._pending_sign = sign
            else:
                self._flip_count = 0
                self._vx_sign = sign
            if (self._flip_count >= config.HIT_CONFIRM_FRAMES
                    and abs(vx_mps) > config.MIN_HIT_SPEED_MPS
                    and tx is not None):
                side = None
                if tx < config.NET_X - config.NET_MARGIN_M and self._pending_sign > 0:
                    side = 0
                elif tx > config.NET_X + config.NET_MARGIN_M and self._pending_sign < 0:
                    side = 1
                if side is not None:
                    self._flip_count = 0
                    self._vx_sign = self._pending_sign
                    if not self.rally_active:
                        self.rally_active = True
                        self.rally_start_t = t
                        self.serve_count += 1
                        events.append(("serve", side))
                    self.rally_hits += 1
                    self.shots[side] += 1
                    self.last_hit_speed = speed_mps
                    events.append(("hit", side, speed_mps))

        # --- bounce: y-velocity flip (down -> up) near the surface ------
        vy_sign = 1 if vy > 60 else (-1 if vy < -60 else 0)
        if (self._vy_sign > 0 and vy_sign < 0 and table_xy is not None
                and self.calib is not None
                and self.calib.on_table(table_xy, config.BOUNCE_TABLE_MARGIN_M)
                and self.calib.near_surface(x, y, radius)):
            side = 0 if table_xy[0] < config.NET_X else 1
            zone = zone_of(table_xy[0])
            self.bounces.append((float(table_xy[0]), float(table_xy[1]), side))
            self.rally_bounces += 1
            events.append(("bounce", side, zone,
                           (float(table_xy[0]), float(table_xy[1]))))
        if vy_sign != 0:
            self._vy_sign = vy_sign

        return events

    def end(self, t):
        """Flush an in-progress rally at end of stream."""
        return self._end_rally(t)

    # ------------------------------------------------------------------ #
    def stats(self) -> dict:
        n = len(self.rallies)
        hits = [r["hits"] for r in self.rallies]
        return {
            "shots_left": self.shots[0],
            "shots_right": self.shots[1],
            "bounces": len(self.bounces),
            "rallies": n,
            "longest_rally": max(hits) if hits else 0,
            "avg_rally": (sum(hits) / n) if n else 0.0,
            "top_speed_mps": self.top_speed,
            "current_rally_hits": self.rally_hits if self.rally_active else 0,
        }
