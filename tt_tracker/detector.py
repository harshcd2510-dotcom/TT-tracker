"""Ball detection: HSV color candidates gated by frame-difference motion.

No ML models - fast enough for real-time on CPU. A detection is strong when a
blob is both the ball's color AND moving; weak color-only or motion-only
candidates near the tracker prediction are accepted as fallbacks.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from . import config


@dataclass
class Candidate:
    x: float
    y: float
    radius: float
    area: float
    score: float          # higher is better
    has_color: bool
    has_motion: bool


class BallDetector:
    def __init__(self, ball_color: str = "orange"):
        self.set_color(ball_color)
        self._prev: list[np.ndarray] = []
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self._dilate = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (config.MOTION_DILATE * 2 + 1,) * 2
        )
        self.fg_motion = None  # latest raw motion mask (for player-movement stats)
        self.expected_radius = None  # learned ball size (px), set via note_accepted

    def note_accepted(self, radius: float):
        """Tracker accepted a detection - fold its size into the ball model."""
        if self.expected_radius is None:
            self.expected_radius = radius
        else:
            self.expected_radius = 0.9 * self.expected_radius + 0.1 * radius

    def _max_area(self) -> float:
        if self.expected_radius is None:
            return config.MAX_BALL_AREA
        # ball can't be >3x its learned size - kills paddles, shirts, posters
        return min(config.MAX_BALL_AREA,
                   np.pi * (self.expected_radius * 3.0) ** 2)

    def set_color(self, name: str):
        lo, hi = config.BALL_COLORS[name]
        self._lo = np.array(lo, dtype=np.uint8)
        self._hi = np.array(hi, dtype=np.uint8)

    # ------------------------------------------------------------------ #
    def detect(self, frame, predicted=None, gate_px=None) -> Candidate | None:
        """Return the best ball candidate, or None.

        predicted: (x, y) tracker prediction used for gating/scoring.
        gate_px:   max distance to prediction to accept a candidate.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        color_mask = cv2.inRange(hsv, self._lo, self._hi)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, self._kernel)

        motion_mask = np.zeros_like(color_mask)
        if len(self._prev) >= 2:
            d1 = cv2.absdiff(gray, self._prev[-1])
            d2 = cv2.absdiff(gray, self._prev[-2])
            diff = cv2.max(d1, d2)
            motion_mask = cv2.threshold(
                diff, config.MOTION_DIFF_THRESH, 255, cv2.THRESH_BINARY
            )[1]
            motion_mask = cv2.dilate(motion_mask, self._dilate)
        self._prev.append(gray)
        if len(self._prev) > 2:
            self._prev.pop(0)
        self.fg_motion = motion_mask

        both = cv2.bitwise_and(color_mask, motion_mask)

        cands = []
        cands += self._blobs(both, color_mask, motion_mask, bonus=3.0)
        cands += self._blobs(color_mask, color_mask, motion_mask, bonus=0.0,
                             exclude=both)
        if not cands:
            return None

        # same-color static objects away from the prediction are not the ball
        if predicted is not None:
            far = (gate_px or config.GATE_BASE_PX)
            cands = [c for c in cands
                     if c.has_motion
                     or np.hypot(c.x - predicted[0], c.y - predicted[1]) < far]
            if not cands:
                return None

        for c in cands:
            if predicted is not None:
                dist = np.hypot(c.x - predicted[0], c.y - predicted[1])
                c.score += max(0.0, (gate_px or config.GATE_BASE_PX) - dist) * 0.05
        cands.sort(key=lambda c: c.score, reverse=True)
        return cands[0]

    # ------------------------------------------------------------------ #
    def _blobs(self, mask, color_mask, motion_mask, bonus, exclude=None):
        out = []
        if exclude is not None:
            mask = cv2.bitwise_and(mask, cv2.bitwise_not(exclude))
        n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
        for i in range(1, n):
            area = stats[i, cv2.CC_STAT_AREA]
            if not (config.MIN_BALL_AREA <= area <= self._max_area()):
                continue
            cx, cy = cents[i]
            sel = labels == i
            if sel.sum() == 0:
                continue
            has_color = bool((color_mask[sel] > 0).any())
            has_motion = bool((motion_mask[sel] > 0).any())
            x, y, w, h = stats[i, :4]
            per = 2 * (w + h)
            circ = 4 * np.pi * area / (per * per) if per else 0.0
            if circ < config.MIN_CIRCULARITY * 0.7:
                continue
            r = float(np.sqrt(area / np.pi))
            score = bonus + min(area, 400) / 400 + circ
            if has_color:
                score += 2.0
            if has_motion:
                score += 1.0
            out.append(Candidate(cx, cy, r, float(area), score,
                                 has_color, has_motion))
        return out
