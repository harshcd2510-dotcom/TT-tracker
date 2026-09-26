"""Draws all annotations onto the frame: ball, trail, court lines, HUD,
placement mini-map, player-activity bars, and transient event toasts."""

import time

import cv2
import numpy as np

from . import config

FONT = cv2.FONT_HERSHEY_SIMPLEX


class Overlay:
    def __init__(self, calib=None):
        self.calib = calib
        self._toasts = []  # (text, color, expiry_time)

    # ------------------------------------------------------------------ #
    def toast(self, text, color=(0, 255, 255), ttl=None):
        ttl = ttl if ttl is not None else config.TOAST_FRAMES / 30.0
        self._toasts.append((text, color, time.time() + ttl))

    def _draw_toasts(self, img):
        now = time.time()
        self._toasts = [t for t in self._toasts if t[2] > now]
        h = img.shape[0]
        for i, (text, color, _) in enumerate(self._toasts[-4:]):
            y = h - 90 - i * 28
            cv2.putText(img, text, (16, y), FONT, 0.8, color, 2, cv2.LINE_AA)

    # ------------------------------------------------------------------ #
    def draw(self, img, tracker, engine, fps, detected, player_motion=(0, 0),
             det_pos=None):
        if self.calib is not None:
            self._draw_table(img)
        self._draw_ball(img, tracker, detected, det_pos)
        self._draw_hud(img, engine, fps, player_motion)
        if self.calib is not None:
            self._draw_minimap(img, engine)
        self._draw_toasts(img)
        return img

    # ------------------------------------------------------------------ #
    def _draw_table(self, img):
        cv2.polylines(img, [self.calib.image_poly], True, (0, 200, 0), 2)
        # net line: table x = NET_X across the width
        a = self.calib.table_to_image(config.NET_X, 0)
        b = self.calib.table_to_image(config.NET_X, config.TABLE_WIDTH)
        cv2.line(img, a, b, (255, 255, 255), 2)
        # zone dividers on the surface
        for frac, col in ((0.45, (90, 90, 90)), (0.95, (90, 90, 90))):
            for sgn in (-1, 1):
                x = config.NET_X + sgn * frac
                p1 = self.calib.table_to_image(x, 0)
                p2 = self.calib.table_to_image(x, config.TABLE_WIDTH)
                cv2.line(img, p1, p2, col, 1, cv2.LINE_AA)

    def _draw_ball(self, img, tracker, detected, det_pos=None):
        trail = list(tracker.trail)[-config.TRAIL_LEN:]
        if len(trail) >= 2:
            pts = np.array([(int(p[0]), int(p[1])) for p in trail], np.int32)
            for i in range(1, len(pts)):
                alpha = i / len(pts)
                cv2.line(img, tuple(pts[i - 1]), tuple(pts[i]),
                         (0, int(200 * alpha), int(255 * alpha)),
                         max(1, int(3 * alpha)), cv2.LINE_AA)
        pos = det_pos if det_pos is not None else tracker.position()
        if pos is None:
            return
        x, y = int(pos[0]), int(pos[1])
        r = max(3, int(tracker.radius) + 3)
        if tracker.lost or tracker.coasted > 0:
            cv2.circle(img, (x, y), r + 4, (0, 165, 255), 1, cv2.LINE_AA)
            cv2.putText(img, "?", (x + r + 4, y), FONT, 0.5,
                        (0, 165, 255), 1, cv2.LINE_AA)
        else:
            cv2.circle(img, (x, y), r, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.drawMarker(img, (x, y), (0, 255, 255),
                           cv2.MARKER_CROSS, r * 2, 1)

    # ------------------------------------------------------------------ #
    def _draw_hud(self, img, engine, fps, player_motion):
        s = engine.stats()
        h, w = img.shape[:2]
        bar = img.copy()
        cv2.rectangle(bar, (0, 0), (w, 62), (15, 15, 15), -1)
        cv2.addWeighted(bar, config.HUD_ALPHA, img, 1 - config.HUD_ALPHA, 0, img)

        state = ("RALLY  hits:%d" % s["current_rally_hits"]
                 if engine.rally_active else "BETWEEN POINTS")
        cv2.putText(img, state, (12, 24), FONT, 0.65,
                    (0, 255, 0) if engine.rally_active else (160, 160, 160),
                    2, cv2.LINE_AA)
        cv2.putText(
            img,
            "Shots L:%d  R:%d   Speed:%.1f m/s  Top:%.1f   Rallies:%d  "
            "Longest:%d  Avg:%.1f   %.0f fps" % (
                s["shots_left"], s["shots_right"], engine.last_speed,
                s["top_speed_mps"], s["rallies"], s["longest_rally"],
                s["avg_rally"], fps),
            (12, 52), FONT, 0.55, (230, 230, 230), 1, cv2.LINE_AA)

        # player activity bars (bottom corners)
        for i, (label, energy) in enumerate(
                zip(("P1 activity", "P2 activity"), player_motion)):
            bx = 12 if i == 0 else w - 212
            e = min(1.0, energy)
            cv2.rectangle(img, (bx, h - 40), (bx + 200, h - 28),
                          (60, 60, 60), -1)
            cv2.rectangle(img, (bx, h - 40), (bx + int(200 * e), h - 28),
                          (0, 200, 255), -1)
            cv2.putText(img, label, (bx, h - 46), FONT, 0.45,
                        (200, 200, 200), 1, cv2.LINE_AA)

    # ------------------------------------------------------------------ #
    def _draw_minimap(self, img, engine):
        """Top-down table inset with bounce placements and hit markers."""
        h, w = img.shape[:2]
        mw, mh = 180, 100
        ox, oy = w - mw - 14, 76
        panel = img.copy()
        cv2.rectangle(panel, (ox - 6, oy - 22), (ox + mw + 6, oy + mh + 6),
                      (15, 15, 15), -1)
        cv2.addWeighted(panel, config.HUD_ALPHA, img, 1 - config.HUD_ALPHA, 0, img)
        cv2.putText(img, "placement", (ox, oy - 6), FONT, 0.45,
                    (200, 200, 200), 1, cv2.LINE_AA)
        cv2.rectangle(img, (ox, oy), (ox + mw, oy + mh), (0, 140, 0), 1)
        cv2.line(img, (ox + mw // 2, oy), (ox + mw // 2, oy + mh),
                 (255, 255, 255), 1)
        for tx, ty, side in engine.bounces[-60:]:
            px = ox + int(tx / config.TABLE_LENGTH * mw)
            py = oy + int(ty / config.TABLE_WIDTH * mh)
            col = (255, 120, 120) if side == 0 else (120, 120, 255)
            cv2.circle(img, (px, py), 3, col, -1, cv2.LINE_AA)
