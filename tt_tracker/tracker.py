"""Kalman-filter ball tracker.

State [x, y, vx, vy] in image pixels, constant-velocity model with a
per-frame dt. The ball accelerates violently at hits/bounces, so the process
noise is high and the filter re-converges within a few frames.
"""

from collections import deque

import cv2
import numpy as np

from . import config


class BallTracker:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32
        )
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * (
            config.KF_MEAS_NOISE ** 2
        )
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 100
        self._initialized = False
        self.coasted = 0          # consecutive frames without a detection
        self.lost = True
        self.trail = deque(maxlen=300)   # (x, y, t) image-space history
        self._dets = deque(maxlen=6)     # recent raw detections (x, y, t)
        self._meas_v = (0.0, 0.0)        # velocity fitted to recent detections
        self.radius = 8.0
        self.last_vx = 0.0
        self.last_vy = 0.0

    # ------------------------------------------------------------------ #
    def _set_dt(self, dt: float):
        # NOTE: cv2 attribute getters return copies - must assign a new matrix.
        self.kf.transitionMatrix = np.array(
            [[1, 0, dt, 0],
             [0, 1, 0, dt],
             [0, 0, 1, 0],
             [0, 0, 0, 1]], dtype=np.float32)
        g = np.array([[dt * dt / 2], [dt * dt / 2], [dt], [dt]], dtype=np.float32)
        self.kf.processNoiseCov = (g @ g.T) * (config.KF_ACCEL_NOISE ** 2)

    # ------------------------------------------------------------------ #
    def predict(self, dt: float):
        if not self._initialized:
            return None
        dt = min(max(dt, 1e-3), 0.1)
        self._set_dt(dt)
        pred = self.kf.predict().flatten()
        return float(pred[0]), float(pred[1])

    def update(self, det, t: float):
        """det: Candidate or None. Returns current (x, y) estimate or None."""
        if det is not None:
            meas = np.array([[det.x], [det.y]], dtype=np.float32)
            if not self._initialized:
                self.kf.statePost = np.array(
                    [[det.x], [det.y], [0.0], [0.0]], dtype=np.float32
                )
                self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 100
                self._initialized = True
            else:
                cur = self.kf.statePost.flatten()
                jump = np.hypot(det.x - cur[0], det.y - cur[1])
                if jump > config.GATE_BASE_PX * 1.6:
                    # far re-acquire: teleport instead of dragging the filter
                    self.kf.statePost = np.array(
                        [[det.x], [det.y], [0.0], [0.0]], dtype=np.float32
                    )
                    self.kf.errorCovPost = np.diag(
                        [25.0, 25.0, 1e4, 1e4]).astype(np.float32)
                    self.trail.clear()
                else:
                    self.kf.correct(meas)
            self.coasted = 0
            self.lost = False
            self.radius = det.radius
            self._dets.append((det.x, det.y, t))
            self._fit_meas_velocity()
        else:
            if self._initialized:
                self.coasted += 1
            if self.coasted > config.COAST_MAX_FRAMES:
                self.lost = True
                self.trail.clear()
            if self.coasted > 3:
                self._dets.clear()

        if not self._initialized:
            return None
        s = self.kf.statePost.flatten()
        x, y = float(s[0]), float(s[1])
        self.last_vx, self.last_vy = float(s[2]), float(s[3])
        tx, ty = (det.x, det.y) if det is not None else (x, y)
        self.trail.append((tx, ty, t, det is not None))
        return x, y

    # ------------------------------------------------------------------ #
    def position(self):
        if not self._initialized:
            return None
        s = self.kf.statePost.flatten()
        return float(s[0]), float(s[1])

    def _fit_meas_velocity(self):
        """Least-squares velocity over the last few raw detections (px/s).
        Responds to direction reversals ~4x faster than the KF velocity."""
        d = list(self._dets)
        if len(d) < 3:
            self._meas_v = (self.last_vx, self.last_vy)
            return
        ts = np.array([p[2] for p in d])
        ts = ts - ts[0]
        if ts[-1] - ts[0] < 1e-3:
            return
        vx = np.polyfit(ts, [p[0] for p in d], 1)[0]
        vy = np.polyfit(ts, [p[1] for p in d], 1)[0]
        self._meas_v = (float(vx), float(vy))

    def measured_velocity(self):
        """Velocity from raw detections if fresh, else KF estimate."""
        if len(self._dets) >= 3 and self.coasted <= 3:
            return self._meas_v
        return self.velocity()

    def velocity(self):
        if not self._initialized:
            return 0.0, 0.0
        s = self.kf.statePost.flatten()
        return float(s[2]), float(s[3])

    def speed_px(self) -> float:
        vx, vy = self.velocity()
        return float(np.hypot(vx, vy))

    def reset(self):
        self._initialized = False
        self.coasted = 0
        self.lost = True
        self.trail.clear()
        self.kf.statePost = np.zeros((4, 1), dtype=np.float32)
        self.kf.statePre = np.zeros((4, 1), dtype=np.float32)
