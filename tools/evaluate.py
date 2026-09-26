#!/usr/bin/env python3
"""Evaluate the tracking pipeline against synthetic-video ground truth.

Usage:
    .venv/bin/python tools/evaluate.py [sim.mp4] [sim_gt.json]
"""

import json
import sys
from collections import defaultdict

import cv2
import numpy as np

sys.path.insert(0, ".")
from tt_tracker import config
from tt_tracker.calibration import Calibration
from tt_tracker.detector import BallDetector
from tt_tracker.events import EventEngine
from tt_tracker.tracker import BallTracker

# table corners used by make_synthetic_video at 1280x720
CORNERS = np.array([[180, 420], [1100, 420], [1180, 500], [100, 500]], float)
EVENT_TOL_FRAMES = 8


def main():
    video = sys.argv[1] if len(sys.argv) > 1 else "tools/sim.mp4"
    gtfile = sys.argv[2] if len(sys.argv) > 2 else "tools/sim_gt.json"
    gt = json.load(open(gtfile))

    cal = Calibration(CORNERS)
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    det, tr, eng = BallDetector("orange"), BallTracker(), EventEngine(cal)
    dt = 1.0 / fps

    pos_err, speeds = [], []
    det_events = []          # (frame, event tuple)
    track_visible = 0
    gt_visible = 0
    for i, rec in enumerate(gt):
        ok, f = cap.read()
        if not ok:
            break
        t = i * dt
        pred = tr.predict(dt)
        gate = config.GATE_BASE_PX + tr.coasted * 10
        c = det.detect(f, pred, gate)
        d = None
        if c is not None:
            if pred is None or tr.coasted > config.REACQUIRE_AFTER:
                d = c
            elif np.hypot(c.x - pred[0], c.y - pred[1]) < gate:
                d = c
        pos = tr.update(d, t)
        evs = eng.step((c.x, c.y) if c is not None else None, t,
                       radius=c.radius if c is not None else tr.radius)
        vel = tr.measured_velocity() if pos else (0.0, 0.0)
        for e in evs:
            det_events.append((i, e))
        if rec["x"] is not None:
            gt_visible += 1
            if pos is not None and not tr.lost:
                track_visible += 1
            dp = d if d is not None else (pos if pos and not tr.lost else None)
            if dp is not None:
                dx = d.x if d is not None else pos[0]
                dy = d.y if d is not None else pos[1]
                pos_err.append(np.hypot(dx - rec["x"], dy - rec["y"]))
            speeds.append(eng.last_speed)

    # ---- match events ------------------------------------------------
    gt_events = [(i, r["event"]) for i, r in enumerate(gt) if r["event"]]
    det_simple = [(i, e[0], e[1]) for i, e in det_events]

    def matched(gt_kind_prefix, det_kind):
        g = [(i, k) for i, k in gt_events if k.startswith(gt_kind_prefix)]
        d = [(i, s) for i, k, s in det_simple if k == det_kind]
        used, tp = set(), 0
        for gi, gk in g:
            side_gt = 0 if gk.endswith("left") else 1
            best = None
            for j, (di, ds) in enumerate(d):
                if j in used or ds != side_gt:
                    continue
                if abs(di - gi) <= EVENT_TOL_FRAMES:
                    if best is None or abs(di - gi) < abs(d[best][0] - gi):
                        best = j
            if best is not None:
                used.add(best)
                tp += 1
        return tp, len(g), len(d)

    print(f"visible frames: gt={gt_visible} tracked={track_visible} "
          f"({100 * track_visible / max(gt_visible, 1):.0f}%)")
    if pos_err:
        print(f"pos err px: median={np.median(pos_err):.1f} "
              f"p90={np.percentile(pos_err, 90):.1f} max={max(pos_err):.1f}")
    if speeds:
        print(f"speed m/s: median={np.median(speeds):.1f} "
              f"p95={np.percentile(speeds, 95):.1f} max={max(speeds):.1f}")
    for prefix, kind in (("hit", "hit"), ("bounce", "bounce")):
        tp, ng, nd = matched(prefix, kind)
        print(f"{kind}: gt={ng} detected={nd} matched={tp} "
              f"recall={tp / max(ng, 1):.0%} precision={tp / max(nd, 1):.0%}")
    serves = [e for e in det_simple if e[1] == "serve"]
    print(f"serves={len(serves)} rallies={len(eng.rallies)} "
          f"rally hits={[r['hits'] for r in eng.rallies]}")
    print("stats:", eng.stats())


if __name__ == "__main__":
    main()
