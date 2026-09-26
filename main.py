#!/usr/bin/env python3
"""Real-time table tennis tracker.

Usage:
    python main.py                              # webcam 0, auto-calibrate
    python main.py --source match.mp4           # video file
    python main.py --record annotated.mp4       # save annotated output
    python main.py --ball-color white
    python main.py --table-corners "x1,y1,x2,y2,x3,y3,x4,y4"
    python main.py --no-display                 # headless (needs --record or
                                                # just prints stats at end)

Keys (display mode):  q quit | space pause | r reset stats | c recalibrate
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

from tt_tracker import config
from tt_tracker.calibration import Calibrator, Calibration
from tt_tracker.detector import BallDetector
from tt_tracker.events import EventEngine
from tt_tracker.overlay import Overlay
from tt_tracker.tracker import BallTracker

SIDE_NAME = {0: "P1(left)", 1: "P2(right)"}
CALIB_FILE = "calibration.json"


def parse_args():
    p = argparse.ArgumentParser(description="Camera-based table tennis tracker")
    p.add_argument("--source", default="0",
                   help="camera index or video file path (default: 0)")
    p.add_argument("--ball-color", default=None,
                   choices=list(config.BALL_COLORS),
                   help="ball color (if omitted, a startup prompt asks)")
    p.add_argument("--calibration", default=CALIB_FILE,
                   help="calibration JSON to load/save")
    p.add_argument("--table-corners", default=None,
                   help='"x1,y1,x2,y2,x3,y3,x4,y4" TL,TR,BR,BL image points - '
                        "skip interactive calibration")
    p.add_argument("--record", default=None, help="write annotated video here")
    p.add_argument("--web", type=int, default=None, metavar="PORT",
                   help="serve annotated feed + stats at http://localhost:PORT")
    p.add_argument("--loop", action="store_true",
                   help="loop a video-file source forever (demo mode)")
    p.add_argument("--no-display", action="store_true", help="headless mode")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    return p.parse_args()


def default_calibration(w, h):
    """Guess: side view, table length horizontal across the frame."""
    pts = np.array([[0.04 * w, 0.42 * h], [0.96 * w, 0.42 * h],
                    [0.99 * w, 0.72 * h], [0.01 * w, 0.72 * h]],
                   dtype=np.float64)
    return Calibration(pts)


def pick_ball_color(args, frame):
    """Ask which ball color to track. GUI prompt if a display is available."""
    if args.ball_color:
        return args.ball_color
    if args.no_display:
        print("[ball] no display - defaulting to orange "
              "(use --ball-color to override)")
        return "orange"
    win = "TT Tracker - ball color"
    cv2.namedWindow(win)
    while True:
        disp = frame.copy()
        cv2.rectangle(disp, (0, 0), (disp.shape[1], 60), (20, 20, 20), -1)
        cv2.putText(disp, "Which ball are you using?", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
                    cv2.LINE_AA)
        cv2.putText(disp, "press  O = orange      W = white",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 165, 255), 2, cv2.LINE_AA)
        cv2.imshow(win, disp)
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("o"), ord("w")):
            cv2.destroyWindow(win)
            color = "orange" if key == ord("o") else "white"
            print(f"[ball] tracking {color} ball")
            return color
        if key == 27:
            cv2.destroyWindow(win)
            sys.exit("cancelled")


def get_calibration(args, frame):
    if args.table_corners:
        vals = [float(v) for v in args.table_corners.split(",")]
        if len(vals) != 8:
            sys.exit("--table-corners needs 8 comma-separated numbers")
        return Calibration(np.array(vals).reshape(4, 2))
    if os.path.exists(args.calibration):
        print(f"[calib] loaded {args.calibration}")
        return Calibration.load(args.calibration)
    if not args.no_display:
        print("[calib] click the 4 table corners (far-left, far-right, "
              "near-right, near-left as seen on screen)")
        cal = Calibrator().run(frame)
        if cal is not None:
            cal.save(args.calibration)
            print(f"[calib] saved to {args.calibration}")
            return cal
    print("[calib] no calibration - using frame-default guess "
          "(run with display for accurate placement/speed)")
    return default_calibration(frame.shape[1], frame.shape[0])


def player_motion(fg_mask, net_x, h):
    """Motion energy on each side of the net, normalized 0..1."""
    if fg_mask is None:
        return 0.0, 0.0
    nx = max(1, min(net_x, fg_mask.shape[1] - 1))
    left = float(cv2.countNonZero(fg_mask[:, :nx]))
    right = float(cv2.countNonZero(fg_mask[:, nx:]))
    denom = nx * h + (fg_mask.shape[1] - nx) * h
    scale = 60.0 / max(denom, 1)   # tuned: ~1.7% of pixels moving -> full bar
    return min(1.0, left * scale), min(1.0, right * scale)


def main():
    args = parse_args()
    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if isinstance(src, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        sys.exit(f"cannot open source {args.source}")

    # warm up camera auto-exposure before showing the calibration frame
    frame = None
    for _ in range(15 if isinstance(src, int) else 1):
        ok, frame = cap.read()
        if not ok:
            sys.exit("cannot read first frame")
    h, w = frame.shape[:2]

    ball_color = pick_ball_color(args, frame)
    calib = get_calibration(args, frame)
    detector = BallDetector(ball_color)
    tracker = BallTracker()
    engine = EventEngine(calib)
    overlay = Overlay(calib)
    net_x = calib.net_image_x()

    web = None
    if args.web:
        from tt_tracker.web import _State, make_server
        web = _State()
        make_server(web, args.web)
        print(f"[web] open http://localhost:{args.web}")

    writer = None
    if args.record:
        fps_out = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if fps_out <= 1 or fps_out > 240:
            fps_out = 30.0
        writer = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps_out, (w, h))

    paused = False
    prev_t = time.perf_counter()
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if not (1 < src_fps <= 240):
        src_fps = 30.0
    fps_ema = src_fps
    file_dt = None if isinstance(src, int) else 1.0 / src_fps
    n_frames = 0

    print("[tt] running - q to quit")
    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok or frame is None:
                if args.loop and file_dt is not None:
                    cap.release()
                    cap = cv2.VideoCapture(src)
                    continue
                break
            n_frames += 1
            now = time.perf_counter()
            dt = file_dt if file_dt is not None else (now - prev_t)
            prev_t = now
            if dt > 0:
                fps_ema = 0.9 * fps_ema + 0.1 / dt

            # -- track --------------------------------------------------
            pred = tracker.predict(dt)
            gate = config.GATE_BASE_PX + tracker.coasted * 10
            cand = detector.detect(frame, pred, gate)
            det = None
            if cand is not None:
                if pred is None or tracker.coasted > config.REACQUIRE_AFTER:
                    det = cand
                elif np.hypot(cand.x - pred[0], cand.y - pred[1]) < gate:
                    det = cand
            t = n_frames * file_dt if file_dt is not None else now
            pos = tracker.update(det, t)
            if det is not None:
                detector.note_accepted(det.radius)

            # -- events (from raw detections, not the lagging KF state) ---
            det_pos = (cand.x, cand.y) if cand is not None else None
            det_r = cand.radius if cand is not None else tracker.radius
            for ev in engine.step(det_pos, t, radius=det_r):
                if ev[0] == "serve":
                    overlay.toast(f"SERVE  {SIDE_NAME[ev[1]]}", (255, 255, 0))
                elif ev[0] == "hit":
                    overlay.toast(
                        f"HIT {SIDE_NAME[ev[1]]}  {ev[2]:.1f} m/s",
                        (0, 255, 255))
                elif ev[0] == "bounce":
                    overlay.toast(f"bounce {SIDE_NAME[ev[1]]} [{ev[2]}]",
                                  (200, 200, 200), ttl=0.8)
                elif ev[0] == "rally_end":
                    overlay.toast(
                        f"POINT OVER - rally of {ev[1]['hits']} hits",
                        (0, 140, 255), ttl=2.0)

            # -- draw ---------------------------------------------------
            pm = player_motion(detector.fg_motion, net_x, h)
            annotated = overlay.draw(frame.copy(), tracker, engine,
                                     fps_ema, det is not None, pm,
                                     det_pos=det_pos)
            if writer is not None:
                writer.write(annotated)
            if web is not None:
                web.push_frame(annotated, engine.stats())
            if not args.no_display:
                cv2.imshow("TT Tracker", annotated)

        if not args.no_display:
            key = cv2.waitKey(1 if not paused else 50) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                paused = not paused
            elif key == ord("r"):
                engine.__init__(calib)
                print("[tt] stats reset")
            elif key == ord("c"):
                cal = Calibrator().run(frame)
                if cal is not None:
                    cal.save(args.calibration)
                    calib, engine.calib, overlay.calib = cal, cal, cal
                    net_x = calib.net_image_x()
        elif args.no_display:
            pass

    engine.end(n_frames * file_dt if file_dt is not None else time.perf_counter())
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    s = engine.stats()
    print("\n=== session summary ===")
    for k, v in s.items():
        print(f"  {k}: {v:.1f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()
