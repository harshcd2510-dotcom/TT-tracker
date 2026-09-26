#!/usr/bin/env python3
"""
make_synthetic_video.py -- synthetic table-tennis match video generator.

Fixed-camera SIDE-ELEVATED view: the table length runs left-right across the
frame. A ball rallies back and forth across the net with ballistic arcs, one
bounce on the opponent's half per shot, occasional faults, and rally gaps.

Used to test CV ball-tracking pipelines. Only OpenCV (cv2) + numpy.

Outputs:
  1. <out>.mp4  (cv2.VideoWriter, 'mp4v')
  2. <out>_gt.json -- per-frame ground truth:
     [{"frame": int, "x": float|null, "y": float|null,
       "event": "hit_left"|"hit_right"|"bounce_left"|"bounce_right"|null}, ...]
     x,y are the ball CENTER in pixels; null when the ball is not visible.
     "hit_left"/"hit_right" = hit made at the left/right end of the table
     (ball then travels right/left). "bounce_left"/"bounce_right" = bounce on
     the left/right half of the table.
"""

import argparse
import json
import math
import os

import cv2
import numpy as np

# --- physical / visual constants -------------------------------------------
GRAVITY = 1.2                 # px / frame^2 (vy increments by this each frame)
BALL_RADIUS = 9               # px
BALL_BGR = (30, 140, 255)     # solid orange
SPECULAR_BGR = (200, 230, 255)
TABLE_BGR = (90, 60, 30)      # dark blue-ish tabletop
BG_GRAY = 36                  # dark gray background
NET_CLEAR_MARGIN = 4          # extra px of clearance above the net top

# Reference table corners (1280x720); scaled to actual frame size.
REF_W, REF_H = 1280.0, 720.0
REF_TL = (180.0, 420.0)
REF_TR = (1100.0, 420.0)
REF_BR = (1180.0, 500.0)
REF_BL = (100.0, 500.0)
REF_NET_RISE = 35.0           # px the net sticks up above the far edge


def build_geometry(W, H):
    """Table quad corners + net info, scaled from the 1280x720 reference."""
    sx, sy = W / REF_W, H / REF_H
    geo = {
        "TL": (REF_TL[0] * sx, REF_TL[1] * sy),
        "TR": (REF_TR[0] * sx, REF_TR[1] * sy),
        "BR": (REF_BR[0] * sx, REF_BR[1] * sy),
        "BL": (REF_BL[0] * sx, REF_BL[1] * sy),
    }
    geo["net_x"] = (geo["TL"][0] + geo["TR"][0]) / 2.0       # == W/2
    geo["net_top"] = geo["TL"][1] - REF_NET_RISE * sy
    return geo


def _edge_y(p1, p2, x):
    if p2[0] == p1[0]:
        return 0.5 * (p1[1] + p2[1])
    t = (x - p1[0]) / (p2[0] - p1[0])
    return p1[1] + t * (p2[1] - p1[1])


def surface_y(geo, x):
    """Playable-surface height at image x: mean of the far and near edges."""
    return 0.5 * (_edge_y(geo["TL"], geo["TR"], x)
                  + _edge_y(geo["BL"], geo["BR"], x))


# ---------------------------------------------------------------------------
# Static scene (background + table + net + constant noise), built once.
# ---------------------------------------------------------------------------
def build_static(W, H, geo, rng):
    img = np.full((H, W, 3), BG_GRAY, np.int16)
    img += rng.integers(-9, 10, size=(H, W, 1), dtype=np.int16)
    img = np.clip(img, 0, 255).astype(np.uint8)

    # table quad with its own constant noise texture
    quad = np.array([geo["TL"], geo["TR"], geo["BR"], geo["BL"]], np.int32)
    mask = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(mask, quad, 255)
    tex = np.clip(
        np.array(TABLE_BGR, np.int16)
        + rng.integers(-7, 8, size=(H, W, 1), dtype=np.int16),
        0, 255).astype(np.uint8)
    img[mask > 0] = tex[mask > 0]

    # subtle edge highlights: far edge thin/dark, near edge brighter
    tl = tuple(np.int32(geo["TL"])); tr = tuple(np.int32(geo["TR"]))
    bl = tuple(np.int32(geo["BL"])); br = tuple(np.int32(geo["BR"]))
    cv2.line(img, tl, tr, (120, 95, 70), 1, cv2.LINE_AA)
    cv2.line(img, bl, br, (170, 150, 120), 2, cv2.LINE_AA)
    # thin white end lines on the near edge (left/right end markings)
    cv2.line(img, tl, bl, (140, 120, 100), 1, cv2.LINE_AA)
    cv2.line(img, tr, br, (140, 120, 100), 1, cv2.LINE_AA)

    # net: thin white vertical bar at table center, from ~35px above the
    # far edge down through the band
    nx = int(round(geo["net_x"]))
    ntop = int(round(geo["net_top"]))
    nbot = int(round(geo["BL"][1]))
    nw = max(3, int(round(W * 0.0025)))
    cv2.rectangle(img, (nx - nw, ntop), (nx + nw, nbot), (235, 235, 235), -1)
    cv2.rectangle(img, (nx - nw - 1, ntop - 2), (nx + nw + 1, ntop),
                  (245, 245, 245), -1)
    geo["net_half_w"] = nw
    return img


# ---------------------------------------------------------------------------
# Shot planning. All trajectories are planned in canonical "left -> right"
# space, then mirrored about x = W/2 for right -> left shots.
# A shot returns a list of (x, y, event) tuples, one per frame it occupies.
# The final endpoint of a normal shot is NOT emitted: it becomes frame 0
# (the hit frame) of the next shot.
# ---------------------------------------------------------------------------
def _seg_pos(x0, y0, vx, vy, t):
    return x0 + vx * t, y0 + vy * t + 0.5 * GRAVITY * t * t


def _plan_first_segment(rng, geo, W, H, xb_lo, xb_hi):
    """Plan launch params from a left-side hit to a bounce at xb.

    Returns (x0, y0, xb, yb, vx, vy, K1) or None."""
    r = BALL_RADIUS
    x0 = rng.uniform(0.10 * W, 0.22 * W)
    y0 = surface_y(geo, x0) - r - rng.uniform(8, 75)
    xb = rng.uniform(xb_lo, xb_hi)
    yb = surface_y(geo, xb) - r
    vxd = rng.uniform(14.0, 30.0)
    K1 = int(round((xb - x0) / vxd))
    if K1 < 10:
        return None
    vx = (xb - x0) / K1
    if not (10.0 <= vx <= 34.0):
        return None
    vy = (yb - y0 - 0.5 * GRAVITY * K1 * K1) / K1
    if vy > -2.0:                      # must launch upward-ish
        return None
    y_min = y0 - vy * vy / (2 * GRAVITY)
    if y_min < 12.0:                   # stay inside top of frame
        return None
    # net clearance: ball bottom edge must stay above the net top
    tn = (geo["net_x"] - x0) / vx
    if not (0.0 < tn < K1):
        return None
    yn = y0 + vy * tn + 0.5 * GRAVITY * tn * tn
    if yn + r + NET_CLEAR_MARGIN > geo["net_top"]:
        return None
    return x0, y0, xb, yb, vx, vy, K1


def plan_normal_shot(rng, geo, W, H):
    """Hit at left end -> bounce on right half -> arrives at right-side hit
    position (emitted as next shot's frame 0)."""
    r = BALL_RADIUS
    seg = None
    K2 = vy2 = xe = ye = None
    for _ in range(400):
        seg = _plan_first_segment(rng, geo, W, H, 0.55 * W, 0.78 * W)
        if seg is None:
            continue
        x0, y0, xb, yb, vx, vy, K1 = seg
        xe = rng.uniform(max(xb + 0.06 * W, 0.80 * W), 0.92 * W)
        ye = surface_y(geo, xe) - r - rng.uniform(10, 70)
        K2 = int(round((xe - xb) / vx))
        if K2 < 5:
            continue
        vy2 = (ye - yb - 0.5 * GRAVITY * K2 * K2) / K2
        if vy2 > -0.6:                 # need a visible upward pop off the table
            continue
        break
    else:
        # deterministic fallback (high gentle arc, always valid)
        x0, xb, xe = 0.15 * W, 0.70 * W, 0.90 * W
        y0 = surface_y(geo, x0) - r - 40
        yb = surface_y(geo, xb) - r
        ye = surface_y(geo, xe) - r - 40
        K1 = 30
        vx = (xb - x0) / K1
        vy = (yb - y0 - 0.5 * GRAVITY * K1 * K1) / K1
        K2 = int(round((xe - xb) / vx))
        vy2 = (ye - yb - 0.5 * GRAVITY * K2 * K2) / K2
        seg = (x0, y0, xb, yb, vx, vy, K1)

    x0, y0, xb, yb, vx, vy, K1 = seg
    frames = []
    for k in range(K1 + K2):
        if k <= K1:
            x, y = _seg_pos(x0, y0, vx, vy, k)
        else:
            x, y = _seg_pos(xb, yb, vx, vy2, k - K1)
        ev = None
        if k == 0:
            ev = "hit_left"
        elif k == K1:
            ev = "bounce_right"
        frames.append((x, y, ev))
    return frames


def plan_fault_shot(rng, geo, W, H):
    """Terminal shot of a rally: ball misses the table or flies out of frame.
    Emits frames only while the ball center is inside the image."""
    r = BALL_RADIUS
    if rng.random() < 0.5:
        # variant A: bounce on the opponent's half, then sails past the
        # would-be hit position and out of frame (whiffed return)
        seg = None
        K2 = vy2 = xe = ye = None
        for _ in range(400):
            seg = _plan_first_segment(rng, geo, W, H, 0.55 * W, 0.76 * W)
            if seg is None:
                continue
            x0, y0, xb, yb, vx, vy, K1 = seg
            xe = rng.uniform(W * 1.05, W * 1.30)
            ye = surface_y(geo, min(xe, W)) - r - rng.uniform(0, 60)
            K2 = int(round((xe - xb) / vx))
            if K2 < 5:
                continue
            vy2 = (ye - yb - 0.5 * GRAVITY * K2 * K2) / K2
            if vy2 > -0.3:
                continue
            break
        else:
            x0, xb, xe = 0.15 * W, 0.70 * W, 1.20 * W
            y0 = surface_y(geo, x0) - r - 40
            yb = surface_y(geo, xb) - r
            ye = surface_y(geo, W) - r - 20
            K1 = 30
            vx = (xb - x0) / K1
            vy = (yb - y0 - 0.5 * GRAVITY * K1 * K1) / K1
            K2 = int(round((xe - xb) / vx))
            vy2 = (ye - yb - 0.5 * GRAVITY * K2 * K2) / K2
            seg = (x0, y0, xb, yb, vx, vy, K1)

        x0, y0, xb, yb, vx, vy, K1 = seg
        frames = []
        for k in range(K1 + K2 + 200):
            if k <= K1:
                x, y = _seg_pos(x0, y0, vx, vy, k)
            else:
                x, y = _seg_pos(xb, yb, vx, vy2, k - K1)
            if not (0.0 <= x <= W and 0.0 <= y <= H):
                break
            ev = None
            if k == 0:
                ev = "hit_left"
            elif k == K1:
                ev = "bounce_right"
            frames.append((x, y, ev))
        return frames

    # variant B: no bounce at all -- hit too long, lands beyond the far end
    # line, then continues until it leaves the frame
    for _ in range(400):
        x0 = rng.uniform(0.10 * W, 0.22 * W)
        y0 = surface_y(geo, x0) - r - rng.uniform(8, 75)
        xl = rng.uniform(0.98 * W, 1.25 * W)     # lands past the table end
        yl = surface_y(geo, min(xl, W))
        vxd = rng.uniform(14.0, 30.0)
        K = int(round((xl - x0) / vxd))
        if K < 10:
            continue
        vx = (xl - x0) / K
        if not (10.0 <= vx <= 34.0):
            continue
        vy = (yl - y0 - 0.5 * GRAVITY * K * K) / K
        if vy > -2.0:
            continue
        if y0 - vy * vy / (2 * GRAVITY) < 12.0:
            continue
        tn = (geo["net_x"] - x0) / vx
        yn = y0 + vy * tn + 0.5 * GRAVITY * tn * tn
        if yn + r + NET_CLEAR_MARGIN > geo["net_top"]:
            continue
        break
    else:
        x0, xl = 0.15 * W, 1.15 * W
        y0 = surface_y(geo, x0) - r - 40
        yl = surface_y(geo, W)
        K = 45
        vx = (xl - x0) / K
        vy = (yl - y0 - 0.5 * GRAVITY * K * K) / K

    frames = []
    for k in range(K + 300):
        x, y = _seg_pos(x0, y0, vx, vy, k)
        if not (0.0 <= x <= W and 0.0 <= y <= H):
            break
        frames.append((x, y, "hit_left" if k == 0 else None))
    return frames


_EVENT_MIRROR = {
    "hit_left": "hit_right", "hit_right": "hit_left",
    "bounce_left": "bounce_right", "bounce_right": "bounce_left",
    None: None,
}


def generate_trajectory(n_frames, rng, geo, W, H):
    """Returns a list of (x|None, y|None, event|None), len == n_frames."""
    recs = []
    side_left = bool(rng.random() < 0.5)   # which end currently hits
    hits = 0                                # normal hits completed in rally
    planned = int(rng.integers(2, 9))       # planned rally length, 2-8 hits
    while len(recs) < n_frames:
        # forced fault on the last planned hit; ~15% random faults otherwise
        fault = (hits >= planned - 1) or (hits >= 1 and rng.random() < 0.15)
        shot = (plan_fault_shot(rng, geo, W, H) if fault
                else plan_normal_shot(rng, geo, W, H))
        if not side_left:                   # mirror to right->left
            shot = [(W - x, y, _EVENT_MIRROR[ev]) for x, y, ev in shot]
        recs.extend(shot)
        if fault:
            gap = int(rng.integers(25, 41))  # 25-40 frames with no ball
            recs.extend([(None, None, None)] * gap)
            side_left = bool(rng.random() < 0.5)
            hits = 0
            planned = int(rng.integers(2, 9))
        else:
            side_left = not side_left
            hits += 1
    return recs[:n_frames]


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def render_frame(static, rec, i, W, H):
    frame = static.copy()

    # two translucent player blobs just beyond the table ends, each
    # oscillating vertically on a slow sine with a different phase/period
    overlay = frame.copy()
    base_y = 0.62 * H
    amp = 0.022 * H
    yl = base_y + amp * math.sin(2 * math.pi * i / 150.0)
    yr = base_y + amp * math.sin(2 * math.pi * i / 173.0 + 1.3)
    rx, ry = int(0.026 * W), int(0.095 * H)
    cv2.ellipse(overlay, (int(0.045 * W), int(yl)), (rx, ry), 0, 0, 360,
                (52, 46, 58), -1, cv2.LINE_AA)
    cv2.ellipse(overlay, (int(0.045 * W), int(yl) - ry - 6),
                (int(0.014 * W), int(0.02 * H)), -25, 0, 360,
                (52, 46, 58), -1, cv2.LINE_AA)
    cv2.ellipse(overlay, (int(0.955 * W), int(yr)), (rx, ry), 0, 0, 360,
                (56, 48, 50), -1, cv2.LINE_AA)
    cv2.ellipse(overlay, (int(0.955 * W), int(yr) - ry - 6),
                (int(0.014 * W), int(0.02 * H)), 25, 0, 360,
                (56, 48, 50), -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    x, y, _ = rec
    if x is not None:
        cx, cy = int(round(x)), int(round(y))
        cv2.circle(frame, (cx, cy), BALL_RADIUS, BALL_BGR, -1, cv2.LINE_AA)
        cv2.circle(frame, (cx - 3, cy - 3), 3, SPECULAR_BGR, -1, cv2.LINE_AA)
    return frame


def main():
    ap = argparse.ArgumentParser(
        description="Synthetic table-tennis match video generator")
    ap.add_argument("--out", required=True, help="output .mp4 path")
    ap.add_argument("--frames", type=int, default=900)
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    W, H, N = args.width, args.height, args.frames
    rng = np.random.default_rng(args.seed)
    geo = build_geometry(W, H)
    static = build_static(W, H, geo, rng)

    recs = generate_trajectory(N, rng, geo, W, H)

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    writer = cv2.VideoWriter(
        args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H))
    if not writer.isOpened():
        raise RuntimeError("cv2.VideoWriter failed to open %s" % args.out)
    for i, rec in enumerate(recs):
        writer.write(render_frame(static, rec, i, W, H))
    writer.release()

    gt = [{"frame": i,
           "x": (round(float(x), 2) if x is not None else None),
           "y": (round(float(y), 2) if y is not None else None),
           "event": ev}
          for i, (x, y, ev) in enumerate(recs)]
    root, ext = os.path.splitext(args.out)
    gt_path = root + "_gt.json"
    with open(gt_path, "w") as f:
        json.dump(gt, f)

    n_vis = sum(1 for x, _, _ in recs if x is not None)
    ev_counts = {}
    for _, _, ev in recs:
        if ev:
            ev_counts[ev] = ev_counts.get(ev, 0) + 1
    print("wrote %s (%d frames, %dx%d @ %dfps)" % (args.out, N, W, H, args.fps))
    print("wrote %s" % gt_path)
    print("ball visible frames: %d/%d" % (n_vis, N))
    print("events: %s" % json.dumps(ev_counts, sort_keys=True))


if __name__ == "__main__":
    main()
