# TT Tracker

Camera-based table tennis analytics. Point a single webcam/USB camera at the
table from an elevated side view and get a **real-time annotated feed** — ball
trajectory, hit/bounce events, rally stats, ball speed, shot placement map, and
player-activity levels — in an OpenCV window and/or a browser.

No ML models, no GPU: pure OpenCV + NumPy on CPU. MIT licensed.

## Features

- **Ball tracking** — HSV color + frame-diff motion detection, Kalman
  smoothing, coast-through-dropout and hard re-acquire. Learns the ball's
  apparent size and rejects other same-color objects (paddles, shirts, posters).
- **Shot & rally detection** — hits are x-velocity reversals on each side of
  the net; bounces are y-velocity flips near the table surface; rallies are
  segmented automatically. Events run on raw detections, so direction changes
  are caught within ~2-4 frames.
- **Ball speed** — estimated from the ball's apparent size (40 mm diameter),
  robust to the ball being off the table plane.
- **Placement mini-map** — top-down inset plotting where each bounce lands
  (short / mid / long zones).
- **Player activity** — motion energy per half of the frame.
- **Browser streaming** — MJPEG feed + live stats at `http://localhost:PORT`.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt   # opencv-python, numpy
```

## Camera setup

- USB webcam or built-in camera, placed **to the side** of the table, elevated
  ~1.5-2 m, far enough back that the table length runs left↔right on screen.
- **Orange or white** 40+ ball — you pick which at startup (`o`/`w` keys).
- Normal room lighting; avoid the camera facing a bright window.

## Run

```bash
.venv/bin/python main.py                        # camera 0
.venv/bin/python main.py --source 1 --web 8080  # camera 1 + browser feed
.venv/bin/python main.py --source match.mp4     # analyze a video file
```

Startup flow:

1. **Ball color prompt** — press `o` (orange) or `w` (white).
   (Skip with `--ball-color orange|white`.)
2. **Table calibration** — click the 4 tabletop corners: far-left → far-right
   → near-right → near-left as seen on screen, then Enter. Saved to
   `calibration.json` for next run; `c` recalibrates anytime.
   (Skip with `--table-corners "x1,y1,x2,y2,x3,y3,x4,y4"`.)

Then play. Keys in the window: `q` quit · `space` pause · `r` reset stats.

## Options

| Flag | Effect |
|---|---|
| `--source N_OR_PATH` | camera index or video file (default `0`) |
| `--ball-color` | skip the color prompt |
| `--web PORT` | serve annotated MJPEG + stats JSON in browser |
| `--record out.mp4` | write annotated video to file |
| `--loop` | loop a file source forever (demo mode) |
| `--no-display` | headless (use with `--web` or `--record`) |
| `--table-corners` | calibration points, no GUI |
| `--width / --height` | requested camera resolution |

## What it tracks

| Metric | Method |
|---|---|
| Ball position/trail | color + motion blob → Kalman `[x,y,vx,vy]` |
| Hits per player | x-velocity sign reversal on each side of the net |
| Bounces/placement | y-velocity flip near surface → homography → mini-map |
| Speed | measured velocity × apparent-size scale (m/s) |
| Rallies | hit sequences; closed on ball loss or dead ball |

## Testing without a table

```bash
# generate a synthetic match with ground truth
.venv/bin/python tools/make_synthetic_video.py --out tools/sim.mp4 --frames 600

# run the tracker on it
.venv/bin/python main.py --source tools/sim.mp4 --ball-color orange \
    --table-corners "180,420,1100,420,1180,500,100,500" --record out.mp4

# score detection/events against ground truth
.venv/bin/python tools/evaluate.py
```

## Limitations

Single side camera → depth placement is approximate and speed is a planar
estimate (misses the toward/away-from-camera component). Tune thresholds in
`tt_tracker/config.py` for unusual lighting or ball colors.

## Layout

```
main.py                     entry point / pipeline loop
tt_tracker/
  config.py                 all tunables
  calibration.py            4-point homography + persistence
  detector.py               ball candidates (color + motion, size learning)
  tracker.py                Kalman filter, coasting, re-acquire
  events.py                 hits, bounces, rallies, speed, zones
  overlay.py                on-video annotations
  web.py                    MJPEG stream + stats page
tools/
  make_synthetic_video.py   synthetic match + ground truth
  evaluate.py               scores pipeline vs ground truth
```

## License

MIT — see [LICENSE](LICENSE).
