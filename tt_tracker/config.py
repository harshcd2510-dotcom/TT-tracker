"""Tunable configuration for the TT tracker."""

# ITTF table dimensions (meters)
TABLE_LENGTH = 2.74
TABLE_WIDTH = 1.525
NET_X = TABLE_LENGTH / 2.0  # net position along the length axis
BALL_DIAMETER_M = 0.040     # ITTF ball: 40 mm

# Ball detection
BALL_COLORS = {
    # name: (lower HSV, upper HSV)
    "orange": ((4, 110, 140), (28, 255, 255)),
    "white": ((0, 0, 175), (180, 80, 255)),
}
MOTION_DIFF_THRESH = 22          # grayscale abs-diff threshold for "moving" pixels
MIN_BALL_AREA = 12               # px^2 (candidate must be at least this)
MAX_BALL_AREA = 2500             # px^2 (reject large blobs — player, paddle)
MIN_CIRCULARITY = 0.45           # 4*pi*A/P^2 ; motion-blurred balls get lenient
MOTION_DILATE = 5                # px dilation of motion mask before ANDing color

# Tracker
KF_ACCEL_NOISE = 25000.0         # px/s^2 process noise (hits are violent accel)
KF_MEAS_NOISE = 4.0              # px measurement std-dev
GATE_BASE_PX = 55.0              # max distance from prediction to accept a detection
COAST_MAX_FRAMES = 10            # frames to coast on prediction before declaring LOST
REACQUIRE_AFTER = 5              # after this many coasted frames, accept a strong
                                 # candidate anywhere in frame (re-acquire)

# Events
MIN_HIT_SPEED_MPS = 0.8          # ignore x-velocity sign flips below this speed
HIT_CONFIRM_FRAMES = 2           # sign must hold this many frames to count as a hit
NET_MARGIN_M = 0.15              # hits must occur this far from the net (table m)
BOUNCE_TABLE_MARGIN_M = 0.25     # slack around table bounds for bounce detection
RALLY_END_FRAMES = 18            # tracker-lost frames before a rally is over
RALLY_END_STILL_MPS = 0.25       # tracked ball moving slower than this while low = point over
RALLY_END_STILL_FRAMES = 12

# Placement zones: split each table half into 3 along the length axis
ZONE_NAMES = ("short", "mid", "long")  # distance from the net

# Overlay
TRAIL_LEN = 42                   # trajectory trail points drawn
TOAST_FRAMES = 45                # how long an event label stays on screen
HUD_ALPHA = 0.55
