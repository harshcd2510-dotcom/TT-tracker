"""Table calibration: map image pixels to real table coordinates via a homography.

The user clicks the 4 visible corners of the tabletop in the image, in this
order (as seen on screen):
    1. top-left     -> table coord (0, 0)              [far edge, left end]
    2. top-right    -> table coord (L, 0)              [far edge, right end]
    3. bottom-right -> table coord (L, W)              [near edge, right end]
    4. bottom-left  -> table coord (0, W)              [near edge, left end]
where L = TABLE_LENGTH (2.74 m) and W = TABLE_WIDTH (1.525 m). The length axis
must run left-to-right in the image (the net is vertical in the image).
"""

import json

import cv2
import numpy as np

from . import config

TABLE_POINTS_M = np.array(
    [
        [0.0, 0.0],
        [config.TABLE_LENGTH, 0.0],
        [config.TABLE_LENGTH, config.TABLE_WIDTH],
        [0.0, config.TABLE_WIDTH],
    ],
    dtype=np.float64,
)

WIN = "TT Tracker - calibrate"


class Calibrator:
    """Interactive 4-point picker on a frozen frame."""

    def __init__(self):
        self.points: list[tuple[int, int]] = []

    def _on_mouse(self, event, x, y, flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < 4:
            self.points.append((x, y))

    def run(self, frame) -> "Calibration | None":
        img = frame.copy()
        cv2.namedWindow(WIN)
        cv2.setMouseCallback(WIN, self._on_mouse)
        labels = [
            "1/4 top-left corner (far edge)",
            "2/4 top-right corner (far edge)",
            "3/4 bottom-right corner (near edge)",
            "4/4 bottom-left corner (near edge)",
        ]
        while True:
            disp = img.copy()
            idx = min(len(self.points), 3)
            if len(self.points) < 4:
                msg = f"Click {labels[idx]}  |  u=undo  Enter=done  Esc=cancel"
            else:
                msg = "Enter=accept  u=undo  Esc=cancel"
            cv2.rectangle(disp, (0, 0), (disp.shape[1], 34), (20, 20, 20), -1)
            cv2.putText(disp, msg, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 1, cv2.LINE_AA)
            for i, p in enumerate(self.points):
                cv2.circle(disp, p, 6, (0, 255, 255), 2)
                cv2.putText(disp, str(i + 1), (p[0] + 8, p[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            if len(self.points) == 4:
                cv2.polylines(disp, [np.array(self.points)], True,
                              (0, 255, 0), 2)
            cv2.imshow(WIN, disp)
            key = cv2.waitKey(30) & 0xFF
            if key == 27:  # Esc
                cv2.destroyWindow(WIN)
                return None
            if key in (13, 10) and len(self.points) == 4:  # Enter
                cv2.destroyWindow(WIN)
                return Calibration(np.array(self.points, dtype=np.float64))
            if key == ord("u") and self.points:
                self.points.pop()


class Calibration:
    """Image <-> table-plane mapping derived from 4 clicked corners."""

    def __init__(self, image_points: np.ndarray):
        self.image_points = np.asarray(image_points, dtype=np.float64)
        self.H_img_to_table = cv2.getPerspectiveTransform(
            self.image_points.astype(np.float32), TABLE_POINTS_M.astype(np.float32)
        )
        self.H_table_to_img = np.linalg.inv(self.H_img_to_table)
        self.image_poly = self.image_points.astype(np.int32)

    # ---- geometry ----------------------------------------------------------
    def image_to_table(self, x: float, y: float) -> np.ndarray:
        pt = np.array([x, y, 1.0])
        out = self.H_img_to_table @ pt
        return out[:2] / out[2]

    def table_to_image(self, x: float, y: float) -> tuple[int, int]:
        pt = np.array([x, y, 1.0])
        out = self.H_table_to_img @ pt
        p = out[:2] / out[2]
        return int(p[0]), int(p[1])

    def meters_per_pixel(self, x: float, y: float) -> float:
        """Approximate local image->table scale at image point (x, y)."""
        eps = 4.0
        p0 = self.image_to_table(x, y)
        px = self.image_to_table(x + eps, y)
        py = self.image_to_table(x, y + eps)
        sx = np.linalg.norm(px - p0) / eps
        sy = np.linalg.norm(py - p0) / eps
        return float((sx + sy) / 2.0)

    def on_table(self, table_xy, margin: float = 0.0) -> bool:
        x, y = table_xy
        return (-margin <= x <= config.TABLE_LENGTH + margin
                and -margin <= y <= config.TABLE_WIDTH + margin)

    def near_surface(self, img_x: float, img_y: float, radius: float) -> bool:
        """True if image point (img_x, img_y) is near the tabletop surface:
        vertically between the far and near table edges at that image x
        (with slack for the ball radius and bounce parallax)."""
        p = self.image_points  # TL, TR, BR, BL
        x_left = min(p[0][0], p[3][0])
        x_right = max(p[1][0], p[2][0])
        if not (x_left - 30 <= img_x <= x_right + 30):
            return False
        fx = np.clip((img_x - p[0][0]) / max(p[1][0] - p[0][0], 1), 0, 1)
        y_top = p[0][1] + fx * (p[1][1] - p[0][1])
        gx = np.clip((img_x - p[3][0]) / max(p[2][0] - p[3][0], 1), 0, 1)
        y_bot = p[3][1] + gx * (p[2][1] - p[3][1])
        return (y_top - 5 * radius) <= img_y <= (y_bot + 2 * radius)

    def net_image_x(self) -> int:
        """Image x-coordinate of the net (table x = L/2, mid width)."""
        return self.table_to_image(config.NET_X, config.TABLE_WIDTH / 2)[0]

    # ---- persistence -------------------------------------------------------
    def save(self, path: str):
        with open(path, "w") as f:
            json.dump({"image_points": self.image_points.tolist()}, f)

    @staticmethod
    def load(path: str) -> "Calibration":
        with open(path) as f:
            data = json.load(f)
        return Calibration(np.array(data["image_points"], dtype=np.float64))
