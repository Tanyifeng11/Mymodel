"""E31：直接在完整估计前景滑窗；不生成 region/panel 标签。"""

import cv2
import numpy as np


WINDOWS = ((48, 48), (64, 64), (80, 80), (96, 96), (64, 96), (96, 64))


def candidate_boxes(mask, stride=16):
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    h, w = mask.shape
    boxes = []
    for cw, ch in WINDOWS:
        for y in range(0, h - ch + 1, stride):
            for x in range(0, w - cw + 1, stride):
                if mask[y:y+ch, x:x+cw].mean() < .95:
                    continue
                # 使用整个窗口的最小内距；比只检查中心更严格。
                if distance[y:y+ch, x:x+cw].min() < 3:
                    continue
                boxes.append((x, y, x+cw, y+ch))
    return boxes, distance


def patch_field(field, box, image_size=(256, 256)):
    x0, y0, x1, y1 = box
    h, w = field.shape[:2]
    iw, ih = image_size
    return field[int(y0*h/ih):max(int(y1*h/ih), int(y0*h/ih)+1),
                 int(x0*w/iw):max(int(x1*w/iw), int(x0*w/iw)+1)]


def unit(value):
    return value / max(float(np.linalg.norm(value)), 1e-8)
