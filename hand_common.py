"""
Shared helper for hand_draw.py and hand_ring.py.
Uses the MediaPipe *Tasks* API (works with the newest MediaPipe / Python 3.14).
The model file (hand_landmarker.task) is downloaded automatically on first run.
"""
import os
import time
import urllib.request

import cv2
import mediapipe as mp

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/"
             "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")

# pairs of landmark indices that form the hand skeleton
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),            # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),            # index
    (5, 9), (9, 10), (10, 11), (11, 12),       # middle
    (9, 13), (13, 14), (14, 15), (15, 16),     # ring
    (13, 17), (17, 18), (18, 19), (19, 20),    # pinky
    (0, 17),                                   # palm edge
]


def _ensure_model():
    if os.path.exists(MODEL_PATH):
        return
    print("Downloading hand_landmarker.task (one time, ~8 MB)...")
    tmp = MODEL_PATH + ".part"
    urllib.request.urlretrieve(MODEL_URL, tmp)
    os.replace(tmp, MODEL_PATH)
    print("Download finished.")


class HandTracker:
    """process(frame_bgr) -> list of hands; each hand is a list of 21 landmarks
    with .x and .y (0..1), the same shape the old mp.solutions API gave."""

    def __init__(self, max_hands=1, detect_conf=0.7, track_conf=0.6):
        _ensure_model()
        vision = mp.tasks.vision
        options = vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=detect_conf,
            min_tracking_confidence=track_conf,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self._last_ts = 0

    def process(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = int(time.time() * 1000)
        if ts <= self._last_ts:              # timestamps must always increase
            ts = self._last_ts + 1
        self._last_ts = ts
        result = self.landmarker.detect_for_video(mp_image, ts)
        return result.hand_landmarks

    def close(self):
        self.landmarker.close()


def draw_skeleton(frame, lm, point_color=(255, 255, 0), line_color=(200, 200, 60)):
    """Draw the 21 points and their connecting lines on the frame."""
    h, w = frame.shape[:2]
    pts = [(int(p.x * w), int(p.y * h)) for p in lm]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], line_color, 2)
    for p in pts:
        cv2.circle(frame, p, 4, point_color, -1)
