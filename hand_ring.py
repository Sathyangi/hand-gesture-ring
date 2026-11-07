import math
import os
import random
import time
from collections import deque

import cv2
import numpy as np

from hand_common import HandTracker, draw_skeleton

# ---------------- Settings (tune these) ----------------
OPEN_ANGLE = 150          # avg finger bend angle above this -> OPEN
FIST_ANGLE = 85           # avg finger bend angle below this -> FIST (between = CLAW)
RING_SMOOTH = 0.35        # 0..1, higher = ring follows hands faster
MIN_RING_R = 25           # ignore tiny rings (pixels)
GRACE = 0.15              # seconds the ring survives a flickering/lost pose

COLOR_MODE = "distance"   # "distance" or "speed"  (press c to toggle)

THROW_SPEED = 500         # hand speed (px/s) needed on release to throw the ring
THROW_BOOST = 1.2         # multiply hand speed when throwing
MAX_THROW_SPEED = 2200
BOUNCE_DAMP = 0.9         # speed kept after hitting an edge
DRAG = 0.25               # slow-down per second while flying
MAX_BOUNCES = 8
THROWN_LIFE = 10.0        # seconds before a flying ring breaks
MAX_THROWN = 4

PUNCH_SCALE_RATE = 1.5    # palm growth per second (punching toward the camera)
PUNCH_SPEED = 1300        # px/s (punching sideways)
PUNCH_COOLDOWN = 0.5      # seconds between punches
REQUIRE_FIST = True       # only count punches made with a fist
SHAKE_TIME = 0.25

# --- message gestures ---
MESSAGE_TIME = 2.2         # seconds a message stays on screen
THUMBS_FINGER_ANGLE = 110  # other fingers must be curled below this angle for a thumbs up
THUMBS_HOLD = 0.35         # seconds to hold a thumbs up
WAVE_WINDOW = 1.2          # seconds of motion examined for a wave
WAVE_AMPLITUDE = 0.35      # swing size needed (x palm size)
WAVE_REVERSALS = 3         # direction changes needed (3 = about two back-and-forth waves)
NAMASTE_HOLD = 0.5         # seconds to hold the prayer pose
NAMASTE_CENTER = 1.3       # max distance between the two palm centers (x palm size)
NAMASTE_TIPS = 0.9         # max distance between the two middle fingertips (x palm size)
NAMASTE_MIN_ANGLE = 125    # fingers must be fairly straight in the prayer pose
SINHALA_IMAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ayubowan.png")
MESSAGES = {               # text shown for each gesture (hue = glow color)
    "THUMBS_UP": {"text": "OKAY!", "hue": 60},
    "WAVE": {"text": "BYE!", "hue": 95},
    "NAMASTE": {"text": "AYUBOWAN", "hue": 25},
}

FONT = cv2.FONT_HERSHEY_DUPLEX


# ---------------- Small helpers ----------------
def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def angle(a, b, c):
    """Angle at point b (degrees) formed by a-b-c."""
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return math.degrees(math.acos(np.clip(cos, -1, 1)))


def hue_bgr(h):
    """OpenCV hue (0-179) -> BGR tuple at full brightness."""
    px = cv2.cvtColor(np.uint8([[[int(h) % 180, 255, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return int(px[0]), int(px[1]), int(px[2])


def lighten(c):
    return tuple(min(255, int(ch * 0.5 + 127)) for ch in c)


def classify(pts):
    """Classify one hand from its 21 pixel landmarks."""
    size = dist(pts[0], pts[9])                                   # palm size
    center = tuple(float(v) for v in np.mean([pts[i] for i in (0, 5, 9, 13, 17)], axis=0))

    joints = [(5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20)]
    avg_angle = float(np.mean([angle(pts[m], pts[p], pts[t]) for m, p, t in joints]))
    tips = [8, 12, 16, 20]
    tip_ratio = float(np.mean([dist(pts[t], center) for t in tips]) / (size + 1e-6))

    if avg_angle > OPEN_ANGLE:
        state = "OPEN"
    elif avg_angle < FIST_ANGLE:
        state = "FIST"
    else:
        state = "CLAW"
    return {"state": state, "center": center, "size": size,
            "angle": avg_angle, "tip_ratio": tip_ratio, "wrist": pts[0], "pts": pts}


# ---------------- Per-hand motion tracking (speed + palm growth) ----------------
prev_hands = []
next_hand_id = [1]


def track_hands(hands_info, now, w):
    """Match hands to the previous frame and add 'vel' (px/s) and 'rate' (palm growth/s)."""
    global prev_hands
    used = set()
    new_prev = []
    for hd in hands_info:
        best, bd = None, 1e9
        for i, ph in enumerate(prev_hands):
            if i in used:
                continue
            d = dist(hd["center"], ph["center"])
            if d < bd:
                best, bd = i, d
        vel, rate, hid = (0.0, 0.0), 0.0, None
        if best is not None and bd < 0.3 * w:
            used.add(best)
            ph = prev_hands[best]
            hid = ph["id"]
            dt = max(now - ph["t"], 1e-3)
            nv = ((hd["center"][0] - ph["center"][0]) / dt,
                  (hd["center"][1] - ph["center"][1]) / dt)
            nr = (hd["size"] - ph["size"]) / max(ph["size"], 1.0) / dt
            vel = (0.5 * nv[0] + 0.5 * ph["vel"][0], 0.5 * nv[1] + 0.5 * ph["vel"][1])
            rate = 0.5 * nr + 0.5 * ph["rate"]
        if hid is None:
            hid = next_hand_id[0]
            next_hand_id[0] += 1
        hd["vel"], hd["rate"], hd["id"] = vel, rate, hid
        new_prev.append({"center": hd["center"], "size": hd["size"],
                         "t": now, "vel": vel, "rate": rate, "id": hid})
    # remember hands that vanished for a moment (fast punches blur the camera image)
    for i, ph in enumerate(prev_hands):
        if i not in used and now - ph["t"] < 0.2:
            new_prev.append(ph)
    prev_hands = new_prev


# ---------------- Effects ----------------
class Particle:
    """A sparkle that moves, slows down, fades and leaves a short streak behind it."""
    __slots__ = ("x", "y", "vx", "vy", "life", "decay", "color")

    def __init__(self, x, y, vx, vy, lifetime, color):
        self.x, self.y, self.vx, self.vy = x, y, vx, vy
        self.life = 1.0
        self.decay = 1.0 / lifetime
        self.color = color

    def update(self, dt):
        self.x += self.vx * dt
        self.y += self.vy * dt
        k = math.exp(-1.8 * dt)
        self.vx *= k
        self.vy *= k
        self.life -= self.decay * dt
        return self.life > 0

    def draw(self, layer):
        col = tuple(int(c * self.life) for c in self.color)
        x, y = int(self.x), int(self.y)
        size = max(1, int(3 * self.life))
        cv2.line(layer, (int(self.x - self.vx * 0.04), int(self.y - self.vy * 0.04)),
                 (x, y), col, size)                       # the streak = sparkle trail
        cv2.circle(layer, (x, y), size, col, -1)


class Shockwave:
    """An expanding circle that thins out and fades."""

    def __init__(self, x, y, color, speed=500, decay=2.0, start_r=10, width=8):
        self.x, self.y, self.color = x, y, color
        self.r, self.speed, self.decay, self.width = start_r, speed, decay, width
        self.life = 1.0

    def update(self, dt):
        self.r += self.speed * dt
        self.life -= self.decay * dt
        return self.life > 0

    def draw(self, layer):
        col = tuple(int(c * self.life) for c in self.color)
        cv2.circle(layer, (int(self.x), int(self.y)), int(self.r), col,
                   max(1, int(self.width * self.life)))


particles, shockwaves, thrown, pow_texts = [], [], [], []
fx = {"shake_until": 0.0, "flash": 0.0, "last_punch": 0.0}


def burst(c, r, n, color, speed=(100, 450)):
    """Sparkles flying outward from a circle of radius r around c."""
    for _ in range(n):
        a = random.uniform(0, 2 * math.pi)
        sp = random.uniform(*speed)
        particles.append(Particle(c[0] + r * math.cos(a), c[1] + r * math.sin(a),
                                  math.cos(a) * sp, math.sin(a) * sp,
                                  random.uniform(0.6, 1.4),
                                  random.choice([color, lighten(color), (255, 255, 255)])))


def emit_trail(c, r, vel, color, n):
    """Sparkles shed from the ring edge, drifting backwards from the motion direction."""
    for _ in range(n):
        a = random.uniform(0, 2 * math.pi)
        particles.append(Particle(c[0] + r * math.cos(a), c[1] + r * math.sin(a),
                                  -vel[0] * 0.2 + random.uniform(-40, 40),
                                  -vel[1] * 0.2 + random.uniform(-40, 40),
                                  random.uniform(0.4, 0.9),
                                  random.choice([color, lighten(color)])))


def break_ring(c, r, color):
    burst(c, r, 90, color)
    shockwaves.append(Shockwave(c[0], c[1], color, speed=600, decay=1.6, start_r=r, width=10))


def draw_ring(layer, c, r, color, spin=2.0):
    t = time.time()
    c = (int(c[0]), int(c[1]))
    r = max(int(r + 3 * math.sin(t * 6)), 1)
    cv2.circle(layer, c, r, color, 8)                                   # colored band
    cv2.circle(layer, c, r, (255, 255, 255), 2)                         # bright core
    cv2.circle(layer, c, max(r - 14, 1), tuple(int(ch * 0.55) for ch in color), 2)
    for k in range(6):                                                  # orbiting dots
        a = t * spin + k * math.pi / 3
        cv2.circle(layer, (int(c[0] + r * math.cos(a)), int(c[1] + r * math.sin(a))),
                   5, (255, 255, 255), -1)


class ThrownRing:
    """A released ring that flies with the hand's velocity and bounces off the edges."""

    def __init__(self, x, y, r, vx, vy, color):
        self.x, self.y, self.r, self.vx, self.vy, self.color = x, y, r, vx, vy, color
        self.born = time.time()
        self.bounces = 0

    def update(self, dt, w, h):
        """Move and bounce. Returns True when the ring should break."""
        self.x += self.vx * dt
        self.y += self.vy * dt
        k = math.exp(-DRAG * dt)
        self.vx *= k
        self.vy *= k

        hit = None
        if self.x - self.r < 0:
            self.x, self.vx, hit = self.r, abs(self.vx) * BOUNCE_DAMP, (0, self.y)
        elif self.x + self.r > w:
            self.x, self.vx, hit = w - self.r, -abs(self.vx) * BOUNCE_DAMP, (w, self.y)
        if self.y - self.r < 0:
            self.y, self.vy, hit = self.r, abs(self.vy) * BOUNCE_DAMP, (self.x, 0)
        elif self.y + self.r > h:
            self.y, self.vy, hit = h - self.r, -abs(self.vy) * BOUNCE_DAMP, (self.x, h)
        if hit:
            self.bounces += 1
            burst(hit, 0, 18, self.color, speed=(80, 350))
            shockwaves.append(Shockwave(hit[0], hit[1], self.color, speed=300,
                                        decay=3.5, start_r=5, width=5))

        emit_trail((self.x, self.y), self.r, (self.vx, self.vy), self.color, 2)
        speed = math.hypot(self.vx, self.vy)
        return (time.time() - self.born > THROWN_LIFE
                or self.bounces > MAX_BOUNCES or speed < 60)


def release_ring(ring, w, h):
    """The pose ended: throw the ring if the hands were moving fast, else break it."""
    c, r, vel, col = ring["c"], ring["r"], ring["vel"], ring["color"]
    if r < MIN_RING_R:
        return
    speed = math.hypot(*vel)
    if speed > THROW_SPEED and len(thrown) < MAX_THROWN:
        vx, vy = vel[0] * THROW_BOOST, vel[1] * THROW_BOOST
        s = math.hypot(vx, vy)
        if s > MAX_THROW_SPEED:
            vx, vy = vx * MAX_THROW_SPEED / s, vy * MAX_THROW_SPEED / s
        thrown.append(ThrownRing(c[0], c[1], min(r, 0.3 * min(w, h)), vx, vy, col))
        shockwaves.append(Shockwave(c[0], c[1], col, speed=400, decay=3.0, start_r=r, width=6))
    else:
        break_ring(c, r, col)


def trigger_punch(c, now):
    col = hue_bgr(15)                                                  # orange
    shockwaves.append(Shockwave(c[0], c[1], (255, 255, 255), speed=1100, decay=2.8,
                                start_r=15, width=12))
    shockwaves.append(Shockwave(c[0], c[1], col, speed=700, decay=1.8, start_r=10, width=16))
    burst(c, 0, 80, col, speed=(300, 1200))
    fx["shake_until"] = now + SHAKE_TIME
    fx["flash"] = 1.0
    pow_texts.append([c[0], c[1], now, random.choice(["POW!", "BAM!", "BOOM!"])])


# ---------------- Message gestures: thumbs up, wave, namaste ----------------
def is_thumbs_up(hd):
    """Thumb pointing up, clearly above the other fingertips, with the four fingers curled."""
    pts, s = hd["pts"], hd["size"]
    if hd["angle"] > THUMBS_FINGER_ANGLE:
        return False
    tip, ip, mcp = pts[4], pts[3], pts[2]
    rise = mcp[1] - tip[1]                                    # y grows downward on screen
    highest_other = min(pts[i][1] for i in (8, 12, 16, 20))
    return (tip[1] < ip[1] < mcp[1]
            and rise > 0.5 * s
            and tip[1] < highest_other - 0.25 * s
            and abs(tip[0] - mcp[0]) < 0.8 * rise)            # mostly vertical thumb


def is_waving(hist, size):
    """Count left/right direction changes of the fingertip over the last moment."""
    if len(hist) < 8:
        return False
    xs = [x for _, x in hist]
    thr = WAVE_AMPLITUDE * size
    direction, ext, reversals = 0, xs[0], 0
    for x in xs[1:]:
        if direction == 0:
            if x - ext > thr:
                direction, ext = 1, x
            elif ext - x > thr:
                direction, ext = -1, x
        elif direction == 1:
            if x > ext:
                ext = x
            elif ext - x > thr:
                direction, ext, reversals = -1, x, reversals + 1
        else:
            if x < ext:
                ext = x
            elif x - ext > thr:
                direction, ext, reversals = 1, x, reversals + 1
    return reversals >= WAVE_REVERSALS


def namaste_metrics(hands_info):
    """Two straight, upright hands pressed together -> (is_namaste, center_ratio, tip_ratio)."""
    if len(hands_info) != 2:
        return False, None, None
    a, b = hands_info
    s = (a["size"] + b["size"]) / 2
    dc = dist(a["center"], b["center"]) / s
    dtip = dist(a["pts"][12], b["pts"][12]) / s
    upright = all(hd["pts"][0][1] - hd["pts"][12][1] > 1.2 * hd["size"] for hd in (a, b))
    straight = all(hd["angle"] > NAMASTE_MIN_ANGLE for hd in (a, b))
    return (dc < NAMASTE_CENTER and dtip < NAMASTE_TIPS and upright and straight), dc, dtip


banner = {"kind": None, "start": 0.0, "until": 0.0}
sinhala = {"img": None, "warned": False}


def load_sinhala():
    """Load the pre-rendered Sinhala text image (OpenCV cannot draw Sinhala letters itself)."""
    if sinhala["img"] is None and os.path.exists(SINHALA_IMAGE):
        img = cv2.imread(SINHALA_IMAGE, cv2.IMREAD_UNCHANGED)
        if img is not None and img.ndim == 3 and img.shape[2] == 4:
            sinhala["img"] = img
    if sinhala["img"] is None and not sinhala["warned"]:
        print("ayubowan.png not found next to the script - showing English text instead.")
        sinhala["warned"] = True
    return sinhala["img"]


def show_message(kind, loc, now):
    banner.update(kind=kind, start=now, until=now + MESSAGE_TIME)
    col = hue_bgr(MESSAGES[kind]["hue"])
    burst(loc, 0, 60, col, speed=(150, 600))
    shockwaves.append(Shockwave(loc[0], loc[1], col, speed=500, decay=2.0, start_r=10, width=8))


def blend(frame, overlay, alpha):
    """Mix an overlay into the frame in place (alpha 0..1)."""
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, dst=frame)


def draw_text(frame, text, cx, cy, scale, fade):
    """White text with a thick dark outline. Drawn on the frame itself (no glow) so it stays sharp."""
    thick = max(int(scale * 2), 2)
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, thick)
    org = (int(cx - tw / 2), int(cy + th / 2))
    overlay = frame.copy()
    cv2.putText(overlay, text, org, FONT, scale, (0, 0, 0), thick + 8)
    cv2.putText(overlay, text, org, FONT, scale, (255, 255, 255), thick)
    blend(frame, overlay, fade)


def draw_image(frame, img, cx, cy, width, fade):
    """Draw a BGRA image on the frame with a dark outline so it stays readable."""
    scale = width / img.shape[1]
    nw, nh = max(int(img.shape[1] * scale), 1), max(int(img.shape[0] * scale), 1)
    im = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    fh, fw = frame.shape[:2]
    x0, y0 = int(cx - nw / 2), int(cy - nh / 2)
    x1, y1, x2, y2 = max(x0, 0), max(y0, 0), min(x0 + nw, fw), min(y0 + nh, fh)
    if x1 >= x2 or y1 >= y2:
        return
    sub = im[y1 - y0:y2 - y0, x1 - x0:x2 - x0]
    a = sub[:, :, 3].astype(np.float32) / 255.0
    outline = cv2.dilate(a, np.ones((7, 7), np.uint8))
    roi = frame[y1:y2, x1:x2].astype(np.float32)
    roi *= 1 - (outline * fade * 0.9)[:, :, None]                 # dark outline
    t = (a * fade)[:, :, None]
    roi = roi * (1 - t) + sub[:, :, :3].astype(np.float32) * t    # the text itself
    frame[y1:y2, x1:x2] = np.clip(roi, 0, 255).astype(np.uint8)


def draw_banner(frame, now, w, h):
    """Message at the top: dark panel + colored border + crisp text. Pops in, holds, fades out."""
    if banner["kind"] is None or now >= banner["until"]:
        return
    kind = banner["kind"]
    age = now - banner["start"]
    fade = min(1.0, age / 0.15) * min(1.0, (banner["until"] - now) / 0.4)
    pop = 1 + 0.25 * max(0.0, 1 - age / 0.25)
    col = hue_bgr(MESSAGES[kind]["hue"])
    cx, cy, pad = w // 2, int(h * 0.2), 28

    img = load_sinhala() if kind == "NAMASTE" else None
    if img is not None:
        width = 0.42 * w * pop
        img_h = img.shape[0] * width / img.shape[1]
        latin_y = cy + img_h / 2 + 38
        left, right = cx - width / 2 - pad, cx + width / 2 + pad
        top, bottom = cy - img_h / 2 - pad, latin_y + 22 + pad
    else:
        text, scale = MESSAGES[kind]["text"], 3.0 * pop
        (tw, th), _ = cv2.getTextSize(text, FONT, scale, max(int(scale * 2), 2))
        left, right = cx - tw / 2 - pad * 1.5, cx + tw / 2 + pad * 1.5
        top, bottom = cy - th / 2 - pad, cy + th / 2 + pad

    p1 = (int(max(left, 0)), int(max(top, 0)))
    p2 = (int(min(right, w - 1)), int(min(bottom, h - 1)))
    overlay = frame.copy()
    cv2.rectangle(overlay, p1, p2, (18, 18, 18), -1)               # dark panel behind the text
    blend(frame, overlay, 0.78 * fade)
    overlay = frame.copy()
    cv2.rectangle(overlay, p1, p2, col, 4)                         # colored border
    blend(frame, overlay, fade)

    if img is not None:
        draw_image(frame, img, cx, cy, width, fade)
        draw_text(frame, "Ayubowan", cx, int(latin_y), 1.2, fade)
    else:
        draw_text(frame, text, cx, cy, scale, fade)


# ---------------- Main loop ----------------
def main():
    global COLOR_MODE
    tracker = HandTracker(max_hands=2)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open the webcam. Close other apps using it (Zoom, Teams) and retry.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    ring = {"active": False, "c": (0.0, 0.0), "r": 0.0, "vel": (0.0, 0.0),
            "last_claw": 0.0, "color": hue_bgr(100)}
    ct = 0.0                 # smoothed 0..1 value that drives the ring color
    debug = True
    prev_t = time.time()
    hold = {"thumbs": 0.0, "namaste": 0.0}          # how long each pose has been held
    latched = {"thumbs": False, "namaste": False}   # stops one pose firing again and again
    wave_hist = {}                                  # hand id -> recent fingertip x positions

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        now = time.time()
        dt = min(max(now - prev_t, 0.001), 0.1)
        prev_t = now
        layer = np.zeros_like(frame)                # glow layer: ring, sparkles, shockwaves

        # ---- detect + classify + track motion ----
        hands_info = []
        for hand in tracker.process(frame):
            pts = [(l.x * w, l.y * h) for l in hand]
            info = classify(pts)
            hands_info.append(info)
            draw_skeleton(frame, hand)
        track_hands(hands_info, now, w)

        for hd in hands_info:
            if debug:
                label = (f"{hd['state']} a={hd['angle']:.0f} "
                         f"v={math.hypot(*hd['vel']):.0f} g={hd['rate']:.1f}")
            else:
                label = hd["state"]
            cv2.putText(frame, label, (int(hd["wrist"][0]) - 50, int(hd["wrist"][1]) + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # ---- punch detection ----
        for hd in hands_info:
            fast = hd["rate"] > PUNCH_SCALE_RATE or math.hypot(*hd["vel"]) > PUNCH_SPEED
            if ((not REQUIRE_FIST or hd["state"] == "FIST") and fast
                    and not is_thumbs_up(hd)
                    and now - fx["last_punch"] > PUNCH_COOLDOWN):
                trigger_punch(hd["center"], now)
                fx["last_punch"] = now

        # ---- message gestures: namaste, thumbs up, wave ----
        busy = now < banner["until"]
        thumb_hand = next((hd for hd in hands_info if is_thumbs_up(hd)), None)
        is_nam, nam_c, nam_t = namaste_metrics(hands_info)

        wave_hand, seen = None, set()
        for hd in hands_info:
            seen.add(hd["id"])
            hist = wave_hist.setdefault(hd["id"], deque())
            if hd["angle"] > 130:                       # only an open hand can wave
                hist.append((now, hd["pts"][12][0]))
            else:
                hist.clear()
            while hist and now - hist[0][0] > WAVE_WINDOW:
                hist.popleft()
            if wave_hand is None and is_waving(hist, hd["size"]):
                wave_hand = hd
        for k in list(wave_hist):
            if k not in seen:
                del wave_hist[k]

        for key, active in (("thumbs", thumb_hand is not None), ("namaste", is_nam)):
            hold[key] = min(hold[key] + dt, 1.5) if active else max(0.0, hold[key] - 2 * dt)
            if hold[key] == 0.0:
                latched[key] = False

        if not busy:
            if is_nam and hold["namaste"] >= NAMASTE_HOLD and not latched["namaste"]:
                a, b = hands_info
                show_message("NAMASTE", ((a["center"][0] + b["center"][0]) / 2,
                                         (a["center"][1] + b["center"][1]) / 2), now)
                latched["namaste"] = True
            elif thumb_hand and hold["thumbs"] >= THUMBS_HOLD and not latched["thumbs"]:
                show_message("THUMBS_UP", thumb_hand["center"], now)
                latched["thumbs"] = True
            elif wave_hand:
                show_message("WAVE", wave_hand["center"], now)
                wave_hist[wave_hand["id"]].clear()

        # ---- ring: both hands in CLAW ----
        claws = [hd for hd in hands_info if hd["state"] == "CLAW"]
        target = None
        if len(claws) >= 2:
            a, b = claws[0], claws[1]
            target = (((a["center"][0] + b["center"][0]) / 2,
                       (a["center"][1] + b["center"][1]) / 2),
                      dist(a["center"], b["center"]) / 2)

        if target:
            tc, tr = target
            if not ring["active"]:
                ring.update(active=True, c=tc, r=tr, vel=(0.0, 0.0))
            else:
                ring["c"] = (ring["c"][0] + (tc[0] - ring["c"][0]) * RING_SMOOTH,
                             ring["c"][1] + (tc[1] - ring["c"][1]) * RING_SMOOTH)
                ring["r"] += (tr - ring["r"]) * RING_SMOOTH
            ring["last_claw"] = now

            # ring velocity = average hand velocity (stays correct when 2 hands become 1)
            mvx = sum(c["vel"][0] for c in claws) / len(claws)
            mvy = sum(c["vel"][1] for c in claws) / len(claws)
            rv = ring["vel"]
            ring["vel"] = (rv[0] + (mvx - rv[0]) * 0.5, rv[1] + (mvy - rv[1]) * 0.5)
            spd = math.hypot(*ring["vel"])

            # color from ring size (distance) or hand speed
            if COLOR_MODE == "speed":
                t = min(spd / 1500.0, 1.0)
                ct += (t - ct) * 0.2
                ring["color"] = hue_bgr(100 - 100 * ct)        # cyan -> green -> yellow -> red
            else:
                t = min(max((ring["r"] - 40) / (0.35 * w), 0.0), 1.0)
                ct += (t - ct) * 0.2
                ring["color"] = hue_bgr(100 + 75 * ct)         # blue -> purple -> pink

            if spd > 120:                                       # sparkle trail while moving
                emit_trail(ring["c"], ring["r"], ring["vel"], ring["color"],
                           min(8, 1 + int(spd / 200)))
        elif ring["active"] and now - ring["last_claw"] > GRACE:
            release_ring(ring, w, h)
            ring["active"] = False
            ring["vel"] = (0.0, 0.0)

        if ring["active"] and ring["r"] > MIN_RING_R:
            draw_ring(layer, ring["c"], ring["r"], ring["color"])

        # ---- thrown rings ----
        for tr in thrown[:]:
            if tr.update(dt, w, h):
                break_ring((tr.x, tr.y), tr.r, tr.color)
                thrown.remove(tr)
            else:
                draw_ring(layer, (tr.x, tr.y), tr.r, tr.color, spin=4.0)

        # ---- sparkles + shockwaves ----
        particles[:] = [p for p in particles if p.update(dt)]
        if len(particles) > 1500:
            del particles[:len(particles) - 1500]
        for p in particles:
            p.draw(layer)
        shockwaves[:] = [s for s in shockwaves if s.update(dt)]
        for s in shockwaves:
            s.draw(layer)

        # ---- composite: add layer, then a blurred half-size copy as glow ----
        small = cv2.resize(layer, (w // 2, h // 2))
        glow = cv2.resize(cv2.GaussianBlur(small, (0, 0), 9), (w, h))
        frame = cv2.add(frame, layer)
        frame = cv2.add(frame, glow)
        frame = cv2.add(frame, glow)

        # ---- punch screen effects: shake, flash, POW! text ----
        if now < fx["shake_until"]:
            amp = 16 * (fx["shake_until"] - now) / SHAKE_TIME
            M = np.float32([[1, 0, random.uniform(-amp, amp)],
                            [0, 1, random.uniform(-amp, amp)]])
            frame = cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        if fx["flash"] > 0.02:
            v = int(110 * fx["flash"])
            frame = cv2.add(frame, (v, v, v, 0))
            fx["flash"] *= math.exp(-9 * dt)
        draw_banner(frame, now, w, h)
        for tx in pow_texts[:]:
            age = now - tx[2]
            if age > 0.7:
                pow_texts.remove(tx)
                continue
            life = 1 - age / 0.7
            scale = 1.5 + 2.0 * age
            (tw, th), _ = cv2.getTextSize(tx[3], FONT, scale, 6)
            x = int(np.clip(tx[0] - tw / 2, 5, max(w - tw - 5, 5)))
            y = int(np.clip(tx[1] - 60, th + 5, h - 5))
            overlay = frame.copy()
            cv2.putText(overlay, tx[3], (x, y), FONT, scale, (0, 0, 0), 12)
            cv2.putText(overlay, tx[3], (x, y), FONT, scale, (0, 200, 255), 6)
            frame = cv2.addWeighted(overlay, life, frame, 1 - life, 0)

        cv2.putText(frame, "2 hands CLAW = ring | release moving = THROW | punch = POW | thumbs up | wave | namaste",
                    (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"color: {COLOR_MODE} (c) | s:save  d:debug  q:quit",
                    (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        if debug and nam_c is not None:
            cv2.putText(frame, f"namaste: centers={nam_c:.2f} (<{NAMASTE_CENTER})  "
                               f"tips={nam_t:.2f} (<{NAMASTE_TIPS})",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.imshow("Hand Skeleton Tracking", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            cv2.imwrite(f"ring_{int(time.time())}.png", frame)
        elif key == ord("d"):
            debug = not debug
        elif key == ord("c"):
            COLOR_MODE = "speed" if COLOR_MODE == "distance" else "distance"

    cap.release()
    cv2.destroyAllWindows()
    tracker.close()


if __name__ == "__main__":
    main()
