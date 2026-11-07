# ✋ HandTrack – Gesture-Controlled Glowing Ring

A real-time hand-tracking app built with **Python, OpenCV and MediaPipe** that turns hand gestures into interactive visual effects using only a webcam.

Bring both hands together in a claw pose and a glowing ring appears between them. Pull your hands apart and it grows. Let go slowly and it bursts into sparkles. Release it while your hands are moving and you throw it across the screen, where it bounces off the edges.

## 🎥 Demo

Watch the demo on LinkedIn: [add your post link here]

<!-- Optional: add a GIF by saving it as docs/demo.gif and uncommenting the next line -->
<!-- ![Demo](docs/demo.gif) -->

---

## ✨ Features

- **Glowing ring** that forms between both hands, follows them, and changes color with its size (or with hand speed)
- **Throw the ring** – release it while your hands are moving and it keeps that velocity and bounces off the screen edges
- **Shockwaves and sparkle trails** when the ring breaks or hits an edge
- **Punch effect** – punch with a fist for a shockwave, particle burst, screen shake, flash and a "POW!"
- **Thumbs up** shows `OKAY!` 👍
- **Wave** shows `BYE!` 👋
- **Prayer pose** (hands pressed together) shows `AYUBOWAN` 🙏, the Sri Lankan greeting

---

## 🎮 Gestures

| Gesture | What happens |
|---|---|
| Both hands in a **claw** pose | A glowing ring appears between the hands |
| Move hands apart / together | The ring grows / shrinks |
| Open hands **slowly** | The ring breaks into sparkles with a shockwave |
| Open hands **while moving fast** | The ring is **thrown** and bounces off the edges |
| **Fist** punched quickly (toward the camera or sideways) | Punch effect |
| **Thumbs up** (hold ~0.35 s) | Shows `OKAY!` |
| **Wave** an open hand side to side | Shows `BYE!` |
| **Hands pressed together**, fingers up (hold ~0.5 s) | Shows `AYUBOWAN` |

**Keyboard controls:** `q` quit · `s` save screenshot · `d` show/hide debug numbers · `c` switch ring color mode (distance / speed)

---

## 🧠 How It Works

1. **Hand tracking** – MediaPipe's Hand Landmarker finds **21 landmarks** on each hand in every webcam frame.
2. **Pose classification** – The bend angle at the middle joint of each finger is averaged. A high angle means an open hand, a medium angle means a claw, and a low angle means a closed fist.
3. **Palm-size normalization** – Distances are measured relative to palm size (wrist to middle knuckle), so gestures work at different distances from the camera.
4. **The ring** – The ring's center is the midpoint between the two palms and its radius is half the distance between them. Exponential smoothing makes it glide, and a short grace period stops one flickering frame from breaking it.
5. **Motion tracking** – Hands are matched between frames to estimate speed and palm growth. A fast release throws the ring using the hands' velocity; a fast fist (or one growing quickly toward the camera) triggers a punch.
6. **Message gestures** – Thumbs up is a thumb pointing up above curled fingers; a wave is counted from left/right direction changes of the fingertip; the prayer pose is two straight, upright hands whose palms and fingertips are close together. Poses must be held briefly (debouncing) to avoid false triggers.
7. **Rendering** – Effects are drawn on a separate layer, and a blurred half-size copy is added back for a fast glow. Messages are drawn on the final frame with a dark panel and outlined text so they stay readable.

---

## 🛠️ Tech Stack

- Python
- OpenCV
- MediaPipe (Tasks API – Hand Landmarker)
- NumPy

---

## 📦 Installation

```bash
# 1. Clone the repository
git clone https://github.com/Sathyangi/hand-gesture-ring.git
cd <hand-gesture-ring>

# 2. Create and activate a virtual environment (Windows)
python -m venv venv
venv\Scripts\activate

# macOS / Linux
# python3 -m venv venv
# source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

The first run automatically downloads the MediaPipe model file (`hand_landmarker.task`, about 8 MB), so you need an internet connection once.

---

## ▶️ Usage

```bash
python hand_ring.py
```

Tips for best results:
- Use good, even lighting and keep your whole hand in view.
- Stay about 40–60 cm from the camera.
- For the prayer pose, face your palms toward the camera, keep your fingers straight and pointing up, and press your hands together slowly.

---

## ⚙️ Configuration

Settings are at the top of `hand_ring.py`. Press `d` while running to see live numbers (finger angle `a`, hand speed `v`, palm growth `g`, and the prayer-pose distances) and tune them to your camera and hands.

| Setting | What it controls |
|---|---|
| `OPEN_ANGLE`, `FIST_ANGLE` | Angle ranges for open hand, claw and fist |
| `THROW_SPEED` | Hand speed needed on release to throw the ring |
| `PUNCH_SCALE_RATE`, `PUNCH_SPEED` | How fast a punch must be |
| `THUMBS_FINGER_ANGLE`, `THUMBS_HOLD` | Thumbs-up tolerance and hold time |
| `WAVE_AMPLITUDE`, `WAVE_REVERSALS` | How big and how many wave swings are needed |
| `NAMASTE_CENTER`, `NAMASTE_TIPS` | How close the hands must be for the prayer pose |
| `MESSAGES` | The text and color shown for each message gesture |

---

## 📁 Project Structure

```
.
├── hand_ring.py        # Main app: ring, throw, punch and message gestures
├── hand_common.py      # MediaPipe hand-tracking helper used by hand_ring.py
├── requirements.txt
└── README.md
```

---

## 🩹 Troubleshooting

| Problem | Fix |
|---|---|
| `No module named 'cv2'` | Activate the virtual environment (`venv\Scripts\activate`) and run `pip install -r requirements.txt` |
| Camera does not open | Close other apps that use the webcam (Zoom, Teams, etc.) |
| Model download fails | Delete any `hand_landmarker.task.part` file and run again, or download the Hand Landmarker model manually from the MediaPipe website and place it next to the scripts |
| Gestures don't trigger | Press `d` to see live values and adjust the thresholds listed above |
| Console says `ayubowan.png not found` | Harmless – the greeting is simply shown in English text |

---

## 🚀 Future Ideas

- Punch a flying ring to knock it in a new direction
- Sound effects (ring hum, hit sounds)
- Rings that bounce off each other
- Train a small classifier on landmark data instead of using angle thresholds
- A browser version using MediaPipe JS and Three.js

---

## 👤 Author

Built by **Subodha** as a computer vision learning project.

- LinkedIn: [(https://www.linkedin.com/in/subodhasathyangiofficial)]
- GitHub: [(https://github.com/Sathyangi)]

Feedback and suggestions are welcome – feel free to open an issue!

---

## 📄 License

This project is released under the MIT License. See the `LICENSE` file for details.
