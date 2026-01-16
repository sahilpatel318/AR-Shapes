# AR Shapes

A production-ready, gesture-driven augmented reality experience that fuses MediaPipe hand tracking, OpenCV video capture, and a ModernGL-style particle renderer powered by PyOpenGL and GLFW. Perform real-time gestures to morph luminous particle sculptures that float beside you, wrapped in a minimalist HUD inspired by sci-fi interfaces.

## Features
- Live 1280×720 webcam capture with threaded ingestion to minimize latency.
- MediaPipe Hands inference with exponential landmark smoothing and gesture stability logic.
- Gesture mapping: thumbs-up → sphere, pinch → torus, open palm → cube.
- Particle-based renderer with 1.8k billboarding sprites, additive glow, and smooth morph transitions.
- Futuristic HUD overlay showing current shape and FPS, plus AR-style corner brackets anchored to the detected hand.
- Separate rendering and vision threads with shared state synchronization for ≥24 FPS stability.
- Graceful shutdown via `ESC` or `Ctrl+C`.

## Installation
1. Install Python 3.10 or newer.
2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
   > On Windows you may need the latest GPU drivers plus the Microsoft Visual C++ Redistributable for GLFW/PyOpenGL.

## Running the Experience
1. Connect a webcam and ensure no other app is using it.
2. From the project root run:
   ```bash
   python main.py
   ```
3. A GLFW window titled **AR Shapes** appears within a few seconds once the first frame is processed. Press `ESC` or `Ctrl+C` to exit cleanly.

## Gesture Vocabulary
| Gesture | Action |
| --- | --- |
| Thumbs up (other fingers curled) | Morph particles into a glowing sphere |
| Thumb-index pinch | Generate a torus field |
| Open palm | Expand into a cube lattice |

Gestures require ~8 stable frames before switching and include a cooldown to prevent flicker.

## Architecture Overview
```
main.py
├─ camera/webcam.py          # Threaded OpenCV capture
├─ hand_tracking/
│  ├─ detector.py            # MediaPipe inference + landmark smoothing
│  └─ gestures.py            # Rule-based classifier + stability checks
├─ rendering/
│  ├─ renderer.py            # GLFW window + OpenGL pipeline
│  ├─ particles.py           # Particle physics + morph targets
│  └─ shapes.py              # Shape sampling utilities
├─ ui/hud.py                 # Sci-fi HUD overlays
└─ utils/
   ├─ config.py              # Dataclasses + shared state helper
   ├─ fps.py                 # Moving FPS tracker
   └─ smoothing.py           # Reusable smoothing utilities
```
- **Vision thread**: captures frames, runs inference, draws HUD, and pushes RGB frames + anchor data into `SharedState`.
- **Rendering thread**: consumes the latest frame, uploads it as a textured quad, and renders particles using GLSL shaders in `shaders/`.
- **Shared state**: small synchronized buffer to avoid blocking either subsystem.

## Configuration
Tune sensitivities and visual parameters via the dataclasses in `utils/config.py` (e.g., particle counts, smoothing alphas, gesture thresholds). The defaults target a balance between responsiveness and cinematic motion.

## Troubleshooting
- **Black window**: ensure your GPU/driver supports OpenGL 3.3+ and that remote desktop is not forcing a fallback renderer.
- **Low FPS**: reduce `particle_count` or window resolution in `RenderConfig`.
- **Gesture misfires**: tweak thresholds in `GestureConfig` or improve ambient lighting for clearer landmarks.

Enjoy crafting futuristic AR sculptures with nothing more than hand gestures.
