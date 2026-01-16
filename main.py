from __future__ import annotations

import signal
import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np

from camera.webcam import WebcamCapture
from hand_tracking.detector import HandDetection, HandDetector
from hand_tracking.gestures import GestureClassifier
from rendering.renderer import ARRenderer
from ui.hud import HUDOverlay
from utils.config import GestureConfig, HUDConfig, RenderConfig, SharedState, VisionConfig
from utils.fps import FPSCounter


def _to_ndc(anchor_norm: Tuple[float, float, float]) -> Tuple[float, float, float]:
    x = anchor_norm[0] * 2.0 - 1.0
    y = 1.0 - anchor_norm[1] * 2.0
    z = np.clip(anchor_norm[2], -1.0, 1.0)
    return x, y, z


def _vision_loop(
    webcam: WebcamCapture,
    detector: HandDetector,
    classifier: GestureClassifier,
    hud: HUDOverlay,
    shared_state: SharedState,
    stop_event: threading.Event,
) -> None:
    fps = FPSCounter()
    while not stop_event.is_set() and not shared_state.shutdown_requested():
        frame = webcam.get_frame()
        if frame is None:
            time.sleep(0.002)
            continue
        detection = detector.process(frame)
        anchor = detection.anchor if detection else np.array([0.5, 0.6, 0.0], dtype=np.float32)
        shape_label = classifier.classify(detection)
        fps_value = fps.tick()
        hud_frame = hud.apply(frame.copy(), shape_label, fps_value, (float(anchor[0]), float(anchor[1])))
        rgb = cv2.cvtColor(hud_frame, cv2.COLOR_BGR2RGB)
        anchor_ndc = _to_ndc((float(anchor[0]), float(anchor[1]), float(anchor[2])))
        shared_state.update(rgb, anchor_ndc, shape_label)
    stop_event.set()


def run() -> None:
    vision_cfg = VisionConfig()
    gesture_cfg = GestureConfig()
    render_cfg = RenderConfig(window_width=vision_cfg.width, window_height=vision_cfg.height)
    hud_cfg = HUDConfig()

    shared_state = SharedState()
    webcam = WebcamCapture(vision_cfg.width, vision_cfg.height)
    detector = HandDetector(vision_cfg)
    classifier = GestureClassifier(gesture_cfg)
    hud = HUDOverlay(hud_cfg)
    vision_stop = threading.Event()

    webcam.start()
    renderer = ARRenderer(render_cfg, shared_state)
    renderer.start()

    worker = threading.Thread(
        target=_vision_loop,
        args=(webcam, detector, classifier, hud, shared_state, vision_stop),
        daemon=True,
    )
    worker.start()

    def _handle_exit(signum, frame):  # pragma: no cover - signal handling
        shared_state.request_shutdown()
        vision_stop.set()

    signal.signal(signal.SIGINT, _handle_exit)
    signal.signal(signal.SIGTERM, _handle_exit)

    try:
        while not shared_state.shutdown_requested():
            time.sleep(0.1)
    finally:
        vision_stop.set()
        shared_state.request_shutdown()
        renderer.stop()
        worker.join(timeout=1.0)
        webcam.stop()
        detector.close()


if __name__ == "__main__":
    run()
