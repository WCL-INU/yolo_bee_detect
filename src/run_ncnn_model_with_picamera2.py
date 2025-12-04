import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import cv2
import dotenv
from flask import Flask, Response, abort, jsonify, render_template_string
from picamera2 import Picamera2
from ultralytics import YOLO

dotenv.load_dotenv()


@dataclass
class AppConfig:
    model_path: Path
    confidence_threshold: float = 0.5
    capture_interval: float = 0.5
    jpeg_quality: int = 80
    show_inference_time: bool = True
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False


def load_config() -> AppConfig:
    """Load runtime configuration from environment variables."""
    return AppConfig(
        model_path=Path(os.getenv("NCNN_MODEL_PATH", "./model.ncnn")),
        confidence_threshold=float(os.getenv("INFERENCE_CONFIDENCE_THRESHOLD", "0.5")),
        capture_interval=float(os.getenv("CAMERA_FRAME_INTERVAL", "0.03")),
        jpeg_quality=int(os.getenv("STREAM_JPEG_QUALITY", "80")),
        show_inference_time=os.getenv("SHOW_INFERENCE_TIME", "1") == "1",
        host=os.getenv("SERVER_HOST", "0.0.0.0"),
        port=int(os.getenv("SERVER_PORT", "8000")),
        debug=os.getenv("FLASK_DEBUG", "0") == "1",
    )


class CameraInferenceService:
    """Continuously grab frames from Picamera2 and run YOLO inference."""

    def __init__(self, model: YOLO, config: AppConfig):
        self.model = model
        self.config = config
        self.picam2 = Picamera2()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._condition = threading.Condition()
        self._latest_original: Optional[bytes] = None
        self._latest_detection: Optional[bytes] = None
        self._updated_at: float = 0.0
        self._frame_seq: int = 0
        # performance stats (ms)
        self._last_inference_ms: float = 0.0
        self._last_encode_ms: float = 0.0
        self._last_loop_ms: float = 0.0

    def start(self) -> None:
        camera_config = self.picam2.create_video_configuration(
            main={"size": (1640, 1232), "format": "RGB888"},
            buffer_count=32,
            sensor={"bit_depth": 8},
            controls={"FrameDurationLimits": (10000, 33333)},
        )
        self.picam2.configure(camera_config)
        self.picam2.start()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        try:
            self.picam2.stop()
        except Exception:
            pass
        try:
            self.picam2.close()
        except Exception:
            pass

    def get_last_updated(self) -> float:
        with self._condition:
            return self._updated_at

    def frame_generator(self, stream_type: str) -> Iterator[bytes]:
        target = "original" if stream_type == "original" else "detection"
        boundary = b"--frame"
        last_seq = -1
        while not self._stop.is_set():
            with self._condition:
                while last_seq == self._frame_seq and not self._stop.is_set():
                    self._condition.wait(timeout=1.0)
                if self._stop.is_set():
                    break
                frame = (
                    self._latest_original
                    if target == "original"
                    else self._latest_detection
                )
                seq = self._frame_seq
            if frame is None:
                continue
            last_seq = seq
            yield (boundary + b"\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

    def get_stats(self) -> dict:
        with self._condition:
            return {
                "inference_ms": self._last_inference_ms,
                "encode_ms": self._last_encode_ms,
                "loop_ms": self._last_loop_ms,
                "frame_seq": self._frame_seq,
            }

    def _loop(self) -> None:
        """Capture frames, store original/detection images, and publish latest paths."""
        jpeg_quality = int(max(10, min(95, self.config.jpeg_quality)))
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
        while not self._stop.is_set():
            loop_start = time.time()
            try:
                bgr = self.picam2.capture_array()
            except Exception as exc:
                print(f"Camera capture failed: {exc}")
                time.sleep(0.2)
                continue

            if bgr is None:
                time.sleep(0.05)
                continue

            try:
                infer_start = time.time()
                results = self.model(bgr, conf=self.config.confidence_threshold)
                infer_end = time.time()
                inference_ms = (infer_end - infer_start) * 1000.0
            except Exception as exc:
                print(f"Inference failed: {exc}")
                time.sleep(0.2)
                continue

            plotted = results[0].plot() if results else bgr

            encode_start = time.time()
            ok_orig, orig_buf = cv2.imencode(".jpg", bgr, encode_params)
            ok_boxed, boxed_buf = cv2.imencode(".jpg", plotted, encode_params)
            encode_end = time.time()
            encode_ms = (encode_end - encode_start) * 1000.0
            if not (ok_orig and ok_boxed):
                print("JPEG encoding failed, skipping frame.")
                continue

            with self._condition:
                self._latest_original = orig_buf.tobytes()
                self._latest_detection = boxed_buf.tobytes()
                self._updated_at = time.time()
                self._frame_seq += 1
                self._last_inference_ms = float(inference_ms)
                self._last_encode_ms = float(encode_ms)
                self._last_loop_ms = float((time.time() - loop_start) * 1000.0)
                self._condition.notify_all()

            if self.config.capture_interval > 0:
                time.sleep(self.config.capture_interval)


def create_app(service: CameraInferenceService) -> Flask:
    app = Flask(__name__)

    template = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Live NCNN YOLO Results</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 20px; background: #f3f3f3; }
    .wrapper { background: #fff; padding: 16px; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.1); }
    .images { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; margin-top: 12px; }
    img { width: 100%; border: 1px solid #ddd; border-radius: 6px; background: #fafafa; min-height: 160px; }
    h1 { margin-bottom: 4px; }
    p { margin: 0; }
  </style>
</head>
<body>
    <div class="wrapper">
    <h1>YOLO Camera Monitor</h1>
    <p>최근 업데이트: <span id="updated-text">대기 중...</span> — 추론: <span id="inference-text">--</span> — 루프(ms): <span id="loop-text">--</span></p>
    <div class="images">
      <div>
        <h3>원본</h3>
        <img id="img-original" src="{{ url_for('stream', stream_type='original') }}" alt="Original stream">
      </div>
      <div>
        <h3>분석 결과</h3>
        <img id="img-detection" src="{{ url_for('stream', stream_type='detection') }}" alt="Detection stream">
      </div>
    </div>
  </div>
  <script>
        const updatedText = document.getElementById("updated-text");
        const inferenceText = document.getElementById("inference-text");
        const loopText = document.getElementById("loop-text");

        async function refreshTimestamp() {
            try {
                const response = await fetch("{{ url_for('status') }}?t=" + Date.now());
                const data = await response.json();
                if (data.updated_at) {
                    const ts = new Date(data.updated_at * 1000);
                    updatedText.textContent = ts.toLocaleString();
                } else {
                    updatedText.textContent = "대기 중...";
                }
                if (data.inference_ms !== undefined) {
                    inferenceText.textContent = data.inference_ms.toFixed(1) + ' ms';
                }
                if (data.loop_ms !== undefined) {
                    loopText.textContent = data.loop_ms.toFixed(1) + ' ms';
                }
            } catch (err) {
                console.error(err);
            }
        }

        refreshTimestamp();
        setInterval(refreshTimestamp, 250);
  </script>
</body>
</html>
"""

    @app.route("/")
    def index():
        return render_template_string(template)

    @app.route("/stream/<string:stream_type>")
    def stream(stream_type: str):
        if stream_type not in {"original", "detection"}:
            abort(404)
        return Response(
            service.frame_generator(stream_type),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.route("/status")
    def status():
        stats = service.get_stats()
        data = {"updated_at": service.get_last_updated()}
        data.update(stats)
        return jsonify(data)

    return app


def main() -> None:
    config = load_config()
    model = YOLO(str(config.model_path))
    service = CameraInferenceService(model, config)
    service.start()

    app = create_app(service)
    print(f"Serving live stream at http://{config.host}:{config.port}")
    try:
        app.run(
            host=config.host, port=config.port, debug=config.debug, use_reloader=False
        )
    finally:
        service.stop()


if __name__ == "__main__":
    main()
