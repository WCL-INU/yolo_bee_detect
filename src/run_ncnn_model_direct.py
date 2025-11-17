import atexit
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import dotenv
import numpy as np
from flask import Flask, render_template_string, send_from_directory

try:
    import ncnn
except ImportError as exc:
    raise SystemExit("ncnn python package is required. Install it first.") from exc

dotenv.load_dotenv()


@dataclass
class AppConfig:
    model_path: Path
    input_data_path: Path
    output_data_path: Path
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    input_size: int = 640
    num_threads: int = 4
    use_vulkan: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False


def load_config() -> AppConfig:
    return AppConfig(
        model_path=Path(os.getenv("NCNN_MODEL_PATH", "./model")),
        input_data_path=Path(os.getenv("INPUT_DATA_PATH", "./input_data")),
        output_data_path=Path(os.getenv("OUTPUT_DATA_PATH", "./output_data")),
        confidence_threshold=float(os.getenv("INFERENCE_CONFIDENCE_THRESHOLD", "0.5")),
        iou_threshold=float(os.getenv("INFERENCE_IOU_THRESHOLD", "0.7")),
        input_size=int(os.getenv("NCNN_INPUT_SIZE", "640")),
        num_threads=int(os.getenv("NCNN_NUM_THREADS", "4")),
        use_vulkan=os.getenv("NCNN_USE_VULKAN", "0") == "1",
        host=os.getenv("SERVER_HOST", "0.0.0.0"),
        port=int(os.getenv("SERVER_PORT", "8000")),
        debug=os.getenv("FLASK_DEBUG", "0") == "1",
    )


def resolve_model_files(base: Path) -> Tuple[Path, Path]:
    if base.is_dir():
        params = sorted(base.glob("*.param"))
        bins = sorted(base.glob("*.bin"))
        if params and bins:
            return params[0], bins[0]
    else:
        stem = base.with_suffix("") if base.suffix else base
        param = stem.with_suffix(".param")
        bin_f = stem.with_suffix(".bin")
        if param.exists() and bin_f.exists():
            return param, bin_f
        param = Path(str(stem) + ".param")
        bin_f = Path(str(stem) + ".bin")
        if param.exists() and bin_f.exists():
            return param, bin_f
    raise FileNotFoundError("NCNN param/bin not found. Set NCNN_MODEL_PATH to the model stem or directory.")


def collect_input_files(input_dir: Path) -> List[Path]:
    return sorted(
        f for f in input_dir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def letterbox(image: np.ndarray, size: int):
    h, w = image.shape[:2]
    r = min(size / w, size / h)
    new_w, new_h = int(round(w * r)), int(round(h * r))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_w, pad_h = size - new_w, size - new_h
    pad_left, pad_right = pad_w // 2, pad_w - pad_w // 2
    pad_top, pad_bottom = pad_h // 2, pad_h - pad_h // 2
    padded = cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    return padded, r, (pad_left, pad_top)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_th: float) -> List[int]:
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep: List[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[np.where(iou <= iou_th)[0] + 1]
    return keep


class NcnnYolo:
    def __init__(self, param: Path, bin_f: Path, cfg: AppConfig):
        self.input_size = cfg.input_size
        self.net = ncnn.Net()
        self.net.opt.num_threads = cfg.num_threads
        if cfg.use_vulkan and hasattr(ncnn, "create_gpu_instance"):
            if getattr(ncnn, "_gpu_created", False) is False:
                ncnn.create_gpu_instance()
                atexit.register(ncnn.destroy_gpu_instance)
                ncnn._gpu_created = True  # type: ignore
            self.net.opt.use_vulkan_compute = True
        self.net.load_param(str(param))
        self.net.load_model(str(bin_f))

    def infer(self, image: np.ndarray, conf_th: float, iou_th: float):
        padded, scale, (pad_x, pad_y) = letterbox(image, self.input_size)
        mat = ncnn.Mat.from_pixels(
            padded, ncnn.Mat.PixelType.PIXEL_BGR2RGB, padded.shape[1], padded.shape[0]
        )
        mat.substract_mean_normalize([0, 0, 0], [1 / 255, 1 / 255, 1 / 255])
        with self.net.create_extractor() as ex:
            ex.input("in0", mat)
            ret, out = ex.extract("out0")
            if ret != 0:
                return []
        # print(out.dims, out.w, out.h, out.c)
        # print(np.array(out).shape)
        data = np.array(out).reshape(out.h, out.w).T
        if data.shape[1] <= 4:
            return []
        boxes_xywh = data[:, :4]
        scores = data[:, 4:]
        cls_scores = scores.max(axis=1)
        cls_ids = scores.argmax(axis=1)
        mask = cls_scores > conf_th
        boxes_xywh = boxes_xywh[mask]
        cls_scores = cls_scores[mask]
        cls_ids = cls_ids[mask]
        if len(boxes_xywh) == 0:
            return []
        boxes = np.empty_like(boxes_xywh)
        boxes[:, 0] = (boxes_xywh[:, 0] - boxes_xywh[:, 2] * 0.5 - pad_x) / scale
        boxes[:, 1] = (boxes_xywh[:, 1] - boxes_xywh[:, 3] * 0.5 - pad_y) / scale
        boxes[:, 2] = (boxes_xywh[:, 0] + boxes_xywh[:, 2] * 0.5 - pad_x) / scale
        boxes[:, 3] = (boxes_xywh[:, 1] + boxes_xywh[:, 3] * 0.5 - pad_y) / scale
        boxes[:, 0::2] = boxes[:, 0::2].clip(0, image.shape[1] - 1)
        boxes[:, 1::2] = boxes[:, 1::2].clip(0, image.shape[0] - 1)
        keep = nms(boxes, cls_scores, iou_th)
        return [(boxes[i], cls_scores[i], cls_ids[i]) for i in keep]


def draw_detections(image: np.ndarray, detections) -> np.ndarray:
    boxed = image.copy()
    for box, score, cls_id in detections:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(boxed, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # cv2.putText(
        #     boxed,
        #     f"{cls_id}:{score:.2f}",
        #     (x1, max(y1 - 5, 0)),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.5,
        #     (0, 0, 0),
        #     3,
        #     cv2.LINE_AA,
        # )
        # cv2.putText(
        #     boxed,
        #     f"{cls_id}:{score:.2f}",
        #     (x1, max(y1 - 5, 0)),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.5,
        #     (255, 255, 255),
        #     1,
        #     cv2.LINE_AA,
        # )
    return boxed


def run_inference(detector: NcnnYolo, images: List[Path], cfg: AppConfig):
    pairs: List[Tuple[str, str]] = []
    for image_path in images:
        img = cv2.imread(str(image_path))
        if img is None:
            continue
        dets = detector.infer(img, cfg.confidence_threshold, cfg.iou_threshold)
        plotted = draw_detections(img, dets)
        orig_name = image_path.name
        boxed_name = f"{image_path.stem}_boxed{image_path.suffix}"
        shutil.copyfile(image_path, cfg.output_data_path / orig_name)
        cv2.imwrite(str(cfg.output_data_path / boxed_name), plotted)
        pairs.append((orig_name, boxed_name))
        print(f"Saved results for {orig_name}")
    return pairs


def create_app(output_dir: Path, pairs: List[Tuple[str, str]]) -> Flask:
    app = Flask(__name__)
    template = """<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>NCNN YOLO Results</title></head>
<body>
  <h1>Results</h1>
  <div>
    {% for original, boxed in pairs %}
      <div style="margin-bottom:16px;">
        <div>{{ original }}</div>
        <img src="{{ url_for('static_file', filename=original) }}" width="320">
        <img src="{{ url_for('static_file', filename=boxed) }}" width="320">
      </div>
    {% endfor %}
  </div>
</body>
</html>
"""

    @app.route("/")
    def index():
        return render_template_string(template, pairs=pairs)

    @app.route("/images/<path:filename>")
    def static_file(filename: str):
        return send_from_directory(output_dir, filename)

    return app


def main() -> None:
    cfg = load_config()
    param, bin_f = resolve_model_files(cfg.model_path)
    detector = NcnnYolo(param, bin_f, cfg)
    images = collect_input_files(cfg.input_data_path)
    if not images:
        print(f"No image files found in {cfg.input_data_path}")
        return
    ensure_output_dir(cfg.output_data_path)
    pairs = run_inference(detector, images, cfg)
    if not pairs:
        print("No results to display.")
        return
    app = create_app(cfg.output_data_path, pairs)
    print(f"Serving results at http://{cfg.host}:{cfg.port}")
    app.run(host=cfg.host, port=cfg.port, debug=cfg.debug)


if __name__ == "__main__":
    main()
