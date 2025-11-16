import os
import shutil
from dataclasses import dataclass
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import List, Tuple

import cv2
import dotenv
from ultralytics import YOLO

dotenv.load_dotenv()


@dataclass
class AppConfig:
    model_path: Path
    input_data_path: Path
    output_data_path: Path
    host: str = "0.0.0.0"
    port: int = 8000


def load_config() -> AppConfig:
    """Load runtime configuration from environment variables."""
    return AppConfig(
        model_path=Path(os.getenv("MODEL_PATH", "./model.pth")),
        input_data_path=Path(os.getenv("INPUT_DATA_PATH", "./input_data")),
        output_data_path=Path(os.getenv("OUTPUT_DATA_PATH", "./output_data")),
        host=os.getenv("SERVER_HOST", "0.0.0.0"),
        port=int(os.getenv("SERVER_PORT", "8000")),
    )


def collect_input_files(input_dir: Path) -> List[Path]:
    """Return sorted image files to process."""
    return sorted(file for file in input_dir.iterdir() if file.suffix.lower() in {".jpg", ".jpeg", ".png"})


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def run_inference(model: YOLO, images: List[Path], output_dir: Path) -> List[Tuple[str, str]]:
    """Run inference and save original and boxed images; returns list of (orig, boxed) file names."""
    saved_pairs: List[Tuple[str, str]] = []

    for image_path in images:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Skipping unreadable file: {image_path.name}")
            continue

        results = model(image)
        plotted = results[0].plot()  # BGR array with boxes and labels

        orig_name = image_path.name
        boxed_name = f"{image_path.stem}_boxed{image_path.suffix}"

        orig_target = output_dir / orig_name
        boxed_target = output_dir / boxed_name

        shutil.copyfile(image_path, orig_target)
        cv2.imwrite(str(boxed_target), plotted)

        saved_pairs.append((orig_name, boxed_name))
        print(f"Saved results for {orig_name} -> {boxed_name}")

    return saved_pairs


def write_gallery_page(output_dir: Path, pairs: List[Tuple[str, str]]) -> None:
    """Generate a simple HTML gallery to view originals and detections."""
    rows = []
    for original, boxed in pairs:
        rows.append(
            f"<div class='item'><div><h3>{original}</h3>"
            f"<div class='images'>"
            f"<div><p>Original</p><img src='{original}' loading='lazy'></div>"
            f"<div><p>Detection</p><img src='{boxed}' loading='lazy'></div>"
            f"</div></div></div>"
        )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>YOLO Results</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 20px; background: #f3f3f3; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(520px, 1fr)); gap: 16px; }}
    .item {{ background: #fff; padding: 12px; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.1); }}
    h3 {{ margin: 0 0 8px 0; font-size: 18px; }}
    .images {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 8px; align-items: start; }}
    img {{ width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; background: #fafafa; }}
    p {{ margin: 0 0 4px 0; font-weight: bold; }}
  </style>
</head>
<body>
  <h1>YOLO Results</h1>
  <p>Showing original images and boxed detections.</p>
  <div class="grid">
    {''.join(rows)}
  </div>
</body>
</html>
"""
    (output_dir / "index.html").write_text(html, encoding="utf-8")


def serve_directory(directory: Path, host: str, port: int) -> None:
    handler = SimpleHTTPRequestHandler
    server = ThreadingHTTPServer((host, port), lambda *args, **kwargs: handler(*args, directory=directory, **kwargs))
    print(f"Serving {directory} at http://{host}:{port}")
    server.serve_forever()


def main() -> None:
    config = load_config()
    model = YOLO(str(config.model_path))

    input_files = collect_input_files(config.input_data_path)
    if not input_files:
        print(f"No image files found in {config.input_data_path}")
        return

    ensure_output_dir(config.output_data_path)
    pairs = run_inference(model, input_files, config.output_data_path)
    if not pairs:
        print("No results to display.")
        return

    write_gallery_page(config.output_data_path, pairs)
    serve_directory(config.output_data_path, config.host, config.port)


if __name__ == "__main__":
    main()
