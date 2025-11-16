from dataclasses import dataclass
from pathlib import Path
from ultralytics import YOLO
import dotenv
import os

dotenv.load_dotenv()


@dataclass
class ExportConfig:
    model_path: Path
    format: str = "tflite"
    imgsz: int = 640
    int8: bool = True
    data: str = ""
    batch: int = 1
    nms: bool = True
    device: str = "cpu"


def load_config() -> ExportConfig:
    """Load runtime configuration from environment variables."""
    return ExportConfig(
        model_path=Path(os.getenv("MODEL_PATH", "./model.pt")),
        format=os.getenv("EXPORT_FORMAT", "tflite"),
        imgsz=int(os.getenv("EXPORT_IMGSZ", "640")),
        int8=os.getenv("EXPORT_INT8", "1") == "1",
        data=os.getenv("EXPORT_DATA", ""),
        batch=int(os.getenv("EXPORT_BATCH_SIZE", "1")),
        nms=os.getenv("EXPORT_NMS", "1") == "1",
        device=os.getenv("EXPORT_DEVICE", "cpu"),
    )


def main() -> None:
    config = load_config()
    model = YOLO(str(config.model_path))

    output_path = model.export(
        format=config.format,
        imgsz=config.imgsz,
        int8=config.int8,
        data=config.data if config.data else None,
        batch=config.batch,
        nms=config.nms,
        device=config.device,
    )

    print(f"Model exported to: {output_path}")


if __name__ == "__main__":
    main()
