from dataclasses import dataclass
from pathlib import Path
from ultralytics import YOLO
import dotenv
import os

dotenv.load_dotenv()


@dataclass
class ExportConfig:
    model_path: Path
    format: str = "ncnn"
    imgsz: int = 640
    half: bool = False
    batch: int = 1
    device: str = "cpu"


def load_config() -> ExportConfig:
    """Load runtime configuration from environment variables."""
    return ExportConfig(
        model_path=Path(os.getenv("MODEL_PATH", "./model.pt")),
        format=os.getenv("EXPORT_FORMAT", "ncnn"),
        imgsz=int(os.getenv("EXPORT_IMGSZ", "640")),
        half=os.getenv("EXPORT_HALF", "0") == "1",
        batch=int(os.getenv("EXPORT_BATCH_SIZE", "1")),
        device=os.getenv("EXPORT_DEVICE", "cpu"),
    )


def main() -> None:
    config = load_config()
    model = YOLO(str(config.model_path))

    output_path = model.export(
        format=config.format,
        imgsz=config.imgsz,
        half=config.half,
        batch=config.batch,
        device=config.device,
    )

    print(f"Model exported to: {output_path}")


if __name__ == "__main__":
    main()
