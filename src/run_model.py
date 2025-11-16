import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

import cv2
import dotenv
import matplotlib
import matplotlib.pyplot as plt
from ultralytics import YOLO

dotenv.load_dotenv()

matplotlib.use("TkAgg")


@dataclass
class AppConfig:
    model_path: Path
    input_data_path: Path
    output_data_path: Path


def load_config() -> AppConfig:
    """Load runtime configuration from environment variables."""
    return AppConfig(
        model_path=Path(os.getenv("MODEL_PATH", "./model.pth")),
        input_data_path=Path(os.getenv("INPUT_DATA_PATH", "./input_data")),
        output_data_path=Path(os.getenv("OUTPUT_DATA_PATH", "./output_data")),
    )


def collect_input_files(input_dir: Path) -> List[Path]:
    """Return sorted image files to process."""
    return sorted(file for file in input_dir.iterdir() if file.suffix.lower() == ".jpg")


def process_images(model: YOLO, images: List[Path]) -> None:
    """Run inference for each image and display the results."""
    figure = plt.figure("YOLO inference")

    for image_path in images:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Skipping unreadable file: {image_path.name}")
            continue

        result = model(image)[0]
        img = result.plot()

        print(f"Processing file: {image_path.name}")
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.pause(1)
        figure.clf()


def main() -> None:
    config = load_config()
    model = YOLO(str(config.model_path))

    input_files = collect_input_files(config.input_data_path)
    if not input_files:
        print(f"No .jpg files found in {config.input_data_path}")
        return

    process_images(model, input_files)


if __name__ == "__main__":
    main()
