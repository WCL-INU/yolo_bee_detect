import os
import dotenv
from ultralytics import YOLO

# Load a model
# model = YOLO("yolo11n.yaml")  # build a new model from YAML//
model = YOLO("yolo11s.pt")  # load a pretrained model (recommended for training)
# model = YOLO("yolo11n.yaml").load("yolo11n.pt")  # build from YAML and transfer weights

dotenv.load_dotenv()
data_dir = os.getenv("DATA_PATH", "./data")
yaml_path = os.getenv("DATA_YAML_PATH", f"{data_dir}/data.yaml")

if __name__ == "__main__":
    # Train the model
    results = model.train(data=yaml_path, epochs=100, imgsz=640, batch=4)
