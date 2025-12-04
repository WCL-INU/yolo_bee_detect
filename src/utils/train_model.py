import os
import dotenv
from ultralytics import YOLO

# Load a model
# model = YOLO("yolo11n.yaml")  # build a new model from YAML//
model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)
# model = YOLO("yolo11n.yaml").load("yolo11n.pt")  # build from YAML and transfer weights

dotenv.load_dotenv()
data_dir = os.getenv("DATA_PATH", "./data")
yaml_path = os.getenv("DATA_YAML_PATH", f"{data_dir}/data.yaml")
hyperparameters_path = os.getenv("HYPERPARAMETERS_PATH", None)
project_dir = os.getenv("PROJECT_PATH", "./runs")

if __name__ == "__main__":
    # Train the model
    results = model.train(data=yaml_path, cfg=hyperparameters_path, epochs=1000, patience=100, imgsz=320, batch=4, project=project_dir)