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
project_dir = os.getenv("PROJECT_PATH", "./runs/detect/")

if __name__ == "__main__":
    # Train the model
    results = model.train(
        data=yaml_path, cfg=hyperparameters_path, epochs=1000, patience=75, imgsz=640, batch=4, project=project_dir,
        
        # === [증강 파라미터 튜닝: 중복 방지] ===
        # 이미 데이터가 어둡기 때문에, 여기서 또 강하게 어둡게 하면 안 됩니다.
        hsv_h=0.015,          # 색조 변화 (기본값 유지 - 벌 색깔은 중요함)
        hsv_s=0.7,            # 채도 변화 (기본값 유지)
        hsv_v=0.4,            # [중요] 밝기 변화를 0.4(기본값)로 낮춤. (0.8은 너무 과함)
        
        # === [작은 객체(벌) 탐지 최적화] ===
        mosaic=1.0,           # 모자이크 100% (작은 객체 학습에 필수)
        mixup=0.1,            # 믹스업 10% (배경과 객체를 섞어서 일반화 성능 향상)
        scale=0.5            # 크기 변화 ±50% (멀리 있는 벌, 가까운 벌)
    )
