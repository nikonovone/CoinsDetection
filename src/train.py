from pathlib import Path

import fire
from clearml import Task
from ultralytics import YOLO


def train_model(
    yaml_path="./data/dataset/data.yaml",
    model_path="./weights/yolov8s.pt",
    epochs=20,
    batch=32,
    imgsz=640,
    device=None,
):
    task = Task.init(project_name="RussianCoinsDetection", task_name="Detection")
    model_variant = Path(model_path).stem
    task.set_parameter("model_variant", model_variant)
    model = YOLO(model_path)
    args = dict(
        data=yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        device=device,
        batch=batch,
    )
    task.connect(args)
    results = model.train(**args)
    return results


if __name__ == "__main__":
    fire.Fire(train_model)
