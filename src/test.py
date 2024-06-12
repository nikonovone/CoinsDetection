import json
import zipfile
from pathlib import Path

import gdown
from ultralytics import YOLO

from src.utils import handle_exceptions, init_logger


@handle_exceptions("Downloading test dataset")
def donwload_test(output_dir):
    zip_path = "split1.zip"
    link_id = "1l5m4kWospsEcE32lpRT_tPnTabF3fPhp"
    gdown.download(f"https://drive.google.com/uc?id={link_id}", zip_path)

    with zipfile.ZipFile(zip_path) as zip_file:
        zip_file.extractall(output_dir)

def compute_relative_error(prediction, gt):
    return abs(prediction - gt) / gt * 100

logger = init_logger('COINS', "INFO")
test_dir = './data/dataset/test'


donwload_test(test_dir)
logger.INFO('Test dataset downloaded')

ground_truth_path = Path(test_dir, "counts.json")
images_dir = Path(test_dir, "images")


with open(ground_truth_path) as f:
    ground_truth_data = json.load(f)



# Load a model
model = YOLO("runs/detect/train/weights/best.pt")  # pretrained YOLOv8n model


# Run batched inference on a list of images
results = model(list(images_dir.rglob('*')))  # return a list of Results objects
print(results[0])

