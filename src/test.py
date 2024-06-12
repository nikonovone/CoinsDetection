import json
import zipfile
from pathlib import Path
from typing import List, Tuple

import fire
import gdown
import numpy as np
from ultralytics import YOLO

from src.const import LABEL2COST
from src.utils import handle_exceptions, init_logger


@handle_exceptions("Downloading test dataset")
def download_test(output_dir: str) -> None:
    """
    Downloads and extracts the test dataset.

    Args:
        output_dir (str): The directory where the test dataset will be extracted.
    """
    zip_path = "split1.zip"
    link_id = "1l5m4kWospsEcE32lpRT_tPnTabF3fPhp"
    gdown.download(f"https://drive.google.com/uc?id={link_id}", zip_path)

    with zipfile.ZipFile(zip_path) as zip_file:
        zip_file.extractall(output_dir)


def compute_relative_error(prediction: float, gt: float) -> float:
    """
    Computes the relative error between a prediction and the ground truth.

    Args:
        prediction (float): The predicted value.
        gt (float): The ground truth value.

    Returns:
        float: The relative error as a percentage.
    """
    return abs(prediction - gt) / gt * 100


def processing_result(results) -> Tuple[float, int]:
    """
    Processes the YOLO results to compute the total amount and count of detected objects.

    Args:
        results: The results from the YOLO model.

    Returns:
        Tuple[float, int]: The total amount and count of detected objects.
    """
    data_list = [res.boxes.data.tolist() for res in results]
    am = [LABEL2COST[int(x[0][5])] for x in data_list]
    amounts = sum(am)
    counts = len(data_list)
    return amounts, counts


def main(
    test_dir: str = "./data/test",
    weights_path="/runs/detect/train/weights/best.pt",
    download=False,
) -> None:
    """
    Main function to run the inference and compute error metrics.

    Args:
        test_dir (str): The directory containing the test dataset.
    """
    logger = init_logger("COINS", "INFO")

    if download:
        download_test(test_dir)
        logger.info("Test dataset downloaded")

    ground_truth_path = Path(test_dir, "counts.json")
    images_dir = Path(test_dir, "images")

    with open(ground_truth_path) as f:
        ground_truth_data = json.load(f)

    # Load a model
    model = YOLO(weights_path)

    # Run batched inference on a list of images
    results = model(list(images_dir.rglob("*.jpg")))  # return a list of Results objects

    count_errors: List[float] = []
    amount_errors: List[float] = []

    for image_name, gt in sorted(ground_truth_data.items(), reverse=True):
        results = model.predict(Path(images_dir, image_name))
        predict_amount, predict_count = processing_result(results[0])
        count_error = compute_relative_error(predict_count, gt["count"])
        amount_error = compute_relative_error(predict_amount, gt["amount"])

        count_errors.append(count_error)
        amount_errors.append(amount_error)

    print(
        f"\033[91mМетрика 1. Средняя относительная ошибка количества монет: {round(np.mean(count_errors), 1)}%",
    )
    print(
        f"\033[91mМетрика 2. Средняя относительная ошибка количества денег: {round(np.mean(amount_errors), 1)}%",
    )


if __name__ == "__main__":
    fire.Fire(main)
