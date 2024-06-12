import random
import warnings
from pathlib import Path

from fire import Fire

from src.utils import (
    generate_data_yaml,
    generate_samples,
    init_logger,
    load_coin_images,
    load_textures,
)


def init(logger_name: str, log_level: str):
    warnings.filterwarnings("ignore")
    logger = init_logger(logger_name, log_level)
    return logger


def generate_data(
    data_dir: str,
    output_dir: str = "output_data",
    num_samples: int = 10,
    val_size=0.2,
):
    """
    Generate synthetic data by adding coins to textures and save images and labels in YOLO format.

    Args:
        texture_dir (str): Directory containing texture images.
        coin_dir (str): Directory containing coin images.
        output_dir (str): Directory to save output images and labels.
        num_samples (int): Number of samples to generate.
    """

    logger = init("COIN_LOGGER", "INFO")

    data_dir_path = Path(data_dir)

    texture_dir = data_dir_path / "textures"
    coin_dir = data_dir_path / "coins"

    texture_paths = list(Path(texture_dir).glob("*"))
    textures = load_textures(texture_paths)

    coin_paths = list(Path(coin_dir).glob("*"))
    coin_data = load_coin_images(coin_paths)

    output_dir_path = Path(output_dir)
    train_dir_path = output_dir_path / "train"
    val_dir_path = output_dir_path / "valid"

    train_images_dir = train_dir_path / "images"
    train_labels_dir = train_dir_path / "labels"

    val_images_dir = val_dir_path / "images"
    val_labels_dir = val_dir_path / "labels"

    train_images_dir.mkdir(parents=True, exist_ok=True)
    train_labels_dir.mkdir(parents=True, exist_ok=True)
    val_images_dir.mkdir(parents=True, exist_ok=True)
    val_labels_dir.mkdir(parents=True, exist_ok=True)

    num_val_samples = int(num_samples * val_size)
    num_train_samples = num_samples - num_val_samples

    random.shuffle(textures)
    logger.info(f"Generating {num_train_samples} train samples")
    generate_samples(
        textures,
        coin_data,
        train_images_dir,
        train_labels_dir,
        num_train_samples,
    )

    logger.info(f"Generating {num_val_samples} valid samples")
    generate_samples(
        textures,
        coin_data,
        val_images_dir,
        val_labels_dir,
        num_val_samples,
    )

    generate_data_yaml(output_dir_path)


if __name__ == "__main__":
    Fire(generate_data)
