import random
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import cv2
import numpy as np
import yaml
from numpy.typing import NDArray
from rich.progress import track

from src.const import BASE_SIZE, COST2LABEL, COST2SIZE

from .annotations import convert_to_yolo_format
from .utils import handle_exceptions


@handle_exceptions("Loading coin images")
def load_coin_images(coin_paths: List[Path]) -> List[Dict[str, Any]]:
    """
    Loads images of coins and extracts metadata from their filenames.

    Each coin image file should be named in the format "cost_side_sample.png",
    where 'cost' is an integer representing the coin's value and 'side' is a string
    representing the coin's side (e.g., "1_front_0.png", ""1_back_0.png").

    Args:
        coin_paths (List[Path]): A list of Path objects pointing to the coin image files.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, each containing:
            - 'image' (Any): The loaded image data.
            - 'side' (str): The side of the coin ('head' or 'tail').
            - 'cost' (int): The value of the coin.
    """
    coin_results = []

    for path in coin_paths:
        coin_data: Dict[str, Union[str, int, NDArray]] = {}
        name = path.stem
        values = name.split("_")
        cost = int(values[0])
        side = values[1]

        coin_image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        coin_data["image"] = coin_image
        coin_data["side"] = side
        coin_data["cost"] = cost

        coin_results.append(coin_data)

    return coin_results


@handle_exceptions("Loading textures")
def load_textures(texture_paths: List[Path]) -> List[np.ndarray]:
    """
    Loads texture images from the provided file paths.

    Args:
        texture_paths (List[Path]): A list of Path objects pointing to the texture image files.

    Returns:
        List[np.ndarray]: A list of loaded texture images as numpy arrays.
    """
    textures = []

    for path in texture_paths:
        texture = cv2.imread(str(path))
        textures.append(texture)

    return textures


@handle_exceptions("Resize coin image")
def resize_coin_image(coin: np.ndarray, new_width: int, new_height: int) -> np.ndarray:
    """
    Resizes the given coin image to the specified width and height.

    Args:
        coin (np.ndarray): The input coin image.
        new_width (int): The desired width of the resized image.
        new_height (int): The desired height of the resized image.

    Returns:
        np.ndarray: The resized coin image.
    """
    return cv2.resize(coin, (new_width, new_height), interpolation=cv2.INTER_AREA)


@handle_exceptions("Adding coins to texture")
def add_coins_to_texture(
    texture: np.ndarray,
    coin_results: List[Dict[str, Any]],
    num_coins: int,
    min_scale: float = 2,
    max_scale: float = 3,
    coin_size: int = 256,
) -> List[Tuple[int, int, int, int, int]]:
    """
    Adds coin images to a texture at random positions and scales, and generates annotations.

    Args:
        texture (np.ndarray): The background texture image.
        coin_results (List[Dict[str, Any]]): List of dictionaries containing coin data.
        num_coins (int): Number of coins to add to the texture.
        min_scale (float): Minimum scale factor for resizing coins.
        max_scale (float): Maximum scale factor for resizing coins.
        fixed_size (int): Fixed size for scaling the coins.

    Returns:
        List[Tuple[int, int, int, int, int]]: A list of annotations containing class labels and bounding box coordinates.
    """

    COINS_SCALE_FACTOR = {
        1: COST2SIZE[1] / BASE_SIZE,
        2: COST2SIZE[2] / BASE_SIZE,
        5: COST2SIZE[5] / BASE_SIZE,
        10: COST2SIZE[10] / BASE_SIZE,
    }

    height, width, _ = texture.shape
    annotations = []

    # Coin size relative to texture size
    scale_t = random.uniform(min_scale, max_scale)

    # Distance between camera and table with coins
    scale_d = random.uniform(1, 4)
    coin_size = int(height / (scale_t * scale_d))

    for _ in range(num_coins):
        coin_data = random.choice(coin_results)

        coin_image = coin_data["image"]
        coin_side = coin_data["side"]
        coin_cost = coin_data["cost"]

        # The size of each type of coin
        coin_scale = COINS_SCALE_FACTOR[coin_cost]
        new_coin_width = int(coin_size * coin_scale)
        new_coin_height = int(coin_size * coin_scale)
        coin_image = resize_coin_image(coin_image, new_coin_width, new_coin_height)

        # Generate random position for the coin
        x = random.randint(0, width - new_coin_width)
        y = random.randint(0, height - new_coin_height)

        # Overlay the coin image onto the texture
        alpha_s = coin_image[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s

        for c in range(3):
            # Define the region of interest in the texture
            roi = texture[y : y + new_coin_height, x : x + new_coin_width, :3]

            # Blend the coin image with the texture using the alpha channel
            blended_roi = (
                alpha_s[:, :, np.newaxis] * coin_image[:, :, :3]
                + alpha_l[:, :, np.newaxis] * roi
            )

            # Place the blended region back into the texture
            texture[y : y + new_coin_height, x : x + new_coin_width, :3] = blended_roi

        cls = COST2LABEL[coin_cost] if coin_side == "front" else 0

        # Save the coordinates and class label for the coin
        annotations.append((cls, x, y, x + new_coin_width, y + new_coin_height))

    return annotations


@handle_exceptions("Generation samples")
def generate_samples(
    textures,
    coin_data,
    output_images_dir,
    output_labels_dir,
    num_samples,
    max_num_coins=10,
):
    """
    Generate samples by adding coins to textures and save images and labels in YOLO format.

    Args:
        textures (list): List of texture images.
        coin_data (list): List of coin images and associated data.
        output_images_dir (Path): Directory to save output images.
        output_labels_dir (Path): Directory to save output labels.
        num_samples (int): Number of samples to generate.
    """
    for i in track(range(num_samples), description=""):
        texture = random.choice(textures).copy()
        height, width, _ = texture.shape
        num_coins = random.randint(1, max_num_coins)
        annotations = add_coins_to_texture(texture, coin_data, num_coins)

        image_filename = f"image_{i}.png"
        label_filename = f"image_{i}.txt"

        cv2.imwrite(str(output_images_dir / image_filename), texture)

        with open(output_labels_dir / label_filename, "w") as f:
            for ann in annotations:
                cls, x_center, y_center, w, h = convert_to_yolo_format(
                    ann,
                    width,
                    height,
                )
                f.write(f"{cls} {x_center} {y_center} {w} {h}\n")


def generate_data_yaml(save_dir: Path):
    names = ["tail"]
    coin_names = [str(x) for x in COST2LABEL.values()]
    names.extend(coin_names)

    yaml_config = {
        "path": '/app/'+ str(save_dir),
        "train": "./train/images",
        "val": "./valid/images",
        "nc": len(COST2LABEL) + 1,
        "names": names,
    }

    # Writing YAML to a file
    with open(save_dir / "data.yaml", "w") as yaml_file:
        yaml.dump(yaml_config, yaml_file)
