from typing import Tuple


def convert_to_yolo_format(
    box: Tuple[int, int, int, int, int],
    img_width: int,
    img_height: int,
) -> Tuple[int, float, float, float, float]:
    """
    Converts annotations to YOLO format.

    Args:
        annotations (Tuple[int, int, int, int, int]): The bounding box coordinates and class label in the format (cls, x_min, y_min, x_max, y_max).
        img_width (int): The width of the image.
        img_height (int): The height of the image.

    Returns:
        Tuple[int, float, float, float, float]: The annotations in YOLO format (cls, x_center, y_center, width, height).
    """
    cls, x_min, y_min, x_max, y_max = box
    x_center = (x_min + x_max) / 2.0 / img_width
    y_center = (y_min + y_max) / 2.0 / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height
    return cls, x_center, y_center, width, height
