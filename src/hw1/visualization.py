from __future__ import annotations

import random
from typing import Sequence

import numpy as np
from PIL import Image
from PIL.Image import Image as ImageType

from src.hw1.main import Data, label_names, combine_images_horizontally, combine_images_vertically


def visualize_dataset_as_image(data: Data) -> ImageType:
    """Visualize the data as an image."""
    images = data["data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    label_images = []
    for label_idx, _ in enumerate(label_names):
        # Choose 10 random images of the label
        images_of_label = [image for image, image_label in zip(images, data["labels"]) if image_label == label_idx]
        rand_10_images = [Image.fromarray(i) for i in random.choices(images_of_label, k=10)]
        label_images.append(combine_images_horizontally(rand_10_images))
    return combine_images_vertically(label_images)


def visualize_single_image(data: Sequence[bytes | int | float]) -> ImageType:
    """Visualize a single image."""

    image = Image.fromarray(np.array(data, dtype=np.uint8).reshape(3, 32, 32).transpose(1, 2, 0))
    image.show()
    return image
