"""Functions to extract features from the data."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, overload, Literal

import numpy as np
from skimage import feature
import torch

from hw1.main import train_data
from hw1.visualization import visualize_grayscale_image


if TYPE_CHECKING:
    pass


logger = logging.getLogger(__name__)


@overload
def rgb_to_grayscale[Pixels: int](
    image_rgb: np.ndarray[tuple[Literal[3], Pixels], np.dtype[np.number[Any]]]
) -> np.ndarray[tuple[Pixels], np.dtype[np.float32]]:
    """Convert an RGB image to grayscale.

    Args:
        image_rgb: RGB image ndarray of shape (3, Pixels)
    """

@overload
def rgb_to_grayscale[Height: int, Width: int](
    image_rgb: np.ndarray[tuple[Literal[3], Height, Width], np.dtype[np.number[Any]]]
) -> np.ndarray[tuple[Height, Width], np.dtype[np.float32]]:
    """Convert an RGB image to grayscale.

    Args:
        image_rgb: RGB image ndarray of shape (3, H, W)
    """


@overload
def rgb_to_grayscale(image_rgb: torch.Tensor) -> torch.Tensor:
    """Convert an RGB image to grayscale.

    Args:
        image_rgb: RGB image tensor of shape (3, H, W) or (3, Pixels)
    """


def rgb_to_grayscale(image_rgb):
    """Convert an RGB image to grayscale.

    Args:
        image_rgb: A sequence of 3 channels (R, G, B) or a tensor of shape (3, H, W), with values in the range [0, 1].
    """
    image_gray = 0.2989 * image_rgb[0] + 0.5870 * image_rgb[1] + 0.1140 * image_rgb[2]
    return image_gray


def detect_edges_canny[Height: int, Width: int](
    image_rgb: np.ndarray[tuple[Literal[3], Height, Width], np.dtype[np.number[Any]]],
    *,
    low_threshold: int = 100,
    high_threshold: int = 200
) -> np.ndarray[tuple[Height, Width], np.dtype[np.bool_]]:
    """Detect edges in an image using the Canny edge detector.

    Args:
        image_rgb: Input RGB image tensor of shape (3, H, W)
        low_threshold (int): Lower threshold for the hysteresis procedure
        high_threshold (int): Higher threshold for the hysteresis procedure

    Returns:
        Edge map tensor of shape (H, W)
    """
    image_gray = rgb_to_grayscale(image_rgb)

    return feature.canny(image_gray, 1, low_threshold, high_threshold)


def main():
    """Quick testing, not part of the library."""
    edges = detect_edges_canny(train_data[1].reshape(3, 32, 32))
    visualize_grayscale_image(edges.reshape(1024) * 255)


if __name__ == "__main__":
    main()
