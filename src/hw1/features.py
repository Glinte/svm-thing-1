"""Functions to extract features from the data."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, overload, Literal

import numpy as np
from skimage import feature
import torch

from hw1 import train_data, test_data
from hw1.visualization import (
    visualize_grayscale_image,
    visualize_rgb_image,
    visualize_images,
)

if TYPE_CHECKING:
    pass


logger = logging.getLogger(__name__)


@overload
def rgb_to_grayscale[Batch: int, Pixels: int](
    image_rgb: np.ndarray[tuple[Batch, Literal[3], Pixels], np.dtype[np.number[Any]]],
) -> np.ndarray[tuple[Batch, Pixels], np.dtype[np.float32]]:
    """Convert RGB images to grayscale.

    Args:
        image_rgb: RGB image ndarray of shape (Batch, 3, Pixels)
    """


@overload
def rgb_to_grayscale[Batch: int, Height: int, Width: int](
    image_rgb: np.ndarray[
        tuple[Batch, Literal[3], Height, Width], np.dtype[np.number[Any]]
    ],
) -> np.ndarray[tuple[Batch, Height, Width], np.dtype[np.float32]]:
    """Convert an RGB image to grayscale.

    Args:
        image_rgb: RGB image ndarray of shape (Batch, 3, H, W)
    """


@overload
def rgb_to_grayscale(image_rgb: torch.Tensor) -> torch.Tensor:
    """Convert an RGB image to grayscale.

    Args:
        image_rgb: RGB image tensor of shape (Batch, 3, H, W) or (Batch, 3, Pixels)
    """


def rgb_to_grayscale(image_rgb):
    """Convert an RGB image to grayscale.

    Args:
        image_rgb: a tensor of shape (Batch, 3, H, W), with values in the range [0, 1].
    """
    image_gray = (
        0.2989 * image_rgb[:, 0] + 0.5870 * image_rgb[:, 1] + 0.1140 * image_rgb[:, 2]
    )
    return image_gray


def detect_edges_canny[Batch: int, Height: int, Width: int](
    image_rgb: np.ndarray[
        tuple[Batch, Literal[3], Height, Width], np.dtype[np.number[Any]]
    ],
    *,
    sigma: float = 0.75,
    low_threshold: int = 80,
    high_threshold: int = 160,
    pickle_path: str | None = None,
) -> np.ndarray[tuple[Batch, Height, Width], np.dtype[np.bool_]]:
    """Detect edges in an image using the Canny edge detector.

    Args:
        image_rgb: Input RGB image tensor of shape (Batch, 3, H, W)
        sigma (float): Standard deviation of the Gaussian filter
        low_threshold (int): Lower threshold for the hysteresis procedure
        high_threshold (int): Higher threshold for the hysteresis procedure
        pickle_path (str): Path to save the edge map to

    Returns:
        Edge map tensor of shape (H, W)
    """
    image_gray = rgb_to_grayscale(image_rgb)

    edges: np.ndarray[tuple[Batch, Height, Width], np.dtype[np.bool_]] = np.zeros_like(
        image_gray, dtype=bool
    )
    for n in range(image_rgb.shape[0]):
        edges[n] = feature.canny(image_gray[n], sigma, low_threshold, high_threshold)

    if pickle_path:
        np.save(pickle_path, edges)
    return edges


def main():
    """Quick testing, not part of the library."""
    from timeit import default_timer as timer

    logging.basicConfig(level=logging.INFO)

    N_SAMPLES = 10000

    time_start = timer()
    edges = detect_edges_canny(train_data[:N_SAMPLES].reshape(N_SAMPLES, 3, 32, 32))
    time_end = timer()
    logger.info(f"Time taken to detect edges: {time_end - time_start:.5f} seconds")

    # visualize_images(edges)
    # visualize_images(train_data[:100])


if __name__ == "__main__":
    main()
