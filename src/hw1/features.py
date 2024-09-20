"""Functions to extract features from the data."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, overload, Literal

import numpy as np
from skimage import feature
import cv2 as cv
import torch

from hw1 import train_data
from hw1.visualization import visualize_images

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
        tuple[Batch, Literal[3], Height, Width], np.dtype[np.float64]
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


def detect_corners_harris[Batch: int, Height: int, Width: int](
    image_rgb: np.ndarray[
        tuple[Batch, Literal[3], Height, Width], np.dtype[np.float64]
    ],
    *,
    method: str = 'k',
    k: float = 0.04,
    eps: float = 0.000001,
    sigma: int = 1,
    pickle_path: str | None = None,
) -> np.ndarray[tuple[Batch, Height, Width], np.dtype[np.float64]]:
    """Detect corners in an image using the Harris corner detector.

    Args:
        image_rgb: Input RGB image tensor of shape (Batch, 3, H, W)
        method: Method to use for the Harris corner detector
        k: Harris detector free parameter
        eps: Harris detector free parameter
        sigma: Standard deviation of the Gaussian filter
        pickle_path: Path to save the corner map to

    Returns:
        Corner map tensor of shape (Batch, H, W)
    """
    image_gray = rgb_to_grayscale(image_rgb)

    corners: np.ndarray[tuple[Batch, Height, Width], np.dtype[np.float64]] = np.zeros_like(image_gray, dtype=np.float64)
    for n in range(image_rgb.shape[0]):
        # The corner_peaks function returns the coordinates of the corner peaks
        peaks = feature.corner_peaks(feature.corner_harris(image_gray[n], method=method, k=k, eps=eps, sigma=sigma))
        corners[n][peaks[:, 0], peaks[:, 1]] = 1

    if pickle_path:
        np.save(pickle_path, corners)
    return corners


def main():
    """Quick testing, not part of the library."""
    from timeit import default_timer as timer

    logging.basicConfig(level=logging.INFO)

    N_SAMPLES = 50000

    time_start = timer()
    corners = detect_corners_harris(train_data[:N_SAMPLES].reshape(N_SAMPLES, 3, 32, 32), pickle_path="train_data_edges.npy")
    time_end = timer()
    logger.info(f"Time taken to detect edges: {time_end - time_start:.5f} seconds")
    # corners = np.load("train_data_edges.npy")

    image_corners = visualize_images(corners[:100].astype(np.bool_), show=False)
    image_originals = visualize_images(train_data[:100], show=False)
    image_corners.putalpha(100)
    image_originals.putalpha(180)
    image_originals.paste(image_corners, (0, 0), image_corners)
    image_originals.show()



if __name__ == "__main__":
    main()
