from __future__ import annotations

import random
from typing import Sequence, TYPE_CHECKING, Any, Annotated, Literal

import numpy as np
from PIL import Image
from PIL.Image import Image as ImageType
import matplotlib as mpl
from beartype import beartype
from beartype.vale import Is
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

from hw1 import label_names, train_data, train_labels
from src.hw1 import DataDict

if TYPE_CHECKING:
    import numpy.typing as npt
    from typings.sklearn._typing import MatrixLike


def visualize_dataset_as_image(data: DataDict) -> ImageType:
    """Visualize the data as an image."""
    images = data["data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    label_images = []
    for label_idx, _ in enumerate(label_names):
        # Choose 10 random images of the label
        images_of_label = [
            image
            for image, image_label in zip(images, data["labels"])
            if image_label == label_idx
        ]
        rand_10_images = [
            Image.fromarray(i) for i in random.choices(images_of_label, k=10)
        ]
        label_images.append(combine_images_horizontally(rand_10_images))
    return combine_images_vertically(label_images)


@beartype
def visualize_rgb_image(
    data: Annotated[np.ndarray[Any, np.dtype[Any]], Is[lambda data: data.size == 3072]],
    *,
    show: bool = True,
) -> ImageType:
    """Visualize a single RGB image."""

    image = Image.fromarray(data.astype(np.uint8).reshape(3, 32, 32).transpose(1, 2, 0))
    if show:
        image.show()
    return image


@beartype
def visualize_grayscale_image(
    data: Annotated[np.ndarray[Any, np.dtype[Any]], Is[lambda data: data.size == 1024]],
    *,
    show: bool = True,
) -> ImageType:
    """Visualize a single 32x32 grayscale image."""

    image = Image.fromarray(data.reshape(32, 32))
    if show:
        image.show()
    return image


@beartype
def visualize_images(
    data: Annotated[
        np.ndarray[Any, np.dtype[Any]],
        Is[lambda data: data[0].size == 1024 or data[0].size == 3072],
    ],
    *,
    show: bool = True,
) -> ImageType:
    """Visualize multiple 32x32 images. Images can be either grayscale or RGB.

    Images are displayed as a square grid of images.
    """

    if data[0].size == 1024:
        images = [Image.fromarray(i.reshape(32, 32)) for i in data]
    else:
        images = [
            Image.fromarray(i.astype(np.uint8).reshape(3, 32, 32).transpose(1, 2, 0))
            for i in data
        ]

    total_images = len(images)
    rows = int(np.sqrt(total_images))
    cols = total_images // rows

    combined_image = combine_images_vertically(
        [
            combine_images_horizontally(images[i : i + cols])
            for i in range(0, total_images, cols)
        ]
    )
    if show:
        combined_image.show()
    return combined_image


def combine_images_horizontally(images: Sequence[ImageType]) -> ImageType:
    """Combine images horizontally."""
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)

    new_image = Image.new("RGB", (total_width, max_height))
    x_offset = 0
    for image in images:
        new_image.paste(image, (x_offset, 0))
        x_offset += image.size[0]

    return new_image


def combine_images_vertically(images: Sequence[ImageType]) -> ImageType:
    """Combine images vertically."""
    widths, heights = zip(*(i.size for i in images))
    max_width = max(widths)
    total_height = sum(heights)

    new_image = Image.new("RGB", (max_width, total_height))
    y_offset = 0
    for image in images:
        new_image.paste(image, (0, y_offset))
        y_offset += image.size[1]

    return new_image


def PCA_visualization(X: MatrixLike, y: npt.ArrayLike) -> None:
    """Visualize the data using PCA."""

    # Set matplotlib to be interactive
    mpl.use("Qt5Agg")
    plt.ion()

    pca = PCA(n_components=3)
    X_r = pca.fit_transform(X)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    colors = [
        "navy",
        "turquoise",
        "darkorange",
        "red",
        "green",
        "blue",
        "purple",
        "yellow",
        "black",
        "pink",
    ]
    for i, target_name in enumerate(label_names):
        ax.scatter(
            X_r[y == i, 0],
            X_r[y == i, 1],
            X_r[y == i, 2],
            color=colors[i],
            label=target_name,
        )
    ax.legend(loc="best", shadow=False, scatterpoints=1)
    ax.set_title("PCA of CIFAR-10 dataset")
    plt.show(block=True)


def main():
    PCA_visualization(train_data[:1000], train_labels[:1000])


if __name__ == "__main__":
    main()
