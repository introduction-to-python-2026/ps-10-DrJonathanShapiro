from PIL import Image
import numpy as np
from scipy.signal import convolve2d


def load_image(path):
    """
    Loads an image.
    - Regular images → grayscale float
    - Edge ground-truth images → boolean
    """
    img = Image.open(path).convert("L")
    img = np.array(img)

    # Ground-truth edge images are binary
    if "edges" in path:
        return img > 0

    return img.astype(np.float64)


def edge_detection(image):
    """
    Sobel edge detection (returns 0–255 scale).
    """
    kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])

    ky = np.array([[-1, -2, -1],
                   [ 0,  0,  0],
                   [ 1,  2,  1]])

    gx = convolve2d(image, kx, mode="same", boundary="symm")
    gy = convolve2d(image, ky, mode="same", boundary="symm")

    edges = np.sqrt(gx**2 + gy**2)

    m = edges.max()
    if m > 0:
        edges = edges / m * 255

    return edges
