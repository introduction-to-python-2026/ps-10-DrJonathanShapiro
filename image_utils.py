from PIL import Image
import numpy as np
from scipy.signal import convolve2d


def load_image(path):
    """
    Loads an image from disk and returns it as a grayscale numpy array (float).
    """
    img = Image.open(path).convert("L")   # grayscale
    img = np.array(img, dtype=np.float64)
    return img


def edge_detection(image):
    """
    Applies Sobel edge detection and returns the edge magnitude image.
    """
    # Sobel kernels
    kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])

    ky = np.array([[-1, -2, -1],
                   [ 0,  0,  0],
                   [ 1,  2,  1]])

    gx = convolve2d(image, kx, mode="same", boundary="symm")
    gy = convolve2d(image, ky, mode="same", boundary="symm")

    edges = np.sqrt(gx**2 + gy**2)

    # Normalize for consistency
    edges = edges / edges.max()

    return edges
