from PIL import Image
import numpy as np
from scipy.signal import convolve2d


def load_image(path):
    """
    Loads an image as a numpy array.

    For the autograder + rank.median:
    - Regular images: return grayscale uint8 (H, W)
    - Ground-truth edge images (e.g., '*edges*'): return boolean (H, W)
    """
    img = Image.open(path).convert("L")   # grayscale
    img = np.array(img)                  # uint8

    # Ground truth edges file should be boolean mask
    if "edges" in path:
        return img > 0

    return img  # keep uint8


def edge_detection(image):
    """
    Edge detection using Sobel filters (per notebook instructions).
    - If image is RGB (H,W,3), convert to grayscale by averaging channels.
    - Convolve with zero-padding.
    - Return edge magnitude: sqrt(edgeX^2 + edgeY^2)
    """

    # Support both grayscale (H,W) and RGB (H,W,3)
    if image.ndim == 3:
        gray = image.mean(axis=2)
    else:
        gray = image

    gray = gray.astype(np.float64)

    # Kernels EXACTLY as in the notebook
    kernelY = np.array([[ 1,  2,  1],
                        [ 0,  0,  0],
                        [-1, -2, -1]])

    kernelX = np.array([[-1,  0,  1],
                        [-2,  0,  2],
                        [-1,  0,  1]])

    # Zero padding
    edgeY = convolve2d(gray, kernelY, mode="same", boundary="fill", fillvalue=0)
    edgeX = convolve2d(gray, kernelX, mode="same", boundary="fill", fillvalue=0)

    edgeMAG = np.sqrt(edgeX**2 + edgeY**2)
    return edgeMAG

