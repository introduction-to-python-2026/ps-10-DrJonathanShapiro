from image_utils import load_image, edge_detection
from skimage.filters import median
from skimage.morphology import ball
from PIL import Image
import numpy as np


def main():
    # 1) Load original (color) image from repo
    input_path = "my_image.jpg"
    image = load_image(input_path)

    # 2) Noise suppression (median filter expects 2D uint8)
    if image.ndim == 3:
        gray = image.mean(axis=2).astype(np.uint8)
    else:
        gray = image.astype(np.uint8)

    clean_image = median(gray, ball(3))

    # 3) Edge detection
    edgeMAG = edge_detection(clean_image)

    # 4) Threshold to binary (adjust threshold if you want)
    threshold = 50
    edge_binary = edgeMAG > threshold

    # 5) Save as PNG (0/255 so it looks correct)
    out = (edge_binary.astype(np.uint8) * 255)
    Image.fromarray(out).save("my_edges.png")


if __name__ == "__main__":
    main()

