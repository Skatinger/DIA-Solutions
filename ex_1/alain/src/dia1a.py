from PIL import Image
import numpy as np


# Wanted to add convolution, but it is not working properly yet...
def resize(img: Image, resize_factor: int = 5) -> Image:
    a = np.asarray(img)
    y, x, _ = a.shape

    if y % resize_factor > 0 or x % resize_factor > 0:
        raise ValueError(f"Can't resize by {resize_factor} as it isn't a divisor of the image dimension")

    kept_rows = [row for row in range(0, y, resize_factor)]
    kept_cols = [row for row in range(0, x, resize_factor)]

    new_image = a[kept_rows]
    new_image = new_image[:, kept_cols]
    return Image.fromarray(new_image, mode=img.mode)



def load_image(file_path: str) -> Image:
    return Image.open(file_path)
