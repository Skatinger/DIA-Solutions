import numpy as np
from PIL import Image

def create_separable_kernel(v: np.array) -> np.array:
    d1 = v.shape[0]
    return v * v.reshape((d1, 1))


# Lanczos
# https://clouard.users.greyc.fr/Pantheon/experiments/rescaling/index-en.html
def create_lanczos_kernel(size: int, a: int = 1) -> np.array:
    X = np.linspace(-a, a, size)
    Y = np.sinc(X) * np.sinc(X / a)
    return create_separable_kernel(Y)



def my_convolve(img: np.array, kernel: np.array) -> Image:
    """
    FIXME! Needs to be corrected to support images, for now, having issues with padding the image

    :param img:
    :param kernel:
    :return:
    """
    k_m_rows, k_n_cols = kernel.shape
    if k_m_rows % 2 == 0 or k_n_cols % 2 == 0:
        raise ValueError("Only accepting odd kernel shapes (2m+1)x(2n+1)")

    # +-----+-----+-----+-----+
    # |  X  | X+1 | X+2 | X+...
    # +-----+-----+-----+-----+
    extra_m = (int)(k_m_rows / 2)
    extra_n = (int)(k_n_cols / 2)

    img_rows = img.shape[0]
    img_cols = img.shape[1]
    np_img = np.pad(np.asarray(img), ((extra_m, extra_m), (extra_n, extra_n)), 'edge')

    output = np.zeros((img_rows, img_cols))

    for row_i in range(img_rows):
        for col_i in range(img_cols):
            img_slice = np_img[row_i:row_i + k_m_rows, col_i:col_i + k_n_cols]
            output[row_i, col_i] = np.sum(kernel * img_slice)

    return output
