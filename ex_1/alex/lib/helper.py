from PIL import Image
import numpy as np
import time
from IPython.display import display

# takes a PIL image (RGB) and the desired scaling factor
# scaling between 0 and 1
def downscale(image, scale):
    inp = np.array(image)
    height = inp.shape[0]
    width = inp.shape[1]
    result = np.zeros((int(height*scale), int(width*scale),3))
    fac = int(1/scale)
    kernel_width = int(fac/2)
    for h in range(0, height, fac):
        for w in range(0, width, fac):
            h_index = int((h-kernel_width)*scale)
            w_index = int((w-kernel_width)*scale)
            for dim in range(3):
                kern = inp[h:h+fac, w:w+fac, dim]
                result[h_index,w_index, dim] = int(np.sum(kern) / fac**2)
    return result

# takes a PIL image (RGB) and the desired scaling factor
# scaling between 0 and 1
def downscale_rough(image, scale):
    inp = np.array(image)
    height = inp.shape[0]
    width = inp.shape[1]
    result = np.zeros((int(height*scale), int(width*scale),3))
    fac = int(1/scale)
    kernel_width = 2 #int(fac/2)
    for h in range(0, height, fac):
        for w in range(0, width, fac):
            h_index = int((h-kernel_width)*scale)
            w_index = int((w-kernel_width)*scale)
            for dim in range(3):
                kern = inp[h:h+kernel_width, w:w+kernel_width, dim]
                result[h_index,w_index, dim] = int(np.sum(kern) / 2**2)
    return result

# takes a PIL image (rgb or greyscale) and convolves with the given parameters
# kernel: numpy array
# some kernels need multiplication
def convolve(image, kernel, stride=1, padding=1):
    img_array = np.asarray(image)
    # kernel width
    k = int(kernel.shape[0] / 2)
    height = img_array.shape[0]
    width = img_array.shape[1]
    dimensions = img_array.shape[2]
    # create padded image
    padded = np.zeros((height + 2*padding, width + 2*padding, dimensions))
    padded[padding:(-padding), padding:(-padding)] = img_array # insert image into empty padded array
    # create result array
    paddedResult = np.zeros((height + 2*padding, width + 2*padding, dimensions))
    # convolve
    for w in range(k, width-k):
        for h in range(k, height-k):
            for dim in range(dimensions):
                paddedResult[h,w,dim] = int((padded[h-k:h+k+1, w-k:w+k+1, dim] * kernel).sum())
    return paddedResult