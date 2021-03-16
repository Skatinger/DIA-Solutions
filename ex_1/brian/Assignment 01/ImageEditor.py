import glob
import os
import cv2
import numpy as np
from PIL import Image


# The convolution algorithm for assignment 01b
def convolve(im_to_convolve, method):
    if method == "edge":
        kernel = np.array([[-1, -1, -1],
                           [-1, 8, -1],
                           [-1, -1, -1]])
        padding = 2
    else:
        kernel = np.array([[1, 1, 1, 1, 1],
                           [1, 1, 2, 1, 1],
                           [1, 2, 4, 2, 1],
                           [1, 1, 2, 1, 1],
                           [1, 1,  1, 1, 1]])/32
        kernel[:, :, np.newaxis]
        padding = 4

    # Size parameters of both kernel and image
    x_kernel, y_kernel = kernel.shape
    x_size, y_size = im_to_convolve.shape[0:2]

    # Initialize Output Convolution
    x_out = int(x_size - x_kernel + padding) + 1
    y_out = int(y_size - y_kernel + padding) + 1

    im_to_convolve_padded = np.pad(im_to_convolve, padding, mode='constant')

    # Iterate through the whole image
    if method == "edge":
        convolution = np.zeros((x_out, y_out))
        for x in range(0, x_size - x_kernel):
            for y in range(0, y_size - y_kernel):
                # dot product
                convolution[x, y] = np.multiply(kernel, im_to_convolve_padded[x: x + x_kernel, y: y + y_kernel]).sum()
    else:
        convolution = np.zeros((x_out, y_out, 3))
        for ch in range(3):
            im_to_convolve_single_channel = im_to_convolve[:, :, ch]
            for x in range(0, x_size - x_kernel):
                for y in range(0, y_size - y_kernel):
                    convolution[x, y, ch] = np.multiply(kernel, im_to_convolve_single_channel[x: x + x_kernel, y: y + y_kernel]).sum()
    return convolution


# Loops over all files matching *.jpg in current directory
for infile in glob.glob("*.jpg"):
    ### Assignment 1a ###
    file, ext = os.path.splitext(infile)
    im = Image.open(infile).convert('RGB')

    # Determining new size
    newWidth = round(im.size[0] / 5)
    newHeight = round(im.size[1] / 5)
    size = (newWidth, newHeight)

    # Declaring a new image
    newIm = Image.new('RGB', size)

    # Nearest Neighbor Algorithm for Resizing
    for col in range(newWidth):
        for row in range(newHeight):
            currentPixel = im.getpixel((col * 5, row * 5))
            newIm.putpixel((col, row), currentPixel)

    # Save edited image with suffix "-resized.jpg"
    newIm.save(file + "-smallerByFactor5.jpg", "JPEG")

    ### Assignment 1b ###
    # blur
    image = cv2.imread(infile)
    cv2.imwrite((file + "_blur.jpg"), convolve(image, "blur"))

    # 2D convolution for edge-detection works best in grayscale
    image = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)
    cv2.imwrite((file + "_edge_detection.jpg"), convolve(image, "edge"))
