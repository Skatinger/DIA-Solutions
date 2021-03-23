# imports
from PIL import Image, ImageOps
import numpy as np
import math
import matplotlib.pyplot as plt


class ImageToolkit:
    path_image: str
    image: Image
    image_gray: Image
    blur_image: Image
    blur_gray_image: Image

    def __init__(self, path_image: str):
        self.path_image = path_image
        print("Loading: ", path_image, " ...")
        self.image = Image.open(path_image)

    def save_image(self, path_target: str):
        # TODO: define what specific "image" to save and how (the original one? a modified one? which one?)
        """
        :param path_target: the path/folder ((incl. file name and its extension) where to save the image
        """
        print("Saving: ", path_target, " ...")
        self.image.save(fp=path_target, format=None)

    def convert_to_gray(self):
        """
        convert the base image (assumed in RGB) into grayscale
        :return:
        """
        self.image_gray = ImageOps.grayscale(self.image)

    @staticmethod
    def gaussian_kernel(size: int, n_channel: int = 1):
        """
        Return a kernel of Gaussian distribution with mean zero and sigma=sqrt(size) (for simplicity)
        :param n_channel: default=1 for grayscale image. set it to 3 if RGB (3-channel) image
        :param size: the desired size of the kernel
        :return: an array (numpy) of the 2D-Gaussian smoothing
        """
        # start the kernel as a vector, based on size argument
        kernel_vector = np.linspace(-(size // 2), size // 2, size)

        # prepare parameters for the Gaussian distribution (standard deviation, denominator of formula)
        std_dev = math.sqrt(size)
        denominator = math.sqrt(2 * np.pi) * std_dev ** 2

        for i in range(size):
            # update the Gaussian distribution
            kernel_vector[i] = (np.e ** (-((kernel_vector[i] / std_dev) ** 2) / 2)) / denominator

        # return the kernel as an array (after out product and averaging)
        kernel_array = np.outer(kernel_vector.T, kernel_vector.T)
        kernel_array *= 1.0 / kernel_array.max()

        # if RGB -> return kernel for 3 channels
        if n_channel == 3:
            kernel_array = np.dstack((kernel_array, kernel_array, kernel_array))

        return kernel_array

    def convolution(self, kernel: np.ndarray):
        """
        Apply the given kernel to the image (self.image) - assuming mode=RGB
        remark: the kernel should be of appropriate dimension ("3D" if mode=RGB, "2D" if mode=grayscale)
        :param kernel:
        """
        # print("image: mode {}, size {}".format(self.image.mode, self.image.size))
        base_img = self.image
        base_row = base_img.height
        base_col = base_img.width
        n_channel = 3  # because we assume the mode is RGB -> 3 channel

        # empty array for output
        image_output = np.zeros((base_row, base_col, n_channel), dtype="float32")
        # print("output image size: {}".format(image_output.shape))

        # padded image
        # print("kernel size: {}".format(kernel.shape))
        pad = kernel.shape[0] - 1 // 2  # remark: the nb of channel is at .shape[2] for the kernel
        image_padded = Image.new(base_img.mode, (base_img.width + 2 * pad, base_img.height + 2 * pad), color=0)
        image_padded.paste(base_img, (pad, pad))
        image_padded_array = np.array(image_padded)

        # print("padded image size: {}".format(image_padded_array.shape))

        # for z in range(n_channel):
        #     print("kernel's 1-channel size: {}".format(kernel[:, :, z].shape))
        #     plt.imshow(kernel[:, :, z], interpolation='none', cmap='gray')
        #     plt.title("Kernel ( {}X{} )".format(pad, pad))
        #     plt.show()

        # Convolution on RGB image: locate an (x,y) and apply the kernels to each 3-channel
        for y in range(base_row):
            for x in range(base_col):
                for z in range(n_channel):
                    # region-of-interest
                    region_of_interest = image_padded_array[y:y + pad, x:x + pad, z]
                    # apply kernel
                    k = (region_of_interest * kernel[:, :, z]).sum()
                    # save at 'pixel target' location (x, y) for the current channel
                    image_output[y - pad, x - pad, z] = k*255

        # print("Output Image size : {}".format(image_output.shape))

        # show image
        # plt.imshow((image_output*255).astype(np.uint8))
        # plt.title("Output Image using {}X{} Kernel".format(base_row, base_col))
        # plt.show()

        # save image
        self.blur_image = Image.fromarray(image_output, mode=base_img.mode)

    def convolution_grayscale(self, kernel: np.ndarray):
        """
        Apply the kernel to a grayscale image
        :param kernel:
        """
        base_img = self.image_gray
        base_row = base_img.height
        base_col = base_img.width

        # empty array: with expected output size
        image_output = np.zeros((base_row, base_col), dtype="float32")

        # Padding: ensure that we return an image of same size as input image
        # because applying the kernel "reduces" the input image
        # (avoid throwing away too much info from border pixels too)
        # (or PIL.ImageOps.pad)
        # https://note.nkmk.me/en/python-pillow-add-margin-expand-canvas/
        pad = kernel.shape[0] - 1 // 2  # the increase necessary to original image, before applying the kernel
        image_padded = Image.new(base_img.mode, (base_img.width + 2 * pad, base_img.height + 2 * pad), color=0)
        image_padded.paste(base_img, (pad, pad))
        image_padded_array = np.array(image_padded)

        # plt.imshow(image_padded_array, cmap='gray')
        # plt.title("Padded Image")
        # plt.show()

        # convolution on grayscale image
        # we go through each pixel of the image (by row, and by column), and apply the kernel to each ROI
        for y in range(base_row):  # np.arange(pad, base_img.height + pad):  # for each row
            for x in range(base_col):  # np.arange(pad, base_img.width + pad):  # for each column
                # region of interest (ROI):
                # it has its center at current (x,y)-pixel, and is of same size as the kernel
                region_of_interest = image_padded_array[y:y + pad, x:x + pad]

                # multiply the ROI with the kernel (element-by-element) and return the sum
                k = (region_of_interest * kernel).sum()

                # the result is saved in the new image at same (x,y)-pixel
                image_output[y - pad, x - pad] = k

        # plt.imshow(image_output, cmap='gray')
        # plt.title("Output Image using {}X{} Kernel".format(base_row, base_col))
        # plt.show()

        # save from plt - as Image.save does not yield expected result
        plt.imshow(image_output, cmap='gray')
        plt.savefig("../assets/edge_"+str(round(kernel[0, 1],0))+"_gray.png")  # use kernel to modify output's name

        # array to PIL.Image
        # conversion -> *255 , uint8: https://stackoverflow.com/questions/47290668
        image_output_uint8 = image_output.astype(np.uint8)
        self.blur_gray_image = Image.fromarray(image_output_uint8, mode=base_img.mode)


if __name__ == "__main__":
    print("***Convolution tests***")

    # paths
    my_image_path = "../assets/"
    my_image_name = "aef-CSN-III-3-1_088-600x900.jpg" # "utp-0101_016v-600x900.jpg"

    # load image
    my_image = ImageToolkit(my_image_path + my_image_name)
    if my_image.image.mode == "RGB":  # assuming our image is RGB, so we save also a grayscale version
        my_image.convert_to_gray()

    # Kernel: Gaussian blur
    my_gaussian_kernel_1d = my_image.gaussian_kernel(size=5, n_channel=1)
    my_gaussian_kernel_3d = my_image.gaussian_kernel(size=5, n_channel=3)
    # Kernel: edge detection (vertical, horizontal)
    kernel_edge_detection_vertical = np.array(((1, 0, -1), (1, 0, -1), (1, 0, -1)))
    kernel_edge_detection_horizontal = np.array(((1, 1, 1), (0, 0, 0), (-1, -1, -1)))

    # Convolution grayscale: blur
    my_image.convolution_grayscale(my_gaussian_kernel_1d)
    my_image.blur_gray_image.save(fp="../assets/blur_gray_image.jpg", mode="L")

    # Convolution grayscale: edge detection
    my_image.convolution_grayscale(kernel_edge_detection_vertical)
    my_image.blur_gray_image.save(fp="../assets/edge_vertical_gray_image.jpg", mode="L")

    my_image.convolution_grayscale(kernel_edge_detection_horizontal)
    my_image.blur_gray_image.save(fp="../assets/edge_horizontal_gray_image.jpg", mode="L")

    # Convolution RGB: blur
    # TODO: wrong output image!
    my_image.convolution(my_gaussian_kernel_3d)
    my_image.blur_image.save(fp="../assets/blur_image.jpg")

