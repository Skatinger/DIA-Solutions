# imports
import argparse
import numpy as np
from PIL import Image


def get_args():
    """
    Gets the arguments from the command line
    :return: parsed arguments
    """

    # setup the parser
    parser = argparse.ArgumentParser(
        description="Image Resizing: given a path to an image, resize it by a factor of 5 and save a copy."
    )

    # helper description
    input_path_desc = "The path to the input file."
    input_image_desc = "The name of the image to open with its extension."
    output_path_desc = "The path where to output the resized file. Optional. Default=redirect to input path."
    output_image_desc = "The name of the resized image with its extension. Optional. Default='resized_image.jpg'"

    # create the argument
    parser.add_argument("-inPath", "--inputPath", help=input_path_desc, required=True)
    parser.add_argument("-inImage", "--inputImage", help=input_image_desc, required=True)
    parser.add_argument("-outPath", "--outputPath", help=output_path_desc, required=False, default="")
    parser.add_argument("-outImage", "--outputImage", help=output_image_desc, required=False,
                        default="resized_image.jpg")
    parsed_args = parser.parse_args()

    return parsed_args


def load_image(path_to_image, image_name):
    """Given path to an image, load it and return it"""
    print("Loading: ", path_to_image + image_name, " ...")
    return Image.open(path_to_image + image_name)


def save_image(path_for_save, image_pillow):
    print("Saving: ", path_for_save, " ...")
    return image_pillow.save(fp=path_for_save, format=None)


def print_image_info(input_image):
    """Printing basic information. void function."""
    print()
    print("Basic Information on image: {}".format(input_image.filename))
    print("Format: {}".format(input_image.format))
    print("Mode: {}".format(input_image.mode))
    print("Size: {}".format(input_image.size))
    print("Width: {}".format(input_image.width))
    print("Height: {}".format(input_image.height))
    print("Palette: {}".format(input_image.palette))
    print()


def resize_image_skipping(image_to_resize, resize_factor=5):
    """
    Resize the given image_to_resize. Assume it has dimensions which are a multiple of resize_factor.
    With this assumption, this function simply 'skips' some pixels based on resize_factor.
    :param image_to_resize: a Pillow image
    :param resize_factor: int representing the factor by which to reduce image.
    :return: resized Pillow image.
    """
    # size info
    input_height = image_to_resize.height  # == input_img_array.shape[0]  ???
    input_width = image_to_resize.width

    # get as Numpy Array
    # transform the Pillow image into a Numpy array to skip rows and columns of pixels
    input_img_array = np.array(copy_image)
    print("dim of input_img_arry: {}".format(input_img_array.shape))

    # RESIZE
    # 1) Reduce by Rows
    rows_to_keep = [row for row in range(0, input_height, resize_factor)]
    img_reduced_by_rows = input_img_array[rows_to_keep, :, :]
    # 2) Reduce by Cols
    cols_to_keep = [col for col in range(0, input_width, resize_factor)]
    img_reduced_by_rows_and_cols = img_reduced_by_rows[:, cols_to_keep, :]

    print("dim of output_img_arry: {}".format(img_reduced_by_rows_and_cols.shape))

    # new resized image
    new_image = Image.fromarray(img_reduced_by_rows_and_cols, mode=copy_image.mode)
    return new_image


if __name__ == '__main__':
    # arguments from user
    user_args = get_args()
    original_image = load_image(user_args.inputPath, user_args.inputImage)

    # print basic info on image loaded
    print_image_info(original_image)

    # copy image
    copy_image = original_image.copy()

    # Resize Image by skipping pixels
    resized_image = resize_image_skipping(copy_image)

    # save copied image
    # if user didn't provide output path => save to same location as original
    if user_args.outputPath == "":
        save_image(user_args.inputPath + user_args.outputImage, resized_image)
    else:
        save_image(user_args.outputPath + user_args.outputImage, resized_image)
