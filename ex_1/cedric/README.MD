# Document Image Analysis - course assignments
Cedric Membrez (07-406-341) cedric.membrez@unifr.ch

Github repo: https://github.com/cmembrez/mcs-doc-image-analysis

## Assignment 1a: Resizing images
A Python program that loads a given image, resizes it and saves a resized copy.

Using Pillow package, the resizing is accomplished by skipping pixels.
As such, the input image's size needs to be a multiple of the resizing factor. 
By default, the factor is 5.

The function 'resize_image_skipping()' take a Pillow image and the resize factor as inputs.
Then, given the resize factor and the size of the image, the function produces two vectors
containing the rows and columns to keep: to do so, it creates a range with step = resize factor.
The function then select the corresponding rows and columns in the original image and return 
the resized image - using Pillow's Image.fromarray

Subfolder "src" contains the code.

### usage
<pre>
usage: main.py [-h] -inPath INPUTPATH -inImage INPUTIMAGE
               [-outPath OUTPUTPATH] [-outImage OUTPUTIMAGE]
</pre>

| arguments | explanation |
| --------- | ----------- |
| -h, --help | show the help message and exit |
| -inPath | The path to the input file. |
| -inImage | The name of the image to open with its extension. |
| -outPath | The path where to output the resized file. Optional. Default=redirect to input path.|
| -outImage | The name of the resized image with its extension. Optional. Default='resized_image.jpg'|


### Helpful Reads
Some helpful read was from: https://courses.cs.vt.edu/~masc1044/L17-Rotation/ScalingNN.html

## Assignment 1b: Convolution

#### Convolution function

Gives some result assuming a grayscale image only.

For this part, the code is setup under the class ImageToolkit().
Assuming the input is a grayscale image, the method convolution_grayscale() takes a kernel as input.
Based on input image's size, an empty array is created. Because applying the kernel would reduce pixels,
a padded version of the image is created (i.e. we add a border around the image). Then using two inner loops,
one for the rows and one for the columns, we iterate over each pixel and find the region-of-interest where to
apply our kernel. Once applied, we save it to the output image at the same location. And finally save the
convoluted image using matplotlib.pyplot (PIL.Image.save is not saving the expected image for now).

#### Blur

For convenience, I used the Gaussian kernel, and it is implemented in the static method gaussian_kernel().
This method starts by using the desired size of the kernel to generate numbers using numpy. Regarding the
Gaussian distribution underlying the kernel, I assume a mean of zero, and a standard deviation based on the 
size of the kernel. Apply the Gaussian distribution to each number generated, compute the outer product and average.


#### Edge detection

Here, I used gradient kernels, one for vertical and one for horizontal edge detections. 
  For simplicity, I only generated them directly with a numpy array (of (1, 0, -1) appropriately). 


#### Current Problems:
- saving an image with PIL's Image.fromarray and Image.save do NOT yield the expected image.
instead, matplotlib has been used.
- the convolution on RGB image is NOT giving proper results.

#### Helpful reads
Some helpful reads were from: 
- convolution over volumes: https://www.youtube.com/watch?v=KTB_OFoAQcc&list=PLkDaE6sCZn6Gl29AoE31iwdVwSG-KnDzF&index=8 
- padding: https://note.nkmk.me/en/python-pillow-add-margin-expand-canvas/
- Smoothing as convolution: https://matthew-brett.github.io/teaching/smoothing_as_convolution.html
- Convolutions with OpenCV and Python: https://www.pyimagesearch.com/2016/07/25/convolutions-with-opencv-and-python/

