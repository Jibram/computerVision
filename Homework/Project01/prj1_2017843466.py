import numpy as np
from skimage import io, color, filters
import scipy.ndimage
import scipy.misc
from queue import *


def imhist(im):
    m, n = im.shape
    h = [0.0] * 256
    for i in range(m):
        for j in range(n):
            h[im[i, j]] += 1
    return np.array(h) / (m * n)


def cumsum(h):
    return [sum(h[:i + 1]) for i in range(len(h))]


def histeq(im):
    h = imhist(im)
    cdf = np.array(cumsum(h))
    transfer = np.uint8(255 * cdf)

    s1, s2 = im.shape
    Y = np.zeros_like(im)
    for i in range(0, s1):
        for j in range(0, s2):
            Y[i, j] = transfer[im[i, j]]
    H = imhist(Y)
    return Y


#Used as provided in Lab 4
def filter(image, kernel):
    radius = kernel.shape[0]//2
    kh, kw = kernel.shape
    height, width = image.shape
    result = np.zeros( image.shape )
    for y in range(height-kh):
        for x in range(width-kw):
            for v in range(kh):
                for u in range(kw):
                    result[y,x] += image[y+v,x+u]*kernel[v,u]
    return result

def edge(image, count):
    #Sobel Kernels
    sobel_x = np.zeros((3,3))
    sobel_x[0,0] = -1
    sobel_x[1,0] = -2
    sobel_x[2,0] = -1
    sobel_x[0,2] = 1
    sobel_x[1,2] = 2
    sobel_x[2,2] = 1
    sobel_y = sobel_x.transpose()

    # Loop should be here for all images
    #Pre-processing
    im_input = color.rgb2gray(image)
    im_input = (im_input+1)/2*255
    im_input = scipy.ndimage.filters.gaussian_filter(im_input, 1.2)

    # Sobel operator
    im_Gy = filter(im_input, sobel_y)
    im_Gx = filter(im_input, sobel_x)
    gradient = np.sqrt(np.square(im_Gy) + np.square(im_Gx))
    orientation = (np.arctan2(im_Gy, im_Gx) * 180 / np.pi) + 180

    # # Orientation fixer
    # for i in range(orientation.shape[0]):
    #     for j in range(orientation.shape[1]):
    #         if orientation[i,j] < 0:
    #             orientation[i,j] += 180
    #         if orientation[i,j] <= 22.5 or 157.5 < orientation[i,j]:
    #             orientation[i,j] = 0
    #         elif 22.5 < orientation[i,j] and orientation[i,j] <= 67.5:
    #             orientation[i,j] = 45
    #         elif 67.5 < orientation[i,j] and orientation[i,j] <= 112.5:
    #             orientation[i,j] = 90
    #         elif 112.5 < orientation[i,j] and orientation[i,j] <= 157.5:
    #             orientation[i,j] = 135

    # # Get all local maximums and thin the lines
    # for i in range(1,gradient.shape[0]-1):
    #     for j in range(1,gradient.shape[1]-1):
    #         if orientation[i,j] == 0:
    #             if gradient[i,j] < gradient[i-1,j] or gradient[i,j] < gradient[i+1,j]:
    #                 gradient[i,j] = 0
    #         elif orientation[i,j] == 45:
    #             if gradient[i,j] < gradient[i-1,j-1] or gradient[i,j] < gradient[i+1,j+1]:
    #                 gradient[i,j] = 0
    #         elif orientation[i,j] == 90:
    #             if gradient[i,j] < gradient[i,j-1] or gradient[i,j] < gradient[i,j+1]:
    #                 gradient[i,j] = 0
    #         elif orientation[i,j] == 135:
    #             if gradient[i,j] < gradient[i-1,j+1] or gradient[i,j] < gradient[i+1,j-1]:
    #                 gradient[i,j] = 0
    gradient = gradient/np.max(gradient) * 255
    gradient = gradient.astype(int)
    gradient = histeq(gradient)
    #
    # # # Hysteresis
    # brightpixels = Queue()
    threshup = 245
    threshdown = 160
    # for i in range(gradient.shape[0]):
    #    for j in range(gradient.shape[1]):
    #        if(gradient[i,j] <= threshdown):
    #            gradient[i,j] = 0
    #        elif(gradient[i,j] >= threshup):
    #            brightpixels.put((i,j))
    #
    #
    # for p in range(brightpixels.qsize()):
    #     # Check directions of edge
    #     tuple = brightpixels.get()
    #     i = tuple[0]
    #     j = tuple[1]
    #     #Check Right
    #     if j+1 < gradient.shape[1]:
    #         if gradient[i,j+1] > threshdown:
    #             gradient[i,j+1] = 255
    #             brightpixels.put((i,j+1))
    #     #Check Left
    #     if j-1 >= 0:
    #         if gradient[i,j-1] > threshdown:
    #             gradient[i,j-1] = 255
    #             brightpixels.put((i,j-1))
    #
    #     #Check Up
    #     if i-1 >= 0:
    #         if gradient[i-1,j] > threshdown:
    #             gradient[i-1,j] = 255
    #             brightpixels.put((i-1,j))
    #
    #     #Check Down
    #     if i+1 < gradient.shape[0]:
    #         if gradient[i+1,j] > threshdown:
    #             gradient[i+1,j] = 255
    #             brightpixels.put((i+1,j))
    #
    #     #Check LeftUp
    #     if i-1 >= 0 and j-1 >= 0:
    #         if gradient[i-1,j-1] > threshdown:
    #             gradient[i-1,j-1] = 255
    #             brightpixels.put((i-1,j-1))
    #
    #     #Check LeftDown
    #     if i-1 >= 0 and j+1 < gradient.shape[1]:
    #         if gradient[i-1, j+1] > threshdown:
    #             gradient[i-1, j+1] = 255
    #             brightpixels.put((i-1,j+1))
    #
    #     #Check RightUp
    #     if i+1 < gradient.shape[0] and j-1 >= 0:
    #         if gradient[i+1, j-1] > threshdown:
    #             gradient[i+1,j-1] = 255
    #             brightpixels.put((i+1,j-1))
    #
    #     #Check RightDown
    #     if i+1 < gradient.shape[0] and j+1 < gradient.shape[1]:
    #         if gradient[i+1,j+1] > threshdown:
    #             gradient[i+1,j+1] = 255
    #             brightpixels.put((i+1,j+1))

    #gradient[gradient < threshup] = 0
    #gradient[gradient >= threshup] = 255

    io.imsave('image' + str(count) + '.jpg', gradient)
    return gradient
