import numpy as np
from skimage import io, color, filters
import scipy.ndimage
import scipy.misc
from queue import *
import time


def imhist(im):
    # calculates normalized histogram of an image
    # you will see the reason for normalization later
    m, n = im.shape
    h = [0.0] * 256
    for i in range(m):
        for j in range(n):
            h[im[i, j]] += 1
    return np.array(h) / (m * n)


def cumsum(h):
    # finds cumulative sum of a numpy array, list
    return [sum(h[:i + 1]) for i in range(len(h))]


def histeq(im):
    # calculate Histogram
    h = imhist(im)

    # cumulative distribution function
    # np.array will enable multiplication below
    cdf = np.array(cumsum(h))

    # transfer function
    transfer = np.uint8(255 * cdf)

    s1, s2 = im.shape
    Y = np.zeros_like(im)
    # apply transfered values for each pixel
    for i in range(0, s1):
        for j in range(0, s2):
            Y[i, j] = transfer[im[i, j]]

    # new histogram
    H = imhist(Y)

    # return transformed image, original and new histogram,
    # and transform function
    return Y


#Used as provided in Lab 4
def filter(image, kernel):
    radius = kernel.shape[0]//2
    kh, kw = kernel.shape
    height, width = image.shape
    result = np.zeros( image.shape )
    # it should be range(radius, height-radius)
    # but omitted for simplicity
    # plz try it by yourself
    # indexing will be quite complicated
    for y in range(height-kh):
        for x in range(width-kw):
            for v in range(kh):
                for u in range(kw):
                    result[y,x] += image[y+v,x+u]*kernel[v,u]
    return result

def edge(image):
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

    # Orientation fixer
    for i in range(orientation.shape[0]):
        for j in range(orientation.shape[1]):
            if orientation[i,j] < 0:
                orientation[i,j] += 180
            if orientation[i,j] <= 22.5 or 157.5 < orientation[i,j]:
                orientation[i,j] = 0
            elif 22.5 < orientation[i,j] and orientation[i,j] <= 67.5:
                orientation[i,j] = 45
            elif 67.5 < orientation[i,j] and orientation[i,j] <= 112.5:
                orientation[i,j] = 90
            elif 112.5 < orientation[i,j] and orientation[i,j] <= 157.5:
                orientation[i,j] = 135

    # Get all local maximums and thin the lines
    for i in range(1,gradient.shape[0]-1):
        for j in range(1,gradient.shape[1]-1):
            if orientation[i,j] == 0:
                if gradient[i,j] < gradient[i-1,j] or gradient[i,j] < gradient[i+1,j]:
                    gradient[i,j] = 0
            elif orientation[i,j] == 45:
                if gradient[i,j] < gradient[i-1,j-1] or gradient[i,j] < gradient[i+1,j+1]:
                    gradient[i,j] = 0
            elif orientation[i,j] == 90:
                if gradient[i,j] < gradient[i,j-1] or gradient[i,j] < gradient[i,j+1]:
                    gradient[i,j] = 0
            elif orientation[i,j] == 135:
                if gradient[i,j] < gradient[i-1,j+1] or gradient[i,j] < gradient[i+1,j-1]:
                    gradient[i,j] = 0
    gradient = gradient/np.max(gradient) * 255
    gradient = gradient.astype(int)
    gradient = histeq(gradient)

    # # Hysteresis
    brightpixels = Queue()
    threshup = 240
    threshdown = 160
    for i in range(gradient.shape[0]):
       for j in range(gradient.shape[1]):
           if(gradient[i,j] <= threshdown):
               gradient[i,j] = 0
           elif(gradient[i,j] >= threshdown): # changed this to threshdown for without canny results
               # brightpixels.put((i,j))
               gradient[i,j] = 1


    # # Iteratively check all bright pixel neighbors
    # for p in range(brightpixels.qsize()):
    #     # Check directions of edge
    #     tuple = brightpixels.get()
    #     i = tuple[0]
    #     j = tuple[1]
    #     # check right and left
    #     if orientation[tuple[0],tuple[1]] == 90:
    #         if tuple[1]-1 > 0:
    #             if gradient[i, j-1] > threshdown:
    #                 gradient[i, j-1] = 255
    #                 brightpixels.put((i, j-1))
    #         if j+1 < gradient.shape[1]:
    #             if gradient[i, j+1] > threshdown:
    #                 gradient[i, j+1] = 255
    #                 brightpixels.put((i, j+1))
    #     #check upright and downleft
    #     elif orientation[tuple[0],tuple[1]] == 135:
    #         if i-1 > 0 and j+1 < gradient.shape[1]:
    #             if gradient[i-1, j+1] > threshdown:
    #                 gradient[i-1, j+1] = 255
    #                 brightpixels.put((i-1,j+1))
    #         if i+1 < gradient.shape[0] and j-1 >= 0:
    #             if gradient[i+1, j-1] > threshdown:
    #                 gradient[i+1, j-1] = 255
    #                 brightpixels.put((i+1,j-1))
    #     #check up and down
    #     elif orientation[tuple[0],tuple[1]] == 0:
    #         if i-1 > 0:
    #             if gradient[i-1, j] > threshdown:
    #                 gradient[i-1, j] = 255
    #                 brightpixels.put((i-1, j))
    #         if i+1 < gradient.shape[0]:
    #             if gradient[i+1, j] > threshdown:
    #                 gradient[i-1, j] = 255
    #                 brightpixels.put((i+1), j)
    #     #check upleft, downright
    #     elif orientation[tuple[0],tuple[1]] == 45:
    #         if i-1 > 0 and j >= 0:
    #             if gradient[i-1, j-1] > threshdown:
    #                 gradient[i-1, j-1] = 255
    #                 brightpixels.put((i-1, j-1))
    #         if i+1 < gradient.shape[0] and j+1 < gradient.shape[1]:
    #             if gradient[i+1, j+1] > threshdown:
    #                 gradient[i+1, j+1] = 255
    #                 brightpixels.put((i+1, j+1))

    #gradient[gradient < threshdown] = 0
    #gradient[gradient >= threshdown] = 1

    return gradient

