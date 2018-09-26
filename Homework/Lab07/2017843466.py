import numpy as np
import math
from scipy import ndimage as nd
from skimage import io, feature, draw


def harris(image, sigma=2, radius=3, alpha=0.04, thresh=0.2):
    kernel_deriv = np.array([[1,0,-1],
                             [2,0,-2],
                             [1,0,-1]])
    im_deriv_x = nd.convolve(image, kernel_deriv, mode='constant', cval=0)
    im_deriv_y = nd.convolve(image, kernel_deriv.transpose(), mode='constant', cval=0)
    igx = nd.gaussian_filter(im_deriv_x**2, sigma, mode='constant', cval=0)
    igy = nd.gaussian_filter(im_deriv_y**2, sigma, mode='constant', cval=0)
    igxy = nd.gaussian_filter(im_deriv_x*im_deriv_y, sigma, mode='constant', cval=0)

    response = (igx*igy - igxy**2) - alpha*(igx+igy)**2
    response_dilated = nd.morphology.grey_dilation(response, size=radius)

    im_max = response == response_dilated
    im_threshed = response > thresh

    im_corner = np.logical_and(im_max, im_threshed)

    return im_corner


im_input = io.imread('geometry.jpg', as_grey=True)

im_response = harris(im_input)

idx = np.nonzero(im_response)

for (r,c) in zip(idx[0],idx[1]):
    rr,cc = draw.circle(r,c,5)
    try:
        im_input[rr,cc] = 0
    except:
        pass

io.imsave('geometry_harris.jpg', im_input)