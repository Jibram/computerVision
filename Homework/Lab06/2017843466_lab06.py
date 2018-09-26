import numpy as np
import math
from scipy import ndimage as nd
from skimage import io, feature

def hough(fname):
    # Read an Input image
    im = io.imread('./' + fname + '.png')

    # Normalize 0~255
    im = (im / im.max()) * 255

    # Detect edges
    im = feature.canny(im, .8)

    # prepare hough space
    RhoLimit = int(math.floor(np.linalg.norm(im.shape)))
    nRho = int(RhoLimit * 2 + 1)  # -limit to +limit
    thetaSamplingFreq = 0.0025
    nTheta = int(math.floor(math.pi / thetaSamplingFreq))
    canvas_HoughSpace = np.zeros((nRho, nTheta))
    threshold = 0.99

    # pre-calculated trigonometric functions
    cosine = np.zeros(nTheta)
    sine = np.zeros(nTheta)
    for theta in range(nTheta):
        cosine[theta] = math.cos(theta * thetaSamplingFreq)
        sine[theta] = math.sin(theta * thetaSamplingFreq)

    # run hough transform
    for y in range(im.shape[0]):
        for x in range(im.shape[1]):
            if im[y, x] != 0:
                for theta in range(nTheta):
                    rho = int(x * cosine[theta] + y * sine[theta])
                    canvas_HoughSpace[rho + RhoLimit, theta] += 1
    canvas_HoughSpace = canvas_HoughSpace / canvas_HoughSpace.max()

    # threshold
    binarized = canvas_HoughSpace > threshold
    ind = np.nonzero(binarized)

    # convert back
    rhos = ind[0] - RhoLimit
    thetas = ind[1] * thetaSamplingFreq

    # draw lines
    eps = 1e-7
    for i in range(len(thetas)):
        for x in range(im.shape[1]):
            y = int((rhos[i] - x * math.cos(thetas[i])) / (math.sin(thetas[i]) + eps))
            if y < 0 or y >= im.shape[0]:
                continue
            im[y, x] = 1
        for y in range(im.shape[0]):
            x = int((rhos[i] - y * math.sin(thetas[i])) / (math.cos(thetas[i]) + eps))
            if x < 0 or x >= im.shape[1]:
                continue
            im[y, x] = 1
    return im, canvas_HoughSpace, binarized


fname = 'points'
im, canvas_HoughSpace, binarized = hough(fname)
im = im * 255
im = im.astype(int)
io.imsave('results/points_result.png', im)
canvas_HoughSpace *= 255
canvas_HoughSpace = canvas_HoughSpace.astype(int)
io.imsave('results/points_HoughSpace.png', canvas_HoughSpace)

fname = 'rect'
im, canvas_HoughSpace, binarized = hough(fname)
im = im * 255
im = im.astype(int)
io.imsave('results/rect_result.png', im)
canvas_HoughSpace *= 255
canvas_HoughSpace = canvas_HoughSpace.astype(int)
io.imsave('results/rect_HoughSpace.png', canvas_HoughSpace)