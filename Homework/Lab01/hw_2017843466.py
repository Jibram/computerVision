import numpy as np
from skimage import io
from skimage import color
def hw1(input,output):
    asdf = io.imread(input)
    asdf[:, :, 2] = 0
    io.imsave(output, asdf)