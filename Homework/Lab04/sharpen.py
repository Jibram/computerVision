import numpy as np
from skimage import io
import scipy

def filter( image, kernel ):
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

# 0.8 is multiplied not to make the result become > 1
im_input = io.imread('_image1.jpg', as_grey=True)*0.8

filter_identity = np.zeros( (3,3) )
filter_identity[1,1] = 1

filter_box = np.ones( (3,3) ) / 9

im_id = filter( im_input, filter_identity )
im_box = filter( im_input, filter_box )
im_detail = im_id - im_box
im_shapen = filter( im_input, filter_identity*2-filter_box )

io.imsave('input.png',im_input)
io.imsave('id.png',im_id)
io.imsave('box.png',im_box)
io.imsave('detail.png',im_detail)

# check below two images are identical
io.imsave('id+detail.png', im_id+im_detail)
io.imsave('shapen.png', im_shapen)
