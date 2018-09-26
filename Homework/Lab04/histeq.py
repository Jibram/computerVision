import numpy as np
import pylab as plt
from skimage import io

def imhist(im):
  # calculates normalized histogram of an image
  # you will see the reason for normalization later
	m, n = im.shape
	h = [0.0] * 256
	for i in range(m):
		for j in range(n):
			h[im[i, j]]+=1
	return np.array(h)/(m*n)

def cumsum(h):
	# finds cumulative sum of a numpy array, list
	return [sum(h[:i+1]) for i in range(len(h))]

def histeq(im):
	# calculate Histogram
	h = imhist(im)

	# cumulative distribution function
	# np.array will enable multiplication below
	cdf = np.array(cumsum(h))

	#transfer function 
	transfer = np.uint8(255 * cdf)
	
	s1, s2 = im.shape
	Y = np.zeros_like(im)
	# apply transfered values for each pixel
	for i in range(0, s1):
		for j in range(0, s2):
			Y[i, j] = transfer[im[i, j]]

        # new histogram
	H = imhist(Y)
	
	#return transformed image, original and new histogram, 
	# and transform function
	return Y , h, H, transfer

img = np.uint8(io.imread('HB.jpg',as_grey=True)*255.0)

# main part
new_img, h, new_h, tf = histeq(img)

# show original image
plt.subplot(121)
plt.imshow(img)
plt.title('original image')
plt.set_cmap('gray')
# show original image
plt.subplot(122)
plt.imshow(new_img)
plt.title('hist. equalized image')
plt.set_cmap('gray')
plt.show()

# plot histograms and transfer function
fig = plt.figure()

fig.add_subplot(221)
plt.plot(h)
plt.title('Original histogram')

fig.add_subplot(222)
plt.plot(new_h)
plt.title('New histogram')

fig.add_subplot(223)
plt.plot(tf)
plt.title('Transfer function')

plt.show()
