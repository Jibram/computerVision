# import the necessary packages
from panorama import Stitcher
import numpy as np
import imutils
import cv2

path = './images/pier/'
imageA = cv2.imread(path + '12.JPG')
imageB = cv2.imread(path + '23.JPG')

# stitch the images together to create a panorama
stitcher = Stitcher()
(result, vis) = stitcher.stitch([imageA, imageB], showMatches=True)

#cv2.imshow("Keypoint Matches", vis)
#cv2.imshow("Result", result)
# shape[0] = y shape[1] = x

#This will cut the bottom black bars off

cut = False
for x in range(0, int(.75* result.shape[1])): # sideways motion
    if cut:
        for y in range(int(.80 * result.shape[0]), result.shape[0] - 1):
            depthsum = 0
            for testlength in range(20):
                if (y + testlength) < result.shape[0]:
                    depthsum += np.sum(result[y + testlength, x])
            average = depthsum / 20
            if average == 0:
                result = result[0:y, 0:result.shape[1]]
                cut = True
#
# for x in range(int(.6 * result.shape[1]), result.shape[1] - 1):
#     depthsum = 0
#     for testlength in range(20):
#         if x + testlength < result.shape[1]:
#             depthsum += np.sum(result[0, x + testlength])
#     average = depthsum / 20
#     if average == 0:
#         result = result[0:result.shape[0], 0:x]

cv2.imwrite(path + "123.JPG", result)