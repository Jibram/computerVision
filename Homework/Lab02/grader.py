import numpy as np
from skimage import io
from math import pi
from numpy.linalg import norm

from hw_2017843466 import transform

p0 = np.array([25,50,1])

s_x = 2
s_y = 3
theta = pi*0.25
t_x = 100
t_y = 30
p1,p2,p3 = transform(p0, s_x, s_y, theta, t_x, t_y)

p1_GT = np.array([50,150,1])
p2_GT = np.array([-70.71067812,141.42135624,1. ])
p3_GT = np.array([29.28932188,171.42135624,1.])

error = sum([norm(p1-p1_GT), norm(p2-p2_GT), norm(p3-p3_GT)])
print(norm(p2-p2_GT))
print(norm(p3-p3_GT))
print(error)
