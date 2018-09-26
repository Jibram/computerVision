import numpy as np
from skimage import io
from skimage import color
from math import cos
from math import sin

# affineArray = np.array([ [s_x*cos(theta),    -sin(theta), t_x],
#                          [    sin(theta), s_y*cos(theta), t_y],
#                          [             0,              0,   1] ])


def transform(p0, s_x, s_y, theta, t_x, t_y):

    # Scale by s_x. s_y
    s = np.array([ [s_x,   0, 0],
                   [  0, s_y, 0],
                   [  0,   0, 1] ])
    p1 = np.dot(s, p0)

    # Rotate by theta
    r = np.array([ [cos(theta), -sin(theta), 0],
                   [sin(theta),  cos(theta), 0],
                   [         0,           0, 1] ])
    p2 = np.dot(r, p1)

    # Translate by t_x, t_y
    t = np.array([ [1, 0, t_x],
                   [0, 1, t_y],
                   [0, 0,   1] ])
    p3 = np.dot(t, p2)
    
    return p1,p2,p3
