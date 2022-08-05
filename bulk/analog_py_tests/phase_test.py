from cmath import pi
import csv
import numpy as np
from scipy.interpolate import CubicSpline,interp1d
import math

t1 = -1.83146095627243
t2 = -2.04915283913527
t3 = 2.46521605512763

top = t3-t2
bottom = 2*(t2-t1)
theta = np.arctan2(top,bottom)
print(theta)
if (np.sign(top) < 0) and (np.sign(bottom) > 0):
    theta = theta+np.pi
if (np.sign(top) > 0) and (np.sign(bottom) > 0):
    theta = theta-np.pi
degree = theta*180/np.pi
print(degree)