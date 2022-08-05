import threading, signal
from ctypes import *
from dwfconstants import *
import math
import time
import matplotlib.pyplot as plt
import sys
import numpy as np
from queue import Queue
import pandas as pd
from scipy.interpolate import CubicSpline,interp1d
import statistics

CSVData = open('test_dataa.csv')

sensors = np.loadtxt(CSVData, delimiter=",")

sensors = np.transpose(sensors)

N = len(sensors[0])
mini_N = 100
energies = np.zeros(int(N/mini_N))

sensor_t = np.zeros(3)

for i in range(int(N/mini_N)):
    x = sensors[0][(i)*mini_N:(i+1)*mini_N-1]
    X = np.fft.fft(x)
    energies[i] = np.mean(np.absolute(X[2:5]))
n_index = np.argmax(energies>=1)+1
print(n_index)
n_index = 5686
print(n_index*mini_N)
print(n_index*mini_N+50)


for i in range(3):
    fft = np.fft.fft(sensors[i][n_index*mini_N-1:n_index*mini_N+50-1])
    fft = fft[1:25]
    ind = np.argmax(np.absolute(fft))
    sensor_t[i] = np.angle(fft[ind])
print(sensor_t)
top = sensor_t[2]-sensor_t[1]
bottom = 2*(sensor_t[1]-sensor_t[0])
theta = np.arctan2(top,bottom)
# if (np.sign(top) < 0) and (np.sign(bottom) > 0):
#     theta = theta+np.pi
# if (np.sign(top) > 0) and (np.sign(bottom) > 0):
#     theta = theta-np.pi
degree = theta*180/np.pi
f = open("test_datae.csv", "w")
for i in range(len(energies)):
    f.write("%s\n" % (energies[i]))#(y1[i],y2[i],y3[i],y4[i]))
f.close()
print(degree)
# plt.plot(sensors[0][n_index*mini_N-1:n_index*mini_N+50-1])
# plt.plot(sensors[1][n_index*mini_N-1:n_index*mini_N+50-1])
# plt.plot(sensors[2][n_index*mini_N-1:n_index*mini_N+50-1])
# plt.show()