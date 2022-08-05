import csv
import numpy as np
from scipy.interpolate import CubicSpline,interp1d
import math
import matplotlib.pyplot as plt
nSamples = 500000
splineFactor = 4

splineI = np.arange(nSamples/splineFactor)
splineII = np.arange(0,nSamples/splineFactor,1.0/splineFactor)
with open('test_data1.csv', newline='') as csvfile:
    raww1 = list(csv.reader(csvfile))
with open('test_data2.csv', newline='') as csvfile:
    raww2 = list(csv.reader(csvfile))
counter = 0
raw1 = np.zeros(len(raww1))
raw2 = np.zeros(len(raww1))
for i in raww1:
    raw1[counter] = (raww1[counter][0])
    raw2[counter] = (raww2[counter][0])
    counter = counter+1

sums = [np.sum(np.abs(raw2[::4])), np.sum(np.abs(raw2[1::4])), np.sum(np.abs(raw2[2::4])), np.sum(np.abs(raw2[3::4]))]
minSumInd = np.argmin(sums)


print(sums)
print(minSumInd)

if minSumInd > 1:
    k = -2
else:
    k = 2

print(splineI[0])
print(splineI[0])
#sensors = [interp1d(splineI,raw1[minSumInd+k::4],fill_value="extrapolate")(splineII), interp1d(splineI,raw1[minSumInd::4],fill_value="extrapolate")(splineII), interp1d(splineI,raw2[minSumInd+k::4],fill_value="extrapolate")(splineII), interp1d(splineI,raw2[minSumInd::4],fill_value="extrapolate")(splineII)]
sensors = [CubicSpline(splineI,raw1[minSumInd+k::4])(splineII), CubicSpline(splineI,raw1[minSumInd::4])(splineII), CubicSpline(splineI,raw2[minSumInd+k::4])(splineII), CubicSpline(splineI,raw2[minSumInd::4])(splineII)]
print(len(sensors[0]))

print(k)
if k > 0:
    sensors[0] = np.concatenate(([0,0,0],sensors[0][:-3]))
    sensors[2] = np.concatenate(([0,0,0],sensors[2][:-3]))
    sensors[1] = np.concatenate(([0],sensors[1][:-1]))
    sensors[3] = np.concatenate(([0],sensors[3][:-1]))
else:
    sensors[1] = np.concatenate(([0,0,0],sensors[1][:-2]))
    sensors[3] = np.concatenate(([0,0,0],sensors[3][:-2]))
    sensors[0] = np.concatenate(([0],sensors[0][:-1]))
    sensors[2] = np.concatenate(([0],sensors[2][:-1]))

top = sensor_t[2]-sensor_t[1]
bottom = 2*(sensor_t[1]-sensor_t[0])
theta = np.arctan(top/bottom)
if (np.sign(top) < 0) and (np.sign(bottom) > 0):
    theta = theta+np.pi
if (np.sign(top) > 0) and (np.sign(bottom) > 0):
    theta = theta-np.pi
degree = theta*180/np.pi