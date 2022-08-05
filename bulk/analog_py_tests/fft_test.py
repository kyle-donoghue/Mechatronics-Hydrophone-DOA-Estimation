import numpy as np
import matplotlib.pyplot as plt
import time

freq = 40000
times = time.time()

f1 = np.fft.rfft(x1)
f2 = np.fft.rfft(x2)
f3 = np.fft.rfft(x3)

i1 = np.argmax(np.absolute(f1))
i2 = np.argmax(np.absolute(f2))
i3 = np.argmax(np.absolute(f3))

t1 = np.angle(f1[i1]) / (2*np.pi*freq)
t2 = np.angle(f2[i2]) / (2*np.pi*freq)
t3 = np.angle(f3[i3]) / (2*np.pi*freq)

theta = np.arctan2(-(t3-t2),-(2*(t3-t1)-(t3-t2)))

degree = theta*180/np.pi
print(time.time()- times)
print(degree)
# plt.plot(np.angle(fft_data))
# plt.show()
# print(test_data)