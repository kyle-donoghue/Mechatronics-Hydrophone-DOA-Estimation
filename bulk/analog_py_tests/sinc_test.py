from time import time
import numpy as np
import matplotlib.pyplot as plot
import time

fs = 2

def sinc_interpolate(x, factor):
    global fs

    Ts = 1/fs
    tc = np.linspace(0,(len(x)-1)/fs, num=(len(x)-1)*factor, endpoint=True)
    tc = np.arange(0,(len(x)-1)*factor)/(fs*factor)
    print(tc)
    N = len(x)

    output = np.zeros(len(tc))

    for ti in range(0,len(tc)) :
        #a = time.time()
        for n in range(0,N) :
            output[ti] = output[ti] + x[n]*np.sinc((tc[ti] - n*Ts)/Ts)
        #print((time.time()-a)/len(range(0,N)))

    return output, tc

td = np.linspace(0,100,num=100*fs,endpoint=False)
td = np.arange(0,100)/fs
print(td)
x = np.sin(np.pi*td)
plot.plot(td,x)
plot.show()
new_x,tc  = sinc_interpolate(x,40)
plot.plot(tc,new_x)
plot.show()