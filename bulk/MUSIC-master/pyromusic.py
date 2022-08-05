from turtle import numinput
import pyroomacoustics as pyro
import numpy as np
import csv
import matplotlib.pyplot as plt
import scipy


microphone_array = np.array([[-.015,0],[0,0]])
fs = 800000
nfft = 64
c = 1500
num_src = 1
mode = 'far'
music_doa = pyro.doa.music.MUSIC(microphone_array,fs,nfft,c,num_src,mode)

with open('smallsignal1.csv', newline='') as csvfile:
    raww1 = list(csv.reader(csvfile))
with open('smallsignal2.csv', newline='') as csvfile:
    raww2 = list(csv.reader(csvfile)) 
raw1 = np.zeros(400)
raw2 = np.zeros(400)
for i in range(400):
    raw1[i] = float(raww1[i][1])
    raw2[i] = float(raww2[i][1])

# fft1 = np.zeros([8,26],dtype = 'complex_')
# fft2 = np.zeros([8,26],dtype = 'complex_')

# for i in range(8):
#     small_fft1 = np.fft.rfft(raw1[i*50:(i+1)*50])
#     small_fft2 = np.fft.rfft(raw2[i*50:(i+1)*50])
#     fft1[i] = small_fft1
#     fft2[i] = small_fft2

fft1 = scipy.signal.stft(raw1,fs=800000,nperseg=64)[2]
print(fft1.shape)
fft2 = scipy.signal.stft(raw2,fs=800000,nperseg=64)[2]
big_fft = np.array([np.transpose(fft1),np.transpose(fft2)])
big_fft = np.array([(fft1),(fft2)])


music_doa.locate_sources(big_fft, freq_range=[10000,50000])

results = music_doa.grid.values
print(results.shape)
plt.plot(results)
plt.show()