#import necessary libraries
#from curses.ascii import FS
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
import scipy.signal
from pyargus.directionEstimation import *

#from bokeh.plotting import figure, output_notebook, show #plotting library
#from bokeh.layouts import column
#from numba import jit, cuda

def killHandler(signum, frame):
    dwf.FDwfDeviceCloseAll()
    exit(1)



bandpass = [-0.000762543314465319,-0.000724305645369472,-0.000984494352211432,-0.00122911862491052,-0.00142389976679423,-0.00153469017950949,-0.00153312604085299,-0.00140122926979750,-0.00113632378613450,-0.000753417235751696,-0.000286489800650055,0.000214701067132880,0.000689916200788670,0.00107695197712447,0.00132013887810251,0.00138098132810039,0.00124673701896968,0.000937570724276891,0.000506486606140155,3.72318061839436e-05,-0.000363421578287391,-0.000577786868271827,-0.000498007372860492,-3.79442522011021e-05,0.000846249093107109,0.00214238905076050,0.00377911190778200,0.00562149764897164,0.00748560411668774,0.00915281955540943,0.0103974678211748,0.0110141175325535,0.0108492330897348,0.00982583726398900,0.00796346518357478,0.00538467001034118,0.00231056418665930,-0.000959935451640414,-0.00408174769036933,-0.00670613221653468,-0.00852639338605883,-0.00932500184424912,-0.00901078633881544,-0.00764448502472991,-0.00544581524733708,-0.00278065221497999,-0.000125365230221961,0.00198359120833701,0.00301777377586703,0.00252652941620181,0.000207061118736044,-0.00403872053998160,-0.0100667765060851,-0.0174800368707797,-0.0256469576266667,-0.0337503735215586,-0.0408622057135920,-0.0460379837719423,-0.0484206226012449,-0.0473423697955369,-0.0424129979838848,-0.0335837179971424,-0.0211771367020029,-0.00587981471861106,0.0113046333130173,0.0291380315008487,0.0462577292801429,0.0613031874060602,0.0730465764463369,0.0805121777189118,0.0830727260034276,0.0805121777189118,0.0730465764463369,0.0613031874060602,0.0462577292801429,0.0291380315008487,0.0113046333130173,-0.00587981471861106,-0.0211771367020029,-0.0335837179971424,-0.0424129979838848,-0.0473423697955369,-0.0484206226012449,-0.0460379837719423,-0.0408622057135920,-0.0337503735215586,-0.0256469576266667,-0.0174800368707797,-0.0100667765060851,-0.00403872053998160,0.000207061118736044,0.00252652941620181,0.00301777377586703,0.00198359120833701,-0.000125365230221961,-0.00278065221497999,-0.00544581524733708,-0.00764448502472991,-0.00901078633881544,-0.00932500184424912,-0.00852639338605883,-0.00670613221653468,-0.00408174769036933,-0.000959935451640414,0.00231056418665930,0.00538467001034118,0.00796346518357478,0.00982583726398900,0.0108492330897348,0.0110141175325535,0.0103974678211748,0.00915281955540943,0.00748560411668774,0.00562149764897164,0.00377911190778200,0.00214238905076050,0.000846249093107109,-3.79442522011021e-05,-0.000498007372860492,-0.000577786868271827,-0.000363421578287391,3.72318061839436e-05,0.000506486606140155,0.000937570724276891,0.00124673701896968,0.00138098132810039,0.00132013887810251,0.00107695197712447,0.000689916200788670,0.000214701067132880,-0.000286489800650055,-0.000753417235751696,-0.00113632378613450,-0.00140122926979750,-0.00153312604085299,-0.00153469017950949,-0.00142389976679423,-0.00122911862491052,-0.000984494352211432,-0.000724305645369472,-0.000762543314465319]


#import SDK
if sys.platform.startswith("win"):
    dwf = cdll.dwf
elif sys.platform.startswith("darwin"):
    dwf = cdll.LoadLibrary("/Library/Frameworks/dwf.framework/dwf")
else:
    dwf = cdll.LoadLibrary("libdwf.so")

def kill():
    dwf.FDwfAnalogOutReset(hdwf, c_int(0))
    dwf.FDwfDeviceCloseAll()
    exit(1)

def killHandler(signum, frame):
    kill()

signal.signal(signal.SIGINT, killHandler)

#declare ctype variables
hdwf = c_int()
sts = c_byte()
hzAcq = c_double(800000)
switching = int(hzAcq.value/4)
fs = hzAcq.value
nSamples = int(1*hzAcq.value)
rawData1 = (c_double*nSamples)()
rawData2 = (c_double*nSamples)()
dataToSend1 = np.zeros(nSamples)
dataToSend2 = np.zeros(nSamples)
cAvailable = c_int()
cLost = c_int()
cCorrupted = c_int()
fLost = 0
fCorrupted = 0
sampleCount = 1
splineFactor = 4
splineI = np.arange(nSamples/splineFactor)
splineII = np.arange(0,nSamples/splineFactor,1.0/splineFactor)
stabilizeTime = 10

continuous = False

class function:
    """ function names """
    pulse = DwfDigitalOutTypePulse
    custom = DwfDigitalOutTypeCustom
    random = DwfDigitalOutTypeRandom
class trigger_source:
    """ trigger source names """
    none = trigsrcNone
    analog = trigsrcDetectorAnalogIn
    digital = trigsrcDetectorDigitalIn
    external = [None, trigsrcExternal1, trigsrcExternal2, trigsrcExternal3, trigsrcExternal4]

def generate(device_handle, channel, function, frequency, duty_cycle=50, data=[], wait=0, repeat=0, trigger_enabled=False, trigger_source=trigger_source.none, trigger_edge_rising=True):
    """
        generate a logic signal
        
        parameters: - channel - the selected DIO line number
                    - function - possible: pulse, custom, random
                    - frequency in Hz
                    - duty cycle in percentage, used only if function = pulse, default is 50%
                    - data list, used only if function = custom, default is empty
                    - wait time in seconds, default is 0 seconds
                    - repeat count, default is infinite (0)
                    - trigger_enabled - include/exclude trigger from repeat cycle
                    - trigger_source - possible: none, analog, digital, external[1-4]
                    - trigger_edge_rising - True means rising, False means falling, None means either, default is rising
    """
    # get internal clock frequency
    internal_frequency = c_double()
    dwf.FDwfDigitalOutInternalClockInfo(device_handle, byref(internal_frequency))
    
    # get counter value range
    counter_limit = c_uint()
    dwf.FDwfDigitalOutCounterInfo(device_handle, c_int(0), c_int(0), byref(counter_limit))
    
    # calculate the divider for the given signal frequency
    divider = int(-(-(internal_frequency.value / frequency) // counter_limit.value))
    
    # enable the respective channel
    dwf.FDwfDigitalOutEnableSet(device_handle, c_int(channel), c_int(1))
    
    # set output type
    dwf.FDwfDigitalOutTypeSet(device_handle, c_int(channel), function)
    
    # set frequency
    dwf.FDwfDigitalOutDividerSet(device_handle, c_int(channel), c_int(divider))
    
    # set wait time
    dwf.FDwfDigitalOutWaitSet(device_handle, c_double(wait))
    
    # set repeat count
    dwf.FDwfDigitalOutRepeatSet(device_handle, c_int(repeat))
    
    # enable triggering
    dwf.FDwfDigitalOutRepeatTriggerSet(device_handle, c_int(trigger_enabled))
    
    if not trigger_enabled:
        # set trigger source
        dwf.FDwfDigitalOutTriggerSourceSet(device_handle, trigger_source)
    
        # set trigger slope
        if trigger_edge_rising == True:
            # rising edge
            dwf.FDwfDigitalOutTriggerSlopeSet(device_handle, DwfTriggerSlopeRise)
        elif trigger_edge_rising == False:
            # falling edge
            dwf.FDwfDigitalOutTriggerSlopeSet(device_handle, DwfTriggerSlopeFall)
        elif trigger_edge_rising == None:
            # either edge
            dwf.FDwfDigitalOutTriggerSlopeSet(device_handle, DwfTriggerSlopeEither)

    # set PWM signal duty cycle
    if function == DwfDigitalOutTypePulse:
        # calculate counter steps to get the required frequency
        steps = int(round(internal_frequency.value / frequency / divider))
        # calculate steps for low and high parts of the period
        high_steps = int(steps * duty_cycle / 100)
        low_steps = int(steps - high_steps)
        dwf.FDwfDigitalOutCounterSet(device_handle, c_int(channel), c_int(low_steps), c_int(high_steps))
    
    # load custom signal data
    elif function == DwfDigitalOutTypeCustom:
        # format data
        buffer = (c_ubyte * ((len(data) + 7) >> 3))(0)
        for index in range(len(data)):
            if data[index] != 0:
                buffer[index >> 3] |= 1 << (index & 7)
    
        # load data
        dwf.FDwfDigitalOutDataSet(device_handle, c_int(channel), byref(buffer), c_int(len(data)))
    
    # start generating the signal
    dwf.FDwfDigitalOutConfigure(device_handle, c_int(True))
    return


def record_set_up():
    global dwf, hdwf, hzAcq, nSamples


    #print(DWF version
    version = create_string_buffer(16)
    dwf.FDwfGetVersion(version)
    print("DWF Version: "+str(version.value))

    #open device
    print("Opening first device")
    dwf.FDwfDeviceOpen(c_int(-1), byref(hdwf))

    if hdwf.value == hdwfNone.value:
        szerr = create_string_buffer(512)
        dwf.FDwfGetLastErrorMsg(szerr)
        print(str(szerr.value))
        print("failed to open device")
        quit()

    device_name = create_string_buffer(32)
    dwf.FDwfEnumDeviceName(c_int(0), device_name)
    print("First Device: " + str(device_name.value))

    


    #set up acquisition
    dwf.FDwfAnalogInBufferSizeSet(hdwf, c_int(8192)) #set buffer to 8kB (max record length = 8192/1M = )
    dwf.FDwfAnalogInChannelEnableSet(hdwf, c_int(0), c_bool(True))
    dwf.FDwfAnalogInChannelRangeSet(hdwf, c_int(0), c_double(10))
    dwf.FDwfAnalogInAcquisitionModeSet(hdwf, acqmodeRecord)
    dwf.FDwfAnalogInFrequencySet(hdwf, hzAcq)
    dwf.FDwfAnalogInRecordLengthSet(hdwf, c_double(-1)) # -1 infinite record length

    #set up pattern generation
    
    #generate(hdwf, 0, function.pulse, switching)

    # set up analog IO channel nodes
    # enable positive supply
    dwf.FDwfAnalogIOChannelNodeSet(hdwf, c_int(0), c_int(0), c_double(True)) 
    # set voltage to 5 V
    dwf.FDwfAnalogIOChannelNodeSet(hdwf, c_int(0), c_int(1), c_double(5.0)) 
    # enable negative supply
    dwf.FDwfAnalogIOChannelNodeSet(hdwf, c_int(1), c_int(0), c_double(True)) 
    # set voltage to -5 V
    dwf.FDwfAnalogIOChannelNodeSet(hdwf, c_int(1), c_int(1), c_double(-5.0)) 
    # master enable
    dwf.FDwfAnalogIOEnableSet(hdwf, c_int(True))
    print("Power supplies set to +/- 5V")


    #wait at least 2 seconds for the offset to stabilize
    time.sleep(stabilizeTime)

class Data_Record(threading.Thread):
    def __init__(self, data_process_thread, q):
        threading.Thread.__init__(self)
        self.data_process_thread = data_process_thread
        self.q = q

    def run(self):
        global dwf, hdwf, hzAcq, nSamples, rawData1, rawData2, cAvailable, cLost, cCorrupted, fLost, fCorrupted, timeElapsed

        print("Starting oscilloscope")
        dwf.FDwfAnalogInConfigure(hdwf, c_int(0), c_int(1))
        
        num = 0
        while True:
            cSamples = 0
            cLostNum = 0
            cCorrNum = 0
            times = time.time()
            while cSamples < nSamples:
                dwf.FDwfAnalogInStatus(hdwf, c_int(1), byref(sts))
                if cSamples == 0 and (sts == DwfStateConfig or sts == DwfStatePrefill or sts == DwfStateArmed) :
                    # Acquisition not yet started.
                    continue

                dwf.FDwfAnalogInStatusRecord(hdwf, byref(cAvailable), byref(cLost), byref(cCorrupted))
                
                cSamples += cLost.value

                if cLost.value :
                    fLost = 1
                    cLostNum += cLost.value
                if cCorrupted.value :
                    fCorrupted = 1
                    cCorrNum += cCorrupted.value
                if cAvailable.value==0 :
                    continue

                if cSamples+cAvailable.value > nSamples :
                    cAvailable = c_int(nSamples-cSamples)

                dwf.FDwfAnalogInStatusData(hdwf, c_int(0), byref(rawData1, sizeof(c_double)*cSamples), cAvailable) # get channel 1 data
                dwf.FDwfAnalogInStatusData(hdwf, c_int(1), byref(rawData2, sizeof(c_double)*cSamples), cAvailable) # get channel 2 data
                cSamples += cAvailable.value

            dataToSend1 = np.ctypeslib.as_array(rawData1)
            dataToSend2 = np.ctypeslib.as_array(rawData2)
            timeElapsed = time.time() - times
            if not continuous:
                num += 1
            queueData = [dataToSend1, dataToSend2, timeElapsed, cLost, cCorrupted, num]
            self.q.put(queueData)
            self.data_process_thread.event.set()
            
            if num == sampleCount and not continuous:
                break
        dwf.FDwfAnalogOutReset(hdwf, c_int(0))
        dwf.FDwfDeviceCloseAll()
        self.data_process_thread.join()

        print("Recording done")
        if fLost:
            print("Samples were lost! Reduce frequency")
        if fCorrupted:
            print("Samples could be corrupted! Reduce frequency")

        return
    """times = time.time() - times
    print(cSamples)
    print(times)
    print(cSamples/times)
    print()
    print(cLostNum)
    print(cCorrNum)
    print()
    print(str(cCorrNum/cSamples*100)+"%")"""
    """print("rolling data:")
    print("\t[totalsamples]\t"+str(totalSamples))
    print("\t[average samples/second]\t"+str(totalSamples/(time.time()-totalTime)))
    print("\t [samples lost]\t"+str(cLostNum))
    print("\t [samples corrupt]\t"+str(cCorrNum))"""

        

class Data_Process(threading.Thread):
    def __init__(self, event, q):
        threading.Thread.__init__(self)
        self.event = event
        self.q = q
        self.bigdata1 = np.empty(0)
        self.bigdata2 = np.empty(0)

                      
    def run(self):
        print("waiting for signal")
        while True:
            self.event.wait()

            queueData = self.q.get()

            procTimeElapsed = time.time()
            
            num = queueData[5]
            raw1 = queueData[0]
            raw2 = queueData[1]
            timeElapsed = queueData[2]
            lostNum = queueData[3].value
            corrNum = queueData[4].value
            





            # sums = [np.sum(np.abs(raw2[::4])), np.sum(np.abs(raw2[1::4])), np.sum(np.abs(raw2[2::4])), np.sum(np.abs(raw2[3::4]))]
            # minSumInd = np.argmin(sums)

            # if minSumInd > 1:
            #     k = -2
            # else:
            #     k = 2


            # sensors = [CubicSpline(splineI,raw1[minSumInd+k::4])(splineII)[199:-201], CubicSpline(splineI,raw1[minSumInd::4])(splineII)[199:-201], CubicSpline(splineI,raw2[minSumInd+k::4])(splineII)[199:-201], CubicSpline(splineI,raw2[minSumInd::4])(splineII)[199:-201]]
            # if k > 0:
            #     sensors[0] = np.concatenate(([0,0],sensors[0][:-2]))
            #     sensors[2] = np.concatenate(([0,0],sensors[2][:-2]))
            # else:
            #     sensors[1] = np.concatenate(([0,0],sensors[1][:-2]))
            #     sensors[3] = np.concatenate(([0,0],sensors[3][:-2]))

            # """power_diff = np.zeros(24688)

            # for i in range(4):
            #     sensors[i] = np.append(sensors[i],power_diff)"""
            

            # sensor_t = np.zeros(3)

            # minFreqInd = int(15000*len(sensors[0])/fs)
            # maxFreqInd = int(45000*len(sensors[0])/fs)

            # #bigfft = np.zeros([3,maxFreqInd-minFreqInd])

            # for i in range(3):
            #     fft = np.fft.fft(sensors[i])
            #     fft = fft[minFreqInd:maxFreqInd]
            #     ind  = np.argmax(np.absolute(fft))
            #     # freq = i*fs/len(fft)
            #     sensor_t[i] = np.angle(fft[ind])
            #     #bigfft[i] = np.absolute(fft)
            
            # theta = np.arctan2((sensor_t[2] - sensor_t[1]),2*(sensor_t[2]-sensor_t[0])-(sensor_t[2]-sensor_t[1]))
            # degree = theta*180/np.pi


            sums = [np.sum(np.abs(raw2[::4])), np.sum(np.abs(raw2[1::4])), np.sum(np.abs(raw2[2::4])), np.sum(np.abs(raw2[3::4]))]
            minSumInd = np.argmin(sums)

            if minSumInd > 1:
                k = -2
            else:
                k = 2

            sensors = [interp1d(splineI,raw1[minSumInd+k::4],fill_value="extrapolate")(splineII), interp1d(splineI,raw1[minSumInd::4],fill_value="extrapolate")(splineII), interp1d(splineI,raw2[minSumInd+k::4],fill_value="extrapolate")(splineII), interp1d(splineI,raw2[minSumInd::4],fill_value="extrapolate")(splineII)]
            #sensors = [CubicSpline(splineI,raw1[minSumInd+k::4])(splineII), CubicSpline(splineI,raw1[minSumInd::4])(splineII), CubicSpline(splineI,raw2[minSumInd+k::4])(splineII), CubicSpline(splineI,raw2[minSumInd::4])(splineII)]

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

            target_frequency = 30e3
            sensor_distance = .015
            sound_speed = 1500

            raw1b = np.convolve(raw1,bandpass,'same')
            raw2b = np.convolve(raw2,bandpass,'same')

            smallN = len(raw1)
            smalln = 50
            energies = np.zeros(int(smallN/smalln))

            for i in range(int(smallN/smalln)):
                x = raw1b[(i*smalln):(i+1)*smalln]
                X = np.fft.rfft(x)
                avg_energy = np.mean(np.absolute(X[1:3]))
                energies[i] = avg_energy

            n_index = np.argmax(energies>=.1)+1
            length = 200
            slength = 50

            d = target_frequency*sensor_distance/sound_speed # Inter element spacing [lambda]
            M = 2  # number of antenna elements in the antenna system (ULA)
            N = int(length/slength)  # sample size used for the simulation  


            small1 = raw1b[n_index*smalln-1:n_index*smalln+length-1]
            small2 = raw2b[n_index*smalln-1:n_index*smalln+length-1]

            samples = np.zeros([int(length/slength),2],dtype = 'complex_')
            for i in range(int(length/slength)):
                small_fft1 = np.fft.rfft(small1[i*slength:(i+1)*slength])
                small_fft2 = np.fft.rfft(small2[i*slength:(i+1)*slength])
                ind = np.argmax(np.absolute(small_fft1))
                samples[i][0] = (small_fft1[ind])/np.absolute(small_fft1[ind])
                ind = np.argmax(np.absolute(small_fft2))
                samples[i][1] = (small_fft2[ind])/np.absolute(small_fft2[ind])

            final_samples =np.transpose(samples)

            R = corr_matrix_estimate(final_samples.T, imp="mem_eff")

            array_alignment = np.arange(0, M, 1)* d
            incident_angles= np.arange(0,181,1)
            ula_scanning_vectors = gen_ula_scanning_vectors(array_alignment, incident_angles)

            MEM = np.absolute(DOA_MEM(R,ula_scanning_vectors, column_select = 1))
            peaks = scipy.signal.find_peaks(MEM)[0]
            if len(peaks) < 1:
                max_peak = np.max(MEM)
                temp_angle = np.argmax(MEM==max_peak)
                if temp_angle == 0:
                    angle = -90
                elif temp_angle == 180:
                    angle = 90
                else:
                    angle = 180
            else:
                angle = peaks[0]-90
            



            # sensor_t = np.zeros(3)
            # minFreqInd = int(15000*len(sensors[0])/fs)
            # maxFreqInd = int(45000*len(sensors[0])/fs)

            # bigfft = np.zeros([3,maxFreqInd-minFreqInd])

            # for i in range(3):
            #     fft = np.fft.fft(sensors[i])
            #     fft = fft[minFreqInd:maxFreqInd]
            #     ind  = np.argmax(np.absolute(fft))
            #     # freq = i*fs/len(fft)
            #     sensor_t[i] = np.angle(fft[ind])
            #     bigfft[i] = np.absolute(fft)
            # print(sensor_t)



            # theta = np.arctan2((sensor_t[2] - sensor_t[1]),2*(sensor_t[2]-sensor_t[0])-(sensor_t[2]-sensor_t[1]))
            # degree = theta*180/np.pi
            # #print(degree)
            # if (np.absolute(degree) > 90):
            #     if (degree > 0):
            #         degree = degree - 180
            #     elif (degree < 0):
            #         degree = degree + 180
            
            # N = len(sensors[0])
            # mini_N = 100
            # energies = np.zeros(int(N/mini_N))

            # sensor_t = np.zeros(3)

            # for i in range(int(N/mini_N)):
            #     x = sensors[0][(i)*mini_N:(i+1)*mini_N-1]
            #     X = np.fft.fft(x)
            #     energies[i] = np.mean(np.absolute(X[2:5]))
            # n_index = np.argmax(energies>=1)+1
            # print(n_index)
            
            # for i in range(3):
            #     fft = np.fft.fft(sensors[i][n_index*mini_N-1:n_index*mini_N+50-1])
            #     fft = fft[1:25]
            #     ind = np.argmax(np.absolute(fft))
            #     sensor_t[i] = np.angle(fft[ind])
            
            # top = sensor_t[2]-sensor_t[1]
            # bottom = 2*(sensor_t[1]-sensor_t[0])
            # theta = np.arctan2(top,bottom)
            # if (np.sign(top) < 0) and (np.sign(bottom) > 0):
            #     theta = theta+np.pi
            # if (np.sign(top) > 0) and (np.sign(bottom) > 0):
            #     theta = theta-np.pi
            #degree = theta*180/np.pi

            # fft1 = np.fft.rfft(raw1)
            # fft2 = np.fft.rfft(raw2)  
            # i1 = np.argmax(np.absolute(fft1))
            # i2 = np.argmax(np.absolute(fft2))
            # t1 = np.angle(fft1[i1]) / (2*np.pi*40000)
            # t2 = np.angle(fft2[i2]) / (2*np.pi*40000)
            # theta = np.arctan2(-(t2),-(2*(t1)-(t2)))
            # degree = theta*180/np.pi

            # procTimeElapsed = time.time() - procTimeElapsed
            # print("data: "+str(num))
            # print("\t[record time]\t\t\t"+str(timeElapsed))
            # print("\t[process time]\t\t\t"+str(procTimeElapsed))
            # print("\t[samples]\t\t\t"+str(len(raw1)))
            # print("\t[average samples/second]\t"+str(nSamples/(timeElapsed)))
            # print("\t[samples lost]\t\t\t"+str(lostNum))
            # print("\t[samples corrupt]\t\t"+str(corrNum))


            
            
            """self.bigdata1 = np.append(self.bigdata1, raw1)
            self.bigdata2 = np.append(self.bigdata2, raw2)"""
            if num == sampleCount and not continuous:
                
                print("starting to export data")
                """f = open("test_dataf1.csv", "w")
                for i in range(len(bigfft[0])):
                     f.write("%s,%s,%s\n" % (bigfft[0][i],bigfft[1][i],bigfft[2][i]))#(y1[i],y2[i],y3[i],y4[i]))
                f.close()    
                f = open("test_data1.csv", "w")
                for i in range(len(rawData1)):
                    f.write("%s\n" % rawData1[i])#(y1[i],y2[i],y3[i],y4[i]))
                f.close()"""
                # f = open("test_data2.csv", "w")
                # for i in range(len(rawData2)):
                #     f.write("%s\n" % rawData2[i])#(y1[i],y2[i],y3[i],y4[i]))
                # f.close()
                # f = open("test_dataa.csv", "w")
                # for i in range(len(sensors[0])):
                #     f.write("%s,%s,%s,%s\n" % (sensors[0][i],sensors[1][i],sensors[2][i],sensors[3][i]))#(y1[i],y2[i],y3[i],y4[i]))
                # f.close()
                # f = open("test_data2sensorRaw.csv", "w")
                # for i in range(len(rawData1)):
                #     f.write("%s,%s\n" % (rawData1[i],rawData2[i]))#(y1[i],y2[i],y3[i],y4[i]))
                # f.close()

                f = open("2sensorRawDeg.csv", "w")
                for i in range(len(rawData1)):
                    f.write("%s,%s\n" % (rawData1[i],rawData2[i]))#(y1[i],y2[i],y3[i],y4[i]))
                f.close()

                # f = open("test_datae.csv", "w")
                # for i in range(len(energies)):
                #     f.write("%s\n" % (energies[i]))#(y1[i],y2[i],y3[i],y4[i]))
                # f.close()
                # testfft = np.fft.fft(sensors[0])
                # f = open("test_dataj.csv", "w")
                # for i in range(len(sensors[0])):
                #     f.write("%s,%s,%s\n" % (sensors[0][i],np.real(testfft)[i],np.imag(testfft)[i]))#(y1[i],y2[i],y3[i],y4[i]))
                # f.close()
                print("exported data")
                """f = open("test_data3.csv", "w")
                for i in range(len(sensor3)):
                    f.write("%s\n" % sensor3[i])#(y1[i],y2[i],y3[i],y4[i]))
                f.close()
                f = open("test_data4.csv", "w")
                for i in range(len(sensor4)):
                    f.write("%s\n" % sensor4[i])#(y1[i],y2[i],y3[i],y4[i]))
                f.close()
                print("exported data")
                kill()"""
            
        return

def data_communicate():
    while True:
        continue
    return

if __name__ == "__main__":

    time.sleep(1)
    record_set_up()

    q = Queue() #define queue to pass data between threads

    data_process_event = threading.Event()
    data_process_thread = Data_Process(data_process_event, q)
    data_process_thread.start() #set up process thead

    data_record_thread = Data_Record(data_process_thread, q)
    data_record_thread.start() #set up record thread

    data_record_thread.join() #start record thread


    #recording_thread = threading.Thread(target=data_record, args=())
    #processing_thread = threading.Thread(target=data_process, args=())
    communicating_thread = threading.Thread(target=data_communicate, args=())

    #recording_thread.start()
    #processing_thread.start()
    communicating_thread.start()

    #recording_thread.join()
    #processing_thread.join()
    communicating_thread.join()

