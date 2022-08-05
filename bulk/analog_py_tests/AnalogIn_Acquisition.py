"""
   DWF Python Example
   Author:  Digilent, Inc.
   Revision:  2018-07-19

   Requires:                       
       Python 2.7, 3
"""

from ctypes import *
from dwfconstants import *
import math
import time
import matplotlib.pyplot as plt
import sys
import numpy

if sys.platform.startswith("win"):
    dwf = cdll.dwf
elif sys.platform.startswith("darwin"):
    dwf = cdll.LoadLibrary("/Library/Frameworks/dwf.framework/dwf")
else:
    dwf = cdll.LoadLibrary("libdwf.so")

#declare ctype variables
hdwf = c_int()
sts = c_byte()
rgdSamples = (c_double*4000)()

version = create_string_buffer(16)
dwf.FDwfGetVersion(version)
print("DWF Version: "+str(version.value))

#open device
print("Opening first device")
dwf.FDwfDeviceOpen(c_int(-1), byref(hdwf))

if hdwf.value == hdwfNone.value:
    szerr = create_string_buffer(512)
    dwf.FDwfGetLastErrorMsg(szerr)
    print(szerr.value)
    print("failed to open device")
    quit()

test_str = create_string_buffer(32)
dwf.FDwfEnumDeviceName(c_int(0), byref(test_str))
print("First Device: " + str(test_str.value))
cBufMax = c_int()
dwf.FDwfAnalogInBufferSizeInfo(hdwf, 0, byref(cBufMax))
print("Device buffer size: "+str(cBufMax.value)) 

#set up acquisition
dwf.FDwfAnalogInFrequencySet(hdwf, c_double(10000000.0))
dwf.FDwfAnalogInBufferSizeSet(hdwf, c_int(4000)) 
#dwf.FDwfAnalogInRecordLengthSet(hdwf, c_double(-1)) # -1 infinite record length
dwf.FDwfAnalogInChannelEnableSet(hdwf, c_int(-1), c_bool(True))
dwf.FDwfAnalogInChannelRangeSet(hdwf, c_int(-1), c_double(5))
dwf.FDwfAnalogInChannelFilterSet(hdwf, c_int(-1), filterDecimate)

#wait at least 2 seconds for the offset to stabilize
time.sleep(2)

while True:


    print("Starting oscilloscope")
    dwf.FDwfAnalogInConfigure(hdwf, c_int(1), c_int(1))
    time_spent = time.time()
    while True:
        dwf.FDwfAnalogInStatus(hdwf, c_int(1), byref(sts))
        if sts.value == DwfStateDone.value :
            break
        time.sleep(0.1)
    time_spent = time.time() - time_spent
    print("Acquisition done: " + str(time_spent))
    print("Samples per second: " + str(4000/time_spent))

    dwf.FDwfAnalogInStatusData(hdwf, 0, rgdSamples, 4000) # get channel 1 data
    #dwf.FDwfAnalogInStatusData(hdwf, 1, rgdSamples, 4000) # get channel 2 data
dwf.FDwfDeviceCloseAll()

