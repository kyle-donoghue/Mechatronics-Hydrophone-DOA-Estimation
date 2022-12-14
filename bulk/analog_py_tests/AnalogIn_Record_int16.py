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
hzAcq = c_double(1e6)
nSamples = 2**19
rgSamples = (c_double*1000000000)()
cAvailable = c_int()
cLost = c_int()
cCorrupted = c_int()
fLost = 0
fCorrupted = 0
deviceName = create_string_buffer(64)

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
dwf.FDwfEnumDeviceName(c_int(0), byref(deviceName))
print("Name: " + str(deviceName.value))

#set up acquisition
dwf.FDwfAnalogInChannelEnableSet(hdwf, c_int(0), c_bool(True))
dwf.FDwfAnalogInChannelRangeSet(hdwf, c_int(0), c_double(5))
dwf.FDwfAnalogInAcquisitionModeSet(hdwf, acqmodeRecord)
dwf.FDwfAnalogInFrequencySet(hdwf, hzAcq)
dwf.FDwfAnalogInRecordLengthSet(hdwf, c_double(-1)) # -1 infinite record length

#wait at least 2 seconds for the offset to stabilize
time.sleep(2)

print("Starting oscilloscope")
dwf.FDwfAnalogInConfigure(hdwf, c_int(0), c_int(1))

cSamples = 0
print(nSamples)
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
        print("uo1")
        break
    if cCorrupted.value :
        fCorrupted = 1
        print("uo2")
        break

    if cAvailable.value==0 :
        continue

    if cSamples+cAvailable.value > nSamples :
        cAvailable = c_int(nSamples-cSamples)
    
    dwf.FDwfAnalogInStatusData16(hdwf, c_int(0), byref(rgSamples, sizeof(c_int16)*cSamples), c_int(0), cAvailable) # get channel 1 data
    #dwf.FDwfAnalogInStatusData16(hdwf, c_int(1), byref(rgSamples, sizeof(c_int16)*cSamples), c_int(c), Available) # get channel 2 data
    cSamples += cAvailable.value
    print(cSamples)

times = time.time() - times
print(times)
print(cSamples/times)
print("Recording done")
if fLost:
    print("Samples were lost! Reduce frequency")
if fCorrupted:
    print("Samples could be corrupted! Reduce frequency")


dwf.FDwfAnalogOutReset(hdwf, c_int(0))
dwf.FDwfDeviceCloseAll()

"""
f = open("record.csv", "w")
for v in rgSamples:
    f.write("%s\n" % v)
f.close()

plt.plot(numpy.fromiter(rgSamples, dtype = numpy.int16))
plt.show()"""


