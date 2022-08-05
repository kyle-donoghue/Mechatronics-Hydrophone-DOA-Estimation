# Mechatronics-Hydrophone-DOA-Estimation
 My work with using passive hydrophones to estimate the direction of arrival of an underwater acoustic ping. This project was made for SDSU's Mechatronics team for deployment on an autonomous submarine competition.

# How It Works
2 passive Aquarius hydrophones are connected to an op-amp network for amplification that is conneected to an Analog Digilent Discovery 2. The ADD2 is used as an ADC with their provided SDK. Python is then used for real-time capturing and processing of hydrophone data using threaded workloads and queues to send data between the workloads. Data is captured at 500kS/s at 12-bits continuously with a 1 second buffer on the recording thread and then is sent to the processing thread.

The processing thread then uses a 127-order bandpass filter from 20k-40kHz. Then, an energy vs. time signal is made in that frequency spectrum to see where in the buffer the spike of the acoustic ping is. Once the spike is located, the first couple of periods of it is taken and used for direction-of-arrival estimation. Next, the STFT is taken of the signal and a new signal is made up of the peaks of every FT in the series. This new complex signal has all the data we need for DOA estimation due to its time relative information on phase differences of the two hydrophones.

Now, Burg's Maximum Entropy Method is used on this new complex signal to calculate the power at each possible degree to make a power vs. degree signal. Finally, the peaks of this signal are found and their indices in the array correspond to the DOA. Finally, the first peak's index is is sent out to the main computer for use in autonomous direction.