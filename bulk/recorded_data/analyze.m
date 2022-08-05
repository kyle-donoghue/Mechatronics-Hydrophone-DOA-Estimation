fileID = fopen('regular_data.csv','r');
format = '%f';
sizeA = [1 Inf];
reg_data = fscanf(fileID,format,sizeA);
fileID = fopen('highpass_data.csv','r');
highpass_data = fscanf(fileID,format,sizeA);
highpass_data = highpass_data(1:2000001);
fileID = fopen('rfft_data.csv','r');
rfft_data = fscanf(fileID,format,sizeA);

t = 0:1:2000000;
t = t./1000000;

clf;
plot(t,reg_data);
hold;
plot(t,highpass_data);

lowpass_data = doLowPass(highpass_data);
plot(t,lowpass_data);


figure;

fft_data1 = fft(highpass_data);
fft_data1 = fft_data1(1:1000001);
fft_data2 = fft(lowpass_data);
fft_data2 = fft_data2(1:1000001);
fft_freq = t.*(1000000/2);
fft_freq = fft_freq(1:1000001);

plot(fft_freq,fft_data1);
hold;
plot(fft_freq,fft_data2);