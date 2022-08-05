clc;clear;close all;
%% Set up the Import Options and import the data
opts = delimitedTextImportOptions("NumVariables", 4);

% Specify range and delimiter
opts.DataLines = [1, Inf];
opts.Delimiter = ",";

% Specify column names and types
opts.VariableNames = ["VarName1", "VarName2", "VarName3", "VarName4"];
opts.VariableTypes = ["double", "double", "double", "double"];

% Specify file level properties
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";
% 1
%2 3
% Import the data
testdataa = readtable("G:\My Drive\mechatronics\analog_py_final\2sensorRawDeg.csv", opts);
load('highpass.mat');
load('bandpass.mat');
load('bandpass2.mat');
load('lowpass.mat');

%% Clear temporary variables
clear opts

A = table2array(testdataa);

%A(:,1) = filter(bandpass,1,A(:,1));
%A(:,2) = filter(bandpass,1,A(:,2));
figure;
plot(A(:,1));hold;
plot(A(:,2));
legend('1','2');

A(:,1) = filter(Hbp.Numerator,1,A(:,1));
A(:,2) = filter(Hbp.Numerator,1,A(:,2));
test = conv(A(:,1),Hbp.Numerator,'Same');
figure;hold;
plot(A(:,1))
plot(test)

% A(:,1) = filter(lowpass,1,A(:,1));
% A(:,2) = filter(lowpass,1,A(:,2));
% A(:,1) = filter(highpass,1,A(:,1));
% A(:,2) = filter(highpass,1,A(:,2));


figure;
plot(A(:,1));hold;
plot(A(:,2));
legend('1','2');

N = length(A(:,1));
n = 50;
energies = zeros(1,N/n);
for i = 1:N/n
	%i=4 for 19200 and i=8 for 44800 @ 125
	%i=2 for 19200 and i=4 for 44800 @ 50
	%i=3 for 16000 and i=7 for 48000 @ 100
	x = A((1+(i-1)*n):i*n,1);
	X = fft(x);
	avg_energy = mean(abs(X(2:4)));
	energies(i) = avg_energy;
end
figure;
plot(energies)
n_index = (find(energies>=max(energies)*.4));
n_index = (find(energies>=.1));
n_index(1)
len = 200;
slength = 50;
start = (n_index(1))*n
small1 = A(start:start+len,1);
small2 = A(start:start+len,2);
figure;
plot(small1);hold;
plot(small2);
legend('1','2');
for i = 1:len/slength
	a1 = small1(1+(i-1)*slength:i*slength);
	a2 = small2(1+(i-1)*slength:i*slength);
	A1 = fft(a1)./length(a1);
	A2 = fft(a2)./length(a2);
	[~,p1] = max(abs(A1));
	[~,p2] = max(abs(A2));
	x_in(1,i) = A1(p1)/(abs(A1(p1)));
	x_in(2,i) = A2(p2)/(abs(A2(p2)));
end



signal = x_in';

fc = 30e3;
numElements = 2;
ula = phased.ULA('NumElements',numElements,'ElementSpacing',0.015);
spatialspectrum = phased.BeamscanEstimator('SensorArray',ula,...
            'OperatingFrequency',fc,'ScanAngles',-90:90,'PropagationSpeed',1500);
spatialspectrum.DOAOutputPort = true;
spatialspectrum.NumSignals = 1;
[~,ang] = spatialspectrum(signal)
figure;
plotSpectrum(spatialspectrum);
mvdrspatialspect = phased.MVDREstimator('SensorArray',ula,...
        'OperatingFrequency',fc,'ScanAngles',-90:90,...
        'DOAOutputPort',true,'NumSignals',1,'PropagationSpeed',1500);
[~,ang] = mvdrspatialspect(signal)
figure;
plotSpectrum(mvdrspatialspect);
musicspatialspect = phased.MUSICEstimator('SensorArray',ula,...
        'OperatingFrequency',fc,'ScanAngles',-90:90,...
        'DOAOutputPort',true,'NumSignalsSource','Property','NumSignals',1,'PropagationSpeed',1500);
[~,ang] = musicspatialspect(signal)
figure;
plotSpectrum(musicspatialspect)