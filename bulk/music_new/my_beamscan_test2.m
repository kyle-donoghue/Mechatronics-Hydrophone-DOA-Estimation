clc;clear;close all;

numElements = 2;

ula = phased.ULA('NumElements',numElements,'ElementSpacing',0.015);


c = physconst('LightSpeed');
c = 1500;
fc = 30e3;              % Operating frequency
lambda = c/fc;
pos = getElementPosition(ula)/lambda;

Fs = 2000e3;
td = [0.00666093090230316,0.00666666666666667,0.00667240243103018];
x_in = zeros(2,100);
t = 0:1/Fs:.0001;
for i = 1:100
	d = rand(1,1);
	a1 = sin(2*pi*30e3.*(t-d-td(1)))+.1*sin(1000e3.*t);
	a2 = sin(2*pi*30e3.*(t-d-td(2)));
	%a3 = sin(2*pi*30e3.*(t-d-td(3)));
	A1 = fft(a1)./length(a1);
	A2 = fft(a2)./length(a2);
	%A3 = fft(a3)./length(a3);
	[~,p1] = max(abs(A1));
	[~,p2] = max(abs(A2));
	%[~,p3] = max(abs(A3));
	x_in(1,i) = A1(p1)/(abs(A1(p1)));
	x_in(2,i) = A2(p2)/(abs(A2(p2)));
	%x_in(3,i) = A3(p3)/(abs(A3(p3)));
end
figure;hold;
plot(a1)
plot(a2)
%plot(a3)

signal = x_in';

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