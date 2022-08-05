clc;clear;close all;

numElements = 2;

ula = phased.ULA('NumElements',numElements,'ElementSpacing',0.5);

ang1 = [40; 0];          % First signal
ang2 = [-20; 0];         % Second signal
angs = [ang1 ang2];
angs = [ang1];
c = physconst('LightSpeed');
fc = 300e6;              % Operating frequency
lambda = c/fc;
pos = getElementPosition(ula)/lambda;

Nsamp = 1000;

nPower = 0.5;

rs = rng(2007);
signal = sensorsig(pos,Nsamp,angs,nPower);


spatialspectrum = phased.BeamscanEstimator('SensorArray',ula,...
            'OperatingFrequency',fc,'ScanAngles',-90:90);

spatialspectrum.DOAOutputPort = true;
spatialspectrum.NumSignals = 1;

[~,ang] = spatialspectrum(signal)

plotSpectrum(spatialspectrum);

mvdrspatialspect = phased.MVDREstimator('SensorArray',ula,...
        'OperatingFrequency',fc,'ScanAngles',-90:90,...
        'DOAOutputPort',true,'NumSignals',1);
[~,ang] = mvdrspatialspect(signal)

figure;
plotSpectrum(mvdrspatialspect);

musicspatialspect = phased.MUSICEstimator('SensorArray',ula,...
        'OperatingFrequency',fc,'ScanAngles',-90:90,...
        'DOAOutputPort',true,'NumSignalsSource','Property','NumSignals',1);
[~,ang] = musicspatialspect(signal)
figure;
plotSpectrum(musicspatialspect)