function [output, tc] = sinc_interpolate(input_signal,Fs,factor)
%SINC_SINTERPOLATE (discrete time signal, sampling frequency, upscale
%factor) -> (interpolated signal)
% Parameters
Ts = 1/Fs;


% Generate "continuous time" signal and discrete time signal
tc = 0:(Ts/factor):((length(input_signal))/Fs);        % CT axis
tc = tc(1:length(tc)-1);
N = length(input_signal);         % number of samples

% Reconstruction by using the formula:
% xr(t) = sum over n=0,...,N-1: x(nT)*sin(pi*(t-nT)/T)/(pi*(t-nT)/T)
% Note that sin(pi*(t-nT)/T)/(pi*(t-nT)/T) = sinc((t-nT)/T)
% sinc(x) = sin(pi*x)/(pi*x) according to MATLAB
output = zeros(size(tc));
for ti = 1:length(tc)
    for n = 0:N-1
        %sinc_train(n+1,:) = sinc((tc-n*Ts)/Ts);
        output(ti) = output(ti) + input_signal(n+1)*sinc((tc(ti)-n*Ts)/Ts);
    end
end
end

