function y = doLowPass(x)
%DOFILTER Filters input x and returns output y.

% MATLAB Code
% Generated by MATLAB(R) 9.10 and DSP System Toolbox 9.12.
% Generated on: 24-Jan-2022 12:51:16

persistent Hd;

if isempty(Hd)
    
    Fpass = 50000;    % Passband Frequency
    Fstop = 60000;    % Stopband Frequency
    Apass = 1;        % Passband Ripple (dB)
    Astop = 60;       % Stopband Attenuation (dB)
    Fs    = 1000000;  % Sampling Frequency
    
    h = fdesign.lowpass('fp,fst,ap,ast', Fpass, Fstop, Apass, Astop, Fs);
    
    Hd = design(h, 'kaiserwin', ...
        'MinOrder', 'any');
    
    
    
    set(Hd,'PersistentMemory',true);
    
end

y = filter(Hd,x);

