clc; clear; close all;
Fs = 800000;
t = 0:1/Fs:.0005;
SIGNAL_DISTANCE = 10;
SIGNAL_THETA = -51
SIGNAL_FREQUENCY = 40000;
SOUND_SPEED = 1500;

for SIGNAL_FREQUENCY = 10000:100:50000
    for SIGNAL_THETA = -100:100
        signal_rad = SIGNAL_THETA*pi/180;
        signal_location = [-SIGNAL_DISTANCE*sin(signal_rad),SIGNAL_DISTANCE*cos(signal_rad)];
        loc = .011;
        sensor_1 = [0,.0075];
        
        d_1 = abs(cos(signal_rad)*(signal_location(2)-sensor_1(2))-sin(signal_rad)*(signal_location(1)-sensor_1(1)));
        
        format longg
        
        t_1 = d_1/SOUND_SPEED;
        
        
        x1 = .001*sin(2*pi*SIGNAL_FREQUENCY*(t-t_1));
        y1 = fft(x1);
        [~,p1] = max(abs(y1));
        t1 = angle(y1(p1));
        
        
        X1 = x1(1:4:end);
        y1 = fft(X1);
        [~,p1] = max(abs(y1));
        t2 = angle(y1(p1));
        
        
        x1 = [0 0 x1];
        y1 = fft(x1);
        [~,p1] = max(abs(y1));
        t3 = angle(y1(p1));
        
        X1 = x1(1:4:end);
        y1 = fft(X1);
        [~,p1] = max(abs(y1));
        t4 = angle(y1(p1));
        
        diff(SIGNAL_THETA+101) = t4-t2;
    end
means(SIGNAL_FREQUENCY/100-99) = mean(diff);
end
plot((10000:100:50000),means)