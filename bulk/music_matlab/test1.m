clc; clear; close all;
%define signals with 45 or 30 degree angle
Fs = 800000;
t = 0:1/Fs:.00005;
%t = 0:.000001:.00050;
%x2 = .001*sin(2*pi*40000*t);
%x2 = .001*sin(2*pi*40000*(t-(.0000070710678))); % 45
%x3 = .001*sin(2x2 = .001*sin(2*pi*40000*(t-(.000005019444856))); % 30
%x1 = .001*sin(2x2 = .001*sin(2*pi*40000*(t-(.000005019444856))); % 30

%new_manual
%x1 = .001*sin(2*pi*40000*(t-.00666219));
%x2 = .001*sin(2*pi*40000*(t-.00666402));
%x3 = .001*sin(2*pi*40000*(t-.00666902));

%new_automatic
SIGNAL_DISTANCE = 10;
SIGNAL_THETA = 35
SIGNAL_FREQUENCY = 30000;
SOUND_SPEED = 1500;

signal_rad = SIGNAL_THETA*pi/180;
signal_location = [-SIGNAL_DISTANCE*sin(signal_rad),SIGNAL_DISTANCE*cos(signal_rad)];
loc = .011;
% sensor_1 = [0,.0075];
% sensor_2 = [-.0075,0];
% sensor_3 = [.0075,0];

sensor_1 = [-.015,0];
sensor_2 = [0,0];
sensor_3 = [.015,0];



d_1 = abs(cos(signal_rad)*(signal_location(2)-sensor_1(2))-sin(signal_rad)*(signal_location(1)-sensor_1(1)));
d_2 = abs(cos(signal_rad)*(signal_location(2)-sensor_2(2))-sin(signal_rad)*(signal_location(1)-sensor_2(1)));
d_3 = abs(cos(signal_rad)*(signal_location(2)-sensor_3(2))-sin(signal_rad)*(signal_location(1)-sensor_3(1)));

format longg

t_1 = d_1/SOUND_SPEED
t_2 = d_2/SOUND_SPEED
t_3 = d_3/SOUND_SPEED

t_1 = 52319/800000;
t_2 = 52316/800000;
t_3 = 52321/800000;

x1 = .001*sin(2*pi*SIGNAL_FREQUENCY*(t-t_1));
x2 = .001*sin(2*pi*SIGNAL_FREQUENCY*(t-t_2));
x3 = .001*sin(2*pi*SIGNAL_FREQUENCY*(t-t_3));

clf;
plot(t,x1);
hold;
plot(t,x2);
plot(t,x3);

x1 = [0 0 x1];
x1 = x1(1:4:end);
x2 = x2(1:4:end);
x3 = x3(1:4:end);


% X1 = x1;
% X2 = x2;
% X3 = x3;
% factor = 4;
% i = 0:length(X1)-1;
% ii = 0:1/factor:length(X1)-1;
% X1 = spline(i,X1,ii);
% i = 0:length(X2)-1;
% ii = 0:1/factor:length(X2)-1;
% X2 = spline(i,X2,ii);X2 = [0 X2];
% i = 0:length(X3)-1;
% ii = 0:1/factor:length(X3)-1;
% X3 = spline(i,X3,ii);X3 = [0 0 X3];
% min = min([length(X1) length(X2) length(X3)]);
% a = 10;
% X1 = X1(a:min-a);
% X2 = X2(a:min-a);
% X3 = X3(a:min-a);
% 
% figure;
% plot(X1);
% hold;
% plot(X2);
% plot(X3);

y1 = fft(x1);
y2 = fft(x2);
y3 = fft(x3);
% y1 = fft(X1);
% y2 = fft(X2);
% y3 = fft(X3);

figure;
plot((0:length(y1)-1)*Fs/length(y1),abs((y1)));

[~,p1] = max(abs(y1));
[~,p2] = max(abs(y2));
[~,p3] = max(abs(y3));

t1 = angle(y1(p1));      
t2 = angle(y2(p2));
t3 = angle(y3(p3));
t1
t2
t3
% t1 = t1 + 0.617399031146796;
% t1 = t1 + 0.474544;

theta = atan((t3-t2)/(2*(t3-t1)-(t3-t2)));
top = sign(t3-t2)
bottom = sign(2*(t3-t1) - (t3-t2))
if( top < 0 && bottom > 0)
        theta = theta+pi;
end
if( top > 0 && bottom > 0)
        theta = theta-pi;
end
degrees = theta*180/pi

