clc; clear; close all;
%define signals with 45 or 30 degree angle
Fs = 150000;
t = 0:1/(4*Fs):.001;
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
SIGNAL_THETA = 26;
SIGNAL_FREQUENCY = 30000;
SOUND_SPEED = 1500;

signal_rad = SIGNAL_THETA*pi/180;
signal_location = [-SIGNAL_DISTANCE*sin(signal_rad),SIGNAL_DISTANCE*cos(signal_rad)];
sensor_1 = [0,.009375];
sensor_2 = [-.009375,0];
sensor_3 = [.009375,0];

d_1 = abs(cos(signal_rad)*(signal_location(2)-sensor_1(2))-sin(signal_rad)*(signal_location(1)-sensor_1(1)));
d_2 = abs(cos(signal_rad)*(signal_location(2)-sensor_2(2))-sin(signal_rad)*(signal_location(1)-sensor_2(1)));
d_3 = abs(cos(signal_rad)*(signal_location(2)-sensor_3(2))-sin(signal_rad)*(signal_location(1)-sensor_3(1)));

format longg

t_1 = d_1/SOUND_SPEED;
t_2 = d_2/SOUND_SPEED;
t_3 = d_3/SOUND_SPEED;

x1 = .001*sin(2*pi*SIGNAL_FREQUENCY*(t-t_1));
x2 = .001*sin(2*pi*SIGNAL_FREQUENCY*(t-t_2));
x3 = .001*sin(2*pi*SIGNAL_FREQUENCY*(t-t_3));
% figure;
% plot(x1);
% hold;
% plot(x2);
% plot(x3);

X1 = x1(1:4:end);
X2 = x2(2:4:end);X2 = [0 X2];
X3 = x3(3:4:end);X3 = [0 0 X3];
factor = 4;
i = 0:length(X1)-1;
ii = 0:1/factor:length(X1)-1;
X1 = spline(i,X1,ii);

x1(2:4:end) = nan;x1(3:4:end) = nan;x1(4:4:end) = nan;
x2(1:4:end) = nan;x2(3:4:end) = nan;x2(4:4:end) = nan;
x3(1:4:end) = nan;x3(2:4:end) = nan;x3(4:4:end) = nan;

plot(x1,'*');
hold;
plot(X1);
