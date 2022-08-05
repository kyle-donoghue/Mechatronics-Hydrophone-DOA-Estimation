clc;clear;close all;
Fs = 120e3;
td = [0.00666093090230316,0.00666666666666667,0.00667240243103018];
x_in = zeros(3,10);
t = 0:1/Fs:.001;
for i = 1:10
	d = rand(1,1);
	a1 = sin(2*pi*30000.*(t-d-t(1)));
	a2 = sin(2*pi*30000.*(t-d-t(2)));
	a3 = sin(2*pi*30000.*(t-d-t(3)));
	A1 = fft(a1)./length(a1);
	A2 = fft(a2)./length(a2);
	A3 = fft(a3)./length(a3);
	[~,p1] = max(abs(A1));
	[~,p2] = max(abs(A2));
	[~,p3] = max(abs(A3));
	x_in(1,i) = A1(p1);
	x_in(2,i) = A2(p2);
	x_in(3,i) = A3(p3);
end