% Create "continuous time" signal, Fc >> f
Fc = 1e6; % very high sample rate to simulate "continuous time"
Tc = 1/Fc; % sampling period
t = (-0.1:Tc:0.1)'; % time axis
f = 10; % signal frequency
xc = sin(2*pi*f*t); % "continuous time" signal

% Create sampled signal
Fs = 100; % sampling rate
Ts = 1/Fs; % sampling period
ratio = round(Ts/Tc);
tn = t(1:ratio:end); % sampled time axis
xn = xc(1:ratio:end); % sampled signal

% Plot the CT signal and sampled signal
figure
hold on
grid on
plot(t, xc)
stem(tn, xn, 'o')
legend('"Continuous time signal"', 'Sampled signal')

% Create and plot sinc train
sincTrain = zeros(length(t), length(xn));
nind = 1;
figure
cmap = colormap(jet(length(-floor(length(xn)/2):floor(length(xn)/2))));
ax = axes('colororder', cmap);
hold on
grid on

plot(t, xc, 'k', 'LineWidth', 3)
for n = -floor(length(xn)/2):floor(length(xn)/2)
   sincTrain(:, nind) = xn(nind)*sinc((t - n*Ts)/Ts);
   p = plot(t, sincTrain(:, nind), 'LineWidth', 2);
   stem(tn(nind), xn(nind), 'Color', p.Color, 'LineWidth', 2)
   nind = nind + 1;
end
xlabel('t')
ylabel('Amplitude')
set(gca, 'FontSize', 20, 'LineWidth', 3, 'FontWeight', 'bold')

xr = sum(sincTrain, 2); % sum(sincTrain, 2) is the interpolated/reconstructed signal, should be equal to xc

figure
hold on
grid on
plot(t, xc)
plot(t, xr) 

error = mean(abs(xc - xr).^2);