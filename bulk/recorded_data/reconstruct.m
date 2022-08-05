fileID = fopen('regular_data.csv','r');
format = '%f';
sizeA = [1 Inf];
reg_data = fscanf(fileID,format,sizeA);
fileID = fopen('highpass_data.csv','r');
highpass_data = fscanf(fileID,format,sizeA);
highpass_data = highpass_data(1:2000001);
fileID = fopen('rfft_data.csv','r');
rfft_data = fscanf(fileID,format,sizeA);

t = 0:1:2000000;
t = t./1000000;

reg_data = reg_data(871001:871500);
t = t(1:500);


downsample = reg_data(1:4:length(reg_data));
td = t(1:4:length(t));
[output_data,tc] = sinc_interpolate(downsample,250000,10);
clf;
stem(td,downsample);
hold;

plot(t, reg_data,'LineWidth',2);
plot(tc, output_data, 'b', 'LineWidth',2);
