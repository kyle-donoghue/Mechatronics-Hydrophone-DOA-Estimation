counter = 1;
for i = 1:60
	if phase(i) < 0
		new_phase(counter) = phase(i);
		counter = counter+1;
	end
end
figure;
plot(new_phase)
format longg
mean(new_phase)
