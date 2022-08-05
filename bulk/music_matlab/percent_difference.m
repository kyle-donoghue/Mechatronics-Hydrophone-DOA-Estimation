function perc_diff = percent_difference(degree)
%PERCENT_ERROR Run a simulation of a degree being passed into the program
%and output the result of |simulated degree-theoretical degree|
    t = 0:.000002:.000050;
    
    SIGNAL_DISTANCE = 10;
    SIGNAL_THETA = degree;
    SIGNAL_FREQUENCY = 40000;
    SOUND_SPEED = 1500;
    
    wavelength = SOUND_SPEED/SIGNAL_FREQUENCY;
    max_dist = wavelength/2;

    signal_rad = SIGNAL_THETA*pi/180;
    signal_location = [-SIGNAL_DISTANCE*sin(signal_rad),SIGNAL_DISTANCE*cos(signal_rad)];
    sensor_1 = [0,max_dist/2];
    sensor_2 = [-max_dist/2,0];
    sensor_3 = [max_dist/2,0];

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
    
    
    %x1 = awgn(x1,10,'measured');
    %x2 = awgn(x2,10,'measured');
    %x3 = awgn(x3,10,'measured');

    plot(t,[x1]);

    y1 = fft(x1);
    y2 = fft(x2);
    y3 = fft(x3);

    [~,p1] = max(abs(y1));
    [~,p2] = max(abs(y2));
    [~,p3] = max(abs(y3));

    t1 = angle(y1(p1)) / (2*pi*SIGNAL_FREQUENCY);
    t2 = angle(y2(p2)) / (2*pi*SIGNAL_FREQUENCY);
    t3 = angle(y3(p3)) / (2*pi*SIGNAL_FREQUENCY);

    theta = atan((t3-t2)/(2*(t3-t1)-(t3-t2)));
    top = sign(t3-t2);
    bottom = sign(2*(t3-t1) - (t3-t2));
    if( top < 0 && bottom > 0)
            theta = theta+pi;
    end
    if( top > 0 && bottom > 0)
            theta = theta-pi;
    end
    degrees = theta*180/pi;
    perc_diff = abs(degrees-degree);

end

