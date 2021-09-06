% Animate tree pattern
%   Execute map_bulbs.m first to create the image (handle gfx)

% view([0, 0])
figure(25)

X = Xb;  % rename for compactness
Y = Yb;
Z = Zb;
Q = atan2(Y, X);
R = sqrt(Y.^2 + X.^2);

pattern_period = 2;  % pattern change period
pattern_mode = 1;  % type of pattern

% background pattern
Kt = 2*pi/2;  % time constant [rad/s]
Kq = 6*pi/(2*pi);  % radial constant [rad/rad] (color cycle per circunference) 
Kz = 6*pi/H;  % waterfall constant [rad/m] (color cycle per m)
Krr = -2*pi/50e-2;  % circle constant [cycle/s] (color cycle per distance from origin)
Kqq = 360/60;  % ray constant [cycle/s] (color cycle per rad)

dt = 0.1;  % plot period
C = zeros(num_bulbs, 3);  % color of each bulb

pause(1)

t0 = tic;
for k = 1 : round(10/dt)
    tic
    time = toc(t0);
    
    % center of background pattern
    if mod(k-1,round(pattern_period/dt)) == 0
        x0 = 0; z0 = H/3;
        RR = sqrt((X-x0).^2 + (Z-z0).^2);  % 
        QQ = atan2(Z-z0, X-x0);
        pattern_mode = mod(pattern_mode + 1, 4); % randi(4);
    end
    
    if pattern_mode == 0
        B = sin(Kt*time + Kqq*QQ).^3;  % rays
    elseif pattern_mode == 1
        B = sin(Kt*time + Krr*RR);  % circles
    elseif pattern_mode == 2
        B = sin(Kt*time + Kq*Q);  % radial
    elseif pattern_mode == 3
        B = sin(Kt*time + Kz*Z).^3;  % waterfall
    end
    
    for j = 1 : 450
        q = mod(Q(j) - Kt*time-pi, 2*pi) - pi;
        C(j,:) = hsv_soft(q * 127/pi) / 255;
    end
    C(B>0,:) = 0;
    hl.CData = C;

    drawnow()
    pause(dt - toc)
end

