clear; clc

H = 76 / 39.3701;  % height
D = 33 / 39.3701;  % base diameter

Q = ([-180, -100, -5, 110, -45, -170, 180, -100, -45, -30] + ...
    cumsum([0, 1, 1, 1, 2, 2, 1, 3, 3, 7])*360)* pi/180;
Z = [0, 6, 9, 13, 18.5, 26, 34, 43, 51, 76] / 39.3701;
Q = flipud(Q(:));
Z = flipud(Z(:));

% Q(1) = Q(1) + 2*pi;

R = D/2 - (Z/H).*D/2;
X = R.*cos(Q);
Y = R.*sin(Q);

QQ = (Q(1) : -1*pi/180 : Q(end))';
ZZ = interp1(Q, Z, QQ);
RR = D/2 - (ZZ/H).*D/2;

XX = RR.*cos(QQ);
YY = RR.*sin(QQ);

LL = [0; cumsum(sqrt(diff(XX).^2 + diff(YY).^2 + diff(ZZ).^2))];
L = interp1(QQ, LL, Q, 'linear', 'extrap');

% bulb position
num_bulbs = 450;
Lb = linspace(LL(1), LL(end), num_bulbs)';
Qb = interp1(LL, QQ, Lb);
Rb = interp1(LL, RR, Lb);
Zb = interp1(LL, ZZ, Lb);
c = ones(num_bulbs, 3);

Qb(1:50) = Qb(1:50) + linspace(1.8*pi, 0, 50)' - pi/3;

Xb = Rb .* cos(Qb);
Yb = Rb .* sin(Qb);

x0 = 0;
z0 = H*1/3;
RRb = sqrt((Xb-x0).^2 + (Zb-z0).^2);  % 
QQb = atan2(Zb-z0, Xb-x0);

[xx, yy] = meshgrid(-D/2:5e-3:D/2, -D/2:5e-3:D/2);
zz = H - sqrt(xx.^2 + yy.^2) * H/(D/2) - 0.05;
zz(zz<0) = NaN;


figure(25)
hold off
s = surf(xx, yy, zz);
s.FaceColor = [0, 0.2, 0];
s.EdgeColor = 'none';
hold on
plot3(XX, YY, ZZ, 'k')
axis equal
xlabel('x'), ylabel('y'), zlabel('z')
plot3(X, Y, Z, 'rs', 'markersize', 10, 'MarkerFaceColor', 'k')
set(gcf, 'Position', [281    76   744   575])
hl = scatter3(Xb, Yb, Zb, 30, c(:,:), 'filled');
hl.MarkerEdgeColor = [0,0,0];

diff(L)

%%
% Background pattern
% view([0, 0])
figure(25)

X = Xb;
Y = Yb;
Z = Zb;
Q = atan2(Y, X);
R = sqrt(Y.^2 + X.^2);

dt = 0.1;
Kt = 2*pi/2;
Kq = 6*pi/(2*pi);
Kz = 6*pi/H;
Krr = -2*pi/50e-2;
Kqq = 360/60;

Kd = -1.5/D;  % completes 1(full bright) in D distance
Kte = 1/0.3;  % completes 1(full bright) in 1 sec

t0 = tic;
for k = 1 : round(20/dt)
    tic
    time = toc(t0);
    
    % define center of explosion
    if mod(k-1,round(1/dt)) == 0
        c = rand(1,3);
        x0 = (rand() - 0.5) * D;
        y0 = (rand() - 0.5) * D;
        z0 = rand() * H*2/3;
        D = sqrt((X-x0).^2 + (Y-y0).^2 + (Z-z0).^2);
        trig_time = time;
    end
    
    % center of background pattern
    if mod(k-1,round(3/dt)) == 0
        x0 = 0; z0 = H/3;
        RR = sqrt((X-x0).^2 + (Z-z0).^2);  % 
        QQ = atan2(Z-z0, X-x0);
        pattern_mode = 4; % randi(4);
    end
    
    if pattern_mode == 1
        B = sin(Kt*time + Kqq*QQ).^3;  % rays
    elseif pattern_mode == 2
        B = sin(Kt*time + Krr*RR);  % circles
    elseif pattern_mode == 3
        B = sin(Kt*time + Kq*Q);  % radial
    elseif pattern_mode == 4
        B = sin(Kt*time + Kz*Z).^3;  % falling rings
    end
    B(B<0) = 0;

    F = sin(Kt*(time-trig_time)) + Kd*D;
    F(F<0) = 0;
    F(F>1) = 1;
    
    C = (0.7*B .* [1, 1, 1] + 2*F .* c) ./ 1;
    C(C<0) = 0;
    C(C>1) = 1;
    for j = 1 : 450
        q = mod(Q(j) - Kt*time-pi, 2*pi) - pi;
        C(j,:) = hsv_soft(q * 127/pi) / 255;
    end
    hl.CData = C .* (1-B);
    
    bin_data = cast(C * 255, 'int8');
    
    drawnow()
    pause(dt - toc)
end

