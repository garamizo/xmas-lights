function plot_tree()

%% Build tree

H = 76 / 39.3701;  % height
D = 33 / 39.3701;  % base diameter

Q = ([-180, -100, -5, 110, -45, -170, 180, -100, -45, -30] + ...
    cumsum([0, 1, 1, 1, 2, 2, 1, 3, 3, 7])*360)* pi/180;
Z = [0, 6, 9, 13, 18.5, 26, 34, 43, 51, 76] / 39.3701;
Q = flipud(Q(:));
Z = flipud(Z(:));

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

%%
% tree
[X, Y] = meshgrid(-D/2:2e-2:D/2, -D/2:2e-2:D/2);
Z = H - sqrt(X.^2 + Y.^2) * H/(D/2);
Z(Z<0) = NaN;

figure
s = surf(X, Y, Z);
s.FaceColor = [0, 0.2, 0];
s.EdgeColor = 'none';
axis equal
xlabel('x'), ylabel('y'), zlabel('z')
set(gcf, 'Position', [281    76   744   575])

% LED string
th = (0 : (360*Nloops))' * pi/180;
r = linspace(D/2, 0, length(th))';
Z = linspace(0, H, length(th))';
X = r.*cos(th);
Y = r.*sin(th);
hold on, plot3(X, Y, Z, 'k', 'linewidth', 2)

% LED bulbs
r = linspace(1.05*D/2, 0, length(th))';
Z = linspace(0, H, length(th))';
X = r.*cos(th);
Y = r.*sin(th);
L = [0; cumsum(sqrt(diff(X).^2 + diff(Y).^2 + diff(Z).^2))];
LL = (L(1) : bulb_dist : L(end))';
num_bulbs = length(LL);
XYZ = interp1(L, [X, Y, Z], LL) + randn(num_bulbs,3)*1e-2;
c = ones(num_bulbs, 3);
hl = scatter3(XYZ(:,1), XYZ(:,2), XYZ(:,3), 50, c(:,:), 'filled');
hl.MarkerEdgeColor = [0,0,0];

% Background pattern
% view([0, 0])

X = XYZ(:,1);
Y = XYZ(:,2);
Z = XYZ(:,3);
Q = atan2(Y, X);
R = sqrt(Y.^2 + X.^2);

dt = 0.1;
Kt = 2*pi/2;
Kq = 6*pi/(2*pi);
Kz = 4*pi/H;
Krr = -2*pi/30e-2;
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
%         x0 = 0; y0 = 0; z0 = H/3;
        x0 = (rand() - 0.5) * D;
        z0 = rand() * H*2/3;
        RR = sqrt((X-x0).^2 + (Z-z0).^2);  % 
        QQ = atan2(Z-z0, X-x0);
        pattern_mode = randi(4);
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
    hl.CData = C;

    drawnow()
    pause(dt - toc)
end
