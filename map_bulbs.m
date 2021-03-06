% Map coordinates of light bulbs
%   1. Enter coordinates of strip connectors manually
%   2. Interpolate light bulbs between connectors
%   3. Plot tree
clear; clc

% Calculate coordinates of LED strip Connectors ===========================
% tree dimensions
H = 76 / 39.3701;  % height
D = 33 / 39.3701;  % base diameter

% cilindrical coordinates of each connector (measure manually)
Q = ([-180, -100, -5, 110, -45, -170, 180, -100, -45, -30] + ...
    cumsum([0, 1, 1, 1, 2, 2, 1, 3, 3, 7])*360)* pi/180; % angle
Z = [0, 6, 9, 13, 18.5, 26, 34, 43, 51, 76] / 39.3701;  % height
Q = flipud(Q(:));  % start on the top
Z = flipud(Z(:));

R = D/2 - (Z/H).*D/2;  % radius (calculate from geometry)
X = R.*cos(Q);  % cartesian coordinates, x
Y = R.*sin(Q);  % cartesian coordinates, y

% Interpolate all light bulb between strip connectors =====================
QQ = (Q(1) : -1*pi/180 : Q(end))';
ZZ = interp1(Q, Z, QQ);
RR = D/2 - (ZZ/H).*D/2;

XX = RR.*cos(QQ);
YY = RR.*sin(QQ);

LL = [0; cumsum(sqrt(diff(XX).^2 + diff(YY).^2 + diff(ZZ).^2))];
L = interp1(QQ, LL, Q, 'linear', 'extrap');  % cummulative length of strips

% bulb position
num_bulbs = 50*9;  % 50 bulbs per strip (9 strips)
Lb = linspace(LL(1), LL(end), num_bulbs)';
Qb = interp1(LL, QQ, Lb);
Rb = interp1(LL, RR, Lb);
Zb = interp1(LL, ZZ, Lb);
c = ones(num_bulbs, 3);

Qb(1:50) = Qb(1:50) + linspace(1.8*pi, 0, 50)' - pi/3;  % small hack fix

Xb = Rb .* cos(Qb);
Yb = Rb .* sin(Qb);

% Cilindrical coordinates rotating along Y axis, with a Z offset
%   Used for ray pattern
x0 = 0;
z0 = H*1/3;  % Z offset
RRb = sqrt((Xb-x0).^2 + (Zb-z0).^2);  % 
QQb = atan2(Zb-z0, Xb-x0);

% gen tree surface =================================================
[xx, yy] = meshgrid(-D/2:5e-3:D/2, -D/2:5e-3:D/2);
zz = H - sqrt(xx.^2 + yy.^2) * H/(D/2) - 0.05;
zz(zz<0) = NaN;

% plot tree ========================================================
figure(25)
hold off
s = surf(xx, yy, zz);
s.FaceColor = [0, 0.2, 0];
s.EdgeColor = 'none';
hold on
plot3(XX, YY, ZZ, 'k')
axis equal
xlabel('x'), ylabel('y'), zlabel('z')
hc = plot3(X, Y, Z, 'rs', 'markersize', 10, 'MarkerFaceColor', 'k');
set(gcf, 'Position', [281    76   744   575])
hl = scatter3(Xb, Yb, Zb, 30, c(:,:), 'filled');
hl.MarkerEdgeColor = [0,0,0];

diff(L)
legend(hc, 'Connector of LED strip', 'Location', 'southoutside')
set(gcf, 'color', 'w')

% save figure ======================================================
[file,path,idxb] = uiputfile('*.png');
if idxb == 1
    print(gcf,fullfile(path, file),'-dpng','-r300'); 
end

