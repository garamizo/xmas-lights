%% Plot tree

H = 1.8;  % height
D = 0.8;  % base diameter
Nloops = 20;  % number of LED loops around tree
bulb_dist = 7e-2; % distance between bulbs

% tree
[X, Y] = meshgrid(-D/2:2e-2:D/2, -D/2:2e-2:D/2);
Z = H - sqrt(X.^2 + Y.^2) * H/(D/2);
Z(Z<0) = NaN;

figure
s = surf(X, Y, Z);
s.FaceColor = [0, 1, 0];
s.EdgeColor = 'none';
axis equal
set(gcf, 'Position', [281    76   744   575])

% LED string
th = (0 : (360*Nloops))' * pi/180;
r = linspace(D/2, 0, length(th))';
Z = linspace(0, H, length(th))';
X = r.*cos(th);
Y = r.*sin(th);
hold on, plot3(X, Y, Z, 'r', 'linewidth', 3)

% LED bulbs
r = linspace(1.05*D/2, 0, length(th))';
Z = linspace(0, H, length(th))';
X = r.*cos(th);
Y = r.*sin(th);
L = [0; cumsum(sqrt(diff(X).^2 + diff(Y).^2 + diff(Z).^2))];
LL = (L(1) : bulb_dist : L(end))';
num_bulbs = length(LL);
XYZ = interp1(L, [X, Y, Z], LL);
c = ones(num_bulbs, 3);
clear hl
for k = 1 : num_bulbs
    hl(k) = scatter3(XYZ(k,1), XYZ(k,2), XYZ(k,3), 50, c(k,:), 'filled');
    hl(k).MarkerEdgeColor = [0,0,0];
    hl(k).MarkerFaceAlpha = 0.5;
end

%% Modify RGBA

for k = 1 : 10
    for i = 1 : num_bulbs
        hl(i).MarkerFaceAlpha = mod(k,2);
    end
    drawnow()
    pause(0.1)
end

