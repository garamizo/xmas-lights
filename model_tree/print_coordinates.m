%% Print each bulb coordinate in C++ (Arduino) code for better performance
%   Paste on xmas_light.ino
clc

fprintf('int8_t X[] = {\n')
for k = 1 : 449
    fprintf('\t%d,\n', round(Xb(k) * 127/(D*0.5)))
end
fprintf('\t% d\n};\n', round(Xb(450) * 127/(D*0.5)))

fprintf('int8_t Y[] = {\n')
for k = 1 : 449
    fprintf('\t% d,\n', round(Yb(k) * 127/(D*0.5)))
end
fprintf('\t% d\n};\n', round(Yb(450) * 127/(D*0.5)))

fprintf('int8_t Z[] = {\n')
for k = 1 : 449
    fprintf('\t% d,\n', round(Zb(k) * 127/(H)))
end
fprintf('\t% d\n};\n', round(Zb(450) * 127/H))

fprintf('int8_t R[] = {\n')
for k = 1 : 449
    fprintf('\t% d,\n', round(Rb(k) * 127/(D*0.5)))
end
fprintf('\t% d\n};\n', round(Rb(450) * 127/(D*0.5)))

fprintf('int8_t Q[] = {\n')
for k = 1 : 449
    fprintf('\t% d,\n', round(mod(Qb(k),2*pi) * 127/(2*pi)))
end
fprintf('\t% d\n};\n', round(mod(Qb(450),2*pi) * 127/(2*pi)))

fprintf('int8_t QQ[] = {\n')
for k = 1 : 449
    fprintf('\t% d,\n', round(mod(QQb(k),2*pi) * 127/(2*pi)))
end
fprintf('\t% d\n};\n', round(mod(QQb(450),2*pi) * 127/(2*pi)))

fprintf('int8_t RR[] = {\n')
for k = 1 : 449
    fprintf('\t% d,\n', round(RRb(k) * 127/(H)))
end
fprintf('\t% d\n};\n', round(RRb(k) * 127/(H)))