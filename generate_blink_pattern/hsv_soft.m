function c = hsv_soft(x)
% x: -127:127

if x < -84
    r = 255;
    g = (x + 127) * 6;
    b = 0;
elseif x <= -42
    r = 255 - (x+84)*6;
    g = 255;
    b = 0;
elseif x <= 0
    r = 0;
    g = 255;
    b = (x+42)*6;
elseif x < 43
    r = 0;
    g = 255-(x-0)*6;
    b = 255;
elseif x <= 84
    r = (x-42.5)*6;
    g = 0;
    b = 255;
else
    r = 255;
    g = 0;
    b = 255 - (x-84.5)*6;
end

c = [r, g, b];