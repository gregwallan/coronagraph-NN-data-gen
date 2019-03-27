function [vortex] = phase_vortex_2d(width, max)
%% PHASE_VORTEX_2D 
% width - width of input in pixels
% max - maximum phase delay (max*pi)
vortex = zeros(width, width);
cx = width / 2;

%% Calculate vortex
for ix = 1:width
    for iy = 1:width
        p_angle = atan2(ix - cx, iy - cx);
        vortex(ix, iy) = (max/2)*(p_angle + pi);
    end
end
end

