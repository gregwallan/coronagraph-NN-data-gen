function [vortex] = phase_vortex_2d(width, charge)
%% PHASE_VORTEX_2D 
% width - width of input in pixels
% charge - topological charge. maximum phase delay = charge*2*pi
vortex = zeros(width, width);
cx = width / 2;

%% Calculate vortex
for ix = 1:width
    for iy = 1:width
        p_angle = atan2(ix - cx, iy - cx);
        vortex(ix, iy) = charge*(p_angle + pi);
    end
end
end

