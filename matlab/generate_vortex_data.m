close all;

N = 512;
z = 30e3; %um (30mm)
f = 100e3; %um (100mm)
Nex = 10;

file = '../zernike_basis.mat';

data = load(file,'Z');
Z = data.Z;
M = size(Z,1)-1; %single index of highest zernike polynomial

A = squeeze(Z(1,:,:)); %use piston mode as shape of aperture
A(isnan(A)) = 0;

Z = Z(2:end,:,:); %remove piston mode
Z(isnan(Z)) = 0;

a = rand(M,Nex);

phase = zeros(N,N,1,Nex);
for i = 1:Nex
    aRep = repmat(a(:,i),1,N,N);
    series = aRep.*Z;
    phase(:,:,1,i) = sum(series,1);
end

Data_generation_vortex(phase,A,z,f,N);