close all;

N = 512;
z = 100e3; %um
f = 100e3; %um
D = 30e3; %um`
Nex = 1;
lambda=0.633; %um   wavelength

desired_rms_um = 0.050;
k = 2*pi/lambda;

desired_rms_phase = k*desired_rms_um;

file = '../zernike_basis.mat';

data = load(file,'Z');
Z = data.Z;
M = size(Z,1)-1; %single index of highest zernike polynomial

A = squeeze(Z(1,:,:)); %use piston mode as shape of aperture
A(isnan(A)) = 0;

%Z = Z(2:end,:,:); %remove piston mode
%Z = Z(4,:,:);
%Z = Z(4,:,:);
Z = zeros(M,N,N);
Z(isnan(Z)) = 0;

a = rand(M,Nex);

phase = zeros(N,N,1,Nex);
rms = zeros(Nex,1);

desired_rms_vec = desired_rms_phase*ones(Nex,1);
for i = 1:Nex
    aRep = repmat(a(:,i),1,N,N);
    series = aRep.*Z;
    phase(:,:,1,i) = sum(series,1);
    rms = sqrt(mean(mean(phase(:,:,1,i).^2)));
    phase(:,:,1,i)= phase(:,:,1,i)*desired_rms_vec(i)/rms;
end




Data_generation_vortex(phase,A,z,D,f,N,lambda,true);