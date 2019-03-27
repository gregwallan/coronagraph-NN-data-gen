close all;
imagefile = 'spongebirb_scaled.jpg';
RGB = imread(imagefile);
im = rgb2gray(RGB);


figure();
imshow(double(im)*1.2/255);

N = 256;
OrgMat = zeros(N,N,1,1);
OrgMat(:,:,1,1) = double(im)*1.22/255;
%OrgMat(:,:,1,1) = TestOrgMat;
%OrgMat(:,:,1,2) = im;

figure();
imagesc(OrgMat(:,:,1,1));


z = 30e3; %um (50mm)
f = 100e3;
%fmm = 100:100:500;

%for i = 1:numel(zmm)
%    f = fmm(i)*1e3;
%    Data_generation_trans_vortex(OrgMat,z,f,N);
%end

Data_generation_trans_vortex(flipud(fliplr(OrgMat)),z,f,N);
ax