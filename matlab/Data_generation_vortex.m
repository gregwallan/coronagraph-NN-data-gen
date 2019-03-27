function [CamMat1,OrgMat1]=Data_generation_vortex(OrgMat,a,z,f,N)  % OrgMat: 4D matrix 256*256*1*N_images   %N=256  % z: um

[~,~,~,Nl]=size(OrgMat);
CamMat1=zeros(N,N,1,Nl);
OrgMat1=zeros(N,N,1,Nl);

normalization = 3e23;
for j=1:Nl
    j
    Im=double(squeeze(OrgMat(:,:,:,j)));
    f_in=Im;
    f_out=vortex_propagation_trans(f_in,a,z,f);
    %f_in=abs(ppval(pp,Im1));
    CamMat1(:,:,1,j)=rot90((f_out/normalization),2);
    OrgMat1(:,:,1,j)=f_in;
end
%figure();
%imagesc(CamMat1);
save(strcat('Images_norm_flip_res_trans_no_noise_',num2str(N),'_',num2str(z/1000),'mm.mat'),'OrgMat1','CamMat1','-v7.3');