 function [I_out]=vortex_propagation_trans(f_in,a_in,z,D,focal,lambda,use_lyot) %f_in is the input phase

%the unit for z is um
%the unit for lambda is um


%parameters
[N1,N2]=size(f_in);   % SBP of the input object
dx=D/N1;%um    pixel size at the object plane


N_min=ceil(lambda*focal/dx/dx); % required minimum SBP to avoid aliasing


if N_min>max(N1,N2)  % zero-padding is required in 2D, ones-padding for phase object?
    g_in=zeros(N_min,N_min);
    a=zeros(N_min,N_min);
    idx1=floor((N_min-N1)/2);
    idx2=floor((N_min-N2)/2);
    g_in(idx1+(1:N1),idx2+(1:N2))=f_in;
    a(idx1+(1:N1),idx2+(1:N2))=a_in;
else if (N_min>N1)&(N_min<=N2)
    g_in=zeros(N_min,N2);
    a=zeros(N_min,N2);
    idx1=floor((N_min-N1)/2);
    g_in(idx1+(1:N1),:)=f_in;
    a(idx1+(1:N1),:)=a_in;
    else if (N_min>N2)&(N_min<=N1)
            g_in=zeros(N1,N_min);
            a=zeros(N1,N_min);
            idx2=floor((N_min-N2)/2);
            g_in(:,idx2+(1:N2))=f_in;
            a(:,idx2+(1:N2))=a_in;
        else 
            g_in=f_in;
            a=a_in;
        end
    end
end

%figure();
%imagesc(g_in);

g_in=a.*exp(1i*pi*g_in); %convert phase to field

[M1,M2]=size(g_in);
X_max=M1*dx;
Y_max=M2*dx;
du=1/X_max;
dv=1/Y_max;

x=(floor(-(M1-1)/2):floor((M1-1)/2))*dx;
y=(floor(-(M2-1)/2):floor((M2-1)/2))*dx;
[Y,X]=meshgrid(y,x);

u=(floor(-(M1-1)/2):floor((M1-1)/2))*du;
v=(floor(-(M2-1)/2):floor((M2-1)/2))*dv;
[V,U]=meshgrid(v,u);

XP = U*lambda*focal;
YP = V*lambda*focal;

if use_lyot
    inv_lyot_stop = ones(M1,M2);
    R = 25e3; %um
    rho = sqrt(XP.^2+YP.^2);
    iCirc = rho<R;
    inv_lyot_stop(iCirc) = 0;
end

% 

%Fraunhofer propagation
Fg_in=fftshift(fft2(ifftshift(g_in)));                      % FT the input field
vtx = exp(1i*pi*phase_vortex_2d(M1,2));

g_2 = Fg_in .* vtx;
%g_2 = Fg_in; %at pupil (mask) plane
g_3 = fftshift(fft2(ifftshift(g_2)));
if use_lyot
    g_3 = g_3 .* inv_lyot_stop;
end
%g_3 = fftshift(fft2(ifftshift(g_3)));
g_out = free_space_propagation_trans_mod(g_3,z,du*lambda*focal);

                                 % Multiply by the Fourier Domain representation of the Fresnel Kernel
                                                                 % use H+eps to avoid NaN? either way there is a problem w/low H values. 


I_out=abs(g_out).^2; %intensity
%I_out_farfield  = abs(g_mid_farfield).^2;



%figure();
%imagesc(angle(g_mid_farfield));
%title('Far Field Phase');


%only crop the central region (determined by the dimension of the CMOS)
%Assume to be the same as the input object
idx1=floor((M1-N1)/2);
idx2=floor((M2-N2)/2);
I_out=I_out(idx1+(1:N1),idx2+(1:N2));

%figure();
%imagesc(I_out);
%title(['Propagated Intensity, f = ',num2str(round(focal/1e3)),' mm']);
% 
% 
% %%
% %Angular spectrum method
% 
% H1_exp = 2*pi*z/lambda*sqrt(1-lambda^2*U.^2-lambda^2*V.^2);   
% H1 = exp(i*H1_exp);
% 
% Fg_out1= Fg_in.*H1;                                  % Multiply by the Fourier Domain representation of the Fresnel Kernel
%                                                                  % use H+eps to avoid NaN? either way there is a problem w/low H values. 
% g_out1 = fftshift(ifft2(ifftshift(Fg_out1)));             % Inverse FT to recover the output field
% 
% I_out1=abs(g_out1).^2; %intensity
% 
% 
% %only crop the central region (determined by the dimension of the CMOS)
% %Assume to be the same as the input object
% idx1=floor((M1-N1)/2);
% idx2=floor((M2-N2)/2);
% I_out1=I_out1(idx1+(1:N1),idx2+(1:N2));
% 
 %%
% %Fresnel propagation by direct convolution 
% 
%determine the appropriate sampling distance
% dx1=lambda*z/(256*20);%required sampling distance
% if dx1<dx % interpolation required
%   N_min=ceil(256*20/dx1);
%   g_in=imresize(f_in,[N_min,N_min],'nearest');
%   dx2=256*20/N_min;
%   x=(floor(-(N_min-1)/2):floor((N_min-1)/2))*dx2;
% else
%   g_in=f_in;
%   x=(floor(-(N1-1)/2):floor((N1-1)/2))*dx;
% end
% 
% 
% g_in=exp(-1*i*g_in); %convert phase to field
% 
% [Y_1,X_1]=meshgrid(x,x);
% 
% Kernel=exp(i*pi/lambda/z*(X_1.^2+Y_1.^2))/(lambda*z)*(dx^2);
% 
% g_out2=conv2(g_in,Kernel,'same');
% I_out2=abs(g_out2).^2; %intensity
% 
% %Interpolation on the CMOS plane
% %Assume to be the same as the input object
% if dx1<dx
%     I_out2=imresize(I_out2,[N1,N2]);
% end








