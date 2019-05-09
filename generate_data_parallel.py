import numpy as np
import poppy
import matplotlib.pyplot as plt
import astropy.units as u
from llowfs import generate_wfe_array, make_coronagraph
import h5py
import multiprocessing

highest_coeff = 15
bounds = [50e-9]*(highest_coeff-1) #piston not included

Nex = 10000 #number of examples

file_out = 'vortex_50nm_256px.hdf5'
#file_out='test.hdf5'

#size of output images is npix_detector
# oversample pads the pupil plane before performing ffts. This gives more accurate simulations of 
# image plane interactions. oversample=2 is generally too low for this application. 
# oversample=4 works well and doesn't take too long (see notebook'Oversample Comparison')
# 'coronagraph' can be 'vortex' or 'fqpm'. See 'Test Responses' notebook for more info.

oversample = 4
wavelength=632e-9*u.m
coronagraph='vortex'
npix_pupil = 512

npix_detector = 256
detector_fov = 0.3 #arcsec
detector_pixelscale = detector_fov/npix_detector

vortex_charge = 2
sensor_defocus = 0.5 #(times wavelength)

processes=8

#------Do not edit below------#

def coronagraph_wrapper(wfe_in):
    llowfs = make_coronagraph(wfe_in,wavelength=wavelength,oversample=oversample,pixelscale=detector_pixelscale,\
                            sensor_defocus=sensor_defocus,llowfs=True,npix_pupil=npix_pupil,\
                            npix_detector=npix_detector, mask_type=coronagraph,\
                            vortex_charge=vortex_charge)
    psf = llowfs.calc_psf(wavelength=wavelength, display_intermediates=False)
    return psf

if __name__ == '__main__':
    wfe_array = generate_wfe_array(bounds,Nex)
    print(wfe_array.shape)
    print(wfe_array[:,:3])

    M = highest_coeff-1 #number of zernike coeffs (not including piston)
    N = Nex; #number of examples to simulate
    D = npix_detector #size of resulting psf images
    
    wfe_iterable = []
    for i in range(N):
        wfe = [0]
        wfe.extend(wfe_array[:,i].tolist())
        wfe_iterable.append(wfe)

    pool = multiprocessing.Pool(processes=processes)
    psf_list = pool.map(coronagraph_wrapper,wfe_iterable)
    pool.close()
    pool.join()

    hf = h5py.File(file_out, "w") #create an hdf5 file to store everything
    hf.create_dataset("zernike_coeffs", data=wfe_array) 

    images_dataset = hf.create_dataset("images",(D,D,N),'f') #create an empty dataset to store images
    for i in range(N):
        images_dataset[:,:,i] = psf_list[i][0].data
    hf.close()
    