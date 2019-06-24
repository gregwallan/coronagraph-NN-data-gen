import numpy as np
import poppy
import matplotlib.pyplot as plt
import astropy.units as u
from llowfs import generate_wfe_array, make_coronagraph
import h5py
import multiprocessing
import time


start_time = time.time()

highest_coeff = 15
zernike_bound = 150e-9

Nex = 10000 #number of examples

file_out = '/Users/gregoryallan/Dropbox (MIT)/Deep Learning + Coronagraph/data/fqpm_150nm_256px_14_individual_zernikes_obsc_training.hdf5'
#fqpm_150nm_256px_14_individual_zernikes_obsc_training
#file_out='test.hdf5'

#size of output images is npix_detector
# oversample pads the pupil plane before performing ffts. This gives more accurate simulations of 
# image plane interactions. oversample=2 is generally too low for this application. 
# oversample=4 works well and doesn't take too long (see notebook'Oversample Comparison')
# 'coronagraph' can be 'vortex' or 'fqpm'. See 'Test Responses' notebook for more info.

oversample = 4
wavelength=632e-9*u.m
coronagraph='fqpm'
npix_pupil = 512

npix_detector = 256
detector_fov = 0.3 #arcsec
detector_pixelscale = detector_fov/npix_detector

vortex_charge = 2
sensor_defocus = 4 #(times wavelength)
obscuration = False

processes=8


#------Do not edit below------#

def coronagraph_wrapper(wfe_in):
    llowfs = make_coronagraph(wfe_in,wavelength=wavelength,oversample=oversample,pixelscale=detector_pixelscale,\
                            sensor_defocus=sensor_defocus,llowfs=True,npix_pupil=npix_pupil,\
                            npix_detector=npix_detector, mask_type=coronagraph,\
                            vortex_charge=vortex_charge, obscuration=obscuration)
    psf = llowfs.calc_psf(wavelength=wavelength, display_intermediates=False)
    return psf

if __name__ == '__main__':
    M = highest_coeff-1 #number of zernike coeffs (not including piston)
    N = Nex; #number of examples to simulate
    D = npix_detector #size of resulting psf images
    
    wfe_array = np.zeros((M,N))
    choices = np.random.randint(M,size=(N,))
    displacements = np.random.uniform(low=-1*zernike_bound,high=zernike_bound,size=(N,))

    #print(wfe_array[choices,:])# = displacements
    for i in range(N):
        choice = choices[i]
        wfe_array[choice,i] = displacements[i]
        
    print(wfe_array.shape)
    print(wfe_array[:,:3])


    
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

    print("Finished processing: ", time.time() - start_time)

    images_dataset = hf.create_dataset("images",(D,D,N),'f',chunks=(D,D,1)) #create an empty dataset to store images
    for i in range(N):
        images_dataset[:,:,i] = psf_list[i][0].data
        
    metadata = {'Date': time.asctime(),
                'Author': 'Greg Allan',
                'oversample': oversample,
                'wavelength': wavelength,
                'coronagraph': coronagraph,
                'npix_pupil': npix_pupil,
                'npix_detector': npix_detector,
                'vortex_charge': vortex_charge,
                'pixelscale': detector_pixelscale,
                'sensor_defocus': sensor_defocus,
                'obscuration': obscuration
            }
    hf.attrs.update(metadata)
    hf.close()
    
    print("Finished storing: ", time.time() - start_time)
    