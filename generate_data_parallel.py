import numpy as np
import poppy
import matplotlib.pyplot as plt
import astropy.units as u
from llowfs import generate_wfe_array, make_coronagraph
import h5py
import multiprocessing
import time

# -- General Parameters -- #

Nex = 10 #number of examples

file_out = 'test.hdf5'

processes=8 #number of workers to spawn

# -- Parameters for the zernike generation -- #
highest_coeff = 15
bounds = [150e-9]*(highest_coeff-1) #a list of M bounds for zernike coeffs starting with piston. Bound on each coefficient can be different if desired.
zernike_distribution = 'sparse' #can be 'uniform' or 'sparse'

# -- Parameters on optical system -- #
oversample = 4 
# oversample pads the pupil plane before performing ffts. This gives more accurate simulations of image plane interactions. oversample=2 is generally too low for this application. oversample=4 works well and doesn't take too long (see notebook'Oversample Comparison')

wavelength=632e-9*u.m
coronagraph='vortex' # can be 'vortex' or 'fqpm'. See 'Test Responses' notebook for more info.
npix_pupil = 512 

npix_detector = 256 #size of output images
detector_fov = 0.3 #arcsec
detector_pixelscale = detector_fov/npix_detector

vortex_charge = 2
sensor_defocus = 4 #(times wavelength)
obscuration = False


#------Do not edit below------#

def generate_wfe_array_sparse(bounds,Nex):
    #For each example, choose a single one of the zernikes, choose its value from a uniform distribution, and set the others to zero.
    #Arguments:
    # - bounds: a list of positive floats b_0...b_M. If zernike m is chosen, its value will be between b_m and -1*b_m.
    M = highest_coeff-1 #number of zernike coeffs (not including piston)
    N = Nex; #number of examples to simulate    
    wfe_array = np.zeros((M,N))
    choices = np.random.randint(M,size=(N,))
    for i in range(N):
        choice = choices[i]
        bound = bounds[choice]
        wfe_array[choice,i] = np.random.uniform(low=-1*bound,high=bound)
    return wfe_array
    
def coronagraph_wrapper(wfe_in):
    llowfs = make_coronagraph(wfe_in,wavelength=wavelength,oversample=oversample,pixelscale=detector_pixelscale,\
                            sensor_defocus=sensor_defocus,llowfs=True,npix_pupil=npix_pupil,\
                            npix_detector=npix_detector, mask_type=coronagraph,\
                            vortex_charge=vortex_charge, obscuration=obscuration)
    psf = llowfs.calc_psf(wavelength=wavelength, display_intermediates=False)
    return psf

if __name__ == '__main__':
    start_time = time.time()
    
    if zernike_distribution == 'sparse':
        wfe_array = generate_wfe_array_sparse(bounds,Nex)
    elif zernike_distribution == 'uniform':
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
                'detector_fov': detector_fov,
                'pixelscale': detector_pixelscale,
                'sensor_defocus': sensor_defocus,
                'obscuration': obscuration,
                'distribution': zernike_distribution,
                'bounds': bounds,
                'examples': N,
                'zernikes': M
            }
    hf.attrs.update(metadata)
    hf.close()
    
    print("Finished storing: ", time.time() - start_time)
    