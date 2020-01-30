import numpy as np
import poppy
import matplotlib.pyplot as plt
import astropy.units as u
from llowfs import generate_wfe_array, make_coronagraph
import h5py
import multiprocessing
import time
import sys

# -- General Parameters -- #

Nex = 800 #number of examples

file_out = 'vortex_15_individual_800st_948nm_128px'
if len(sys.argv) > 1:
	file_out +=  str(sys.argv[1])
file_out += '.hdf5'

processes=8 #number of workers to spawn

# -- Parameters for the zernike generation -- #
highest_coeff = 15
bounds = [632e-9*1.5]*(highest_coeff-1) #a list of M bounds for zernike coeffs starting with piston. Bound on each coefficient can be different if desired.
zernike_distribution = 'individual' #can be 'uniform', 'sparse', or 'uniform-overall'
overall_max_wfe = 632e-9*1.5;
# -- Parameters on optical system -- #
oversample = 4 
# oversample pads the pupil plane before performing ffts. This gives more accurate simulations of image plane interactions. oversample=2 is generally too low for this application. oversample=4 works well and doesn't take too long (see notebook'Oversample Comparison')

wavelength=632e-9*u.m
coronagraph='vortex' # can be 'vortex' or 'fqpm'. See 'Test Responses' notebook for more info.
npix_pupil = 512 

npix_detector = 128 #size of output images
detector_fov = 0.3 #arcsec
detector_pixelscale = detector_fov/npix_detector

vortex_charge = 2
sensor_defocus = 4 #(times wavelength)
obscuration = False

#so the metadata is less confusing, but hdf5 doesn't support 'None'
if coronagraph == 'fqpm':
    vortex_charge = 0 
if zernike_distribution == 'uniform-overall':
    bounds = 0
else:
    overall_max_wfe = 0

#------Do not edit below------#

def generate_wfe_array_individual(bounds, NexPerZ):
     M = highest_coeff-1
     N = NexPerZ
     Ntot = N*M
     wfe_array = np.zeros((M, Ntot))
     vals = np.linspace(-1*bounds[0],bounds[0],N)
     for i in range(M):
          wfe_array[i,i*N:(i+1)*N] = vals
     return wfe_array

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

def generate_wfe_array_uniform_overall(overall_bound,Nex):
    bounds = [100e-9]*(highest_coeff-1) #doesn't matter as long as it's uniform
    wfe_array = generate_wfe_array(bounds,Nex)
    rms_wfe_calc = np.sqrt(np.sum(np.power(wfe_array,2),(0)))
    desired_wfe = np.random.uniform(high=overall_bound,size=(Nex))
    wfe_array = np.multiply(np.divide(wfe_array,rms_wfe_calc),desired_wfe)
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
    elif zernike_distribution == 'uniform-overall':
        wfe_array = generate_wfe_array_uniform_overall(overall_max_wfe,Nex)
    elif zernike_distribution == 'individual':
        wfe_array = generate_wfe_array_individual(bounds,Nex)
        Nex = Nex*(highest_coeff-1)
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
    
