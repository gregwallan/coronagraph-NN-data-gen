import poppy
import astropy.units as u
import numpy as np
import h5py

def generate_coefficients(wfe_budget):
    #takes rms wfe coeffs in units of meters
    coefficients = []
    for term in wfe_budget:
        coefficients.append(
            np.random.uniform(low=-1*term, high=term)
        )
    return coefficients

def generate_wfe_array(wfe_bounds,n_samples):
    #numbered by the nool convention, but dropping piston
    M = len(wfe_bounds) #number of zernike coeffs (not including piston)
    N = n_samples; #number of wf simulations to generate
    
    wfe_array = np.zeros((M,N))
    for i in range(N):
        wfe_array[:,i] = generate_coefficients(wfe_bounds)
        
    return wfe_array

def simulate_multiple_llowfs(wfe_array,filename,oversample=4,wavelength=632e-9*u.m,coronagraph='vortex',npix_pupil=512,npix_detector=128,vortex_charge=2,pixelscale=0.005,sensor_defocus=0.5):
    #input wfe_array should start at tip, not piston
    #sensor_defocus specified in wavelengths
    
    M = wfe_array.shape[0] #number of zernike coeffs (not including piston)
    N = wfe_array.shape[1]; #number of examples to simulate
    D = npix_detector #size of resulting psf images
    
    
    
    if filename is None:
        images_dataset = np.zeros((D,D,N))
    else: 
        hf = h5py.File(filename, "w") #create an hdf5 file to store everything
        hf.create_dataset("zernike_coeffs", data=wfe_array) 
        images_dataset = hf.create_dataset("images",(D,D,N),'f') #create an empty dataset to store images
        
    for i in range(N):
        wfe = [0]
        wfe.extend(wfe_array[:,i].tolist())
        llowfs = make_coronagraph(wfe,wavelength=wavelength,oversample=oversample,pixelscale=pixelscale,\
                                sensor_defocus=sensor_defocus,llowfs=True,npix_pupil=npix_pupil,\
                                npix_detector=npix_detector, mask_type=coronagraph,\
                                vortex_charge=vortex_charge)
        psf = llowfs.calc_psf(wavelength=wavelength, display_intermediates=False)
        images_dataset[:,:,i] = psf[0].data
    
    if filename is None:
        return images_dataset
    else:
        hf.close()
        
    
    

class VortexMask(poppy.AnalyticOpticalElement):
    def __init__(self, name="unnamed Vortex ", charge=1, wavelength=10.65e-6 * u.meter, **kwargs):
        self.charge = charge
        self.central_wavelength = wavelength
        self._wavefront_display_hint='phase'
        poppy.AnalyticOpticalElement.__init__(self,name=name,**kwargs)
    
    def get_opd(self,wave):
        if not isinstance(wave, poppy.poppy_core.Wavefront):
            raise ValueError("get_opd must be called with a Wavefront to define the spacing")
        assert (wave.planetype == poppy.poppy_core.PlaneType.image)
        
        y, x = self.get_coordinates(wave)
        angle = np.arctan2(x,y)
        opd = self.charge*angle/(2*np.pi)*self.central_wavelength.to(u.meter).value
        return opd

def make_coronagraph(wfe_coeffs,npix_pupil=512,npix_detector=128,wavelength=1e-6,oversample=4,pixelscale=0.01,sensor_defocus=0.5,vortex_charge=2,llowfs=False,mask_type='fqpm',obscuration=False):
    #sensor_defocus: defocus of llowfs detector in waves peak-to-valley
    
    #these values are picked rather arbitrarily, but seem to work
    pupil_radius = 3
    lyot_radius=2.8
    
    #create the optical system
    osys = poppy.OpticalSystem("LLOWFS", oversample=oversample, npix=npix_pupil)
    osys.add_pupil(poppy.CircularAperture(radius=pupil_radius,pad_factor=1.5))
    if obscuration:
        osys.add_pupil(poppy.AsymmetricSecondaryObscuration(secondary_radius=0.5,support_width=0.2*u.meter,support_angle=(0,120,240)))
    
    #inject wavefrotn error at the pupil
    error = poppy.ZernikeWFE(radius=pupil_radius, coefficients=wfe_coeffs)
    osys.add_pupil(error)
    
    #correct for fqpm (and vvc) alignment
    osys.add_pupil(poppy.FQPM_FFT_aligner())
    osys.add_image() #helper for plotting intermediates
    
    #select mask type
    if mask_type is 'fqpm':
        cgph_mask = poppy.IdealFQPM(wavelength=wavelength,name='FQPM Mask')
    elif mask_type is 'vortex':
        cgph_mask = VortexMask(charge=vortex_charge,wavelength=wavelength,name='Vortex Mask')
    else:
        raise ValueError("mask_type must be 'fqpm' or 'vortex'")
    cgph_mask._wavefront_display_hint='phase'
    osys.add_image(cgph_mask)
    
    #correct alignment back the other way
    osys.add_pupil(poppy.FQPM_FFT_aligner(direction='backward'))
    osys.add_pupil()
    
    #add lyot stop
    lyot = poppy.CircularAperture(radius=lyot_radius,name='Lyot Stop')
    lyot._wavefront_display_hint='intensity'
    if obscuration:
        obsc = poppy.AsymmetricSecondaryObscuration(secondary_radius=0.7,support_width=0.30*u.meter,support_angle=(0,120,240))
        obsc._wavefront_display_hint='intensity'
    if llowfs:
        #take the rejected light for the LLOWFS
        lyot = poppy.InverseTransmission(lyot)
        osys.add_pupil(lyot)
        if obscuration:
            obsc = poppy.InverseTransmission(obsc)
            osys.add_pupil(obsc)

        
        #Add a defocus term to the sensor
        #Calc of peak-to-valley WFE: https://poppy-optics.readthedocs.io/en/stable/wfe.html
        defocus_coeff = 1/2/np.sqrt(3)*sensor_defocus*wavelength.to(u.m).value
        defocus = poppy.ZernikeWFE(radius=pupil_radius,coefficients=[0,0,0,defocus_coeff])
        osys.add_pupil(defocus)
        osys.add_detector(pixelscale=pixelscale, fov_pixels=npix_detector, oversample=1)
    else:
        osys.add_pupil(lyot)
        if obscuration:
            osys.add_pupil(obsc)
        osys.add_detector(pixelscale=pixelscale, fov_arcsec=1)
        
    return osys