
from lenspack.image.inversion import ks93, ks93inv
from lenspack.utils import sigma_critical
import numpy as np
from astropy import cosmology
from astropy.units.core import Unit

from copy import deepcopy as cp
from numpy.random import poisson

from .tools import rebin
from scipy.interpolate import interp2d

from scipy.ndimage import zoom

def weak_lensing( images, params, zl=0.3, zs=2.0, \
                 ngal_per_sq_arcmin=100., \
                 kpc_per_pixel=20., ell_disp=0.3, 
                 gals_per_bin=2., interpolate=False, **kwargs ):
    '''
    add weak lensing noise 
    '''
    
    kwargs = kwargs['kwargs']
    print(kwargs)

    if 'zl' in kwargs.keys():
        zl = kwargs['zl']
    if 'zs' in kwargs.keys():
        zs = kwargs['zs']
    if 'ngal_per_sq_arcmin' in kwargs.keys():
        ngal_per_sq_arcmin = kwargs['ngal_per_sq_arcmin']
    if 'kpc_per_pixel' in kwargs.keys():
        kpc_per_pixel = kwargs['kpc_per_pixel']
    if 'ell_disp' in kwargs.keys():
        ell_disp = kwargs['ell_disp']
    if 'e1_bias' in kwargs.keys():
        e1_bias = kwargs['e1_bias']     
    if 'e2_bias' in kwargs.keys():
        e2_bias = kwargs['e2_bias']  
    if 'interpolate' in kwargs.keys():
        interpolate = kwargs['interpolate']
        
    ngal_per_sq_arcmin  /= Unit('arcminute')**2
    
    kpc_per_pixel *= Unit('kpc')
    
    kpc_per_arcmin = \
        1*Unit("arcminute").to("radian")*cosmology.Planck18.angular_diameter_distance(0.3).to("kpc")/Unit("arcminute")

    arcmin_per_pixel = kpc_per_pixel / kpc_per_arcmin 
    
    sq_arcmin_per_pixel = arcmin_per_pixel * arcmin_per_pixel  
    
    ngalaxies_per_pixel =  ngal_per_sq_arcmin * sq_arcmin_per_pixel
    
    sigma_crit = sigma_critical(zl, zs, cosmology.Planck18)
    
    peak = params['lensing_norm']*Unit("solMass")/(Unit("Mpc")*Unit("Mpc"))
   
    peak_pc_sq = peak.to(Unit("solMass")/(Unit("pc")*Unit("pc")))/sigma_crit
    rebin_pix = int(np.ceil(gals_per_bin/ngalaxies_per_pixel))

    rebin_pix = [ i for i in range(1,rebin_pix+1) if images[0].shape[0]//i == images[0].shape[0]/i ][-1]
    
    all_wl_images = []
    for idx, image in enumerate(images):
        convergence = image*peak_pc_sq[idx]
        ampl = np.mean(convergence)

        e1,e2 = ks93inv(convergence,convergence*0.)

        e1 += np.random.randn( e1.shape[0],e1.shape[0] )*ell_disp/np.sqrt(2.)/np.sqrt(ngalaxies_per_pixel)
        e2 += np.random.randn( e2.shape[0],e2.shape[0] )*ell_disp/np.sqrt(2.)/np.sqrt(ngalaxies_per_pixel) 
    
        e1 = e1_bias[0] + (1.+e1_bias[1])*e1
        e2 = e2_bias[0] + (1.+e2_bias[1])*e2

        
        kappaE, kappB = ks93(e1,e2)
        
        #renormliase back to 0->1
        kappaE -= np.min(kappaE)
        kappaE += ampl
        kappaE /= np.max(kappaE)
        
        if interpolate & (rebin_pix > 1):
            kappaE = rebin_wl(kappaE, rebin_pix)
        
        all_wl_images.append(kappaE)
        
    return np.array(all_wl_images)

def rebin_wl( kappaE, rebin_pix):
    
    new_shape = kappaE.shape[0]//rebin_pix
    
    new_image = rebin( kappaE, (new_shape,new_shape))
    
    z = zoom( new_image, rebin_pix)
    
    return z

def xray( xray_emission_dimensionless, params, exposure_time=10_000, kpc_per_pixel=20, **kwargs ):
    kwargs = kwargs['kwargs']
    print(kwargs)
    xray_norm = params['xray_norm']
    
    if 'exposure_time' in kwargs.keys():
        exposure_time = kwargs['exposure_time']
    if 'kpc_per_pixel' in kwargs.keys():
        kpc_per_pixel = kwargs['kpc_per_pixel']
    if 'zl' in kwargs.keys():
        zl = kwargs['zl']      
    xray_emission = xray_norm[:,np.newaxis,np.newaxis]* \
        xray_emission_dimensionless*Unit('erg') / Unit('s') / Unit('cm')**2 / Unit('arcmin')**2
         
    kpc_per_arcmin = \
        1*Unit("arcminute").to("radian")*cosmology.Planck18.angular_diameter_distance(zl).to("kpc")/Unit("arcminute")

    exposure_time = exposure_time*Unit('s') # defualt 10 ks
    
    #the size of a pixel collecting area in arcminutes
    fov_arcmin = (kpc_per_pixel*Unit('kpc')/kpc_per_arcmin)**2 
    
    #the aperture size of Chandra
    aperture = np.pi*(60.*Unit('cm'))**2
    
    #Now calculate the total integrate photon count
    xray_energy_per_pixel = xray_emission * exposure_time * fov_arcmin * aperture
    
    #Then convert to kev
    total_energy_in_kev = xray_energy_per_pixel.to("keV")
    
    
    #Assuming a 2kev photo, calculate the number of photons
    number_photons = total_energy_in_kev/(2.*Unit("keV"))
    
    all_xray_images = []
    for iImage in range(xray_emission_dimensionless.shape[0]):
        
        
        image_inted = np.round(number_photons[iImage])
        exposure = poisson( image_inted.reshape(np.prod(image_inted.shape)), \
                       size=np.prod(image_inted.shape) ).reshape(image_inted.shape)  
        
        bkgrd = poisson( np.ones(image_inted.shape).reshape(np.prod(image_inted.shape)), \
                       size=np.prod(image_inted.shape) ).reshape(image_inted.shape)  
        all_image = (exposure+bkgrd).view(float)
        
        all_image /= np.max(all_image)
        
    
        all_xray_images.append(all_image)
        
    return np.array(all_xray_images)

    
def add_noise_to_images( images, params, channels, noise_parameters={} ):
    
    print("Adding noise to :", list(noise_parameters.keys()))
    
    default_noise_params = {}
    default_noise_params['total'] = \
        {'zl':0.3, 'zs':2.0, \
        'ngal_per_sq_arcmin':100., \
        'kpc_per_pixel':20., 
         'ell_disp':0.3,
         'e1_bias':[0.,0.], 
         'e2_bias':[0.,0.],
        'interpolate':False}
        
    default_noise_params['xray'] = {\
            'exposure_time':10_000,\
            'kpc_per_pixel':20,\
            'zl':default_noise_params['total']['zl']}
    noisy_images = cp(images)
    
    for noise_component in noise_parameters.keys():
        if noise_component == 'total' :
            for i in noise_parameters['total'].keys():
                default_noise_params['total'][i] = noise_parameters['total'][i]
            idx = [ i for i in range(len(channels)) if channels[i] == 'total'][0]
            noisy_images[:,:,:,idx] = weak_lensing( images[:,:,:,idx], params, kwargs=default_noise_params['total'] )
            
        if noise_component == 'xray':
            for i in noise_parameters['xray'].keys():
                default_noise_params['xray'][i] = noise_parameters['xray'][i]
                
            idx = [ i for i in range(len(channels)) if channels[i] == 'xray'][0]
            noisy_images[:,:,:,idx] = xray( images[:,:,:,idx], params, kwargs=default_noise_params['xray'] )
    
    
    return noisy_images


def get_einstein_radius( cluster_mass, zl=0.3, zs=2.0 ):
    
    cluster_mass  = cluster_mass*Unit('solMass')
    sigma_crit = sigma_critical(zl, zs, cosmology.Planck18)

    einstein_rad = np.sqrt( cluster_mass / sigma_crit).to(Unit('Mpc'))
    
    return einstein_rad
    
    