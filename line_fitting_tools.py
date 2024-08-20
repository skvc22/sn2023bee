import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import optimize
from scipy.stats import skewnorm
import scipy.stats as stats
import numpy as np
from astropy import constants as const
import random
from astropy.table import Table, Column
from astropy.io import ascii
from astropy.modeling import models
from astropy import units as u
import scipy.optimize as opt
import glob
import matplotlib.patheffects as PathEffects
import matplotlib.gridspec as gridspec
from astropy import constants as const
import emcee
import math
import corner
from uncertainties import ufloat
from uncertainties.umath import *  # sin(), etc.
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import scipy.interpolate
import scipy.ndimage
from astropy.nddata import block_reduce
from astropy.convolution import Gaussian1DKernel, convolve

#rest wave, Aki, j_f
line_dict = {'H':       ([6562.79, 4861.35, 4340.472, 4101.734],'#4477AA'),
             '[O III]': ([4958.911, 5006.843, 4363.210], '#EE6677'),
             '[O II]':  ([3726.032, 3728.815], '#AA3377'),
             '[N II]':  ([6548.050, 6583.460], 
                         [9.84e-04, 2.91e-03], #not sure about these
                         [2.,2.], '#CCBB44'), #not sure about these
             '[S II]':  ([6716.440, 6730.810],
                         [1.88e-04, 1.21e-04], #not sure about these
                         [2.5, 1.5],'#228833'), #not sure about these
             '[S III]':  ([9068.6, 9531.10],
                         [1.88e-04, 1.88e-04], #not sure about these
                         [2.5, 2.5],'#228833'), #not sure about these
             'Ca H':    ([3968.5]),
             'Ca K':    ([3933.7]),
             'G-band':  ([4304.4]),
             'Mg':      ([5175.3]),
             'Na ID':   ([5894.0]),
             '[Fe II]': ([7155.1742, 7171.9985, 7388.1673, 7452.5611],
                         [1.46e-01, 5.51e-02, 4.21e-02, 4.77e-02],
                         [4.5, 3.5, 3.5, 4.5], 'mediumblue'),
             '[Fe II] NIR': ([12567, 12704, 12788, 12943, 12978, 13206, 13278],
                         [4.7e-3, 3.32e-3, 2.45e-3, 1.98e-3, 1.08e-3, 1.31e-3, 1.17e-3],
                         [4.5, 0.5, 1.5, 2.5, 0.5, 3.5, 1.5], 'mediumblue'),

             '[Fe II] NIR2': ([16440],
                              [4.7e-3],
                              [4.5],'mediumblue'),
             '[Co II] NIR': ([15470],
                              [4.7e-3],
                              [4.5],'green'),

             '[Ca II]': ([7291.47, 7323.89], 
                         [1.3e+00, 1.3e+00], 
                         [2.5, 1.5], 'magenta'),
             '[O I]': ([6300.304, 6363.776], 
                         [5.63e-03, 1.82e-03], 
                         [2, 2], 'seagreen'),

             '[Ni II]': ([7377.83, 7411.61], 
                         [2.3e-01, 1.8e-01], 
                         [3.5, 2.5],'darkorange'),
             'Ca NIR':  ([8498.02, 8542.09, 8662.14],
                         [1.11e6, 9.9e6, 1.06e7],
                         [1.,1.,1.], 'magenta'),#not sure about these
             'Ca NIR1': ([8498.02],
                         [1.11e6],
                         [1.], 'magenta'),
             'Ca NIR2': ([8542.09],
                         [9.9e6],
                         [1.], 'magenta'),
             'Ca NIR3': ([8662.14],
                         [1.06e7],
                         [1.], 'magenta'),
             '[Ar III]': ([7136.],
                         [1.06e7],
                         [1.], 'maroon'),
             '[Fe III]': ([4701.],
                         [1.0e7],
                         [1.], 'blue')
             } 


def basic_format(size=[8,8]):
    plt.rc('font', family='serif')
    fig, ax = plt.subplots(1,1)
    fig.set_size_inches(size[0], size[1], forward = True)
    plt.minorticks_on()
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.tick_params(
        which='major', 
        bottom='on', 
        top='on',
        left='on',
        right='on',
        direction='in',
        length=20)
    plt.tick_params(
        which='minor', 
        bottom='on', 
        top='on',
        left='on',
        right='on',
        direction='in',
        length=10)


def gsmooth(x_array, y_array, var_y, vexp , nsig = 5.0):
    """Function gsmooth() is an inverse variance weighted Gaussian smoothing of spectra
       Optional imputs are smoothing velocity (vexp) and number of sigma (nsig)
       Syntax: new_y_array = gsmooth(x_array, y_array, var_y, vexp = 0.01, nsig = 5.0)
    """
    
    # Check for zero variance points, and set to 1E-20

    if var_y is None:
        var_y = 1.e-31*np.ones(len(y_array))
    
    # Output y-array
    new_y = np.zeros(len(x_array), float)
    
    # Loop over y-array elements
    for i in range(len(x_array)):
        
        # Construct a Gaussian of sigma = vexp*x_array[i]
        gaussian = np.zeros(len(x_array), float)
        sigma = vexp*x_array[i]
        
        # Restrict range to +/- nsig sigma
        sigrange = np.nonzero(abs(x_array-x_array[i]) <= nsig*sigma)
        gaussian[sigrange] = (1/(sigma*np.sqrt(2*np.pi)))*np.exp(-0.5*((x_array[sigrange]-x_array[i])/sigma)**2)
        
        # Multiply Gaussian by 1 / variance
        W_lambda = gaussian / var_y
        
        # Perform a weighted sum to give smoothed y value at x_array[i]
        W0 = np.sum(W_lambda)
        W1 = np.sum(W_lambda*y_array)
        new_y[i] = W1/W0

    # Return smoothed y-array
    return new_y


def spectres(new_wavs, spec_wavs, spec_fluxes, spec_errs=None, fill=None,
             verbose=True):

    """
    Function for resampling spectra (and optionally associated
    uncertainties) onto a new wavelength basis.
    Parameters
    ----------
    new_wavs : numpy.ndarray
        Array containing the new wavelength sampling desired for the
        spectrum or spectra.
    spec_wavs : numpy.ndarray
        1D array containing the current wavelength sampling of the
        spectrum or spectra.
    spec_fluxes : numpy.ndarray
        Array containing spectral fluxes at the wavelengths specified in
        spec_wavs, last dimension must correspond to the shape of
        spec_wavs. Extra dimensions before this may be used to include
        multiple spectra.
    spec_errs : numpy.ndarray (optional)
        Array of the same shape as spec_fluxes containing uncertainties
        associated with each spectral flux value.
    fill : float (optional)
        Where new_wavs extends outside the wavelength range in spec_wavs
        this value will be used as a filler in new_fluxes and new_errs.
    verbose : bool (optional)
        Setting verbose to False will suppress the default warning about
        new_wavs extending outside spec_wavs and "fill" being used.
    Returns
    -------
    new_fluxes : numpy.ndarray
        Array of resampled flux values, first dimension is the same
        length as new_wavs, other dimensions are the same as
        spec_fluxes.
    new_errs : numpy.ndarray
        Array of uncertainties associated with fluxes in new_fluxes.
        Only returned if spec_errs was specified.
    """

    # Rename the input variables for clarity within the function.
    old_wavs = spec_wavs
    old_fluxes = spec_fluxes
    old_errs = spec_errs

    # Arrays of left hand sides and widths for the old and new bins
    old_lhs = np.zeros(old_wavs.shape[0])
    old_widths = np.zeros(old_wavs.shape[0])
    old_lhs = np.zeros(old_wavs.shape[0])
    old_lhs[0] = old_wavs[0]
    old_lhs[0] -= (old_wavs[1] - old_wavs[0])/2
    old_widths[-1] = (old_wavs[-1] - old_wavs[-2])
    old_lhs[1:] = (old_wavs[1:] + old_wavs[:-1])/2
    old_widths[:-1] = old_lhs[1:] - old_lhs[:-1]
    old_max_wav = old_lhs[-1] + old_widths[-1]

    new_lhs = np.zeros(new_wavs.shape[0]+1)
    new_widths = np.zeros(new_wavs.shape[0])
    new_lhs[0] = new_wavs[0]
    new_lhs[0] -= (new_wavs[1] - new_wavs[0])/2
    new_widths[-1] = (new_wavs[-1] - new_wavs[-2])
    new_lhs[-1] = new_wavs[-1]
    new_lhs[-1] += (new_wavs[-1] - new_wavs[-2])/2
    new_lhs[1:-1] = (new_wavs[1:] + new_wavs[:-1])/2
    new_widths[:-1] = new_lhs[1:-1] - new_lhs[:-2]

    # Generate output arrays to be populated
    new_fluxes = np.zeros(old_fluxes[..., 0].shape + new_wavs.shape)

    if old_errs is not None:
        if old_errs.shape != old_fluxes.shape:
            raise ValueError("If specified, spec_errs must be the same shape "
                             "as spec_fluxes.")
        else:
            new_errs = np.copy(new_fluxes)

    start = 0
    stop = 0

    # Calculate new flux and uncertainty values, looping over new bins
    for j in range(new_wavs.shape[0]):

        # Add filler values if new_wavs extends outside of spec_wavs
        if (new_lhs[j] < old_lhs[0]) or (new_lhs[j+1] > old_max_wav):
            new_fluxes[..., j] = fill

            if spec_errs is not None:
                new_errs[..., j] = fill

            if (j == 0) and verbose:
                print("\nSpectres: new_wavs contains values outside the range "
                      "in spec_wavs. New_fluxes and new_errs will be filled "
                      "with the value set in the 'fill' keyword argument (nan "
                      "by default).\n")
            continue

        # Find first old bin which is partially covered by the new bin
        while old_lhs[start+1] <= new_lhs[j]:
            start += 1

        # Find last old bin which is partially covered by the new bin
        while old_lhs[stop+1] < new_lhs[j+1]:
            stop += 1

        # If new bin is fully inside an old bin start and stop are equal
        if stop == start:
            new_fluxes[..., j] = old_fluxes[..., start]
            if old_errs is not None:
                new_errs[..., j] = old_errs[..., start]

        # Otherwise multiply the first and last old bin widths by P_ij
        else:
            start_factor = ((old_lhs[start+1] - new_lhs[j])
                            / (old_lhs[start+1] - old_lhs[start]))

            end_factor = ((new_lhs[j+1] - old_lhs[stop])
                          / (old_lhs[stop+1] - old_lhs[stop]))

            old_widths[start] *= start_factor
            old_widths[stop] *= end_factor

            # Populate new_fluxes spectrum and uncertainty arrays
            f_widths = old_widths[start:stop+1]*old_fluxes[..., start:stop+1]
            new_fluxes[..., j] = np.sum(f_widths, axis=-1)
            new_fluxes[..., j] /= np.sum(old_widths[start:stop+1])

            if old_errs is not None:
                e_wid = old_widths[start:stop+1]*old_errs[..., start:stop+1]

                new_errs[..., j] = np.sqrt(np.sum(e_wid**2, axis=-1))
                new_errs[..., j] /= np.sum(old_widths[start:stop+1])

            # Put back the old bin widths to their initial values
            old_widths[start] /= start_factor
            old_widths[stop] /= end_factor

    # If errors were supplied return both new_fluxes and new_errs.
    if old_errs is not None:
        return np.array([new_wavs, new_fluxes, new_errs])
        # return new_fluxes, new_errs

    # Otherwise just return the new_fluxes spectrum array
    else:
        # return new_fluxes
        return np.array([new_wavs, new_fluxes])


c = const.c.to('km/s').value
def calculate_wave_from_velocity(velocity, rest_wave):
    wave = rest_wave*np.sqrt((c+velocity)/(c-velocity))
    return wave
    
def calculate_velocity(wave, rest_wave):
    v = -1.*c*((rest_wave/wave)**2. - 1)/(1+((rest_wave/wave)**2.))
    return v


def calculate_gf(Aki, wave, j_f):
    # get Aki from table
    #wave is wavelength in A
    #j_f is total angular momentum of the upper level
    return 1.499e-8*(Aki)*(wave**2.)*((2.0*j_f+1.0))

def calc_ebv_from_H_decrement(line_ratio):
    ebv_gal = 1.97*np.log10(line_ratio/2.86)
    if ebv_gal < 0 or np.isinf(ebv_gal):
        ebv_gal = 0.
    return ebv_gal

def get_all_fluxes(mgaus_solved, error=True, sf=10**(-15)): #scaling right for firefly?
    if error:
        oii_a_flux = ufloat(mgaus_solved[3]['[O II]_a_3726'][0], mgaus_solved[3]['[O II]_a_3726'][1])*sf
        oii_b_flux = ufloat(mgaus_solved[3]['[O II]_b_3728'][0], mgaus_solved[3]['[O II]_b_3728'][1])*sf

        oiii_a_flux = ufloat(mgaus_solved[3]['[O III]_a_4958'][0], mgaus_solved[3]['[O III]_a_4958'][1])*sf
        oiii_b_flux = ufloat(mgaus_solved[3]['[O III]_b_5006'][0], mgaus_solved[3]['[O III]_b_5006'][1])*sf

        nii_a_flux = ufloat(mgaus_solved[3]['[N II]_a_6548'][0], mgaus_solved[3]['[N II]_a_6548'][1])*sf
        nii_b_flux = ufloat(mgaus_solved[3]['[N II]_b_6583'][0], mgaus_solved[3]['[N II]_b_6583'][1])*sf

        halpha_flux = ufloat(mgaus_solved[3]['H_a_6562'][0], mgaus_solved[3]['H_a_6562'][1])*sf
        hbeta_flux = ufloat(mgaus_solved[3]['H_b_4861'][0], mgaus_solved[3]['H_b_4861'][1])*sf

        sii_a_flux = ufloat(mgaus_solved[3]['[S II]_a_6716'][0], mgaus_solved[3]['[S II]_a_6716'][1])*sf
        sii_b_flux = ufloat(mgaus_solved[3]['[S II]_b_6730'][0], mgaus_solved[3]['[S II]_b_6730'][1])*sf
    else:
        oii_a_flux = ufloat(mgaus_solved[3]['[O II]_a_3726'][0], 0)*sf
        oii_b_flux = ufloat(mgaus_solved[3]['[O II]_b_3728'][0], 0)*sf

        oiii_a_flux = ufloat(mgaus_solved[3]['[O III]_a_4958'][0], 0)*sf
        oiii_b_flux = ufloat(mgaus_solved[3]['[O III]_b_5006'][0], 0)*sf

        nii_a_flux = ufloat(mgaus_solved[3]['[N II]_a_6548'][0], 0)*sf
        nii_b_flux = ufloat(mgaus_solved[3]['[N II]_b_6583'][0], 0)*sf

        halpha_flux = ufloat(mgaus_solved[3]['H_a_6562'][0], 0)*sf
        hbeta_flux = ufloat(mgaus_solved[3]['H_b_4861'][0], 0)*sf

        sii_a_flux = ufloat(mgaus_solved[3]['[S II]_a_6716'][0], 0)*sf
        sii_b_flux = ufloat(mgaus_solved[3]['[S II]_b_6730'][0], 0)*sf

    flux_arr = [oii_a_flux,oii_b_flux,oiii_a_flux,oiii_b_flux,nii_a_flux,nii_b_flux,halpha_flux,hbeta_flux,sii_a_flux,sii_b_flux]
    for i,f in enumerate(flux_arr):
        if f <= 0.:
            flux_arr[i] = ufloat(0, f.std_dev)
    oii_a_flux,oii_b_flux,oiii_a_flux,oiii_b_flux,nii_a_flux,nii_b_flux,halpha_flux,hbeta_flux,sii_a_flux,sii_b_flux = flux_arr
    return (oii_a_flux,oii_b_flux,oiii_a_flux,oiii_b_flux,nii_a_flux,nii_b_flux,halpha_flux,hbeta_flux,sii_a_flux,sii_b_flux)

def calc_logr23(oii_a_flux, oii_b_flux, oiii_a_flux, oiii_b_flux, hbeta_flux, error=True):

    oii = oii_a_flux+oii_b_flux
    oiii = oiii_a_flux+oiii_b_flux
    hbeta = hbeta_flux
    try:
        logR23 = log10((oii+oiii)/hbeta)
    except:
        logR23 = ufloat(float('nan'), float('nan'))
    try:
        logO3Hbeta = log10(oiii/hbeta)
    except:
        logO3Hbeta = ufloat(float('nan'), float('nan'))
    try:
        logO32 = log10(oiii/oii)
    except:
        logO32 = ufloat(float('nan'), float('nan'))

    return logR23, logO32, logO3Hbeta

def calc_logN2Halpha(nii_b_flux, halpha_flux,error=True):
    nii = nii_b_flux
    halpha = halpha_flux
    try:
        logN2Halpha = log10(nii/halpha)
    except:
        logN2Halpha = ufloat(float('nan'), float('nan'))

    return logN2Halpha

def calc_N2O2(nii_b_flux, oii_a_flux, oii_b_flux, error=True):

    nii = nii_b_flux
    oii = oii_a_flux+oii_b_flux

    if nii.nominal_value ==0.:
        return np.nan
    
    try:
        logN2O2 = log10(nii/oii)
    except:
        logN2O2 = ufloat(float('nan'), float('nan'))

    return logN2O2

def calc_metallicity_two_branch(logR23, logO32, logN2O2):
    #Kewley 2008, using N2O2 to break degeneracy, accuracy ~.15 dex
    if logN2O2 < -1.2 or isnan(logN2O2):
        return (12 - 4.944 + 0.767*logR23 + 0.602*logR23**2 - logO32*(0.29 + 0.332*logR23 - 0.331*logR23**2))
    else:
        return (12 - 2.939 - 0.2*logR23 - 0.237*logR23**2 - 0.305*logR23**3 - 0.0283*logR23**4 
                - logO32*(0.0047 - 0.0221*logR23 - 0.102*logR23**2 - 0.0817*logR23**3 - 0.00717*logR23**4)) 

def calc_metallicity(logR23):
    #tremonti 2004
    return (9.185 - 0.313*logR23 - 0.264*logR23**2 - 0.321*logR23**3)

def calc_metallicity_N2(logN2):
    #pettini & pagel 2004
    # if np.isnan(logN2.nominal_value):
    #     return 'N/A'
    # return (8.90 + 0.57*logN2)
    return (9.37 + 2.03*logN2 + 1.26*logN2**2 + .32*logN2**3)

def gaus(x,a,x0,sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def gconvolve(x,a,x0,sigma,lsf_width=30):
    g1 = a*np.exp(-(x-x0)**2/(2*sigma**2))
    g2 = Gaussian1DKernel(stddev=lsf_width/np.diff(x)[0])
    gconv = convolve(g1, g2)
    # gconv_norm = gconv*(np.amax(g1)/np.amax(gconv))
    # plt.plot(x,g1)
    # plt.plot(x,gconv_norm)
    # plt.plot(x,gconv)
    # plt.show()
    return gconv

# def gconvolve(x,a,x0,sigma,lsf_width=30):
#     print(x[0], x[-1])
#     x1 = x[0]
#     x2 = x[-1]
#     diff = (np.absolute(np.absolute(x1) - np.absolute(x2)))/2.
#     print (diff)
#     if np.absolute(x1) > np.absolute(x2):
#         diff = 1.*diff
#     else:
#         diff = -1.*diff
#     x_conv = x+diff

#     g1 = a*np.exp(-(x-x0)**2/(2*sigma**2))
#     g2 = a*np.exp(-(x-0)**2/(2*lsf_width**2))
#     gconv = np.convolve(g1, g2,mode='same')
#     gconv_norm = gconv*(np.amax(g1)/np.amax(gconv))
#     plt.plot(x,g1)
#     plt.plot(x,gconv_norm)
#     plt.show()
#     return gconv_norm

############################### Eva added skew function
def gaus(x,a,x0,sigma,skew):
    # return a*skewnorm.pdf(x,skew,x0,sigma+1)
    #Matt: some function have default values, in this case, loc and scale need to be defined this way
    #      added if statement so the skew is always zero if not in our initial guess
    #      changed how we scale, since a pdf has area = 1 initialized with a guess of say .4 lead to a curve 
    #      with a peak that was very small, this is why your curves all looked flat
    #      now we can use the same guess and the function scales the peak appropriately
    if skew == None:
        temp_dist = skewnorm.pdf(x,0,loc=x0,scale=sigma)
        peak_val = np.amax(temp_dist)
        if np.isnan(peak_val) or peak_val == 0.:
            scale_fac = 0.
        else:
            scale_fac = (a/peak_val)
        return scale_fac*skewnorm.pdf(x,0,loc=x0,scale=sigma)
    else:
        temp_dist = skewnorm.pdf(x,skew,loc=x0,scale=sigma)
        peak_val = np.amax(temp_dist)
        if np.isnan(peak_val) or peak_val == 0.:
            scale_fac = 0.
        else:
            scale_fac = (a/peak_val)
        return scale_fac*skewnorm.pdf(x,skew,loc=x0,scale=sigma)


class element_line(object):
    def __init__(self, name = None, rest_wave = None, Aki = None, j_f=None, parent_line=None, master_line=None):
        if rest_wave is not None:
            self.name = name
            self.rest_wave = rest_wave
            self.Aki = Aki
            self.j_f = j_f
            self.parent_line = parent_line
            if master_line:
                self.master_line = master_line
            else:
                self.master_line = self
            
    def set_gaus_params(self, amp, vel, v_width, skew = None):
        #params in line frame
        self.a = amp #amp same in both frames
        self.vel = vel
        self.v_width = v_width
        self.skew = skew # Eva added skew parameter
        
        #determine parameters for drawing line in master frame
        # v_wave = calculate_wave_from_velocity(vel, self.rest_wave)#wave of current line corresponding to vel
        # self.x0 = calculate_velocity(v_wave, self.master_line.rest_wave)#v_wave in frame of master line (gaus center)
        # v_diff_wave = calculate_wave_from_velocity(vel - v_width, self.rest_wave)#wave of current line corresponding to vel-v_width
        # v_diff_master = calculate_velocity(v_diff_wave, self.master_line.rest_wave)#v_diff_wave in frame of master line
        # self.sigma = v_diff_master - self.x0#gaus width in master frame

def setup_gal_fit(verbose=False, all_params = False):
    fit_param_indices = {}
    fit_lines = []
    fit_params = []

    width = 1000
    H_param_guess = {'amp': .65, 'vel': 0., 'width': width}
    fit_lines, fit_params, fit_param_indices = add_line_to_fit('H_a', fit_lines, fit_params, fit_param_indices, 
                                                            components=[0], param_guess = H_param_guess)
    if all_params:
        H_param_guess = {'amp': .65, 'vel': 0., 'width': width}
    else:
        H_param_guess = {'amp': .65, 'vel': 'H_a', 'width': 'H_a'}
    fit_lines, fit_params, fit_param_indices = add_line_to_fit('H_b', fit_lines, fit_params, fit_param_indices, 
                                                            components=[1], param_guess = H_param_guess)
    # # if all_params:
    # #     H_param_guess = {'amp': .1, 'vel': 0., 'width': 100.}
    # # else:   
    # #     H_param_guess = {'amp': .1, 'vel': 'H_a', 'width': 'H_a'}
    # # fit_lines, fit_params, fit_param_indices = add_line_to_fit('H_c', fit_lines, fit_params, fit_param_indices, 
    # #                                                         components=[2], param_guess = H_param_guess)
    # # if all_params:
    # #     H_param_guess = {'amp': .1, 'vel': 0., 'width': 0.}
    # # else:
    # #     H_param_guess = {'amp': .1, 'vel': 'H_a', 'width': 'H_a'}
    # # fit_lines, fit_params, fit_param_indices = add_line_to_fit('H_d', fit_lines, fit_params, fit_param_indices, 
    # #                                                         components=[3], param_guess = H_param_guess)

    OIII_param_guess = {'amp': .4, 'vel': 'H_a', 'width': width}
    fit_lines, fit_params, fit_param_indices = add_line_to_fit('[O III]_a', fit_lines, fit_params, fit_param_indices, 
                                                            components=[0], param_guess = OIII_param_guess)
    if all_params:
        OIII_param_guess = {'amp': .4, 'vel': 0, 'width': width}
    else:
        OIII_param_guess = {'amp': .4, 'vel': 'H_a', 'width': '[O III]_a'}
    fit_lines, fit_params, fit_param_indices = add_line_to_fit('[O III]_b', fit_lines, fit_params, fit_param_indices, 
                                                            components=[1], param_guess = OIII_param_guess)


    # OII_param_guess = {'amp': .5, 'vel': 0., 'width': 100.}
    # fit_lines, fit_params, fit_param_indices = add_line_to_fit('[O II]_a', fit_lines, fit_params, fit_param_indices, 
    #                                                         components=[0], param_guess = OII_param_guess)
    # if all_params:
    #     OII_param_guess = {'amp': .5, 'vel': 0., 'width': 100.}
    # else:
    #     OII_param_guess = {'amp': '[O II]_a', 'vel': '[O II]_a', 'width': '[O II]_a'}
    # fit_lines, fit_params, fit_param_indices = add_line_to_fit('[O II]_b', fit_lines, fit_params, fit_param_indices, 
    #                                                         components=[1], param_guess = OII_param_guess)



    # N_param_guess = {'amp': .1, 'vel': 'H_a', 'width': width}
    # fit_lines, fit_params, fit_param_indices = add_line_to_fit('[N II]_a', fit_lines, fit_params, fit_param_indices, 
    #                                                         components=[0], param_guess = N_param_guess)
    # if all_params:
    #     N_param_guess = {'amp': .1, 'vel': 0., 'width': width}
    # else:
    #     N_param_guess = {'amp': .1, 'vel': 'H_a', 'width': '[N II]_a'}
    # fit_lines, fit_params, fit_param_indices = add_line_to_fit('[N II]_b', fit_lines, fit_params, fit_param_indices, 
    #                                                         components=[1], param_guess = N_param_guess)


    S_param_guess = {'amp': .1, 'vel': 'H_a', 'width': width}
    fit_lines, fit_params, fit_param_indices = add_line_to_fit('[S II]_a', fit_lines, fit_params, fit_param_indices, 
                                                            components=[0], param_guess = S_param_guess)
    if all_params:
        S_param_guess = {'amp': .1, 'vel': 0., 'width': width}
    else:
        S_param_guess = {'amp': .1, 'vel': 'H_a', 'width': '[S II]_a'}
    fit_lines, fit_params, fit_param_indices = add_line_to_fit('[S II]_b', fit_lines, fit_params, fit_param_indices, 
                                                            components=[1], param_guess = S_param_guess)

    # S3_param_guess = {'amp': .1, 'vel': 'H_a', 'width': width}
    # fit_lines, fit_params, fit_param_indices = add_line_to_fit('[S III]_a', fit_lines, fit_params, fit_param_indices, 
    #                                                         components=[0], param_guess = S3_param_guess)
    # if all_params:
    #     S3_param_guess = {'amp': .1, 'vel': 0., 'width': width}
    # else:
    #     S3_param_guess = {'amp': .1, 'vel': 'H_a', 'width': '[S III]_a'}
    # fit_lines, fit_params, fit_param_indices = add_line_to_fit('[S III]_b', fit_lines, fit_params, fit_param_indices, 
    #                                                         components=[1], param_guess = S3_param_guess)


    if verbose:
        for line in fit_lines:
            print (line.name, line.rest_wave)
        print (fit_params)
        print (fit_param_indices)
    return fit_lines, fit_params, fit_param_indices

def setup_gal_fit_test(verbose=False):
    fit_param_indices = {}
    fit_lines = []
    fit_params = []

    H_param_guess = {'amp': .65, 'vel': 0., 'width': 100}
    fit_lines, fit_params, fit_param_indices = add_line_to_fit('H_a', fit_lines, fit_params, fit_param_indices, 
                                                            components=[0], param_guess = H_param_guess)
    H_param_guess = {'amp': .65, 'vel': 'H_a', 'width': 'H_a'}
    fit_lines, fit_params, fit_param_indices = add_line_to_fit('H_b', fit_lines, fit_params, fit_param_indices, 
                                                            components=[1], param_guess = H_param_guess)
    H_param_guess = {'amp': .1, 'vel': 'H_a', 'width': 'H_a'}
    fit_lines, fit_params, fit_param_indices = add_line_to_fit('H_c', fit_lines, fit_params, fit_param_indices, 
                                                            components=[2], param_guess = H_param_guess)
    H_param_guess = {'amp': .1, 'vel': 'H_a', 'width': 'H_a'}
    fit_lines, fit_params, fit_param_indices = add_line_to_fit('H_d', fit_lines, fit_params, fit_param_indices, 
                                                            components=[3], param_guess = H_param_guess)

    N_param_guess = {'amp': .1, 'vel': 0., 'width': 100}
    fit_lines, fit_params, fit_param_indices = add_line_to_fit('[N II]_a', fit_lines, fit_params, fit_param_indices, 
                                                            components=[0], param_guess = N_param_guess)
    N_param_guess = {'amp': .1, 'vel': '[N II]_a', 'width': '[N II]_a'}
    fit_lines, fit_params, fit_param_indices = add_line_to_fit('[N II]_b', fit_lines, fit_params, fit_param_indices, 
                                                            components=[1], param_guess = N_param_guess)
    
    if verbose:
        for line in fit_lines:
            print (line.name, line.rest_wave)
        print (fit_params)
        print (fit_param_indices)
    return fit_lines, fit_params, fit_param_indices


def set_all_line_params(fit_params, fit_lines, fit_param_indices):
    #assumes parent lines are set first
    for line in fit_lines:
        if line.Aki != None and line.j_f != None:
            line.osc_strength = calculate_gf(line.Aki, line.rest_wave, line.j_f)
        if line.parent_line:
            #non-parent lines gaus params determined from parent
            factor = line.osc_strength/line.parent_line.osc_strength#determine strength relative to parent line
            line.set_gaus_params(factor*line.parent_line.a, line.parent_line.vel, line.parent_line.v_width, line.parent_line.skew)
        else:
            amp_ind = fit_param_indices[line.name+'_amp']
            vel_ind = fit_param_indices[line.name+'_vel']
            width_ind = fit_param_indices[line.name+'_width']
            ############################### Eva added skew
            if line.name+'_skew' in fit_param_indices.keys(): #Matt: only define skew if it is in the dictionary, else set it to None
                skew_ind = fit_param_indices[line.name+'_skew']
                skew = fit_params[skew_ind]
            else:
                skew = None

            amp = fit_params[amp_ind]
            vel = fit_params[vel_ind]
            width = fit_params[width_ind]

            
#             if line.skew = None:
#                 skew == 0
#             else:
#                 skew_ind = fit_param_indices[line.name+'_skew']
#                 'skew' == 0
                
            ############################## Eva added all kinds of stuff in different attempts
            #Matt added some more variables to make this clear, skew will be none if it isn't in the intitial guess
            line.set_gaus_params(amp, vel, width, skew)
            #line.set_gaus_params(fit_params[amp_ind], fit_params[vel_ind], fit_params[width_ind])
    return
 
#             amp_ind = fit_param_indices[line.name+'_amp']
#             vel_ind = fit_param_indices[line.name+'_vel']
#             width_ind = fit_param_indices[line.name+'_width']
#             skew_ind = fit_param_indices[line.name+'_skew']
                
            
#             line.set_gaus_params(fit_params[amp_ind], fit_params[vel_ind], fit_params[width_ind], fit_params[skew_ind])
            
#             if 'skew' in fit_param_indices:
# #                 skew_ind = fit_param_indices[line.name+'_skew']
#                 set_gaus_params.append(fit_params[skew_ind]
                
            
               
# #             for param_name in param_guess.keys():
# #                 if type(param_guess[param_name]) != str:
# #                     fit_params.append(param_guess[param_name])
# #                     fit_param_indices[name+'_'+param_name] = len(fit_params) - 1 #just appended so located at last index
# #                 else:
# #                     fit_param_indices[name+'_'+param_name] = fit_param_indices[param_guess[param_name]+'_'+param_name]
#     return 


def add_line_to_fit(name, fit_lines, fit_params, fit_param_indices, components='all', param_guess = {}):
    init_params = []
    line_data = line_dict[name.split('_')[0]]
    if components == 'all': #use all lines for this element in the dictionary
        components = np.arange(len(line_data[0]))
    for i, x in enumerate(components):
        if i == 0: #first line is the parent line
            if len(line_data) == 2:
                Aki = None
                j_f = None
            else:
                Aki = line_data[1][x]
                j_f = line_data[2][x]
            if len(fit_lines) == 0: #first line of fit_lines is master line
                line_obj = element_line(name=name, rest_wave = line_data[0][x], Aki = Aki, 
                                        j_f = j_f, parent_line=None, master_line=None)
                parent_line = line_obj
            else:
                line_obj = element_line(name=name, rest_wave = line_data[0][x], Aki = Aki, 
                                        j_f = j_f, parent_line=None, master_line=fit_lines[0])
                parent_line = line_obj
        else:
            if len(line_data) == 2:
                Aki = None
                j_f = None
            else:
                Aki = line_data[1][x]
                j_f = line_data[2][x]
            line_obj = element_line(name=name, rest_wave = line_data[0][x], Aki = Aki, 
                                    j_f = j_f, parent_line=parent_line, master_line=fit_lines[0])
        
        for param_name in param_guess.keys():
            if 'amp' in param_name:
                if type(param_guess[param_name]) == str:
                    line_obj.amp_ref = param_guess[param_name]
                else:
                    line_obj.amp_ref = None
            elif 'vel' in param_name:
                if type(param_guess[param_name]) == str:
                    line_obj.vel_ref = param_guess[param_name]
                else:
                    line_obj.vel_ref = None
            elif 'width' in param_name:
                if type(param_guess[param_name]) == str:
                    line_obj.wid_ref = param_guess[param_name]
                else:
                    line_obj.wid_ref = None

        fit_lines.append(line_obj)
    
    for param_name in param_guess.keys():
        if type(param_guess[param_name]) != str:
            fit_params.append(param_guess[param_name])
            fit_param_indices[name+'_'+param_name] = len(fit_params) - 1 #just appended so located at last index
        else:
            fit_param_indices[name+'_'+param_name] = fit_param_indices[param_guess[param_name]+'_'+param_name]

    return fit_lines, fit_params, fit_param_indices


#TODO: update
def min_func(params, w_range, flux, flux_error, fit_lines, fit_param_indices, fit_cont, lsf_width):
#     params = (params[0], params[1]*5000, params[2]*10000, params[3], params[4]*5000, params[5]*5000, params[6], 
#               params[7]*5000, params[8]*5000, params[9]*7000, 
#               params[10]*7700)
    # params = (params[0], params[1]*5000, params[2]*10000, params[3], params[4]*5000, params[5]*5000, params[6], 
    #           params[7]*5000, params[8]*5000, params[9], params[10]*5000, params[11]*5000, params[12]*7000, 
    #           params[13]*7700)
    
    set_all_line_params(params, fit_lines, fit_param_indices)
    #print (params)
    model_flux, flux_subcont, w_range_fit = multiple_gaussians(fit_lines, flux, w_range, params, flux_error, fit_cont, lsf_width)
    return np.sum(((flux_subcont - model_flux)/flux_error)**2.)

def calc_vel_ranges(w_range, fit_lines):
    for line in fit_lines:
        line_vel_range = calculate_velocity(w_range, line.rest_wave)
        line.vel_range = line_vel_range
    return

def fit_continuum(w_range, flux, fit_params):
    wave_min = fit_params[-2]
    wave_max = fit_params[-1]
    roi = (w_range > wave_min) & (w_range < wave_max)
    spec_smooth = gsmooth(w_range[roi], flux[roi], var_y=None, vexp=.002)
    m = (spec_smooth[-1]-spec_smooth[0]) / (w_range[roi][-1]-w_range[roi][0])
    b = spec_smooth[-1] - m*w_range[roi][-1]
    continuum = m*w_range + b   
    flux_subcont = flux - continuum
    return flux_subcont

def multiple_gaussians(fit_lines, flux, w_range, fit_params, flux_error, fit_cont, fit_param_indices = None, 
                        emcee_samples = None, plot=False, plot_cont=False, plot_ivar=False, plot_emcee=False, 
                        plot_resid=False, plot_inset=False, legend=False, text='', txt_color='k',store_line_fluxes = False, 
                        savename=None, xlim = None, ylim = None, resid_ylim=None, fit_dw_plot=None, lsf_width=2):
    if emcee_samples:
        set_all_line_params(fit_params, fit_lines, fit_param_indices)


    deltafit=w_range[1]-w_range[0]
    if fit_dw_plot is not None:
        w_range_fit = np.arange(w_range[0],w_range[-1], fit_dw_plot)
        deltafit = fit_dw_plot
    else:
        w_range_fit = w_range

    if fit_cont:
        wave_min = fit_params[-2]
        wave_max = fit_params[-1]

        roi = (w_range > wave_min) & (w_range < wave_max)
        spec_smooth = gsmooth(w_range[roi], flux[roi], var_y=None, vexp=.002)
        m = (spec_smooth[-1]-spec_smooth[0]) / (w_range[roi][-1]-w_range[roi][0])
        b = spec_smooth[-1] - m*w_range[roi][-1]
        continuum = m*w_range + b   
        flux_subcont = flux - continuum
    else:
        flux_subcont = flux
    
    # vel = calculate_velocity(w_range, fit_lines[0].rest_wave)

    # total_flux = np.zeros(len(vel))
    total_flux = np.zeros(len(w_range_fit))
    

    if plot_cont and fit_cont:
        plt.plot(w_range, flux)
        plt.plot(w_range, continuum)
        plt.scatter(w_range[roi][0], spec_smooth[0], color='r', s=80)
        plt.scatter(w_range[roi][-1], spec_smooth[-1], color='r', s=80)
        plt.show()
        
    if plot:
        plt.rc('font', family='serif')

        if plot_resid:
            fig, ax1 = plt.subplots(figsize=[12,12], sharex=True)
            ax1.set_axis_off()
            fig.subplots_adjust(hspace=0)
            ax1 = fig.add_subplot(211)
            
        else:
            # fig, ax1 = plt.subplots(figsize=[12,8])
            fig, ax1 = plt.subplots(figsize=[12,12])

        plt.minorticks_on()
        plt.xticks(fontsize = 20)
        plt.yticks(fontsize = 20)
        plt.tick_params(
            which='major', 
            bottom='on', 
            top='on',
            left='on',
            right='on',
            direction='in',
            length=20)
        plt.tick_params(
            which='minor', 
            bottom='on', 
            top='on',
            left='on',
            right='on',
            direction='in',
            length=10)

        plt.plot(w_range, flux_subcont, drawstyle='steps-mid', color='k', lw=2)
        if plot_ivar:
            plt.fill_between(w_range, flux_subcont-flux_error, 
                flux_subcont+flux_error, color='black', alpha = 0.5)
    
    unique_elements = []
    for line in fit_lines:
        unique_elements.append(line.name)
    unique_elements = set(unique_elements)
####gaus function called
    if store_line_fluxes:
        flux_dict = {}
        if emcee_samples:
            flux_emcee_dict = {}
    for el in unique_elements:
        el_flux = np.zeros(len(w_range_fit))
        for line in fit_lines:
            if line.name == el:
                if fit_dw_plot is None:
                    if line.skew is not None:
                        line_flux = gaus(line.vel_range, line.a, line.vel, line.v_width, line.skew)
                    else:
                        line_flux = gconvolve(line.vel_range, line.a, line.vel, line.v_width, lsf_width=lsf_width)
                else:
                    vel_range_fit = calculate_velocity(w_range_fit, line.rest_wave)
                    if line.skew is not None:
                        line_flux = gaus(vel_range_fit, line.a, line.vel, line.v_width, line.skew)
                    else:
                        line_flux = gconvolve(vel_range_fit, line.a, line.vel, line.v_width, lsf_width=lsf_width)
                line.line_flux = line_flux

                if emcee_samples:
                    line_fluxes_samps = []
                    for esamp in emcee_samples:
                        if line.amp_ref is None:
                            amp_ind = fit_param_indices[line.name+'_amp']
                        else:
                            amp_ind = fit_param_indices[line.amp_ref+'_amp']
                        if line.vel_ref is None:
                            vel_ind = fit_param_indices[line.name+'_vel']
                        else:
                            vel_ind = fit_param_indices[line.vel_ref+'_vel']
                        if line.wid_ref is None:
                            wid_ind = fit_param_indices[line.name+'_width']
                        else:
                            wid_ind = fit_param_indices[line.wid_ref+'_width']
                        if line.parent_line != None:
                            factor = line.osc_strength/line.parent_line.osc_strength#determine strength relative to parent line
                        else:
                            factor=1.
                        if fit_dw_plot is None:
                            # lflux = gaus(line.vel_range, factor*esamp[amp_ind], esamp[vel_ind], esamp[wid_ind], None)
                            lflux = gconvolve(line.vel_range, factor*esamp[amp_ind], esamp[vel_ind], esamp[wid_ind], lsf_width=lsf_width)
                        else:
                            # lflux = gaus(vel_range_fit, factor*esamp[amp_ind], esamp[vel_ind], esamp[wid_ind], None)
                            lflux = gconvolve(vel_range_fit, factor*esamp[amp_ind], esamp[vel_ind], esamp[wid_ind], lsf_width=lsf_width)
                        line_fluxes_samps.append(np.sum(lflux*deltafit))
                        if plot_emcee:
                            plt.plot(w_range_fit, lflux, line_dict[line.name.split('_')[0]][-1], lw=1, linestyle = '--',alpha=.5)

                if store_line_fluxes:
                    flux_dict[line.name+'_'+str(line.rest_wave).split('.')[0]] = np.sum(line_flux*deltafit)
                    if emcee_samples:
                        mcmc = np.nanpercentile(line_fluxes_samps, [16, 50, 84])
                        q = np.diff(mcmc)
                        val = mcmc[1]
                        err = np.average(q)
                        flux_emcee_dict[line.name+'_'+str(line.rest_wave).split('.')[0]] = [val,err]
                el_flux = el_flux + line_flux
                if plot:
                    plt.plot(w_range_fit, line_flux, line_dict[line.name.split('_')[0]][-1], lw=1, linestyle = '--')
                total_flux = total_flux + line_flux
        if plot:
            if '_b' in el or '_c' in el:
                label = ''
            else:
                label=el.split('_')[0]
            plt.plot(w_range_fit, el_flux, line_dict[el.split('_')[0]][-1], label = label, lw=3)
    if plot:
        if xlim:
            plt.xlim(xlim[0],xlim[1])
        else:
            plt.xlim(w_range[0],w_range[-1])
        if ylim:
            plt.ylim(ylim[0],ylim[1])

        plt.plot(w_range_fit, total_flux, 'r', lw=3, label = 'Total Model Flux', zorder=10)

        handles, labels = plt.gca().get_legend_handles_labels()
        if len(handles)==4:
            order = [3,1,0,2]
        else:
            order =[2,0,1]
        # print (labels)
        
        if legend:
            border = .3
            xpos = 7020
            if '2020hvf 240d' in text:
                ypos=.82
            else:
                ypos=.85
            if '2012dn' in text:
                xpos = 6995
                ypos = .87
            if '15pz' in text:
                xpos = 6995
                ypos = .85
            if '12cg' in text:
                xpos = 7375
                ypos = .71
            if '09dc' in text:
                xpos = 7020
                ypos = .87
            txt = plt.text(xpos, ypos, text, fontsize=25, horizontalalignment='left', color=txt_color)
            txt.set_path_effects([PathEffects.withStroke(linewidth=border, foreground='k')])
            if '2009dc' in text:
                plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], 
                    bbox_to_anchor=(.03, 0.43, 0.5, 0.5), fontsize = 20, frameon=False)
            elif '15pz' in text:
                plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], 
                    bbox_to_anchor=(0.01, 0.38, 0.5, 0.5), fontsize = 20, frameon=False)
            elif '12cg' in text:
                plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], 
                    bbox_to_anchor=(0.5, 0.38, 0.5, 0.5), fontsize = 20, frameon=False)
            else:
                plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], 
                        bbox_to_anchor=(0.03, 0.38, 0.5, 0.5), fontsize = 20, frameon=False)
        plt.ylabel('Relative Flux + Constant', fontsize=30)

        if plot_resid:
#            ax.set_xticklabels([])

            ax2 = fig.add_subplot(212, sharex=ax1)
            fig.subplots_adjust(wspace=0)

            plt.minorticks_on()
            plt.xticks(fontsize = 20)
            plt.yticks(fontsize = 20)
            plt.tick_params(
                which='major', 
                bottom='on', 
                top='on',
                left='on',
                right='on',
                direction='in',
                length=20)
            plt.tick_params(
                which='minor', 
                bottom='on', 
                top='on',
                left='on',
                right='on',
                direction='in',
                length=10)
            
            if fit_dw_plot:
                total_flux_interp = spectres(w_range, w_range_fit, total_flux, spec_errs=None, fill=None, verbose=True)[1]
            else:
                total_flux_interp = total_flux

            plt.plot(w_range, flux_subcont-total_flux_interp, drawstyle='steps-mid', color='grey', lw=2)
            # plt.plot(w_range, flux_subcont-total_flux, drawstyle='steps-mid', color='grey', lw=2)
            bin_size = 3.0
#            rebin_wave = block_reduce(np.array(w_range), bin_size)
            rebin_wave = block_reduce(w_range, bin_size)/bin_size
            rebin_resid = block_reduce(flux_subcont-total_flux_interp, bin_size)/bin_size
            plt.plot(rebin_wave, rebin_resid, drawstyle='steps-mid', color='k', lw=2)

            plt.axhline(y=0, color='r', linestyle='dashed')

            f_range = np.max(flux_subcont-total_flux_interp) - np.min(flux_subcont-total_flux_interp)
            l1 = 7254.0
            l2 = 7286.254
            i1 = (w_range > l1 - 10) & (w_range < l1 + 10)
            i2 = (w_range > l2 - 10) & (w_range < l2 + 10)
            i3 = (w_range > l1 - 10) & (w_range < l2 + 10)

            f1_max = np.max(flux_subcont[i1]-total_flux_interp[i1])
            f2_max = np.max(flux_subcont[i2]-total_flux_interp[i2])
            f3_max = np.max(flux_subcont[i3]-total_flux_interp[i3])

            min1 = f1_max + 0.05*f_range
            min2 = f2_max + 0.05*f_range
            max12 = np.max([f1_max*1.05, f2_max*1.05, f3_max]) + 0.05*f_range

            plt.plot((l1, l1), (min1, max12), c='magenta')
            plt.plot((l2, l2), (min2, max12), c='magenta')
            plt.plot((l1, l2), (max12, max12), c='magenta')

            plt.text((l1+l2)/2, max12*1.05, '[Ca II]', fontsize=18, ha='center')
            ymin, ymax = plt.ylim()
            plt.ylim(resid_ylim[0], resid_ylim[1])
            
            plt.xlabel('Rest Wavelength ($\mathrm{\AA}$)', fontsize=30)
            plt.ylabel('Residuals', fontsize=15)

            if plot_inset:
                # axins = zoomed_inset_axes(ax1, 2.3, loc=1, bbox_to_anchor=[760,830,5,1])
                axins = zoomed_inset_axes(ax1, 1.5, bbox_to_anchor=[740,740])
        else:
            plt.xlabel('Rest Wavelength ($\mathrm{\AA}$)', fontsize=30)
            if plot_inset:
                # axins = zoomed_inset_axes(ax1, 1.3, loc=1, bbox_to_anchor=[1050,650])
                axins = ax1.inset_axes([0.68, 0.65, 0.3, 0.3])
                axins.set_xticklabels([])

                axins2 = ax1.inset_axes([0.68, 0.45, 0.3, 0.2])
                if fit_dw_plot:
                    total_flux_interp = spectres(w_range, w_range_fit, total_flux, spec_errs=None, fill=None, verbose=True)[1]
                else:
                    total_flux_interp = total_flux

                axins2.plot(w_range, flux_subcont-total_flux_interp, drawstyle='steps-mid', color='grey', lw=1)
                bin_size = 3.0
                rebin_wave = block_reduce(w_range, bin_size)/bin_size
                rebin_resid = block_reduce(flux_subcont-total_flux_interp, bin_size)/bin_size
                axins2.plot(rebin_wave, rebin_resid, drawstyle='steps-mid', color='k', lw=1)

        if plot_inset:
            axins.plot(w_range, flux_subcont, drawstyle='steps-mid', color='k', lw=1)
            axins.plot(w_range_fit, total_flux, 'r', lw=2)
            axins.axvline(7254.0, ls = '--', color='magenta', zorder=-10)
            axins.axvline(7286.254, ls = '--', color='magenta', zorder=-10)
            axins2.axvline(7254.0, ls = '--', color='magenta', zorder=-10)
            axins2.axvline(7286.254, ls = '--', color='magenta', zorder=-10)

        plt.minorticks_on()
        plt.xticks(fontsize = 25)
        plt.yticks(fontsize = 25)
        plt.tick_params(
            which='major', 
            bottom='on', 
            top='on',
            left='on',
            right='on',
            direction='in',
            length=10)
        plt.tick_params(
            which='minor', 
            bottom='on', 
            top='on',
            left='on',
            right='on',
            direction='in',
            length=5)

        if plot_inset:
            # x1, x2, y1, y2 = 7248, 7310, 0.6, 0.95 #update to variables
            x1, x2, y1, y2 = 7241, 7339, 0.61, 0.94 #update to variables
            x1, x2, y1, y2 = 7241, 7310, 0.56, 0.94 #update to variables
            axins.set_xlim(x1, x2)
            axins2.set_xlim(x1, x2)
            axins.set_ylim(y1, y2)
            mark_inset(ax1, axins, loc1=2, loc2=3, fc="none", ec="0.5")

            # axins.set_xticks([7260, 7280, 7300], minor=False, direction='in')
            # axins2.set_xticks([7260, 7280, 7300], minor=False, direction='in')
            # axins.minorticks_on()
            # axins.xticks(fontsize = 12)
            # axins.yticks(fontsize = 12)
            axins.tick_params(
                which='major', 
                bottom='on', 
                top='on',
                left='on',
                right='on',
                direction='in',
                length=5)
            # axins.tick_params(
            #     which='minor', 
            #     bottom='on', 
            #     top='on',
            #     left='on',
            #     right='on',
            #     direction='in',
            #     length=5)
            # axins2.minorticks_on()
            # axins.xticks(fontsize = 12)
            # axins.yticks(fontsize = 12)
            axins2.tick_params(
                which='major', 
                bottom='on', 
                top='on',
                left='on',
                right='on',
                direction='in',
                length=5)
            # axins2.tick_params(
            #     which='minor', 
            #     bottom='on', 
            #     top='on',
            #     left='on',
            #     right='on',
            #     direction='in',
            #     length=5)

    
    #Matt: uncomment this code if you want to see the functions for every line in velocity space
    # for line in fit_lines:
    #     print (line.name, line.skew)
    #     plt.plot(line.vel_range, line.line_flux)
    #     plt.show()

    if savename:
        fig.savefig('/Users/msiebert/Documents//Research/UCSC/2020hvf_analysis/final_plots/'+savename+'.png',dpi = 300, bbox_inches = 'tight')
        fig.savefig('/Users/msiebert/Documents//Research/UCSC/2020hvf_analysis/final_plots/'+savename+'.pdf',dpi = 300, bbox_inches = 'tight')
    plt.show()
    if store_line_fluxes:
        if emcee_samples:
            return total_flux, flux_subcont, flux_dict, flux_emcee_dict, w_range_fit
        else:
            return total_flux, flux_subcont, flux_dict, w_range_fit
    else:
        return total_flux, flux_subcont, w_range_fit


def plot_fit(w_range, flux, fit_lines, flux_err = None, xlim = None, plot_err=False):
    plt.figure(figsize=[15,8])
    if xlim:
        roi = (w_range > xlim[0]) & (w_range < xlim[-1])
    else:
        roi = (w_range > w_range[0]) & (w_range < w_range[-1])

    plt.plot(w_range[roi], flux[roi], drawstyle='steps-mid', color='k', lw=2, zorder=-10)
    if plot_err:
        plt.fill_between(w_range, flux-flux_err, 
            flux+flux_err, color='black', alpha = 0.5)

    total_flux = np.zeros(len(w_range[roi]))
    unique_elements = []
    for line in fit_lines:
        unique_elements.append(line.name)
    unique_elements = set(unique_elements)
    for el in unique_elements:
        el_flux = np.zeros(len(w_range[roi]))
        for line in fit_lines:
            if line.name == el:
                el_flux = el_flux + line.line_flux[roi]
                plt.plot(w_range[roi], line.line_flux[roi], line_dict[line.name.split('_')[0]][-1], lw=1, linestyle = '--')
                plt.axvline(line.rest_wave, lw=2, linestyle='--', color=line_dict[line.name.split('_')[0]][-1])
                total_flux = total_flux + line.line_flux[roi]
        plt.plot(w_range[roi], el_flux, line_dict[el.split('_')[0]][-1], label = el, lw=3)
    plt.plot(w_range[roi], total_flux, 'r', lw=3, label = 'Total Model Flux', zorder=-9)
    plt.xlabel('Rest Wavelength ($\mathrm{\AA}$)', fontsize=15)
    plt.ylabel('Relative Flux + Constant', fontsize=15)
    plt.legend(bbox_to_anchor=(0.02, 0.48, 0.5, 0.5), fontsize = 10, frameon=False)
    if xlim:
        plt.xlim(xlim[0],xlim[-1])
    else:
        plt.xlim(w_range[0],w_range[-1])
#    plt.show()

def plot_fit_fancy(w_range, flux, fit_lines, emcee_samples=None, fit_param_indices = None, xlim = None, ylim = None, save=False, suf = ''):
    basic_format(size=[10,10])
    if xlim:
        roi = (w_range > xlim[0]) & (w_range < xlim[-1])
    else:
        roi = (w_range > w_range[0]) & (w_range < w_range[-1])

    plt.plot(w_range[roi], flux[roi], drawstyle='steps-mid', color='k', lw=2, zorder=-10,label= 'Gas-only')
    # if plot_ivar:
    #     plt.fill_between(w_range, flux_subcont-flux_error, 
    #         flux_subcont+flux_error, color='black', alpha = 0.5)

    total_flux = np.zeros(len(w_range[roi]))
    unique_elements = []
    for line in fit_lines:
        if line.rest_wave > xlim[0] and line.rest_wave < xlim[1]:
            unique_elements.append(line.name.split('_')[0])
    unique_elements = set(unique_elements)

    if emcee_samples:
        inds = list(np.random.randint(len(emcee_samples), size=50))
        emcee_samples_rand = []
        for i in inds:
            emcee_samples_rand.append(emcee_samples[i])


    for el in unique_elements:
        el_flux = np.zeros(len(w_range[roi]))
        for line in fit_lines:
            if line.name.split('_')[0] == el:
                el_flux = el_flux + line.line_flux[roi]
                # plt.plot(w_range[roi], line.line_flux[roi], line_dict[line.name.split('_')[0]][-1], lw=1, linestyle = '--')
                # plt.axvline(line.rest_wave, lw=2, linestyle='--', color=line_dict[line.name.split('_')[0]][-1])
                if emcee_samples:
                    line_fluxes_samps = []
                    for esamp in emcee_samples_rand:
                        if line.amp_ref is None:
                            amp_ind = fit_param_indices[line.name+'_amp']
                        else:
                            amp_ind = fit_param_indices[line.amp_ref+'_amp']
                        if line.vel_ref is None:
                            vel_ind = fit_param_indices[line.name+'_vel']
                        else:
                            vel_ind = fit_param_indices[line.vel_ref+'_vel']
                        if line.wid_ref is None:
                            wid_ind = fit_param_indices[line.name+'_width']
                        else:
                            wid_ind = fit_param_indices[line.wid_ref+'_width']
                        if line.parent_line != None:
                            factor = line.osc_strength/line.parent_line.osc_strength#determine strength relative to parent line
                        else:
                            factor=1.
                        if esamp[amp_ind] > 0:
                            # lflux = gaus(line.vel_range, factor*esamp[amp_ind], esamp[vel_ind], esamp[wid_ind], None)
                            lflux = gconvolve(line.vel_range, factor*esamp[amp_ind], esamp[vel_ind], esamp[wid_ind], lsf_width=lsf_width)

                            plt.plot(w_range, lflux, line_dict[line.name.split('_')[0]][-1], lw=1, linestyle = '--',alpha=.5)
                total_flux = total_flux + line.line_flux[roi]
        plt.plot(w_range[roi], el_flux, line_dict[el.split('_')[0]][-1], label = el, lw=3, alpha=.9)
    # plt.plot(w_range[roi], total_flux, 'r', lw=3, label = 'Total Model Flux', zorder=-9)
    plt.xlabel('Rest Wavelength ($\mathrm{\AA}$)', fontsize=25)
    plt.ylabel(r'$F_{\lambda}$ ($10^{-15}$ ergs cm$^{-2}$ s$^{-1}$ $\AA^{-1}$)', fontsize = 25)
    # plt.ylabel('Relative Flux', fontsize=25)
    plt.legend(loc=1, fontsize =20, frameon=False)
    if xlim:
        plt.xlim(xlim[0],xlim[-1])
    else:
        plt.xlim(w_range[0],w_range[-1])
    if ylim:
        plt.ylim(ylim[0],ylim[-1])

    if save:
        plt.savefig('/Users/msiebert/Documents/UCSC/Research/Foundation_Hosts/plots/neb_fit_'+suf+'.png',dpi = 300, bbox_inches = 'tight')
        plt.savefig('/Users/msiebert/Documents/UCSC/Research/Foundation_Hosts/plots/neb_fit_'+suf+'.pdf',dpi = 300, bbox_inches = 'tight')
    plt.show()


def set_bounds(fit_params, fit_param_indices, fit_cont, custom_bnds = None, amp_bnd = [0.,2.], vel_bnd = [-5000.,5000.], 
               width_bnd = [0.,5000.], wave_bnd = [-100,100], skew_bnd = [-15,15]):
    fit_bounds = [() for _ in range(len(fit_params))]
    for param in fit_param_indices:
        # print (param)
        if 'amp' in param:
            fit_bounds[fit_param_indices[param]] = (amp_bnd[0], amp_bnd[1])
#             fit_bounds[fit_param_indices[param]][1] = amp_bnd[1]
        elif 'vel' in param:
            fit_bounds[fit_param_indices[param]] = (vel_bnd[0], vel_bnd[1])
        elif 'width' in param:
            fit_bounds[fit_param_indices[param]] = (width_bnd[0], width_bnd[1])
        #Matt: we need default boundaries for skewness
        elif 'skew' in param:
            fit_bounds[fit_param_indices[param]] = (skew_bnd[0], skew_bnd[1])
        else:
            print (param)

    print (custom_bnds)
    if custom_bnds:
        for param in custom_bnds:
            if (param != 'wave_min') & (param != 'wave_max'):
                fit_bounds[fit_param_indices[param]] = (custom_bnds[param][0], custom_bnds[param][1])

    print (custom_bnds['wave_min'])
    if fit_cont:
        fit_bounds[-2] = custom_bnds['wave_min']
        fit_bounds[-1] = custom_bnds['wave_max']

    print(fit_bounds)
#        fit_bounds[-2] = (fit_params[-2] + wave_bnd[0], fit_params[-2] + wave_bnd[1])
#        fit_bounds[-1] = (fit_params[-1] + wave_bnd[0], fit_params[-1] + wave_bnd[1])
    return tuple(fit_bounds)

def set_bounds_gal(fit_params, fit_param_indices, fit_cont, custom_bnds = None, amp_bnd = [0.,2.], vel_bnd = [-1000.,1000.], 
               width_bnd = [0.,300.], wave_bnd = [-100,100], skew_bnd = [-15,15]):
    fit_bounds = [() for _ in range(len(fit_params))]
    for param in fit_param_indices:
        # print (param)
        if 'amp' in param:
            fit_bounds[fit_param_indices[param]] = (amp_bnd[0], amp_bnd[1])
#             fit_bounds[fit_param_indices[param]][1] = amp_bnd[1]
        elif 'vel' in param:
            fit_bounds[fit_param_indices[param]] = (vel_bnd[0], vel_bnd[1])
        elif 'width' in param:
            fit_bounds[fit_param_indices[param]] = (width_bnd[0], width_bnd[1])
        #Matt: we need default boundaries for skewness
        elif 'skew' in param:
            fit_bounds[fit_param_indices[param]] = (skew_bnd[0], skew_bnd[1])
        else:
            print (param)

    if custom_bnds:
        for param in custom_bnds:
            fit_bounds[fit_param_indices[param]] = (custom_bnds[param][0], custom_bnds[param][1])
    if fit_cont:
        fit_bounds[-2] = (fit_params[-2] + wave_bnd[0], fit_params[-2] + wave_bnd[1])
        fit_bounds[-1] = (fit_params[-1] + wave_bnd[0], fit_params[-1] + wave_bnd[1])
    return tuple(fit_bounds)

def fit_galaxy(neb_data, fit_range=[3700,6800]):
    fit_lines, fit_params, fit_param_indices = setup_gal_fit()
    set_all_line_params(fit_params, fit_lines, fit_param_indices)
#         gal_spec = np.genfromtxt(neb_file, unpack=True)
    gal_wave = neb_data[0]
    gal_flux = neb_data[1]
    gal_fluxerr = neb_data[2]

    wave_min = fit_range[0]
    wave_max = fit_range[1]
    fit_cont=False
    buff = 0
    roi = (gal_wave >= wave_min-buff) & (gal_wave <= wave_max+buff)
    w_range = gal_wave[roi]
    vel_ranges = calc_vel_ranges(w_range,fit_lines)

    print ('Fitting...')
    fit_bounds = set_bounds_gal(fit_params, fit_param_indices, fit_cont)
    scale_factor = (1./np.amax(gal_flux[roi]))
    fit_flux = gal_flux[roi]*scale_factor
    fit_err = gal_fluxerr[roi]*scale_factor
    optim = opt.minimize(min_func, fit_params, args = (w_range, fit_flux, fit_err, fit_lines, fit_param_indices, fit_cont, lsf_width), bounds=fit_bounds, 
                                                             method='L-BFGS-B', options={'ftol': 1e-7, 'eps': 1e-4, 'maxiter': 10000, 'disp': True})
    errors = np.diag(optim.hess_inv.todense())

    fit_params_solved = optim.x

    mgaus_solved = multiple_gaussians(fit_lines, fit_flux, w_range, fit_params_solved, fit_err, fit_cont, plot=True, plot_cont=True,
                                        plot_ivar=True, store_line_fluxes = True, savename=None)

    return mgaus_solved, w_range, fit_lines


def fit_lines(data, gal_fit=True, fit_setup = None, custom_bnds = None, fit_cont=False, s_f = True,lsf_width=2,
              width_bnd = [0.,5000.], wave_range = [3700,6800], mc_params = None, interp_dw = None, all_params = False, verbose=False):
    if gal_fit:
        # fit_lines, fit_params, fit_param_indices = setup_gal_fit_test()
        fit_lines, fit_params, fit_param_indices = setup_gal_fit(all_params = all_params, verbose=verbose)
        fit_bounds = set_bounds_gal(fit_params, fit_param_indices, fit_cont, width_bnd=width_bnd)
    else:
        fit_lines, fit_params, fit_param_indices = fit_setup
        fit_bounds = set_bounds(fit_params, fit_param_indices, fit_cont, custom_bnds = custom_bnds)

    set_all_line_params(fit_params, fit_lines, fit_param_indices)
    data_wave = data[0]
    data_flux = data[1]
    if gal_fit:
        print ('here')
        # data_fluxerr = np.ones(len(data_wave))*1.*np.std(data_flux) #quick fix since small err causing fit to fail
        # data_fluxerr = np.ones(len(data_wave))*1.stats.median_abs_deviation(data_flux)
        data_fluxerr = data[2]

    else:
        data_fluxerr = data[2]
    
    # data_fluxerr = data[2]
    wave_min = wave_range[0]
    wave_max = wave_range[1]
    buff = 0
    roi = (data_wave > wave_min-buff) & (data_wave < wave_max+buff)
    w_range = data_wave[roi]

    if s_f:
        scale_factor = (1./np.amax(data_flux[roi]))
    else:
        scale_factor = 1.

    fit_flux = data_flux[roi]*scale_factor
    fit_err = data_fluxerr[roi]*scale_factor

    if interp_dw:
        print ('Interpolating...')
        interp_wave = np.arange(math.ceil(w_range[0])+3.*interp_dw, math.floor(w_range[-1])-3.*interp_dw, dtype=float, step=interp_dw)
        # interp_wave = np.arange(math.ceil(w_range[0])+10.*(data_wave[1]-data_wave[1]), math.floor(w_range[-1])-10.*(data_wave[1]-data_wave[1]), dtype=float, step=interp_dw)
        binned_data = spectres(interp_wave, w_range, fit_flux, spec_errs=fit_err, fill=None, verbose=True)
        wave_interp = binned_data[0]
        fit_flux_interp = binned_data[1]
        fit_err_interp = binned_data[2]
        vel_ranges = calc_vel_ranges(wave_interp,fit_lines)
    else:
        wave_interp = w_range
        fit_flux_interp = fit_flux
        fit_err_interp = fit_err
        vel_ranges = calc_vel_ranges(wave_interp,fit_lines)

    print ('Fitting...')
    optim = opt.minimize(min_func, fit_params, args = (wave_interp, fit_flux_interp, fit_err_interp, fit_lines, fit_param_indices, fit_cont, lsf_width), bounds=fit_bounds,
                                                             method='L-BFGS-B', options={'ftol': 1e-7, 'eps': 1e-4, 'maxiter': 10000, 'disp': True})
    # optim = opt.minimize(min_func, fit_params, args = (wave_interp, fit_flux_interp, fit_err_interp, fit_lines, fit_param_indices, fit_cont), bounds=fit_bounds, 
    #                                                          method='L-BFGS-B')
    # print (optim)
    fit_params_solved = optim.x
    fit_hes_matrix = optim.hess_inv

    print (fit_params_solved)
    mgaus_solved = multiple_gaussians(fit_lines, fit_flux_interp, wave_interp, fit_params_solved, fit_err_interp, fit_cont, plot=True, plot_cont=True, 
                                        plot_ivar=True, store_line_fluxes = True, savename=None,lsf_width=lsf_width)

    if gal_fit:
        plot_fit(wave_interp, mgaus_solved[1], fit_lines, flux_err= fit_err_interp, xlim = [3650, 3800], plot_err=True)
        plot_fit(wave_interp, mgaus_solved[1], fit_lines, flux_err= fit_err_interp, xlim = [4750, 5100], plot_err=True)
        plot_fit(wave_interp, mgaus_solved[1], fit_lines, flux_err= fit_err_interp, xlim = [6500, 6800], plot_err=True)

    if mc_params:
        err_scale = stats.median_abs_deviation(fit_flux_interp)
        nwalkers, ndiscard, nchain = mc_params
        param_mc_scales = np.zeros(len(fit_params_solved))+1.
        for param in fit_param_indices.keys(): #might want to make these scales customizable
            if 'amp' in param:
                # param_mc_scales[fit_param_indices[param]] = err_scale 
                param_mc_scales[fit_param_indices[param]] = 0.05 #this could be better 
            if 'vel' in param:
                param_mc_scales[fit_param_indices[param]] = 5.
            if 'width' in param:
                param_mc_scales[fit_param_indices[param]] = 20.
        # pos = fit_params_solved + (fit_params_solved*1e-2 * np.random.randn(nwalkers, len(fit_params_solved)))
        pos = fit_params_solved + (param_mc_scales* np.random.randn(nwalkers, len(fit_params_solved)))

        for walker in pos:
            for param in fit_param_indices.keys(): #might want to make these scales customizable
                if 'amp' in param:
                    if walker[fit_param_indices[param]] < 0:
                        walker[fit_param_indices[param]] = 0.

        nwalkers, ndim = pos.shape
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(fit_bounds, wave_interp, fit_flux_interp, fit_err_interp, fit_lines, fit_param_indices, fit_cont))
        state = sampler.run_mcmc(pos, ndiscard, progress=True)
        # state = sampler.run_mcmc(pos, ndiscard, progress=True, skip_initial_state_check=True)
        sampler.reset()
        sampler.run_mcmc(state, nchain, progress=True)
        # sampler.run_mcmc(state, nchain, progress=True, skip_initial_state_check=True)
        return sampler, mgaus_solved, fit_params_solved, fit_lines, fit_params, fit_param_indices, fit_flux_interp, wave_interp, fit_err_interp, fit_cont, scale_factor, fit_hes_matrix
    else:
        return None, mgaus_solved, fit_params_solved, fit_lines, fit_params, fit_param_indices, fit_flux_interp, wave_interp, fit_err_interp, fit_cont, scale_factor, fit_hes_matrix


def lnlike(params, w_range, flux, flux_error, fit_lines, fit_param_indices, fit_cont):
    set_all_line_params(params, fit_lines, fit_param_indices)
    model_flux, flux_subcont, w_range_fit = multiple_gaussians(fit_lines, flux, w_range, params, flux_error, fit_cont)
    inv_sigma2 = 1.0/(flux_error**2)
    return -0.5*(np.sum((flux_subcont-model_flux)**2*inv_sigma2 - np.log(inv_sigma2)))

def lnprior(fit_params, fit_bounds):
    for i, fp in enumerate(fit_params):
        if fp > fit_bounds[i][0] and fp < fit_bounds[i][1]:
            pass
        else:
            return -np.inf
    return 0.0

def lnprob(fit_params, fit_bounds, w_range, flux, flux_error, fit_lines, fit_param_indices, fit_cont):
    lp = lnprior(fit_params, fit_bounds)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(fit_params, w_range, flux, flux_error, fit_lines, fit_param_indices, fit_cont)

def plot_chain(all_fit_data,fit_cont=False):
    sampler, mgaus_solved, fit_params_solved, fit_lines, fit_params, fit_param_indices, fit_flux_interp, wave_interp, fit_err_interp, fit_cont, scale_factor, fit_hes_matrix = all_fit_data
    fig, axes = plt.subplots(len(fit_params), figsize=(10, 30), sharex=True)
    samples = sampler.get_chain()
    #list(fit_param_indices.keys())[list(fit_param_indices.values()).index(i)]
    labels = []
    for i in range(len(fit_params)):
        if i in list(fit_param_indices.values()):
        # if list(fit_param_indices.values()).index(i) in list(fit_param_indices.keys()):
            labels.append(list(fit_param_indices.keys())[list(fit_param_indices.values()).index(i)])
        elif 'wloc1' in labels:
            labels.append('wloc2')
        else:
            labels.append('wloc1')
    for i in range(len(fit_params)):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3, )
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
    axes[-1].set_xlabel("step number")
    plt.show()
    return labels

def get_flat_samples(sampler, labels=None, plot=False):
    flat_samples = sampler.get_chain(discard=50, thin=15, flat=True)
    if plot:
        fig = corner.corner(
            flat_samples, labels=labels
        )
        plt.show()
    return flat_samples

def estimate_params(all_fit_data, flat_samples, burn_in=400):
    sampler, mgaus_solved, fit_params_solved, fit_lines, fit_params, fit_param_indices, fit_flux_interp, wave_interp, fit_err_interp, fit_cont, scale_factor, fit_hes_matrix = all_fit_data
    fit_params_solved_emcee = []
    fit_params_errs_emcee = []    
    for i in range(len(fit_params)):
        if i in list(fit_param_indices.values()):
            param = list(fit_param_indices.keys())[list(fit_param_indices.values()).index(i)] #might not be right every time
        elif param == 'wloc1':
            param = 'wloc2'
        else:
            param = 'wloc1'
        mcmc = np.percentile(flat_samples[burn_in:, i], [16, 50, 84])
        q = np.diff(mcmc)
        val = mcmc[1]
        err = np.average(q)
        # print (param, np.round(val,4), np.round(err,4))
        fit_params_solved_emcee.append(val)
        fit_params_errs_emcee.append(err)
    return fit_params_solved_emcee, fit_params_errs_emcee

def get_rand_samples(flat_samples, n=50):
    inds = np.random.randint(len(flat_samples), size=n)
    emcee_samples = []
    for ind in inds:
        sample = flat_samples[ind]
        emcee_samples.append(sample)
    return emcee_samples











