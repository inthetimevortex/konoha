'''
functions for color correction
'''

import os
import sys
import numpy as np
import atpy
import pyhdust.phc as phc
import pyhdust.beatlas as bat
from glob import glob
from scipy.interpolate import griddata
from konoha.constants import G, pi, mH, Msun, Rsun, Lsun, pc
from scipy import stats


def jy2cgs(flux, lbd, inverse=False):
    '''
    Converts from Jy units to erg/s/cm2/micron, and vice-versa

    [lbd] = micron

    Usage:
    flux_cgs = jy2cgs(flux, lbd, inverse=False)
    '''
    if not inverse:
        flux_new = 3e-9 * flux / lbd**2
    else:
        flux_new = lbd**2 * flux / 3e-9

    return flux_new






# COLOR CORRECTION
def color_corr(lbd, flux, flag):
    '''
    Computes color correction factors iteractively

    [lbd]=micron
    [flux]=erg/s/cm2/micron
    flag='akari' OR 'iras' OR 'wise'

    if 'wise':
        K = {int R lbd dlbd}/{int R g(lbd) lbd dlbd}
    else:
        K = {int R (lbd/lbdi)^-1 dlbd}/{int R g(lbd) dlbd}

    where
    g(lbd)=exp[Pn(ln lbd_i,ln f_i)]
    g(lbd_i)=flux_i
    norma = int R dlambda

    Usage:
    flux_corr = color_corr(lbd, flux, flag):
    '''
    from goodboi import integral, poly_interp

    # make sure input are numpy arrays
    lbd = np.array([lbd]).reshape((-1))
    flux = np.array([flux]).reshape((-1))

    # path to bandpasses transmission files
    dir0 = 'defs/'

    # either select filter files, or apply default correction
    # for single fluxes
    if (flag.lower() == 'akari') or (flag.lower() == 'akari/irc'):
        if len(lbd) == 1:
            print('Just one data point: assuming slope=3')
            K = np.array([1.096, 0.961])
            lbdi = np.array([9., 18.])
            flux_corr = flux / K[np.where(np.abs(lbd - lbdi) \
                        == np.min(np.abs(lbd - lbdi)))]
            return flux_corr
        else:
            band = [dir0 + 'bandpasses/irc-s9w.dat', \
                    dir0 + 'bandpasses/irc-l18w.dat']
    elif flag.lower() == 'iras':
        if len(lbd) == 1:
            print('Just one data point: assuming slope=3')
            K = np.array([1.25, 1.23, 1.15, 1.04])
            lbdi = np.array([12., 25., 60., 100.])
            flux_corr = flux / K[np.where(np.abs(lbd - lbdi) \
                        == np.min(np.abs(lbd - lbdi)))]
            return flux_corr
        else:
            band = []
            lbd_arr = np.array(['12', '25', '60', '100'])
            delt = 9.
            for i in range(len(lbd_arr)):
                if (np.abs(lbd - np.float(lbd_arr[i])) < delt).any():
                    band.append(dir0 + 'bandpasses/IRAS_IRAS.' \
                                + lbd_arr[i] + 'mu.dat')
    elif flag.lower() == 'wise':
        if len(lbd) == 1:
            print('Just one data point: assuming slope=3')
            K = np.array([0.9961, 0.9976, 0.9393, 0.9934])
            lbdi = np.array([3.3526, 4.6028, 11.5608, 22.0883])
            flux_corr = flux / K[np.where(np.abs(lbd - lbdi) \
                        == np.min(np.abs(lbd - lbdi)))]
            return flux_corr
        else:
            band = []
            lbdi = np.array([3.3526, 4.6028, 11.5608, 22.0883])
            delt = 2.
            for i in range(4):
                if (np.abs(lbd - lbdi[i]) < delt).any():
                    band.append(dir0 + 'bandpasses/RSR-W' \
                                + '{:1d}'.format(i + 1) + '.txt')
    else:
        print('unknown input flag')
        flux_corr = np.zeros(len(flux))
        return flux_corr

    # iterative color-correction
    nband = len(band)
    K = np.ones(nband)
    K1 = np.zeros(nband)
    delt = 1e-5
    res = 1.
    while res > delt:
        for iband in range(nband):
            tab = np.loadtxt(band[iband])
            lbdi, R = tab[:, 0], tab[:, 1]
            logG = poly_interp(np.log(lbd), np.log((flux/K) \
                   / (flux[iband] / K[iband])), np.log(lbdi))
            G = np.exp(logG)
            if flag.lower() == 'wise':
                K1[iband] = integral(lbdi, R * G * lbdi) \
                            / integral(lbdi, R * lbdi)
            else:
                K1[iband] = integral(lbdi, R * G) \
                            / integral(lbdi, R * (lbd[iband] / lbdi))

        res = np.sum(np.abs(K - K1) / K1)
        K = K1

    flux_corr = flux / K

    return flux_corr
    
    
    


# VOSA TO CATALOGUE VALUES
def vosa2catvalues(lbd, flux, flag):
    '''
    Converts fluxes back to the catalog values,
    i.e., at the formal nominal wavelengths

    currently available flags: 'akari', 'iras', 'wise'

    Usage:
    lbd_cat, flux_cat = vosa2catvalues(lbd_vosa, flux_vosa, flag)
    '''

    # make sure everybody is a numpy array
    lbd = np.array([lbd]).reshape((-1))
    flux = np.array([flux]).reshape((-1))

    # definitions
    lcat = np.zeros(len(lbd))
    fcat = np.zeros(len(flux))

    # wavelength tolerance range
    delt = 1. # wavelength tolerance range

    # conversion
    if (flag.lower() == 'akari') or (flag.lower() == 'akari/irc'):
        i09 = np.abs(lbd - 8.22) < delt
        i18 = np.abs(lbd - 17.61)  < delt
        if i09.any():
            lcat[i09] = 9.
            fcat[i09] = flux[i09] * (lbd[i09] / 9.)**2
        if i18.any():
            lcat[i18] = 18.
            fcat[i18] = flux[i18] * (lbd[i18] / 18.)**2
    elif flag.lower() == 'akari/fis':
        i65 = np.abs(lbd - 62.95) < delt
        i90 = np.abs(lbd - 76.90)  < delt
        i140 = np.abs(lbd - 140.86)  < delt
        i160 = np.abs(lbd - 159.47)  < delt
        if i65.any():
            lcat[i65] = 65.
            fcat[i65] = flux[i65] * (lbd[i65] / 65.)**2
        if i90.any():
            lcat[i90] = 90.
            fcat[i90] = flux[i90] * (lbd[i90] / 90.)**2
        if i140.any():
            lcat[i140] = 140.
            fcat[i140] = flux[i140] * (lbd[i140] / 140.)**2
        if i160.any():
            lcat[i160] = 160.
            fcat[i160] = flux[i160] * (lbd[i160] / 160.)**2
    elif flag.lower() == 'iras':
        i12 = np.abs(lbd - 10.15) < delt
        i25 = np.abs(lbd - 21.73) < delt
        i60 = np.abs(lbd - 51.99) < delt
        i100= np.abs(lbd - 95.30) < delt
        if i12.any():
            lcat[i12] = 12.
            fcat[i12] = flux[i12] * (lbd[i12] / 12.)**2
        if i25.any():
            lcat[i25] = 25.
            fcat[i25] = flux[i25] * (lbd[i25] / 25.)**2
        if i60.any():
            lcat[i60] = 60.
            fcat[i60] = flux[i60] * (lbd[i60] / 60.)**2
        if i100.any():
            lcat[i100] = 100.
            fcat[i100] = flux[i100] * (lbd[i100] / 100.)**2
    elif flag.lower() == 'wise':
        lcat = lbd
        fcat = flux
    else:
        print('unknown input flag')
        lcat, fcat = lbd, flux

    return lcat, fcat
    
