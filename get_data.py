#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  get_data.py
#
#  Copyright 2020 Amanda Rubio <amanda.rubio@usp.br>
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#
#
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from astropy.io import fits as fits
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import SkyCoord, EarthLocation
from collections import OrderedDict
import pyhdust.spectools as spt
from PyAstronomy import pyasl
from konoha import linesDict, constants
import seaborn as sns
import socket
from konoha.utils import bin_data

sns.set_style("white", {"xtick.major.direction": "in", "ytick.major.direction": "in"})


PCname = socket.gethostname()
if PCname == "Pakkun":
    direc = "/Users/amanda/Drive/"
elif PCname == "BISUKE.local":
    direc = "/Volumes/GoogleDrive/Meu Drive/"
else:
    direc = "/home/amanda/"


def get_iue():
    """
    Get IUE data, at input wavelength grid

    Usage:
    lbd, flux, dflux = get_data(lbd_grid=None, stddev=False)

    if stddev=True, errors are computed from stddev around power-law fit
    """
    flist = glob("/home/amanda/Dropbox/Amanda/Data/IUE-INES/*")
    lbd_iue = np.array([])
    flux_iue = np.array([])
    dflux_iue = np.array([])
    for i in range(len(flist)):
        fname = flist[i]
        hdr_list = fits.open(fname)
        fits_data = hdr_list[1].data
        fits_header = hdr_list[0].header
        lbd_iue = np.hstack([lbd_iue, 1e-4 * fits_data["WAVELENGTH"]])
        flux_iue = np.hstack([flux_iue, 1e4 * fits_data["FLUX"]])
        dflux_iue = np.hstack([dflux_iue, 1e4 * fits_data["SIGMA"]])
    ordem = lbd_iue.argsort()
    lbd_iue = lbd_iue[ordem]
    flux_iue = flux_iue[ordem]
    dflux_iue = dflux_iue[ordem]
    keep = flux_iue > 0.0
    lbd_iue = lbd_iue[keep]
    flux_iue = flux_iue[keep]
    dflux_iue = dflux_iue[keep]

    nbins = 200
    xbin, ybin, dybin = bin_data(lbd_iue, flux_iue, nbins, exclude_empty=True)

    return lbd_iue, flux_iue, dflux_iue


def get_lines(line):

    """
    get_halpha, but more general
    """

    flux_all = []
    vel_all = []
    MJD_all = []
    flag_all = []
    ra = 84.9122543
    dec = -34.07410972

    if line == "Ha":
        USE = [
            "ESPaDOnS",
            "BeSS",
            "BeSOS",
            "UVES",
            "FEROS",
            "OPD - Musicos",
            "OPD - Ecass",
            "NRES",
        ]
    elif line == "Hb":
        USE = ["ESPaDOnS", "BeSOS", "FEROS", "NRES"]
    else:
        USE = ["ESPaDOnS", "BeSOS", "FEROS", "NRES"]

    lbd0 = linesDict.line_names[line][1]

    # plot ESPaDOnS
    flag = "ESPaDOnS"
    if flag in USE:
        lines = glob(direc + "Dropbox/Amanda/Data/ESPaDOnS/new/*i.fits.gz")
        MJD = np.zeros([len(lines)])
        JD = np.zeros([len(lines)])
        ####
        # FROM THE HEADER
        ###COMMENT Correcting wavelength scale from Earth motion...
        ###COMMENT Coordinates of object : 5:39:38.94 & -34: 4:26.9
        ###COMMENT Time of observations : 2011 11 9 @ UT 13:34:33
        ###COMMENT  (hour angle = 0.775 hr, airmass = 1.742 )
        ###COMMENT Total exposure time : 25.0 s
        ###COMMENT Cosine latitude of observatory : 0.941
        ###COMMENT Heliocentric velocity of observer towards star : 9.114 km/s

        for n in range(len(lines)):
            fname = lines[n]
            # read fits
            hdr_list = fits.open(fname)
            fits_data = hdr_list[0].data
            fits_header = hdr_list[0].header
            # f = open('{0}_{1}.txt'.format(flag, n), 'wb')
            # fits_header.totextfile('{0}_{1}.txt'.format(flag, n))
            # read MJD
            MJD[n] = fits_header["MJDATE"]
            #
            # lat = fits_header['LATITUDE']
            # lon = fits_header['LONGITUD']
            lbd = fits_data[0, :]
            ordem = lbd.argsort()
            lbd = lbd[ordem]
            flux_norm = fits_data[1, ordem]
            vel, flux = spt.lineProf(lbd, flux_norm, lbc=lbd0)
            # cut = asas(vel, flux, line)
            # cut_all.append(cut)
            vel_all.append(vel)
            # corr_all.append(corr)
            flux_all.append(flux)
            MJD_all.append(MJD[n])
            flag_all.append(flag)

    ## plot BeSS
    flag = "BeSS"
    if flag in USE:
        lines = glob(direc + "Dropbox/Amanda/Data/BeSS/new/*fits")
        MJD = np.zeros([len(lines)])
        JD = np.zeros([len(lines)])

        # FROM HEADER
        #           BSS_VHEL shows the applied correction in km/s. If BSS_VHEL=0,
        # COMMENT   no correction has been applied. The required correction given
        # COMMENT   in BSS_RQVH (in km/s) is an escape velocity (redshift). To apply
        # COMMENT   it within Iraf, the keyword redshift must be set to -BSS_RQVH
        # COMMENT   and isvelocity must be set to "yes" in the dopcor task.

        for n in range(len(lines)):
            fname = lines[n]
            # read fits
            hdr_list = fits.open(fname)
            fits_data = hdr_list[0].data
            fits_header = hdr_list[0].header
            # fits_header.totextfile('{0}_{1}.txt'.format(flag, n))
            # read MJD
            MJD[n] = fits_header["MID-HJD"]  # HJD at mid-exposure
            t = Time(MJD[n], format="jd", scale="utc")
            MJD[n] = t.mjd
            # lat = fits_header['BSS_LAT']
            # lon = fits_header['BSS_LONG']
            # elev = fits_header['BSS_ELEV']
            corr = -fits_header["BSS_RQVH"]
            lbd = fits_header["CRVAL1"] + fits_header["CDELT1"] * np.arange(
                len(fits_data)
            )
            vel, flux = spt.lineProf(lbd, fits_data, lbc=lbd0 * 10)
            vel = vel + corr
            # cut = asas(vel, flux, line)
            # cut_all.append(cut)
            vel_all.append(vel)
            # corr_all.append(corr)
            flux_all.append(flux)
            MJD_all.append(MJD[n])
            flag_all.append(flag)

    # plot MUSICOS
    flag = "OPD - Musicos"
    if flag in USE:
        lines = glob(
            direc + "Dropbox/Amanda/Data/MUSICOS/andre/spec_*/acol/*halpha.fits"
        )
        MJD = np.zeros([len(lines)])
        #######################################
        # Andre disse q estao corrigidos
        #######################################
        for n in range(len(lines)):
            fname = lines[n]
            # read fits
            hdr_list = fits.open(fname)
            fits_data = hdr_list[0].data
            fits_header = hdr_list[0].header
            # fits_header.totextfile('{0}_{1}.txt'.format(flag, n))
            # read MJD
            MJD[n] = fits_header["JD"]  # JD
            t = Time(MJD[n], format="jd", scale="utc")
            MJD[n] = t.mjd
            # corr = fits_header['VHELIO']
            lbd = fits_header["CRVAL1"] + fits_header["CDELT1"] * np.arange(
                len(fits_data)
            )
            vel, flux = spt.lineProf(lbd, fits_data, lbc=lbd0 * 10)
            vel_all.append(vel)
            # cut = asas(vel, flux, line)
            # cut_all.append(cut)
            # corr_all.append(corr)
            flux_all.append(flux)
            MJD_all.append(MJD[n])
            flag_all.append(flag)

    ## plot Moser
    flag = "OPD - Ecass"
    if flag in USE:
        lines = glob(direc + "Dropbox/Amanda/Data/ecass_musicos/data/alpCol*")
        MJD = np.zeros([len(lines)])

        ## NO INFO ON HEADER!
        # Latitude: 22° 32' 04" S	Longitude: 45° 34' 57" W
        lat = -22.53444
        lon = -45.5825
        alt = 1864.0
        for n in range(len(lines)):
            fname = lines[n]
            # read fits
            hdr_list = fits.open(fname)
            fits_data = hdr_list[0].data
            fits_header = hdr_list[0].header
            # fits_header.totextfile('{0}_{1}.txt'.format(flag, n))
            # read MJD
            MJD[n] = fits_header["MJD"]  # JD
            # t = Time(MJD[n], format = 'jd', scale='utc')
            # MJD[n] = t.mjd
            t = Time(MJD[n], format="mjd", scale="utc")
            JD[n] = t.jd
            corr, hjd = pyasl.helcorr(lon, lat, alt, ra, dec, JD[n])
            # corr = 0.
            lbd = fits_header["CRVAL1"] + fits_header["CDELT1"] * np.arange(
                len(fits_data)
            )
            vel, flux = spt.lineProf(lbd, fits_data, lbc=lbd0 * 10)
            vel = vel + corr
            vel_all.append(vel)
            # cut = asas(vel, flux, line)
            # cut_all.append(cut)
            # corr_all.append(corr)
            flux_all.append(flux)
            MJD_all.append(MJD[n])
            flag_all.append(flag)

    # plot UVES
    flag = "UVES"
    if flag in USE:
        lines = glob(direc + "Dropbox/Amanda/Data/UVES/new/*.fits")
        MJD = np.zeros([len(lines)])
        JD = np.zeros([len(lines)])

        # lat = -29.257778
        # lon = -70.736667

        # FROM HEADER
        ###################################################################################
        # HIERARCH ESO QC VRAD BARYCOR = -8.401681 / Barycentric radial velocity correctio
        # HIERARCH ESO QC VRAD HELICOR = -8.398018 / Heliocentric radial velocity correcti
        ###################################################################################

        for n in range(len(lines)):
            fname = lines[n]
            # read fits
            hdulist = fits.open(fname)
            # print column information
            # hdulist[1].columns
            # get to the data part (in extension 1)
            scidata = hdulist[1].data
            wave = scidata[0][0]
            arr1 = scidata[0][1]
            arr2 = scidata[0][2]
            fits_header = hdulist[0].header
            # fits_header.totextfile('{0}_{1}.txt'.format(flag, n))
            MJD[n] = fits_header["MJD-OBS"]
            vel, flux = spt.lineProf(wave, arr1, lbc=lbd0 * 10)
            # vel = vel + corr
            # cut = asas(vel, flux, line)
            # cut_all.append(cut)
            vel_all.append(vel)
            # corr_all.append(corr)
            flux_all.append(flux)
            MJD_all.append(MJD[n])
            flag_all.append(flag)

    ##plot BeSOS
    flag = "BeSOS"
    if flag in USE:
        lines = glob(direc + "Dropbox/Amanda/Data/BeSOS/2018/*.fits")
        MJD = np.zeros([len(lines)])
        JD = np.zeros([len(lines)])

        # FROM WEBSITE
        # All the spectra available are reduced and corrected with the heliocentric velocity

        for n in range(len(lines)):
            fname = lines[n]
            # read fits
            hdr_list = fits.open(fname)
            fits_data = hdr_list[0].data
            fits_header = hdr_list[0].header
            # fits_header.totextfile('{0}_{1}.txt'.format(flag, n))
            # read MJD
            MJD[n] = fits_header["MJD"]
            # corr = 0.
            # t = Time(MJD[n], format = 'mjd', scale='utc')
            # JD[n] = t.jd
            # corr, hjd = pyasl.helcorr(lon, lat, 2400., ra, dec, JD[n])
            lbd = fits_header["CRVAL1"] + fits_header["CDELT1"] * np.arange(
                len(fits_data)
            )
            vel, flux = spt.lineProf(lbd, fits_data, lbc=lbd0 * 10)
            # corr = fits_header['BVEL']
            # vel = vel + corr
            # cut = asas(vel, flux, line)
            # cut_all.append(cut)
            vel_all.append(vel)
            # corr_all.append(corr)
            flux_all.append(flux)
            MJD_all.append(MJD[n])
            flag_all.append(flag)

    # plot prof_nelson
    flag = "FEROS"
    if flag in USE:
        lines = glob(direc + "Dropbox/Amanda/Data/prof_nelson/*/*")
        MJD = np.zeros([len(lines)])
        JD = np.zeros([len(lines)])

        # lat = -29.257778
        # lon = -70.736667

        # FROM HEADER
        ###################################################################
        # HISTORY  'BARY_CORR'      ,'R*4 '   ,    1,    1,'5E14.7',' ',' '
        # HISTORY  -1.5192770E+01
        ###################################################################

        for n in range(len(lines)):
            fname = lines[n]
            # read fits
            hdr_list = fits.open(fname)
            fits_data = hdr_list[0].data
            fits_header = hdr_list[0].header
            # fits_header.totextfile('{0}_{1}.txt'.format(flag, n))
            # read MJD
            MJD[n] = fits_header["MJD-OBS"]
            t = Time(MJD[n], format="mjd", scale="utc")
            JD[n] = t.jd
            # corr = 0.
            lbd = fits_header["CRVAL1"] + fits_header["CDELT1"] * np.arange(
                len(fits_data)
            )
            vel, flux = spt.lineProf(lbd, fits_data, lbc=lbd0 * 10)
            # vel = vel + corr
            vel_all.append(vel)
            # corr_all.append(corr)
            flux_all.append(flux)
            MJD_all.append(MJD[n])
            flag_all.append(flag)

    flag = "NRES"
    if flag in USE:
        lines = glob(direc + "Dropbox/Amanda/Data/NRES/*fits*")
        MJD = np.zeros([len(lines)])
        JD = np.zeros([len(lines)])

        # lat = -29.257778
        # lon = -70.736667
        barycorr_list = []
        # FROM HEADER
        ###################################################################
        # HISTORY  'BARY_CORR'      ,'R*4 '   ,    1,    1,'5E14.7',' ',' '
        # HISTORY  -1.5192770E+01
        ###################################################################
        order = linesDict.line_names[line][0]
        wl_list = []
        flx_list = []
        BJD_list = []
        for n in range(len(lines)):
            fname = lines[n]

            hdulist = fits.open(fname)
            headers = hdulist[0].header
            BJD_mid_exp = headers["BJD"]
            # t = Time(BJD_mid_exp, format = 'jd', scale='utc')
            # MJD[n] = t.mjd
            DAY_OBS = headers["DATE-OBS"]
            SITE = headers["SITE"]
            LONG1 = headers["LONG1"]
            LAT1 = headers["LAT1"]
            HT1 = headers["HT1"]
            SpecRaw = hdulist[1]
            SpecFlat = hdulist[2]
            SpecBlaze = hdulist[3]
            ThArRaw = hdulist[4]
            ThArFlat = hdulist[5]
            WaveSpec = hdulist[6]
            WaveThAr = hdulist[7]
            SpecXcor = hdulist[8]
            RVBlockFit = hdulist[9]

            wl_list.append(WaveSpec)
            flx_list.append(SpecFlat)
            # flx_list.append(SpecBlaze)
            BJD_list.append(BJD_mid_exp)

            obs_loc = EarthLocation.from_geodetic(
                lat=LAT1 * u.deg, lon=LONG1 * u.deg, height=HT1 * u.m
            )
            sc = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
        wl_list = np.array(wl_list)[np.argsort(BJD_list)]
        flx_list = np.array(flx_list)[np.argsort(BJD_list)]
        BJD_list = np.array(BJD_list)[np.argsort(BJD_list)]
        for i, x in enumerate(wl_list):
            vel, flux = spt.lineProf(
                wl_list[i].data[order], flx_list[i].data[order], lbc=lbd0
            )
            # vel = vel + corr
            # vel_all.append(vel)
            # corr_all.append(corr)
            flux_all.append(flux)
            t = Time(BJD_list[i], format="jd", scale="utc")
            MJD[n] = t.mjd
            MJD_all.append(MJD[n])

            flag_all.append(flag)
            # barycentric correction is more precise, but might be more difficult to apply
            barycorr = sc.radial_velocity_correction(
                kind="barycentric",
                obstime=Time(BJD_list[i], format="jd"),
                location=obs_loc,
            )
            barycorr = barycorr.value / 1000.0
            barycorr_list.append(barycorr)

            # heliocentric correction is easier, but less precise at a level of like 10 m/s
            # heliocorr = sc.radial_velocity_correction(kind='heliocentric',obstime=Time(BJD_list[i], format='jd'), location=obs_loc)
            # helcorr_list.append(heliocorr.value* au.value / (60*60*24)/ 1000.0)
            vl = vel + barycorr + vel * barycorr / (c.value / 1000.0)
            vel_all.append(vl)

    return MJD_all, vel_all, flux_all, flag_all


def get_halpha():
    flux_all = []
    vel_all = []
    MJD_all = []
    flag_all = []
    corr_all = []
    ra = 84.9122543
    dec = -34.07410972

    line = "Ha"

    USE = [
        "ESPaDOnS",
        "BeSS",
        "BeSOS",
        "UVES",
        "FEROS",
        "OPD - Musicos",
        "OPD - Ecass",
        "NRES",
    ]
    # USE = ['NRES']

    lbd0 = 656.28

    # plot ESPaDOnS
    flag = "ESPaDOnS"
    if flag in USE:
        lines = glob(direc + "Dropbox/Amanda/Data/ESPaDOnS/new/*i.fits.gz")
        MJD = np.zeros([len(lines)])
        JD = np.zeros([len(lines)])
        ####
        # FROM THE HEADER
        ###COMMENT Correcting wavelength scale from Earth motion...
        ###COMMENT Coordinates of object : 5:39:38.94 & -34: 4:26.9
        ###COMMENT Time of observations : 2011 11 9 @ UT 13:34:33
        ###COMMENT  (hour angle = 0.775 hr, airmass = 1.742 )
        ###COMMENT Total exposure time : 25.0 s
        ###COMMENT Cosine latitude of observatory : 0.941
        ###COMMENT Heliocentric velocity of observer towards star : 9.114 km/s

        for n in range(len(lines)):
            fname = lines[n]
            # read fits
            hdr_list = fits.open(fname)
            fits_data = hdr_list[0].data
            fits_header = hdr_list[0].header
            hdr_list.close()

            # f = open('{0}_{1}.txt'.format(flag, n), 'wb')
            # fits_header.totextfile('{0}_{1}.txt'.format(flag, n))
            # read MJD
            MJD[n] = fits_header["MJDATE"]

            lat = fits_header["LATITUDE"]
            lon = fits_header["LONGITUD"]
            ht = 4200  # m
            obs_loc = EarthLocation.from_geodetic(
                lat=lat * u.deg, lon=lon * u.deg, height=ht * u.m
            )
            sc = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
            lbd = fits_data[0, :]
            ordem = lbd.argsort()
            lbd = lbd[ordem]
            flux_norm = fits_data[1, ordem]
            vel, flux = spt.lineProf(lbd, flux_norm, lbc=lbd0)
            barycorr = sc.radial_velocity_correction(
                kind="barycentric", obstime=Time(MJD[n], format="mjd"), location=obs_loc
            )
            barycorr = barycorr.value / 1000.0
            # print(barycorr)
            vel_all.append(vel)
            corr_all.append(barycorr)
            flux_all.append(flux)
            MJD_all.append(MJD[n])
            flag_all.append(flag)

    ## plot BeSS
    flag = "BeSS"
    if flag in USE:
        lines = glob(direc + "Dropbox/Amanda/Data/BeSS/new/*fits")
        MJD = np.zeros([len(lines)])
        JD = np.zeros([len(lines)])

        # FROM HEADER
        #           BSS_VHEL shows the applied correction in km/s. If BSS_VHEL=0,
        # COMMENT   no correction has been applied. The required correction given
        # COMMENT   in BSS_RQVH (in km/s) is an escape velocity (redshift). To apply
        # COMMENT   it within Iraf, the keyword redshift must be set to -BSS_RQVH
        # COMMENT   and isvelocity must be set to "yes" in the dopcor task.

        for n in range(len(lines)):
            fname = lines[n]
            # read fits
            hdr_list = fits.open(fname)
            fits_data = hdr_list[0].data
            fits_header = hdr_list[0].header
            hdr_list.close()

            # fits_header.totextfile('{0}_{1}.txt'.format(flag, n))
            # read MJD
            MJD[n] = fits_header["MID-HJD"]  # HJD at mid-exposure
            t = Time(MJD[n], format="jd", scale="utc")
            MJD[n] = t.mjd
            # lat = fits_header['BSS_LAT']
            # lon = fits_header['BSS_LONG']
            # elev = fits_header['BSS_ELEV']
            corr = -fits_header["BSS_RQVH"]
            lbd = fits_header["CRVAL1"] + fits_header["CDELT1"] * np.arange(
                len(fits_data)
            )
            vel, flux = spt.lineProf(lbd, fits_data, lbc=lbd0 * 10)
            vel = vel + corr
            # cut = asas(vel, flux, line)
            # cut_all.append(cut)
            vel_all.append(vel)
            # corr_all.append(corr)
            flux_all.append(flux)
            MJD_all.append(MJD[n])
            flag_all.append(flag)

    # plot MUSICOS
    flag = "OPD - Musicos"
    if flag in USE:
        lines = glob(
            direc + "Dropbox/Amanda/Data/MUSICOS/andre/spec_*/acol/*halpha.fits"
        )
        MJD = np.zeros([len(lines)])
        #######################################
        # Andre disse q estao corrigidos
        #######################################
        for n in range(len(lines)):
            fname = lines[n]
            # read fits
            hdr_list = fits.open(fname)
            fits_data = hdr_list[0].data
            fits_header = hdr_list[0].header
            hdr_list.close()

            # fits_header.totextfile('{0}_{1}.txt'.format(flag, n))
            # read MJD
            MJD[n] = fits_header["JD"]  # JD
            t = Time(MJD[n], format="jd", scale="utc")
            MJD[n] = t.mjd
            # corr = fits_header['VHELIO']
            lbd = fits_header["CRVAL1"] + fits_header["CDELT1"] * np.arange(
                len(fits_data)
            )
            vel, flux = spt.lineProf(lbd, fits_data, lbc=lbd0 * 10)
            vel_all.append(vel)
            # cut = asas(vel, flux, line)
            # cut_all.append(cut)
            # corr_all.append(corr)
            flux_all.append(flux)
            MJD_all.append(MJD[n])
            flag_all.append(flag)

    ## plot Moser
    flag = "OPD - Ecass"
    if flag in USE:
        lines = glob(direc + "Dropbox/Amanda/Data/ecass_musicos/data/alpCol*")
        MJD = np.zeros([len(lines)])

        ## NO INFO ON HEADER!
        # Latitude: 22° 32' 04" S	Longitude: 45° 34' 57" W
        lat = -22.53444
        lon = -45.5825
        alt = 1864.0
        for n in range(len(lines)):
            fname = lines[n]
            # read fits
            hdr_list = fits.open(fname)
            fits_data = hdr_list[0].data
            fits_header = hdr_list[0].header
            hdr_list.close()

            # fits_header.totextfile('{0}_{1}.txt'.format(flag, n))
            # read MJD
            MJD[n] = fits_header["MJD"]  # JD
            # t = Time(MJD[n], format = 'jd', scale='utc')
            # MJD[n] = t.mjd
            t = Time(MJD[n], format="mjd", scale="utc")
            JD[n] = t.jd
            corr, hjd = pyasl.helcorr(lon, lat, alt, ra, dec, JD[n])
            # corr = 0.
            lbd = fits_header["CRVAL1"] + fits_header["CDELT1"] * np.arange(
                len(fits_data)
            )
            vel, flux = spt.lineProf(lbd, fits_data, lbc=lbd0 * 10)
            vel = vel + corr
            vel_all.append(vel)
            # cut = asas(vel, flux, line)
            # cut_all.append(cut)
            # corr_all.append(corr)
            flux_all.append(flux)
            MJD_all.append(MJD[n])
            flag_all.append(flag)

    # plot UVES
    flag = "UVES"
    if flag in USE:
        lines = glob(direc + "Dropbox/Amanda/Data/UVES/new/*.fits")
        MJD = np.zeros([len(lines)])
        JD = np.zeros([len(lines)])

        # lat = -29.257778
        # lon = -70.736667

        # FROM HEADER
        ###################################################################################
        # HIERARCH ESO QC VRAD BARYCOR = -8.401681 / Barycentric radial velocity correctio
        # HIERARCH ESO QC VRAD HELICOR = -8.398018 / Heliocentric radial velocity correcti
        ###################################################################################

        for n in range(len(lines)):
            fname = lines[n]
            # read fits
            hdulist = fits.open(fname)
            # print column information
            # hdulist[1].columns
            # get to the data part (in extension 1)
            scidata = hdulist[1].data
            wave = scidata[0][0]
            arr1 = scidata[0][1]
            arr2 = scidata[0][2]
            fits_header = hdulist[0].header
            hdulist.close()

            # fits_header.totextfile('{0}_{1}.txt'.format(flag, n))
            MJD[n] = fits_header["MJD-OBS"]
            vel, flux = spt.lineProf(wave, arr1, lbc=lbd0 * 10)
            # vel = vel + corr
            # cut = asas(vel, flux, line)
            # cut_all.append(cut)
            if MJD[n] not in [53012.15712866, 53012.17133498]:
                # print(fname)
                vel_all.append(vel)
                # corr_all.append(corr)
                flux_all.append(flux)
                MJD_all.append(MJD[n])
                flag_all.append(flag)

    ##plot BeSOS
    flag = "BeSOS"
    if flag in USE:
        lines = glob(direc + "Dropbox/Amanda/Data/BeSOS/2018/*.fits")
        MJD = np.zeros([len(lines)])
        JD = np.zeros([len(lines)])

        # FROM WEBSITE
        # All the spectra available are reduced and corrected with the heliocentric velocity

        for n in range(len(lines)):
            fname = lines[n]
            # read fits
            hdr_list = fits.open(fname)
            fits_data = hdr_list[0].data
            fits_header = hdr_list[0].header
            hdr_list.close()
            # fits_header.totextfile('{0}_{1}.txt'.format(flag, n))
            # read MJD
            MJD[n] = fits_header["MJD"]
            # corr = 0.
            # t = Time(MJD[n], format = 'mjd', scale='utc')
            # JD[n] = t.jd
            # corr, hjd = pyasl.helcorr(lon, lat, 2400., ra, dec, JD[n])
            lbd = fits_header["CRVAL1"] + fits_header["CDELT1"] * np.arange(
                len(fits_data)
            )
            vel, flux = spt.lineProf(lbd, fits_data, lbc=lbd0 * 10)
            # corr = fits_header['BVEL']
            # vel = vel + corr
            # cut = asas(vel, flux, line)
            # cut_all.append(cut)
            vel_all.append(vel)
            # corr_all.append(corr)
            flux_all.append(flux)
            MJD_all.append(MJD[n])
            flag_all.append(flag)

    # plot prof_nelson
    flag = "FEROS"
    if flag in USE:
        lines = glob(direc + "Dropbox/Amanda/Data/prof_nelson/*/*")
        MJD = np.zeros([len(lines)])
        JD = np.zeros([len(lines)])

        # lat = -29.257778
        # lon = -70.736667

        # FROM HEADER
        ###################################################################
        # HISTORY  'BARY_CORR'      ,'R*4 '   ,    1,    1,'5E14.7',' ',' '
        # HISTORY  -1.5192770E+01
        ###################################################################

        for n in range(len(lines)):
            fname = lines[n]
            # read fits
            hdr_list = fits.open(fname)
            fits_data = hdr_list[0].data
            fits_header = hdr_list[0].header
            hdr_list.close()

            # fits_header.totextfile('{0}_{1}.txt'.format(flag, n))
            # read MJD
            MJD[n] = fits_header["MJD-OBS"]
            t = Time(MJD[n], format="mjd", scale="utc")
            JD[n] = t.jd
            # corr = 0.
            lbd = fits_header["CRVAL1"] + fits_header["CDELT1"] * np.arange(
                len(fits_data)
            )
            vel, flux = spt.lineProf(lbd, fits_data, lbc=lbd0 * 10)
            # vel = vel + corr
            vel_all.append(vel)
            # corr_all.append(corr)
            flux_all.append(flux)
            MJD_all.append(MJD[n])
            flag_all.append(flag)

    barycorr_list = []
    flag = "NRES"
    if flag in USE:
        lines = glob(direc + "Dropbox/Amanda/Data/NRES/*fits*")
        MJD = np.zeros([len(lines)])
        JD = np.zeros([len(lines)])

        order = linesDict.line_names["Ha"][0]
        wl_list = []
        flx_list = []
        BJD_list = []
        for n in range(len(lines)):
            fname = lines[n]

            hdulist = fits.open(fname)
            headers = hdulist[0].header
            BJD_mid_exp = headers["BJD"]

            DAY_OBS = headers["DATE-OBS"]
            SITE = headers["SITE"]
            LONG1 = headers["LONG1"]
            LAT1 = headers["LAT1"]
            HT1 = headers["HT1"]

            SpecRaw = hdulist[1]
            SpecFlat = hdulist[2]
            SpecBlaze = hdulist[3]
            ThArRaw = hdulist[4]
            ThArFlat = hdulist[5]
            WaveSpec = hdulist[6]
            WaveThAr = hdulist[7]
            SpecXcor = hdulist[8]
            RVBlockFit = hdulist[9]
            # hdulist.close()

            wl_list.append(WaveSpec)
            flx_list.append(SpecFlat)
            # flx_list.append(SpecBlaze)
            BJD_list.append(BJD_mid_exp)
            # MJD_all.append(MJD[n])
            obs_loc = EarthLocation.from_geodetic(
                lat=LAT1 * u.deg, lon=LONG1 * u.deg, height=HT1 * u.m
            )
            sc = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)

        wl_list = np.array(wl_list)[np.argsort(BJD_list)]
        flx_list = np.array(flx_list)[np.argsort(BJD_list)]
        BJD_list = np.array(BJD_list)[np.argsort(BJD_list)]
        for i, x in enumerate(wl_list):
            vel, flux = spt.lineProf(
                wl_list[i].data[order], flx_list[i].data[order], lbc=lbd0
            )
            # hdulist.close()
            # vel = vel + corr
            # vel_all.append(vel)
            # corr_all.append(corr)
            flux_all.append(flux)
            t = Time(BJD_list[i], format="jd", scale="utc")

            MJD_all.append(t.mjd)
            flag_all.append(flag)
            # barycentric correction is more precise, but might be more difficult to apply
            barycorr = sc.radial_velocity_correction(
                kind="barycentric",
                obstime=Time(BJD_list[i], format="jd"),
                location=obs_loc,
            )
            barycorr = barycorr.value / 1000.0
            barycorr_list.append(barycorr)

            # heliocentric correction is easier, but less precise at a level of like 10 m/s
            # heliocorr = sc.radial_velocity_correction(kind='heliocentric',obstime=Time(BJD_list[i], format='jd'), location=obs_loc)
            # helcorr_list.append(heliocorr.value* au.value / (60*60*24)/ 1000.0)
            vl = vel + barycorr + vel * barycorr / (constants.c / 1000.0)
            vel_all.append(vl)
        hdulist.close()
    return MJD_all, vel_all, flux_all, flag_all


def get_sed():
    """
    Retrieves available wavelengths, fluxes and errors for SED

    USAGE: lbd, flux, dflux = get_data(obj,flag)
    """

    # data file name
    fname = "/home/amanda/Dropbox/Amanda/GRID/data/HD37795.sed.dat"  # aCol data

    # read data
    data = np.loadtxt(
        fname,
        dtype={
            "names": ("filter", "lbd", "flux", "dflux"),
            "formats": ("S20", np.float, np.float, np.float),
        },
        skiprows=10,
    )

    filt = data["filter"][:-1]
    filt = filt.astype(str)
    lbd_vosa = data["lbd"]
    flux_vosa = data["flux"]
    dflux_vosa = data["dflux"]

    lbd = lbd_vosa[:-1] * 1e-4
    flux = flux_vosa[:-1] * 1e4
    dflux = dflux_vosa[:-1] * 1e4

    # selection criteria
    flag_list = np.array(["iras", "akari/irc", "wise"])
    for flag in flag_list:
        iflux = np.array([flag.lower() in filt[i].lower() for i in range(len(filt))])
        if iflux.any():
            # data of interest
            lbd_tmp = lbd[iflux]
            flux_tmp = flux[iflux]
            # convert from vosa to catalogue nominal values
            lbd_tmp, flux_tmp = vosa2catvalues(lbd_tmp, flux_tmp, flag)
            # color correction
            flux_tmp = color_corr(lbd_tmp, flux_tmp, flag)
            lbd[iflux] = lbd_tmp
            flux[iflux] = flux_tmp

    # additional data
    fname = "/home/amanda/Dropbox/Amanda/GRID/data/alfCol.txt"
    data = np.loadtxt(
        fname,
        dtype={
            "names": ("lbd", "flux", "dflux", "source"),
            "formats": (np.float, np.float, np.float, "|S20"),
        },
    )
    # lbd = np.hstack([lbd, data['lbd']])
    # flux = np.hstack([flux, jy2cgs(1e-3*data['flux'], data['lbd'])])
    # dflux = np.hstack([dflux, jy2cgs(1e-3*data['dflux'], data['lbd'])])
    # filt = np.hstack([filt, data['source']])
    lbd = np.hstack([lbd, data["lbd"]])
    flux = np.hstack([flux, jy2cgs(1e-3 * data["flux"], data["lbd"])])
    dflux = np.hstack([dflux, jy2cgs(1e-3 * data["dflux"], data["lbd"])])
    filt = np.hstack([filt, data["source"]])

    # increasing order
    ordem = lbd.argsort()
    lbd = lbd[ordem]
    flux = flux[ordem]
    dflux = dflux[ordem]
    filt = filt[ordem]

    return lbd, flux, dflux


def plot_lines(line, togle):

    if line == "Ha":
        MJD_all, vel_all, flux_all, flag_all = get_halpha()
        sns.set_palette("husl", len(set(flag_all)))
        cor = sns.color_palette("husl", len(set(flag_all)))
    else:
        MJD_all, vel_all, flux_all, flag_all = get_lines(line)
        sns.set_palette("Set2", len(set(flag_all)))
        cor = sns.color_palette("Set2", len(set(flag_all)))

    selects = [[0, 15], [15, 30], [30, 45], [45, 60], [60, len(MJD_all)]]

    for k in range(len(selects)):
        fig = plt.figure(k, figsize=(5, 10))
        flag_names = list(OrderedDict.fromkeys(flag_all))
        ax1 = fig.add_axes([0.1, 0.1, 0.6, 0.75])

        flags = np.zeros(len(flag_all))
        for i in range(len(flag_all)):
            for j in range(len(flag_names)):
                if flag_all[i] == flag_names[j]:
                    flags[i] = j
        vel_new = []
        # n_MJD = len(MJD_all)
        for i_MJD in range(selects[k][0], selects[k][1]):
            if line == "Ha":
                radv = Ha_delta_v(vel_all[i_MJD], flux_all[i_MJD], "Ha")
                vel = vel_all[i_MJD] - radv
                # vel_new.append(vel)
                togle = 5.0
            elif line == "Hb" or "Hd" or "Hg":
                radv = delta_v(vel_all[i_MJD], flux_all[i_MJD], line)
                vel = vel_all[i_MJD] - radv
                # vel_new.append(vel)
                togle = 1.0
            else:
                vel = vel_all[i_MJD]

            # aux = (MJD_all[i_MJD] - np.array(MJD_all).min()) / (np.array(MJD_all).max() - np.array(MJD_all).min())
            # ax1.plot(vel_new[i_MJD] , flux_all[i_MJD] + togle * aux, color=cor[int(flags[i_MJD])], lw=0.8)
            ax1.plot(
                vel,
                flux_all[i_MJD] + 0.01 * i_MJD,
                color=cor[int(flags[i_MJD])],
                lw=0.8,
            )

            ax1.set_xlim(-800, 800)

        patches = [
            mpatches.Patch(color=cor[i], label=flag_names[i])
            for i in range(len(flag_names))
        ]
        ax1.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0, handles=patches)
        ax1.set_xlabel("$\mathrm{vel\,[km\,s^{-1}]}$")
        ax1.set_ylabel("$\mathrm{Relative\, flux}$")
        plt.savefig(
            "{0}_{1}_profiles.png".format(line, k), dpi=200, bbox_inches="tight"
        )
        # plt.show()
    return "Image {0}_profiles.png saved!".format(line)


def gen_plot_lines(line, MJD_all, vel_all, flux_all, flag_all):

    MJD_to_sort = np.array(MJD_all)
    sort = MJD_to_sort.argsort()
    flag_all = np.array(flag_all)[sort]
    vel_all = np.array(vel_all)[sort]
    flux_all = np.array(flux_all)[sort]

    if line == "Ha":
        # MJD_all,vel_all, flux_all, flag_all = get_halpha()
        sns.set_palette("husl", len(set(flag_all)))
        cor = sns.color_palette("husl", len(set(flag_all)))
    else:
        # MJD_all,vel_all, flux_all, flag_all = get_lines(line)
        sns.set_palette("Set2", len(set(flag_all)))
        cor = sns.color_palette("Set2", len(set(flag_all)))

    # selects = [[0,15], [15,30], [30,45], [45,60], [60, len(MJD_all)]]

    # for k in range(len(selects)):
    fig = plt.figure(1, figsize=(5, 10))
    flag_names = list(OrderedDict.fromkeys(flag_all))
    ax1 = fig.add_axes([0.1, 0.1, 0.6, 0.75])

    flags = np.zeros(len(flag_all))
    for i in range(len(flag_all)):
        for j in range(len(flag_names)):
            if flag_all[i] == flag_names[j]:
                flags[i] = j
    vel_new = []
    n_MJD = len(MJD_all)
    # for i_MJD in range(selects[k][0],selects[k][1]):
    for i_MJD in range(n_MJD):
        if line == "Ha":
            radv = Ha_delta_v(vel_all[i_MJD], flux_all[i_MJD], "Ha")
            vel = vel_all[i_MJD] - radv
            # vel_new.append(vel)
            togle = 5.0
        # elif line == 'Hb' or line =='Hd' or line =='Hg':
        #    radv = delta_v(vel_all[i_MJD], flux_all[i_MJD], line)
        #    vel = vel_all[i_MJD] - radv
        #    #vel_new.append(vel)
        #    togle = 1.
        else:
            vel = vel_all[i_MJD]

        aux = (MJD_all[i_MJD] - np.array(MJD_all).min()) / (
            np.array(MJD_all).max() - np.array(MJD_all).min()
        )
        # ax1.plot(vel , flux_all[i_MJD] + 6 * aux, color=cor[int(flags[i_MJD])], lw=0.8)
        ax1.plot(
            vel, flux_all[i_MJD] + 0.2 * i_MJD, color=cor[int(flags[i_MJD])], lw=0.8
        )

        ax1.set_xlim(-800, 800)

    patches = [
        mpatches.Patch(color=cor[i], label=flag_names[i])
        for i in range(len(flag_names))
    ]
    ax1.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0, handles=patches)
    ax1.set_xlabel("$\mathrm{vel\,[km\,s^{-1}]}$")
    ax1.set_ylabel("$\mathrm{Relative\, flux}$")
    plt.savefig("{0}_profiles.png".format(line), dpi=200, bbox_inches="tight")
    # plt.show()
    return "Image {0}_profiles.png saved!".format(line)
