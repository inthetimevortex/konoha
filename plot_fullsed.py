#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  plot_fullsed.py
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


'''
Reads sed2 files and combines them into one fullsed file
Plots flux and Ha
'''

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pyhdust.spectools as spt
import pyhdust as phd
from glob import glob
from scipy.interpolate import griddata
import numpy as np
import pyhdust.beatlas as bat
import pyhdust.input as inp
import pyhdust.phc as phc
from functools import partial
import seaborn as sns

def plot_fullsed(mm, modnum, dist=0.0):
    ################
    # Ploting the temp files
    ################

    tfiles, tlabels = phd.gentemplist(mm+'/mod'+modnum+'_b20.temp',tfrange=[15,20])
    phd.plottemp(tfiles, xax=1, tlabels=tlabels, fmt=['png'])

    ################
    # Ploting the populations
    ################
    # theta=0
    # pos = [ 1,  2,  3,  4,  5]
    # r, interp = phd.temp_interp(tfiles[-1], theta, pos=pos)
    #
    # for i in range(len(pos)):
    #     plt.plot(r, interp[:,i], label=pos[i])
    # #plt.plot(r, interp[:,1], label='1')
    # plt.legend()
    # plt.yscale('log')
    # plt.xlabel('R')
    # plt.ylabel('Pop.')
    # plt.savefig('populations1.png', dpi=150)
    # plt.close()
    #
    # pos = [6,  7,  8,  9, 10]
    # r, interp = phd.temp_interp(tfiles[-1], theta, pos=pos)
    #
    # for i in range(len(pos)):
    #     plt.plot(r, interp[:,i], label=pos[i])
    # #plt.plot(r, interp[:,1], label='1')
    # plt.legend()
    # plt.yscale('log')
    # plt.xlabel('R')
    # plt.ylabel('Pop.')
    # plt.savefig('populations2.png', dpi=150)
    # plt.close()
    #
    #
    # pos = [11, 12, 13, 14, 15, 16]
    # r, interp = phd.temp_interp(tfiles[-1], theta, pos=pos)
    #
    # for i in range(len(pos)):
    #     plt.plot(r, interp[:,i], label=pos[i])
    # #plt.plot(r, interp[:,1], label='1')
    # plt.legend()
    # plt.yscale('log')
    # plt.xlabel('R')
    # plt.ylabel('Pop.')
    # plt.savefig('populations3.png', dpi=150)
    # plt.close()
    #
    # pos = [17, 18, 19, 20, 21, 22, 23, 24]
    # r, interp = phd.temp_interp(tfiles[-1], theta, pos=pos)
    #
    # for i in range(len(pos)):
    #     plt.plot(r, interp[:,i], label=pos[i])
    # #plt.plot(r, interp[:,1], label='1')
    # plt.legend()
    # plt.yscale('log')
    # plt.xlabel('R')
    # plt.ylabel('Pop.')
    # plt.savefig('populations4.png', dpi=150)
    # plt.close()
    #################
    # To normalize the spectra, add the distance (paralax)
    #################

    if dist != 0:
        dist = 1.e3/dist
        norma = (10. / dist)**2
    else:
        norma = 1.


    # CHANGE TO YOUR MAIN DISK FILE (USUAL FORMAT IS mod01 PLn2.0 sig1.00e+11 h72.0 Rd40.0_Be M3.00 ob1.20 H0.10 Z0.014 bE Ell.txt)
    models = mm+'/mod'+modnum+'_b.txt'

    # CHANGE TO YOUR SOURCE FILE (USUAL FORMAT IS Be M3.00 ob1.20 H0.10 Z0.014 bE Ell.txt)
    #source = 'source/Be_M15.00_W0.87_t0.50_Z0.014_bE_Ell.txt'
    inp_file = glob(mm+'/*.inp')[0]
    print(inp_file)
    keyword = 'SOURCE'
    file = open(inp_file)
    for line in file:
        line.strip().split('/n')
        if line.startswith(keyword):
            source = line.split()[-1][1:-1]
    file.close()

    source = 'source/'+source+'.txt'

    # Reading Mass, Rpole and W from source file...

    f0 = open(source).read().splitlines()
    try:
        Ms = float(f0[3].split()[2]) * phc.Msun.cgs
        Rp = float(f0[4].split()[2]) * phc.Rsun.cgs
        W = float(f0[5].split()[2])
        L = float(f0[6].split()[2]) * phc.Lsun.cgs
    except:
        Ms = float(f0[7].split()[2]) * phc.Msun.cgs
        Rp = float(f0[8].split()[2]) * phc.Rsun.cgs
        W = float(f0[9].split()[2])
        L = float(f0[10].split()[2]) * phc.Lsun.cgs

    print('Stellar params')
    print('M = {:.2e}'.format(Ms))
    print('Rp = {:.2e}'.format(Rp))
    print('W = {:.2f}'.format(W))
    print('L = {:.2e}'.format(L))

    # Finding vrot
    oblat = W**2./2. + 1.
    Req = Rp * oblat
    vorb = np.sqrt(phc.G.cgs * Ms/Req)
    vrot = vorb * W* 1e-5 # cm/s to km/s
    vrot = np.array(vrot) # We add vrot to broaden the line profile!

    print('vrot = {:.2f}'.format(vrot))



    # This merges every simulation output it finds in the 'mod01' directory
    # for the 'models' file you chose
    phd.mergesed2(models, [vrot])


    # Reading the fullsed file...
    tab = phd.readfullsed2('fullsed/fullsed_mod'+modnum+'_b.sed2')
    lbd = tab[0, :, 2]
    nlbd = len(lbd)
    obs = tab[:, 0, 0] # = cos(incl)
    nobs = len(obs)
    incl = np.arccos(obs) * 180. / np.pi
    sed = tab[:, :, 3]
    pol = tab[:, :, 7]

    sns.set_palette("Set2", nobs)

    # HDUST's flux output is adimentional, as it is Flambda/int_0^inf Flambda dlambda
    # To get the flux in erg cm-2 s-1 um-1
    flux = sed * L/(4. * np.pi * (10 * phc.pc.cgs)**2)

    # Normalizing the flux
    flux = flux*norma

    ## PLOT SED
    for iobs in range(nobs):
        plt.plot(lbd, lbd * flux[iobs, :], ls='-', label=r'{:.2f} $\degree$'.format(incl[iobs]))

    plt.legend(title='Inclination angle', loc='best', fontsize='small', ncol=2)
    plt.xlabel(r'$\lambda \, [\mu m]$')
    plt.ylabel('$\lambda F_{\lambda}\, \mathrm{[erg\, s^{-1}\, cm^{-2}]}$')
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig('sed_'+mm+'.png', dpi=150)
    #plt.show()

    plt.close()

    ## PLOT Ha
    lbd0 = 0.65646 # Ha centre in vacuum
    for iobs in range(nobs-1):
        if (71 >incl[iobs] > 70):
            vel_ha, flux_ha = spt.lineProf(lbd, flux[iobs, :], lbc=lbd0) #wavelength to velocity
            plt.plot(vel_ha, flux_ha, label=r'{:.2f} $\degree$'.format(incl[iobs]))

    plt.legend(title='Inclination angle', loc='best', fontsize='small', ncol=2)
    plt.xlabel('Velocity [km/s]')
    plt.ylabel('Normalized flux')
    plt.savefig('halpha_'+mm+'.png', dpi=150)
    plt.close()
